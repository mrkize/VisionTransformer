import tim
import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
import warmup_scheduler
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import time
from masking_generator import JigsawPuzzleMaskedRegion
from mymodel import ViT, ViT_mask, ViT_ape, ViT_mask_avg, Swin, Swin_mask_avg, Swin_mask


def rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)    #(128,197,197) -> (128, 38809)
            _, indices = flat[:,1:].topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices + torch.ones(indices.shape).to(indices.device)
            indices = indices.to(torch.int64)
            flat.scatter_(1, indices, 0)

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2

            a = a / a.sum(dim=-1)[..., None]

            result = torch.matmul(a, result.to(attention_heads_fused.device))

    return result[:, 0, 1:]

def calculate_entropy(heatmap, epsilon = 1e-8):
    heatmap = heatmap + epsilon

    probabilities = heatmap / heatmap.sum(-1)[...,None]
    entropy = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)
    # torch.nonzero(torch.isnan(probabilities))
    return entropy

def predict(model, dataloaders, dataset_sizes, device, is_img=False):
    model.to(device)
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    attn_metric = []
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(dataloaders):
        inputs, labels = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
            attn = []
            for block in model.blocks:
                attn.append(block.attn.attn_value)
            cls_attn = rollout(attn)
            cls_attn = cls_attn/cls_attn.max(-1)[0][...,None]

            attn_metric.append(calculate_entropy(cls_attn).sum(0))
        running_corrects += preds.eq(labels).sum().item()
    acc = 1.0 * running_corrects / dataset_sizes
    print("acc:{:.5f}".format(acc))
    print("attn entropy:{:.5f}".format(sum(attn_metric)/dataset_sizes))
    return acc, sum(attn_metric)/dataset_sizes



def switch_fused_attn(model):
    for name, block in model.blocks.named_children():
        block.attn.fused_attn =True
    return model


def get_mask(nums, size):
    mask = np.zeros([1,size],dtype=int)
    idx = list(range(1, size))
    np.random.shuffle(idx)
    mask[:,idx[:nums]] = 1
    return mask


def mask_train(model, loader, size, criterion, scheduler, optimizer, mixup_fn, jigsaw_pullzer, config, opt):
    print("DATASET SIZE", size)
    val_criterion = nn.CrossEntropyLoss()
    since = time.time()
    ret_value = np.zeros((4, config.learning.epochs))
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    #print('-' * 10)
    # print(config.learning.epochs)
    switch_fused_attn(model.module if config.learning.DP else model)
    for epoch in range(config.learning.epochs):
        if opt.local_rank == 0:
            print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
    # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if config.learning.DDP:
                    loader[phase].sampler.set_epoch(epoch)
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for batch_idx, (data, target) in enumerate(loader[phase]):
                inputs, labels = data.to(opt.device), target.to(opt.device)
                unk_mask = None
                if phase == 'train':
                    if mixup_fn is not None:
                        inputs, labels = mixup_fn(inputs, labels)
                    if epoch >= config.mask.warmup_epoch:
                        unk_mask = get_mask(opt.nums, 197)
                        if unk_mask is not None:
                            unk_mask = torch.from_numpy(unk_mask).long().to(opt.device)
                else:
                    criterion = val_criterion
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, unk_mask=unk_mask)
                    _, preds = torch.max(outputs, 1)
                    # labels = torch.squeeze(labels)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(target.to(opt.device)).sum().item()
            epoch_loss = running_loss / size[phase]
            epoch_acc = 1.0 * running_corrects / size[phase]
            if phase == 'train':
                if opt.local_rank == 0:
                    print('train acc:{:.3f}'.format(epoch_acc), end=' ')
                ret_value[0][epoch] = epoch_loss
                ret_value[1][epoch] = epoch_acc

            else:
                if opt.local_rank == 0:
                    print('val acc:{:.3f}'.format(epoch_acc))
                ret_value[2][epoch] = epoch_loss
                ret_value[3][epoch] = epoch_acc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, ret_value

def mask_train_model(model_type, opt, config, data_loader, data_size, is_target = True):
    config.set_subkey('general', 'type', model_type)
    config.set_subkey('mask', 'mask_ratio', opt.mask_ratio)
    if 'ape' in model_type and 'init' in model_type:
        model = ViT_ape.creat_VIT(config)
    elif 'init' in model_type:
        model = ViT_mask.creat_VIT(config)
    else:
        model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
        model.load_state_dict(torch.load('./vit_small_patch16_224_{}.pth'.format(config.patch.num_classes)))
        model.ape = opt.ape
    model = model.to(opt.device)
    model.train_method = opt.pe_aug
    if config.learning.DDP:
        # config.set_subkey('learning', 'learning_rate', config.learning.learning_rate)
        # print('lr:',config.learning.learning_rate)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1], output_device=opt.device)

    criterion = nn.CrossEntropyLoss()
    mixup_fn = None
    if opt.defence == "label_smoothing":
        mixup_fn = Mixup(
            mixup_alpha=config.general.mixup_alpha,
            # cutmix_alpha=config.general.cutmix_alpha,
            num_classes=config.patch.num_classes)
        criterion = timm.loss.SoftTargetCrossEntropy()
    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.image_size,
                                              patch_size=config.patch.patch_size,
                                              num_masking_patches=int(opt.nums),
                                              min_num_patches=config.mask.min_num_patches,
                                              mask_type = opt.img_aug,
                                              pub_data_dir =config.path.public_path,
                                              channels = config.patch.channels,
                                              device = opt.device)
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    model, ret = mask_train(model, data_loader, data_size, criterion, base_scheduler, optimizer, mixup_fn, jigsaw_pullzer, config, opt)
    pad = "" if is_target else "_shadow"
    if config.learning.DDP or config.learning.DP:
        if opt.local_rank == 0:
            torch.save(model.module.state_dict(),config.path.model_path + model_type + "_{:.3f}{}".format(opt.mask_ratio, pad) + '.pth')
    else:
        torch.save(model.state_dict(),config.path.model_path + model_type + "_{:.3f}{}".format(opt.mask_ratio, pad) + '.pth')

    np.save(config.path.result_path + model_type + "_{:.3f}{}".format(opt.mask_ratio, pad), ret)
    return

