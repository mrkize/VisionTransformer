import sys
sys.path.append("..")
import argparse
from PIL import Image
import tim
from dataloader import get_loader
import torch
from utils import MyConfig
import os
import tim
import torch
import torch.nn as nn
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from masking_generator import JigsawPuzzleMaskedRegion


def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
    parser.add_argument('--n_class', type=int, default=10, help='')
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--istarget", action="store_false", help="defence method")
    parser.add_argument('--img_aug', type=str, default='pbf')
    parser.add_argument('--nums', type=int, default=0)
    opt = parser.parse_args()
    return opt




def gpu_init(opt):
    if config.learning.DDP:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py
save_path = { 'cifar10': "../defence_model/cifar10/DP/",
              'cifar100': "../defence_model/cifar100/DP/",
              'ImageNet100': "../defence_model/ImageNet100/DP/"}


dataset_class_dict = {
    "STL10": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "CelebA": 2,
    "Place365": 2,
    "Place100": 2,
    "Place50": 2,
    "Place20": 2,
    "ImageNet100": 100,
    "ImageNet10": 10,
    "Fmnist": 10
}


config_dict = { 'cifar10': "./config/cifar10/",
                'cifar100': "./config/cifar100/",
                'ImageNet100': "./config/ImageNet100/",
                }


def predict(model, dataloaders, dataset_sizes, device, is_img=False):
    model.to(device)
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(dataloaders):
        inputs, labels = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
        running_corrects += preds.eq(labels).sum().item()
    acc = 1.0 * running_corrects / dataset_sizes
    print("acc:{:.5f}".format(acc))
    return acc




def mask_train(model, loader, size, criterion, scheduler, optimizer, jigsaw_pullzer, config, opt):
    scalar = GradScaler()
    print("DATASET SIZE", size)
    since = time.time()
    for epoch in range(config.learning.epochs):
        if opt.local_rank == 0:
            print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                if config.learning.DDP:
                    loader[phase].sampler.set_epoch(epoch)
                model.train()
                scheduler.step()
            else:
                model.eval()
            running_corrects = 0
            for batch_idx, (data, target) in enumerate(loader[phase]):
                inputs, labels = data.to(opt.device), target.to(opt.device)
                unk_mask = None
                if phase == 'train':
                    if epoch >= config.mask.warmup_epoch:
                        inputs, unk_mask = jigsaw_pullzer(inputs)
                        unk_mask = torch.from_numpy(unk_mask).long().to(opt.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs, unk_mask=unk_mask)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            scalar.scale(loss).backward()
                            scalar.step(optimizer)
                            scalar.update()
                running_corrects += preds.eq(target.to(opt.device)).sum().item()
            epoch_acc = 1.0 * running_corrects / size[phase]
            if phase == 'train':
                if opt.local_rank == 0:
                    print('train acc:{:.3f}'.format(epoch_acc), end=' ')
            else:
                if opt.local_rank == 0:
                    print('val acc:{:.3f}'.format(epoch_acc))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")
    return model

def mask_train_model(opt, config, data_loader, data_size, is_target = True):
    model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
    model.load_state_dict(torch.load('../vit_small_patch16_224_{}.pth'.format(opt.n_class)))
    model = model.to(opt.device)
    if config.learning.DDP:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1], output_device=opt.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )                         
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.image_size,
                                              patch_size=config.patch.patch_size,
                                              num_masking_patches=int(opt.nums),
                                              min_num_patches=config.mask.min_num_patches,
                                              mask_type = opt.img_aug,
                                              pub_data_dir =config.path.public_path,
                                              channels = config.patch.channels,
                                              device = opt.device)
    model = mask_train(model, data_loader, data_size, criterion, base_scheduler, optimizer, jigsaw_pullzer, config, opt)
    pad = "" if is_target else "_shadow"
    if config.learning.DDP or config.learning.DP:
        if opt.local_rank == 0:
            torch.save(model.module.state_dict(), '../defence_model/{}/PEdrop/mask_{}{}.pth'.format(opt.dataset, opt.nums, pad))
    else:
        torch.save(model.state_dict(), '../defence_model/{}/PEdrop/mask_{}{}.pth'.format(opt.dataset, opt.nums, pad))
    return


opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]
# torch.random.manual_seed(3407)
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)
opt.device = gpu_init(opt)
# config.set_subkey('learning', 'epochs', 0)
print("PEdrop dataset : {} nums: {}".format(opt.dataset, opt.nums))
data_loader, data_size = get_loader(opt.dataset, config, is_target=opt.istarget)
mask_train_model(opt, config, data_loader, data_size, is_target=opt.istarget)