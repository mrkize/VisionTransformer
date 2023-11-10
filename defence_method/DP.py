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


def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
    parser.add_argument('--n_class', type=int, default=10, help='')
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--istarget", action="store_false", help="defence method")
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--iterations', type=int, default=14000)
    parser.add_argument('--l2_norm_clip', type=float, default=1.)
    parser.add_argument('--l2-penalty', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--microbatch-size', type=int, default=1)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--noise_multiplier', type=float, default=1.)
    parser.add_argument('--epsilon', type=float, default=1.1)
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
    return acc




def mask_train(model, loader, size, criterion, scheduler, optimizer, mixup_fn, config, opt):
    scalar = GradScaler()
    print("DATASET SIZE", size)
    val_criterion = nn.CrossEntropyLoss()
    since = time.time()
    ret_value = np.zeros((4, config.learning.epochs))
    for epoch in range(config.learning.epochs):
        if opt.local_rank == 0:
            print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
    # Each epoch has a training and validation phase
        for phase in ['train']:
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
                optimizer.zero_grad()
                if epoch >= 20:
                    for sample in zip(inputs, labels):
                        with autocast():
                            x,y = sample
                            optimizer.zero_grad()
                            outputs = model(x[None], unk_mask=None)
                            loss = criterion(outputs, y[None])
                        loss.backward()
                        for param in model.parameters():
                            torch.nn.utils.clip_grad_norm_(param.grad, max_norm=opt.l2_norm_clip)  # in-place

                    for param in model.parameters():
                        noise = torch.empty_like(param.grad).data.normal_(0, opt.noise_multiplier *opt.l2_norm_clip).to(opt.device)
                        param.grad = (param.grad + noise)/config.learning.batch_size
                        optimizer.step()
                else:
                    with autocast():
                        outputs = model(inputs, unk_mask=unk_mask)
                        loss = criterion(outputs, labels)
                        
                
                with autocast():
                    scalar.scale(loss).backward()
                    scalar.step(optimizer)
                    scalar.update()
                # running_corrects += preds.eq(target.to(opt.device)).sum().item()
            # epoch_acc = 1.0 * running_corrects / size[phase]
            # if phase == 'train':
            #     if opt.local_rank == 0:
            #         print('train acc:{:.3f}'.format(epoch_acc), end=' ')
            # else:
            #     if opt.local_rank == 0:
            #         print('val acc:{:.3f}'.format(epoch_acc))
        print("val{} acc: {:.5f}".format(opt.local_rank, predict(model, loader["val"], size["val"], opt.device)))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def mask_train_model(opt, config, data_loader, data_size, is_target = True):
    model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
    model.load_state_dict(torch.load('../vit_small_patch16_224_{}.pth'.format(opt.n_class)))
    model = model.to(opt.device)
    if config.learning.DDP:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1], output_device=opt.device)
    mixup_fn = None
    # mixup_fn = Mixup(
    #     mixup_alpha=config.general.mixup_alpha,
    #     label_smoothing=opt.ls,
    #     num_classes=config.patch.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )                         
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    model = mask_train(model, data_loader, data_size, criterion, base_scheduler, optimizer, mixup_fn, config, opt)
    pad = "" if is_target else "_shadow"
    if config.learning.DDP or config.learning.DP:
        if opt.local_rank == 0:
            torch.save(model.module.state_dict(), '{}DP_{}{}.pth'.format(save_path[opt.dataset], opt.noise_multiplier, pad))
    else:
        torch.save(model.state_dict(), '{}DP_{}{}.pth'.format(save_path[opt.dataset], opt.noise_multiplier, pad))
    return


opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]
# torch.random.manual_seed(3407)
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)
opt.device = gpu_init(opt)
# config.set_subkey('learning', 'epochs', 0)
print("DP dataset : {}  para: {}".format(opt.dataset, opt.noise_multiplier))
data_loader, data_size = get_loader(opt.dataset, config, is_target=opt.istarget)
mask_train_model(opt, config, data_loader, data_size, is_target=opt.istarget)