import argparse
import sys
import os

import torch
import argparse
import sys
import os

import torch

from defence import seg_train_dp, seg_train_crop
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
sys.path.append('../')
from trainer import predict
from dataloader import get_loader
from utils import MyConfig
from mymodel import ViT

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--multigpu', action="store_true", help='whether use mask')
parser.add_argument('--epochs', type=int, default=50, help='training epoch')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
parser.add_argument('--model_type', type=str, default='ViT_dpsgd', help='model name')
parser.add_argument('--mask_type', type=str, default='pub_fill', help='if fill')
parser.add_argument('--mix_up', action="store_true", help='use Mixup')
parser.add_argument('--modelpath', type=str, default='Network/adv_label/', help='mask ratio')
parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--defence", type=str, default='dpsgd', help="defence method")
opt = parser.parse_args()

config_dict = { 'ViT': {
                'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'Fmnist': "config/fashion-mnist/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'cifar10-8': "config/cifar10-8*8/",
                'ImageNet10-8': "config/ImageNet10-8*8/",
                },
                'Swin': {
                'cifar10': "config/Swin-cifar10/",
                'cifar100': "config/Swin-cifar100/",
                'ImageNet10': "config/Swin-ImageNet10/",
                'ImageNet100': "config/Swin-ImageNet100/"
                }
}

def gpu_init(opt, config):
    if config.learning.DDP or opt.multigpu:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py

def train_dpsgd(config, opt, istarget):

    data_loader, data_size = get_loader(opt.dataset, config, is_target=istarget)
    model = ViT.creat_VIT(config).to(opt.device)
    if opt.multigpu:
        model = DPDDP(model)
    model, loss = seg_train_dp.train_segmentation_model_dp(data_size['train'], model, data_loader['train'], data_loader['val'], config.learning.epochs, config, opt)
    pad = '' if istarget else '_shadow'
    torch.save(model, opt.modelpath + '{}_dpsgd{}.pth'.format(opt.dataset, pad))

def train_crop(config, opt, istarget):

    data_loader, data_size = get_loader(opt.dataset, config, is_target=istarget)
    model = ViT.creat_VIT(config).to(opt.device)
    if opt.multigpu:
        model = DPDDP(model)
    model, loss = seg_train_crop.train_segmentation_model_crop(model, data_loader['train'], data_loader['val'], config.learning.epochs, config, opt)
    pad = '' if istarget else '_shadow'
    torch.save(model, opt.modelpath + '{}_crop{}.pth'.format(opt.dataset, pad))


if opt.defence == 'dpsgd':
    config_path = config_dict['Swin'][opt.dataset] if 'Swin' in opt.model_type else config_dict['ViT'][opt.dataset]
    config = MyConfig.MyConfig(path=config_path)
    config.set_subkey('learning', 'DDP', False)
    opt.device = gpu_init(opt, config)
    for flag in [True, False]:
        train_dpsgd(config, opt, flag)
else:
    config_path = config_dict['Swin'][opt.dataset] if 'Swin' in opt.model_type else config_dict['ViT'][opt.dataset]
    config = MyConfig.MyConfig(path=config_path)
    opt.device = gpu_init(opt, config)
    for flag in [True, False]:
        train_crop(config, opt, flag)