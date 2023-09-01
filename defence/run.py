import argparse
import sys
import os

import torch

from VisionTransformer.defence.seg_train_dp import train_segmentation_model_dp

sys.path.append('../')
from trainer import predict
from dataloader import get_loader
from utils import MyConfig
from mymodel import ViT

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--ordinary_train', action="store_true", help='whether use mask')
parser.add_argument('--epochs', type=int, default=50, help='training epoch')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
parser.add_argument('--model_type', type=str, default='ViT_dpsgd', help='model name')
parser.add_argument('--mask_type', type=str, default='pub_fill', help='if fill')
parser.add_argument('--mix_up', action="store_true", help='use Mixup')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--defence", type=str, default='dpsgd', help="defence method")
opt = parser.parse_args()

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


config = MyConfig.MyConfig(path="config/cifar10/")
device = gpu_init(opt)
data_loader, data_size = get_loader("cifar10", config, is_target=True)
model = ViT.creat_VIT(config).to(device)
model, loss = train_segmentation_model_dp(data_size['train'], model, data_loader['train'], data_loader['val'], config.learning.epochs, config)