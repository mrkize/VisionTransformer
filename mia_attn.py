import argparse
import os
import time

import numpy as np
import torch.nn.functional as F
import torch

from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import TensorDataset
from tqdm import tqdm
import tim
from dataloader import get_loader
from utils import MyConfig
from vit_rollout import rollout, VITAttentionRollout
from torch import nn


config_dict = {'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'Fmnist': "config/fashion-mnist/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'cifar10-8': "config/cifar10-8*8/",
                'ImageNet10-8': "config/ImageNet10-8*8/"
                }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--dataset', type=str, default='cifar100',help='attack dataset')
    parser.add_argument('--model', type=str, default="ViT_mask_0.051.pth")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--noise_repeat', type=int, default=1)
    parser.add_argument('--head_fusion', type=str, default='max')
    parser.add_argument('--discard_ratio', type=float, default=0.9)
    parser.add_argument('--addnoise', action='store_true', default=False)
    parser.add_argument('--atk_output', action='store_true', default=True)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args


def add_gaussian_noise_to_patches(image_tensor, patch_size=16, num_patches=196, mean=0, stddev=0.25):
    batch_size, channels, height, width = image_tensor.size()
    unfolded = torch.nn.functional.unfold(image_tensor, patch_size, stride=patch_size)
    selected_patches = torch.randperm(unfolded.size(2))[:num_patches]
    noise = torch.randn(batch_size, unfolded.shape[1], num_patches) * stddev + mean
    noise = noise.cuda()
    unfolded = unfolded.cuda()
    unfolded[:, :, selected_patches] += noise
    unfolded = unfolded.clamp(0, 1)
    image_tensor = torch.nn.functional.fold(unfolded, (height, width), patch_size, stride=patch_size)
    return image_tensor

def calculate_entropy(heatmap, epsilon = 1e-8):
    heatmap = heatmap + epsilon
    probabilities = heatmap / heatmap.sum(-1)[...,None]
    entropy = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)
    return entropy


def init_config_model_attn(args):
    config = MyConfig.MyConfig(path=config_dict[args.dataset])
    target_model = tim.create_model('vit_small_patch16_224', num_classes=args.num_class)
    target_model.load_state_dict(torch.load(config.path.model_path + args.model))
    shadow_model = tim.create_model('vit_small_patch16_224', num_classes=args.num_class)
    shadow_model.load_state_dict(torch.load(config.path.model_path + args.model[:-4]+"_shadow.pth"))

    target_rollout = VITAttentionRollout(target_model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
    shadow_rollout = VITAttentionRollout(shadow_model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
    return config, target_model, shadow_model, target_rollout, shadow_rollout


def get_attn(model, dataloaders, attention_rollout, noise=False, out_atk=False):
    model.to(device)
    model.eval()
    attn_metric = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            if noise:
                inputs = add_gaussian_noise_to_patches(inputs)
            if out_atk:
                mask = model.forward_features(inputs, None)[:,0,:]
            else:
                mask = attention_rollout(inputs)
            attn_metric.append(mask.cpu())
        attn = torch.cat(attn_metric, dim=0)
    return attn


def gpu_init(config, opt):
    config.set_subkey('learning', 'DP', False)
    config.set_subkey('learning', 'DDP', False)
    if config.learning.DDP:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def find_best_threshold(data, labels):
    # data = torch.tensor(data, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int32)
    sorted_data, indices = torch.sort(data)
    best_accuracy = 0.0
    best_threshold = 0.0
    for i in range(len(sorted_data) - 1):
        threshold = (sorted_data[i] + sorted_data[i+1]) / 2.0
        predicted_labels = (data <= threshold).to(torch.int32)
        accuracy = (predicted_labels == labels).to(torch.float32).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def calculate_accuracy(data, labels, threshold):
    predicted_labels = (data <= threshold).to(torch.int32)
    accuracy = (predicted_labels == labels).to(torch.float32).mean()
    return accuracy


def get_data(model, attention_rollout, loader, out_atk = False):
    attn_orain_train = get_attn(model, loader["train"], attention_rollout, noise=False, out_atk=out_atk)
    attn_orain_val = get_attn(model, loader["val"], attention_rollout, noise=False, out_atk=out_atk)
    # entropy_train = calculate_entropy(attn_orain_train)
    # entropy_val = calculate_entropy(attn_orain_val)
    cross = SoftTargetCrossEntropy()
    cos_train = []
    cos_val = []
    cross_train = []
    cross_val = []
    for i in range(args.noise_repeat):
        attn_train= get_attn(model, loader["train"], attention_rollout, noise=True, out_atk=out_atk)
        cos_sim_train = torch.cosine_similarity(attn_orain_train,attn_train,dim=1)
        # cross_train.append(cross(attn_orain_train, attn_train))
        cos_train.append(cos_sim_train)

        attn_val= get_attn(model, loader["val"], attention_rollout, noise=True, out_atk=out_atk)
        cos_sim_val = torch.cosine_similarity(attn_orain_val,attn_val, dim=1)
        # cross_val.append(cross(attn_orain_val, attn_val))
        cos_val.append(cos_sim_val)

    cos_train_res = sum(cos_train) / args.noise_repeat
    cos_val_res = sum(cos_val) / args.noise_repeat
    # cross_train_res = sum(cross_train) / args.noise_repeat
    # cross_val_res = sum(cross_val) / args.noise_repeat

    # cos_train_list = torch.stack([entropy_train, cos_train_res], dim=-1)
    # cos_val_list = torch.stack([entropy_val, cos_val_res], dim=-1)
    data = torch.cat([cos_train_res, cos_val_res])
    target = torch.cat([torch.ones(cos_train_res.shape[0]), torch.zeros(cos_val_res.shape[0])])
    # dataset = TensorDataset(data, target)
    # return dataset
    return data, target


def get_opt(model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5)
    return criterion, optimizer, scheduler


def train(model, loader, size, criterion, optimizer, scheduler, epochs):
    since = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        scheduler.step()
        running_corrects = 0
        for batch_idx, (data, target) in enumerate(loader):
            inputs, labels = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_corrects += preds.eq(target.to(device)).sum().item()
        epoch_acc = 1.0 * running_corrects / size
    time_elapsed = time.time() - since
    print("Training used {:.5f}s.".format(time_elapsed))
    print("Train acc {:.5f}.".format(epoch_acc))
    return model


def predict(model, loader, size):
    model.eval()
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(loader):
        inputs, labels = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += preds.eq(target.to(device)).sum().item()
    epoch_acc = 1.0 * running_corrects / size
    print("Val acc {:.5f}.".format(epoch_acc))
    return model

torch.random.manual_seed(1333)
args = get_args()
config, target_model, shadow_model, target_rollout, shadow_rollout = init_config_model_attn(args)
# target_model=torch.nn.DataParallel(target_model, device_ids=[0,1])
# shadow_model=torch.nn.DataParallel(shadow_model, device_ids=[0,1])
device = gpu_init(config, args)
target_model, shadow_model = target_model.to(device), shadow_model.to(device)
config.set_subkey('learning', 'DP', False)
config.set_subkey('learning', 'DDP', False)
# config.set_subkey('learning', 'batch_size', 1)
sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
sha_dataset, sha_target = get_data(shadow_model, shadow_rollout, sha_loader, args.atk_output)
sha_dataset, sha_target = sha_dataset.to(device), sha_target.to(device)
thr = find_best_threshold(sha_dataset, sha_target)
tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
tar_dataset, tar_target = get_data(target_model, target_rollout, tar_loader, args.atk_output)
tar_dataset, tar_target = tar_dataset.to(device), tar_target.to(device)
val_acc = calculate_accuracy(tar_dataset, tar_target, thr)

print("{}".format(args.model))
print("Attack acc:{:.4f}".format(val_acc))
