import copy
from random import random

import torchvision
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset,Subset,ConcatDataset
from torchvision import transforms, datasets
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image

image_size = 224
def public_data(root_dir, img_size=224, patch_size=16, length = 256, random_seed = 101, channels=3):
    np.random.seed(random_seed)
    im_to_patches = torch.nn.Unfold((patch_size, patch_size), stride=patch_size)
    if channels == 3 :
        Transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        Transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    dataset = datasets.ImageFolder(root_dir, Transform)
    idx = np.random.choice(np.arange(len(dataset)), size=length, replace=False)
    subset = torch.utils.data.Subset(dataset, idx)
    data = torch.stack([subset[i][0] for i in range(length)], dim=0)
    data = im_to_patches(data)
    return data


sp_num = 2


def data_split(dataset, num_class, nums_per_class, istarget, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    if istarget:
        idx_1 = np.array(idx)[:int(nums_per_class / 4)]
        idx_2 = np.array(idx)[int(nums_per_class / 4):int(nums_per_class / 4)*2]
    else:
        idx_1 = np.array(idx)[int(nums_per_class / 4)*2:int(nums_per_class / 4)*3]
        idx_2 = np.array(idx)[int(nums_per_class / 4)*3:]
    index_1 = []
    index_2 = []
    for i in range(num_class):
        index_1 += idx_1.tolist()
        idx_1 += nums_per_class
        index_2 += idx_2.tolist()
        idx_2 += nums_per_class
    set_1 = Subset(dataset, index_1)
    set_2 = Subset(dataset, index_2)
    return set_1, set_2


def data_split_2(dataset, num_class, nums_per_class, num, istarget, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    if istarget:
        idx_1 = np.array(idx)[:num]
    else:
        idx_1 = np.array(idx)[num:num*2]
    index_1 = []
    for i in range(num_class):
        index_1 += idx_1.tolist()
        idx_1 += nums_per_class
    set_1 = Subset(dataset, index_1)
    return set_1


def data_split_3(dataset, num_class, nums_per_class, istarget, seed=1011):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    if istarget:
        idx_1 = np.array(idx)[:int(nums_per_class / 2)]
    else:
        idx_1 = np.array(idx)[int(nums_per_class / 2):]
    index_1 = []
    for i in range(num_class):
        index_1 += idx_1.tolist()
        idx_1 += nums_per_class
    set_1 = Subset(dataset, index_1)
    return set_1


def dataset_split(dataset, num_class, nums_1, num_2, istarget, seed=101):
    np.random.seed(seed)
    nums_per_class = nums_1 + num_2
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    if istarget:
        idx_1 = np.array(idx)[:int(nums_1)]
        idx_2 = np.array(idx)[int(nums_1):nums_1+num_2]
    else:
        idx_1 = np.array(idx)[num_2:nums_1+num_2]
        idx_2 = np.array(idx)[:num_2]
    index_1 = []
    index_2 = []
    for i in range(num_class):
        index_1 += idx_1.tolist()
        idx_1 += nums_per_class
        index_2 += idx_2.tolist()
        idx_2 += nums_per_class
    set_1 = Subset(dataset, index_1)
    set_2 = Subset(dataset, index_2)
    return set_1, set_2

sp_num = 2
#seed = 101



class VITdataset(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True):
        self.Transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_dir = root_dir
        self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform) + datasets.ImageFolder(self.data_dir+'val', self.Transform)
        train_set_1, train_set_2 = data_split(datasets.ImageFolder(self.data_dir+'train', self.Transform), num_class, nums_per_class[0], is_target, random_seed)
        val_set_1, val_set_2 = data_split(datasets.ImageFolder(self.data_dir+'val', self.Transform), num_class, nums_per_class[1], is_target, random_seed)
        if split == 'train':
            self.dataset = train_set_1 + val_set_1
        else:
            self.dataset = train_set_2 + val_set_2


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class imageNet10(Dataset):
    def __init__(self, root_dir, split, num_class=10, nums_per_class=700, is_target = True, seed = 101):
        # Output of pretransform should be PIL images
        self.Transform = transforms.Compose([transforms.Resize([224,224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.root_dir = root_dir
        self.dataset = datasets.ImageFolder(self.root_dir, self.Transform)
        set1, set2 = dataset_split(self.dataset, num_class, 500, 800, is_target, seed)
        if split == 'train':
            self.dataset = set1
        else:
            self.dataset = set2
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class STL10_dataset(Dataset):
    def __init__(self, root_dir, split, is_target=True ,seed=1001):
        self.Transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.root_dir = root_dir
        self.train_set = datasets.STL10("../data", "train", transform=self.Transform, download=True)
        self.val_set = datasets.STL10("../data", "val", transform=self.Transform, download=True)
        set1, set2 = dataset_split(self.val_set, 10, 300, 500, True)

        if split == 'train':
            self.dataset = set1 + set2 if is_target else self.train_set + set1
        elif split == 'val':
            self.dataset = self.train_set if is_target else set2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class imageNet100(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True, set=0):
        self.Transform = transforms.Compose([transforms.Resize([224, 224]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        if split == 'train':
            self.dataset = data_split_2(datasets.ImageFolder(self.data_dir+'train', self.Transform), num_class, 1000, 150, is_target, random_seed)
        else:
            set1 = data_split_3(datasets.ImageFolder(self.data_dir+'test', self.Transform), num_class, 100, is_target, random_seed)
            set2 = data_split_3(datasets.ImageFolder(self.data_dir + 'val', self.Transform), num_class, 200, is_target,random_seed)
            self.dataset = set1+ set2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class imageNet(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True, set=0):
        self.Transform = transforms.Compose([transforms.Resize([224, 224]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        if split == 'train':
            self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform)
        else:
            self.dataset = datasets.ImageFolder(self.data_dir+'test', self.Transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]
    

class cinic_dataset(Dataset):

    def __init__(self, root_dir, split, num_class=10, nums_per_class=9000,random_seed = 1001, is_target=True ,seed=1001):
        self.Transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.root_dir = root_dir
        self.train_set = datasets.ImageFolder(self.root_dir+'train', self.Transform)
        self.val_set = datasets.ImageFolder(self.root_dir+'val', self.Transform)

        if split == 'train':
            self.dataset = self.train_set if is_target else self.val_set
        elif split == 'val':
            self.dataset = self.val_set if is_target else self.train_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class cinic_split_dataset(Dataset):

    def __init__(self, root_dir, split, num_class, nums_per_class, first = True, is_target=True ,seed=1001):
        self.Transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.root_dir = root_dir
        train_set = data_split_2(datasets.ImageFolder(self.root_dir+'train', self.Transform), num_class, 9000,1000, seed)
        # val_set_1, val_set_2 = data_split_2(datasets.ImageFolder(self.root_dir+'val', self.Transform), num_class, 9000,1000, seed)
        self.dataset = train_set
        # if first:
        #     if split == 'train':
        #         if is_target:
        #             self.dataset = train_set_1
        #         else:
        #             self.dataset = val_set_1
        #     elif split == 'val':
        #         if is_target:
        #             self.dataset = val_set_1
        #         else:
        #             self.dataset = train_set_1
        # else:
        #     if split == 'train':
        #         if is_target:
        #             self.dataset = train_set_2
        #         else:
        #             self.dataset = val_set_2
        #     elif split == 'val':
        #         if is_target:
        #             self.dataset = val_set_2
        #         else:
        #             self.dataset = train_set_2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


def set2loader(config, train_set, val_set):
    if config.learning.DDP:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config.learning.batch_size,
                                                   # shuffle=True,
                                                   num_workers=config.general.num_workers,
                                                   pin_memory=True,
                                                   sampler=DistributedSampler(train_set))
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=config.learning.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.general.num_workers,
                                                 pin_memory=True,
                                                 sampler=DistributedSampler(val_set))
    else:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config.learning.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.general.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=config.learning.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.general.num_workers,
                                                 pin_memory=True)
    data_loader = {'train': train_loader, 'val': val_loader}
    data_size = {"train": len(train_set), "val": len(val_set)}
    return data_loader, data_size

def get_loader(dataset, config, is_target, cif=True, set = 0):
    global image_size
    image_size = config.patch.image_size
    if 'cifar' in dataset:
        train_set = VITdataset(root_dir=config.path.data_path,
                               split='train',
                               num_class=config.patch.num_classes,
                               nums_per_class=config.patch.nums_per_class,
                               random_seed=config.general.seed,
                               is_target=is_target)
        val_set = VITdataset(root_dir=config.path.data_path,
                             split='val',
                             num_class=config.patch.num_classes,
                             nums_per_class=config.patch.nums_per_class,
                             random_seed=config.general.seed,
                             is_target=is_target)
        # train_set = Subset(train_set, range(10))
        # val_set = Subset(val_set, range(10))

    elif dataset == 'cinic':
        train_set = cinic_dataset(root_dir=config.path.data_path, split='train', is_target=is_target,
                                  seed=config.general.seed)[:1]
        val_set = cinic_dataset(root_dir=config.path.data_path, split='val', is_target=is_target,
                                seed=config.general.seed)[:1]
    elif dataset == 'ImageNet':
        train_set = imageNet(root_dir=config.path.data_path, split='train',
                             num_class=config.patch.num_classes,
                             nums_per_class=config.patch.nums_per_class,
                             is_target=is_target,
                             random_seed=config.general.seed,
                             set=set)
        val_set = imageNet(root_dir=config.path.data_path, split='val',
                           num_class=config.patch.num_classes,
                           nums_per_class=config.patch.nums_per_class,
                           is_target=is_target,
                           random_seed=config.general.seed,
                           set=set)
    elif dataset == 'ImageNet10':
        train_set = imageNet10(root_dir=config.path.data_path, split='train', is_target=is_target,
                               seed=config.general.seed)
        val_set = imageNet10(root_dir=config.path.data_path, split='val', is_target=is_target, seed=config.general.seed)
    elif dataset == 'ImageNet100':
        train_set = imageNet100(root_dir=config.path.data_path, split='train',
                             num_class=config.patch.num_classes,
                             nums_per_class=config.patch.nums_per_class,
                             is_target=is_target,
                             random_seed=config.general.seed,
                             set=set)
        val_set = imageNet100(root_dir=config.path.data_path, split='val',
                           num_class=config.patch.num_classes,
                           nums_per_class=config.patch.nums_per_class,
                           is_target=is_target,
                           random_seed=config.general.seed,
                           set=set)
        # train_set = Subset(train_set, range(1000))
        # val_set = Subset(val_set, range(1000))
    elif dataset == 'cinic10':
        train_set = cinic_split_dataset(root_dir=config.path.data_path, split='train', num_class =config.patch.num_classes,
                                        nums_per_class=config.patch.nums_per_class,first=cif,
                                        is_target=is_target,seed=config.general.seed)
        val_set = cinic_split_dataset(root_dir=config.path.data_path, split='val',num_class=config.patch.num_classes,
                                      nums_per_class=config.patch.nums_per_class,first=cif,
                                      is_target=is_target, seed=config.general.seed)

    else:
        train_set = mnist_dataset(root_dir=config.path.data_path,
                                  split='train',
                                  num_class=config.patch.num_classes,
                                  nums_per_class=config.patch.nums_per_class,
                                  is_target=is_target,
                                  seed=config.general.seed)
        val_set = mnist_dataset(root_dir=config.path.data_path,
                                split='val',
                                num_class=config.patch.num_classes,
                                nums_per_class=config.patch.nums_per_class,
                                is_target=is_target,
                                seed=config.general.seed)
    data_loader, data_size = set2loader(config, train_set, val_set)
    return data_loader, data_size

