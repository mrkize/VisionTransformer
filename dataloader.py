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

def data_spilt(dataset, num_class, nums_per_class, is_target=True, seed=101):
    idx = list(range(int(len(dataset)/num_class)))
    np.random.seed(seed)
    np.random.shuffle(idx)
    idx_train = np.array(idx)[:nums_per_class] if is_target else np.array(idx)[-nums_per_class:]
    idx_val = np.array(idx)[nums_per_class:] if is_target else np.array(idx)[:-nums_per_class]
    index_train = []
    index_val = []
    for i in range(num_class):
        index_train += idx_train.tolist()
        idx_train += nums_per_class
        index_val += idx_val.tolist()
        idx_val += nums_per_class
    train_set = Subset(dataset, index_train)
    val_set = Subset(dataset, index_val)
    return train_set, val_set



def dataset_split(dataset, num_class, nums_1, num_2, seed=101):
    np.random.seed(seed)
    nums_per_class = nums_1 + num_2
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    idx_1 = np.array(idx)[:int(nums_1)]
    idx_2 = np.array(idx)[int(nums_1):]
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

def dataset_split_2(dataset, num_class, nums_per_class, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    idx_1 = np.array(idx)[:int(nums_per_class / 2)]
    idx_2 = np.array(idx)[int(nums_per_class / 2):]
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


class VITdataset(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True):
        self.Transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_dir = root_dir
        self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform) + datasets.ImageFolder(self.data_dir+'val', self.Transform)
        train_set_1, train_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'train', self.Transform), num_class, nums_per_class[0], random_seed)
        val_set_1, val_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'val', self.Transform), num_class, nums_per_class[1], random_seed)
        if split == 'train':
            if is_target:
                self.dataset = train_set_1 + val_set_1
            else:
                self.dataset = train_set_2 + val_set_2
        elif split == 'val':
            if is_target:
                self.dataset = train_set_2 + val_set_2
            else:
                self.dataset = train_set_1 + val_set_1
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
        set1, set2 = data_spilt(self.dataset, num_class, nums_per_class, is_target, seed)
        if split == 'train':
            self.dataset = set1
        else:
            self.dataset = set2
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


class imageNet(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True):
        self.Transform = transforms.Compose([transforms.Resize([224, 224]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        train_set_1, train_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'train', self.Transform), num_class, nums_per_class[0], random_seed)
        val_set_1, val_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'val', self.Transform), num_class, nums_per_class[1], random_seed)
        test_set_1, test_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir + 'test', self.Transform), num_class, nums_per_class[2], random_seed)
        if split == 'train':
            if is_target:
                self.dataset = train_set_1 + val_set_1 + test_set_1
            else:
                self.dataset = train_set_2 + val_set_2 + test_set_2
        elif split == 'val':
            if is_target:
                self.dataset = train_set_2 + val_set_2 + test_set_2
            else:
                self.dataset = train_set_1 + val_set_1 + test_set_1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

class imageNet100(Dataset):
    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True, set=1):
        self.Transform = transforms.Compose([transforms.Resize([224, 224]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        train_set_1, train_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'train', self.Transform), num_class, nums_per_class[0], random_seed)
        val_set_1, val_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir+'val', self.Transform), num_class, nums_per_class[1], random_seed)
        test_set_1, test_set_2 = dataset_split_2(datasets.ImageFolder(self.data_dir + 'test', self.Transform), num_class, nums_per_class[2], random_seed)
        if split == 'train':
            if is_target:
                self.dataset = train_set_1 + val_set_1 + test_set_1
            else:
                self.dataset = train_set_2 + val_set_2 + test_set_2
        elif split == 'val':
            if is_target:
                self.dataset = train_set_2 + val_set_2 + test_set_2
            else:
                self.dataset = train_set_1 + val_set_1 + test_set_1

        if set == 1:
            if split == 'train':
                self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform)
            elif split == 'val':
                self.dataset = datasets.ImageFolder(self.data_dir+'test', self.Transform)
        elif set == 2:
            self.dataset = datasets.ImageFolder(self.data_dir + 'train', self.Transform)
            set1, set2 = dataset_split(self.dataset, 100, 900, 100)
            if split == 'train':
                self.dataset = datasets.ImageFolder(self.data_dir+'test', self.Transform)
                self.dataset = ConcatDataset([self.dataset,set1])
            elif split == 'val':
                self.dataset = set2

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

class mnist_dataset(Dataset):

    def __init__(self, root_dir, split, num_class, nums_per_class, is_target=True,seed = 1001):
        self.Transform = transforms.Compose([# transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])]
                                            )
        # Output of pretransform should be PIL images
        self.root_dir = root_dir
        self.train_set = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=self.Transform, download = True)
        self.val_set = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=self.Transform, download = True)

        train_set_1, train_set_2 = dataset_split_2(self.train_set, num_class, nums_per_class[0], seed=seed)
        val_set_1, val_set_2 = dataset_split_2(self.val_set, num_class, nums_per_class[1], seed=seed)

        if split == 'train':
            if is_target:
                self.dataset = train_set_1 + val_set_1
            else:
                self.dataset = train_set_2 + val_set_2
        elif split == 'val':
            if is_target:
                self.dataset = train_set_2 + val_set_2
            else:
                self.dataset = train_set_1 + val_set_1
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
        train_set_1, train_set_2 = dataset_split_2(datasets.ImageFolder(self.root_dir+'train', self.Transform), num_class, nums_per_class[0], seed)
        val_set_1, val_set_2 = dataset_split_2(datasets.ImageFolder(self.root_dir+'val', self.Transform), num_class, nums_per_class[1], seed)
        if first:
            if split == 'train':
                if is_target:
                    self.dataset = train_set_1
                else:
                    self.dataset = val_set_1
            elif split == 'val':
                if is_target:
                    self.dataset = val_set_1
                else:
                    self.dataset = train_set_1
        else:
            if split == 'train':
                if is_target:
                    self.dataset = train_set_2
                else:
                    self.dataset = val_set_2
            elif split == 'val':
                if is_target:
                    self.dataset = val_set_2
                else:
                    self.dataset = train_set_2

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

    elif dataset == 'cinic10':
        train_set = cinic_dataset(root_dir=config.path.data_path, split='train', is_target=is_target,
                                  seed=config.general.seed)
        val_set = cinic_dataset(root_dir=config.path.data_path, split='val', is_target=is_target,
                                seed=config.general.seed)
    elif dataset == 'ImageNet':
        train_set = imageNet(root_dir=config.path.data_path, split='train',
                             num_class=config.patch.num_classes,
                             nums_per_class=config.patch.nums_per_class,
                             is_target=is_target,
                             random_seed=config.general.seed)
        val_set = imageNet(root_dir=config.path.data_path, split='val',
                           num_class=config.patch.num_classes,
                           nums_per_class=config.patch.nums_per_class,
                           is_target=is_target,
                           random_seed=config.general.seed)
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
    elif dataset == 'cinic':
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

