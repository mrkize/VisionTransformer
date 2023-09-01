# from https://github.com/Lab41/cyphercat/blob/master/Defenses/Adversarial_Regularization.ipynb
import argparse
import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import warmup_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset

from trainer import predict
from dataloader import get_loader
from utils import MyConfig
from mymodel import ViT

sys.path.insert(0, '../Utils')

# import models
# from train import *
# from metrics import *
# from data_downloaders import *

print("Python: %s" % sys.version)
print("Pytorch: %s" % torch.__version__)

# determine device to run network on (runs on gpu if available)

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

batch_size = 128
lr_classification = 0.0001
lr_inference = 0.001
lr_attack = 0.001

# define series of transforms to pre process images
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
n_classes = 10


# load test set
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# helper function to unnormalize and plot image
def imshow(img):
    img = np.array(img)
    img = img / 2 + 0.5
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)

def model2device(model):
    if config.learning.DDP:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                                                          find_unused_parameters=True)  # 这句加载到多GPU上
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1])
    return model

# display sample from dataset
# imgs, labels = iter(D_loader).next()
# imshow(torchvision.utils.make_grid(imgs))


class inference_attack(nn.Module):
    def __init__(self, n_classes):
        super(inference_attack, self).__init__()

        self.n_classes = n_classes

        self.prediction_vector_block = nn.Sequential(
            nn.Linear(n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )

        self.label_block = nn.Sequential(
            nn.Linear(n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )

        self.common_block = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, prediction_vector, one_hot_label):
        prediction_block_out = self.prediction_vector_block(prediction_vector)
        label_block_out = self.label_block(one_hot_label)
        out = F.sigmoid(self.common_block(torch.cat((prediction_block_out, label_block_out), dim=1)))
        return out


def eval_target_net(model, loader, classes):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    dataset_sizes = 0
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(loader):
        inputs, labels = data.to(device), target.to(device)
        dataset_sizes += inputs.shape[0]
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
        running_corrects += preds.eq(labels).sum().item()
    acc = 1.0 * running_corrects / dataset_sizes
    return acc



def label_to_onehot(labels, num_classes=10):
    one_hot = torch.eye(num_classes).to(device)
    return one_hot[labels]


def adversarial_train(inference_net, classification_net, train_set, out_set, test_set,
                      infer_optim, infer_loss, class_optim, target_scheduler, class_loss, n_epochs, k, privacy_theta, data_size, verbose=False):
    losses = []
    scalar = torch.cuda.amp.GradScaler()
    scalar1 = torch.cuda.amp.GradScaler()
    inference_net.train()
    classification_net.train()
    train_iter = iter(train_set)
    out_iter = iter(out_set)
    train_iter2 = iter(train_set)
    for epoch in range(n_epochs):
        # if epoch%batch == 0:
        if opt.local_rank ==0 and epoch%batch == 0:
            print('epoch:{}/{}'.format(epoch+1,n_epochs))
            val_acc = predict(classification_net, out_set, data_size['val'], device)
            print('val acc:{:0.3f}'.format(val_acc))
        if epoch%batch == 0:
            target_scheduler.step()
        train_top = np.array([])
        out_top = np.array([])

        train_p = np.array([])
        out_p = np.array([])

        total_inference = 0
        total_correct_inference = 0

        inference_losses = np.array([])
        classification_losses = np.array([])
        for k_count in range(k):
            # train inference network
            try:
                train_imgs, train_lbls = next(train_iter)
            except StopIteration:
                train_iter = iter(train_set)
                train_imgs, train_lbls = next(train_iter)
            train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)
            try:
                out_imgs, out_lbls = next(out_iter)
            except StopIteration:
                out_iter = iter(out_set)
                out_imgs, out_lbls = next(out_iter)
            out_imgs, out_lbls = out_imgs.to(device), out_lbls.to(device)
            trainlbl = train_lbls
            outlbl = out_lbls
            if mixup:
                train_imgs, train_lbls = mixup_fn(train_imgs, train_lbls)
                out_imgs, out_lbls = mixup_fn(out_imgs, out_lbls)
            mini_batch_size = train_imgs.shape[0]

            train_lbl = torch.ones(train_imgs.shape[0]).to(device)
            out_lbl = torch.zeros(out_imgs.shape[0]).to(device)

            train_posteriors = F.softmax(classification_net(train_imgs), dim=1)
            out_posteriors = F.softmax(classification_net(out_imgs), dim=1)

            '''
            t_p = train_posteriors.cpu().detach().numpy().flatten()
            o_p = out_posteriors.cpu().detach().numpy().flatten()

            train_p = np.concatenate((train_p, t_p))
            out_p = np.concatenate((out_p, o_p))
            '''

            train_sort, _ = torch.sort(train_posteriors, descending=True)
            out_sort, _ = torch.sort(out_posteriors, descending=True)

            t_p = train_sort[:, :4].cpu().detach().numpy().flatten()
            o_p = out_sort[:, :4].cpu().detach().numpy().flatten()

            train_p = np.concatenate((train_p, t_p))
            out_p = np.concatenate((out_p, o_p))

            train_top = np.concatenate((train_top, train_sort[:, 0].cpu().detach().numpy()))
            out_top = np.concatenate((out_top, out_sort[:, 0].cpu().detach().numpy()))

            infer_optim.zero_grad()

            train_inference = torch.squeeze(inference_net(train_posteriors, label_to_onehot(trainlbl).to(device)))
            out_inference = torch.squeeze(inference_net(out_posteriors, label_to_onehot(outlbl).to(device)))

            total_inference += 2 * mini_batch_size
            total_correct_inference += torch.sum(train_inference > 0.5).item() + torch.sum(out_inference < 0.5).item()

            loss_train = infer_loss(train_inference, train_lbl)
            loss_out = infer_loss(out_inference, out_lbl)

            loss = privacy_theta * (loss_train + loss_out) / 2
            scalar.scale(loss).backward()
            scalar.step(infer_optim)
            scalar.update()

            # train classifiction network
        try:
            train_imgs, train_lbls = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_set)
            train_imgs, train_lbls = next(train_iter2)
        train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)
        trainlbl = train_lbls
        if mixup:
            train_imgs, train_lbls = mixup_fn(train_imgs, train_lbls)
        class_optim.zero_grad()

        outputs = classification_net(train_imgs)
        train_posteriors = F.softmax(outputs, dim=1)

        loss_classification = class_loss(outputs, train_lbls)
        train_lbl = torch.ones(train_imgs.shape[0]).to(device)

        train_inference = torch.squeeze(inference_net(train_posteriors, label_to_onehot(trainlbl).to(device)))
        loss_infer = infer_loss(train_inference, train_lbl)
        loss = loss_classification - privacy_theta * loss_infer

        scalar1.scale(loss).backward()
        scalar1.step(class_optim)
        scalar1.update()
        if epoch + 1 % batch == 0:
            val_acc = predict(classification_net, out_set, data_size['val'], device)
            if opt.local_rank ==0:
                print('val acc:{:0.3f}'.format(val_acc))


# tuples of (lambda, attack_accuracy, classification_train_accuracy, classification_test_accuracy)
adv_reg_metrics = []

def adv_reg(istarget, lambd, data ,mixup):
    adversarial_regularization_net = inference_attack(n_classes).to(device)
    adversarial_regularization_net = model2device(adversarial_regularization_net)
    # adversarial_regularization_net.apply(models.weights_init)

    adversarial_regularization_loss = nn.BCELoss()
    adversarial_regularization_optim = optim.Adam(adversarial_regularization_net.parameters(), lr=lr_inference)


    # net = resnet18.to(device)
    target_net = ViT.creat_VIT(config).to(device)
    target_net = model2device(target_net)
    # target_net.apply(models.weights_init)
    target_loss = nn.CrossEntropyLoss()
    if mixup:
        target_loss = SoftTargetCrossEntropy()
    # target_optim = optim.Adam(target_net.parameters(), lr=lr_classification)
    target_optim = torch.optim.Adam(target_net.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )
    # optimizer = optim.SGD(model_vit.parameters(),
    #                       lr=config.learning.learning_rate,
    #                       momentum=config.learning.momentum,
    #                       # weight_decay=config.learning.weight_decay
    #                       )

    # learning rate adopt
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(target_optim,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    target_scheduler = warmup_scheduler.GradualWarmupScheduler(target_optim, multiplier=1.,
                                                        total_epoch=config.learning.warmup_epoch,
                                                        after_scheduler=base_scheduler)

    data_loader, data_size = get_loader(data, config, is_target=istarget)
    print("train/val size:{}/{}".format(data_size['train'], data_size['val']))
    D_loader, D_prime_loader = data_loader['train'], data_loader['val']

    adversarial_train(adversarial_regularization_net, target_net, D_loader, D_prime_loader, D_prime_loader,
                      adversarial_regularization_optim, adversarial_regularization_loss, target_optim, target_scheduler,
                      target_loss, n_epochs, 7, lambd, data_size)

    pad = '' if istarget else '_shadow'
    torch.save(target_net.module.state_dict(), './Network/adv_label/{}{}_{}.pth'.format(data, pad, lambd))


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



n_epochs = 2750
batch = 55
config = MyConfig.MyConfig(path="config/ImageNet10/")
device = gpu_init(opt)
mixup_fn = Mixup(
    mixup_alpha=config.general.mixup_alpha,
    # cutmix_alpha=config.general.cutmix_alpha,
    num_classes=config.patch.num_classes)
# adv_reg(False, 10, 'cifar10')
# n_epochs = 5500
mixup = True
config.set_subkey("learning","batch_size",128)
config.set_subkey("learning","learning_rate",0.00075)
for lambd in [10]:
    adv_reg(True, lambd, 'ImageNet10', mixup)
    # adv_reg(False, lambd, 'ImageNet10', mixup)

# n_epochs = 23500
# batch = 235
# config = MyConfig.MyConfig(path="config/cifar10/")
# config.set_subkey("learning","learning_rate",0.00185)
# # device = gpu_init(opt)
# for lambd in [10]:
#     adv_reg(True, lambd, 'cifar10', mixup)
#     adv_reg(False, lambd, 'cifar10', mixup)



