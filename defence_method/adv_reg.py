import sys

import warmup_scheduler

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
import torch.nn.functional as F

def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
    parser.add_argument('--n_class', type=int, default=10, help='')
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--istarget", action="store_false")
    opt = parser.parse_args()
    return opt


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


def model2device(model):
    if config.learning.DDP:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                                                          find_unused_parameters=True)  # 这句加载到多GPU上
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1])
    return model


def gpu_init(opt):
    if config.learning.DDP:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


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


def label_to_onehot(labels, opt):
    one_hot = torch.eye(opt.n_class).to(device)
    return one_hot[labels]


def adversarial_train(inference_net, classification_net, train_set, out_set, test_set,
                      infer_optim, infer_loss, class_optim, target_scheduler, class_loss, n_epochs, k, privacy_theta, data_size, verbose=False):
    losses = []
    batch = len(train_set)
    scalar = torch.cuda.amp.GradScaler()
    scalar1 = torch.cuda.amp.GradScaler()
    inference_net.train()
    classification_net.train()
    train_iter = iter(train_set)
    out_iter = iter(out_set)
    train_iter2 = iter(train_set)
    for epoch in range(n_epochs):
        if opt.local_rank == 0:
            print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
        target_scheduler.step()
        running_corrects = 0
        for train_imgss, train_lblss in train_set:
            train_top = np.array([])
            out_top = np.array([])

            train_p = np.array([])
            out_p = np.array([])

            total_inference = 0
            total_correct_inference = 0

            inference_losses = np.array([])
            classification_losses = np.array([])
            for k_count in range(k):
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
                with torch.cuda.amp.autocast():
                    train_inference = torch.squeeze(inference_net(train_posteriors, label_to_onehot(trainlbl, opt).to(device)))
                    out_inference = torch.squeeze(inference_net(out_posteriors, label_to_onehot(outlbl, opt).to(device)))

                    total_inference += 2 * mini_batch_size
                    total_correct_inference += torch.sum(train_inference > 0.5).item() + torch.sum(out_inference < 0.5).item()

                    loss_train = infer_loss(train_inference, train_lbl)
                    loss_out = infer_loss(out_inference, out_lbl)

                    loss = privacy_theta * (loss_train + loss_out) / 2
                    scalar.scale(loss).backward()
                    scalar.step(infer_optim)
                    scalar.update()

                # train classifiction network
            # try:
            #     train_imgs, train_lbls = next(train_iter2)
            # except StopIteration:
            #     train_iter2 = iter(train_set)
            #     train_imgs, train_lbls = next(train_iter2)
            train_imgss, train_lblss = train_imgss.to(device), train_lblss.to(device)
            trainlbl = train_lblss
            class_optim.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = classification_net(train_imgss)
                train_posteriors = F.softmax(outputs, dim=1)

                loss_classification = class_loss(outputs, train_lblss)
                train_lbl = torch.ones(train_imgss.shape[0]).to(device)

                train_inference = torch.squeeze(inference_net(train_posteriors, label_to_onehot(trainlbl, opt).to(device)))
                loss_infer = infer_loss(train_inference, train_lbl)
                loss = loss_classification - privacy_theta * loss_infer
                _, preds = torch.max(outputs, 1)
                running_corrects += preds.eq(train_lblss.to(opt.device)).sum().item()
                scalar1.scale(loss).backward()
                scalar1.step(class_optim)
                scalar1.update()

        print('train{} acc:{:0.4f}'.format(epoch, 1.0 * running_corrects / data_size["train"]))
        val_acc = predict(classification_net, out_set, data_size['val'], device)
        print('val{} acc:{:0.4f}'.format(epoch, val_acc))


# tuples of (lambda, attack_accuracy, classification_train_accuracy, classification_test_accuracy)
adv_reg_metrics = []
batch_size = 128
lr_classification = 0.0001
lr_inference = 0.001
lr_attack = 0.001


def adv_reg(opt, config):
    adversarial_regularization_net = inference_attack(opt.n_class).to(device)
    adversarial_regularization_net = model2device(adversarial_regularization_net)
    # adversarial_regularization_net.apply(models.weights_init)

    adversarial_regularization_loss = nn.BCEWithLogitsLoss()
    adversarial_regularization_optim = torch.optim.Adam(adversarial_regularization_net.parameters(), lr=lr_inference)


    target_net = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class).to(device)
    target_net.load_state_dict(torch.load('../vit_small_patch16_224_{}.pth'.format(opt.n_class)))
    target_net = model2device(target_net)
    # target_net.apply(models.weights_init)
    target_loss = nn.CrossEntropyLoss()
    # target_optim = optim.Adam(target_net.parameters(), lr=lr_classification)
    target_optim = torch.optim.Adam(target_net.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )
    # learning rate adopt
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(target_optim,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    # target_scheduler = warmup_scheduler.GradualWarmupScheduler(target_optim, multiplier=1.,
    #                                                     total_epoch=config.learning.warmup_epoch,
    #                                                     after_scheduler=base_scheduler)

    data_loader, data_size = get_loader(opt.dataset, config, is_target=opt.istarget)
    print("train/val size:{}/{}".format(data_size['train'], data_size['val']))
    D_loader, D_prime_loader = data_loader['train'], data_loader['val']

    adversarial_train(adversarial_regularization_net, target_net, D_loader, D_prime_loader, D_prime_loader,
                      adversarial_regularization_optim, adversarial_regularization_loss, target_optim, base_scheduler,
                      target_loss, config.learning.epochs, 2, opt.lambd, data_size)

    pad = '' if opt.istarget else '_shadow'
    if config.learning.DDP or config.learning.DP:
        if opt.local_rank == 0:
            torch.save(target_net.module.state_dict(), '../defence_model/{}/adv_reg/ar_{}{}.pth'.format(opt.dataset, opt.lambd, pad))
    else:
        torch.save(target_net.state_dict(), '../defence_model/{}/adv_reg/ar_{}{}.pth'.format(opt.dataset, opt.lambd, pad))
    return target_net


opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]

# torch.random.manual_seed(3407)
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)
config.set_subkey('learning', 'batch_size', 64)
# config.set_subkey('learning', 'learning_rate', config.learning.learning_rate/4)
device = gpu_init(opt)
opt.device = device
print("adv_reg dataset : {}  para: {}".format(opt.dataset, opt.lambd))
adv_reg(opt, config)