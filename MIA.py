import csv
import datetime

import tim
import torch
import torchvision

# from Testtool import gain_data, audit_data, conf
from utils import MyConfig
from atk_models.attack_model import MLP_CE
from utils.mia.attackTraining import attackTraining
from utils.mia.metric_based_attack import AttackTrainingMetric
from utils.mia.label_only_attack import AttackLabelOnly
from mymodel import ViT,ViT_mask,ViT_ape,ViT_mask_avg,Swin,Swin_mask, Swin_mask_avg,Swin_ape
from dataloader import get_loader

import numpy as np
import os
import argparse
torch.manual_seed(0)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--adaptive', action='store_true', default=False)
    parser.add_argument('--maskPE', action = "store_true",
                        help='whether use PE')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--shf_dim', type=int, default=0,
                        help='shuffle dim for image, <=0 means donot shuffle')
    parser.add_argument('--ptr', type=float, default=0.1,
                        help='patches shuffle ratio')
    parser.add_argument('--pxr', type=float, default=0.2,
                        help='pixel shuffle ratio')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='pbf_mask_0.000.pth')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')


    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', "CE"], help='choose method')


    # linear training epochs
    parser.add_argument('--linear_epoch', type=int, default=0,
                        help='conduct MIA over a specific models 0 means the original target model, other numbers like 10 or 20 means target model linear layer only trained for 10 or 20 epochs')
    # label
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')
    parser.add_argument('--single_label_dataset', type=list, default=["cifar10", "cifar100", "STL10", "ImageNet10", "ImageNet100", "cinic10", "Fmnist"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")
    parser.add_argument('--mia_type', type=str, default="ft_nn",
                        help="nn-based, lebel-only, metric-based")
    parser.add_argument('--select_posteriors', type=int, default=3,
                        help='how many posteriors we select, if -1, we remains the original setting')

    parser.add_argument('--mia_defense', type=str,
                        default="ft_nn", help='None or memGuard')

    parser.add_argument('--modeldir', type=str, default="", help='None or memGuard')
    parser.add_argument('--valmode', action = "store_true", help='None or memGuard')
    opt = parser.parse_args()
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
    opt.n_class = dataset_class_dict[opt.dataset]
    return opt



def write_res(wf, attack_name, res):
    line = "%s:\t" % attack_name

    line += ",".join(["%.4f" % (row) for row in res])
    line += "\n"
    wf.write(line)

def write_copy(wf, res):
    line = "res_copy:"

    line += ",".join(["%.4f" % (row) for row in res])
    line += "\n"
    wf.write(line)

def write_spilt(wf):
    wf.write('-'*120+"\n")

def write_time(wf):
    wf.write(str(datetime.datetime.now())[:19]+'\n')

def write_config(wf, opt):
    line = "%s, %s, use_PE: %s, shf_dim: %d, ptr: %f, pxr: %f\n" % (
        opt.dataset, opt.model, not opt.maskPE, opt.shf_dim, opt.ptr, opt.pxr)
    wf.write(line)

# def wr(acc):
#     with open("log/model/exp_attack/res.txt", "a") as wf:
#         wf.write(acc)


opt = parse_option()
opt.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

config_dict = { 'ViT': {
                'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'Fmnist': "config/fashion-mnist/",
                },
                'Swin': {
                'cifar10': "config/Swin-cifar10/",
                'cifar100': "config/Swin-cifar100/",
                'ImageNet10': "config/Swin-ImageNet10/",
                'ImageNet100': "config/Swin-ImageNet100/"
                }
}

# torch.random.manual_seed(1001)
config_path = config_dict['ViT'][opt.dataset]


config = MyConfig.MyConfig(path=config_path)
config.set_subkey('learning','DDP', False)
config.set_subkey('learning','DP', False)
target_loader, target_size = get_loader(opt.dataset, config, is_target=True)
shadow_loader, shadow_size = get_loader(opt.dataset, config, is_target=False)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = target_loader['train'], target_loader['val'], shadow_loader['train'], shadow_loader['val']


def freeze_parameters(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze


def replace_last_fc_layer(model, num_classes):
    model.head = torch.nn.Linear(384, num_classes)
    return model


def ft_model(model, dataloader, num_class, num_epochs=1, learning_rate=0.001, freeze=True):
    if opt.dataset =="cifar100":
        learning_rate = 0.0001
    freeze_parameters(model, freeze)
    model.head = torch.nn.Linear(384, 10).to(opt.device)
    model.head.weight.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        # print(epoch)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            optimizer.zero_grad()
            outputs = model(inputs, unk_mask=None)
            _, preds = torch.max(outputs, 1)
            # labels = torch.squeeze(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def load_model(opt, config):
    if opt.modeldir != "":
        opt.model = 'ImageNet10_10'
        target_model = ViT.load_VIT(opt.modeldir+opt.model+'.pth', config)
        shadow_model = ViT.load_VIT(opt.modeldir+opt.model[:10]+'_shadow'+opt.model[10:]+'.pth', config)
    elif 'ViT_mask_avg' in opt.model:
        path = config.path.model_path+opt.model[:12]
        target_model = ViT_mask_avg.load_VIT(path + opt.model[12:] + '.pth', config)
        shadow_model = ViT_mask_avg.load_VIT(path + '_shadow' + opt.model[12:] + '.pth', config)
    elif 'ViT_ape' in opt.model:
        path = config.path.model_path + opt.model
        target_model = ViT_ape.load_VIT(path + '.pth', config)
        shadow_model = ViT_ape.load_VIT(path + '_shadow.pth', config)
    elif 'mask' in opt.model:
        path = config.path.model_path + opt.model
        target_model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
        target_model.load_state_dict(torch.load(path))
        shadow_model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
        if opt.adaptive:
            shadow_model.load_state_dict(torch.load(config.path.model_path + 'pbf_mask_0.000_shadow.pth'))
        else:
            shadow_model.load_state_dict(torch.load(path[:-4] + '_shadow.pth'))
    else :
        path = config.path.model_path + opt.model
        # target_model = ViT.load_VIT(path + '.pth', config)
        # shadow_model = ViT.load_VIT(path + '_shadow.pth', config)
        target_model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
        target_model.load_state_dict(torch.load(path + '.pth'))
        shadow_model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
        if opt.adaptive:
            shadow_model.load_state_dict(torch.load('_shadow.pth'))
        shadow_model.load_state_dict(torch.load(path + '_shadow.pth'))
    target_model, shadow_model = target_model.to(opt.device), shadow_model.to(opt.device)
    ft_config = MyConfig.MyConfig(path=config_dict['ViT']["cinic10"])
    ft_config.set_subkey('learning', 'DDP', False)
    ft_config.set_subkey('learning', 'DP', False)
    loader1, size = get_loader("cinic10", ft_config, is_target=True)
    loader2, size = get_loader("cinic10", ft_config, is_target=False)
    target_model = ft_model(target_model, loader1["train"], opt.n_class)
    shadow_model = ft_model(shadow_model, loader2["train"], opt.n_class)
    if opt.maskPE:
        target_model.PE = False
        shadow_model.PE = False
    if config.learning.DP:

        target_model = torch.nn.DataParallel(target_model, device_ids=[0, 1])
        shadow_model = torch.nn.DataParallel(shadow_model, device_ids=[0, 1])
    return target_model, shadow_model


def change_csv(dir, value):
    rows = []
    with open(dir, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    for row in rows:
        if row['atk'] == opt.mia_type:
            row[opt.model] = round(value, 4)

    fieldnames = rows[0].keys()
    with open(dir, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


ratio = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]

attack_model = MLP_CE(selected_posteriors=opt.select_posteriors)
os.makedirs("log/model/exp_attack/", exist_ok=True)
# attack_model = [MLP_CE(opt.select_posteriors) for i in range(len(ratio))]
for i in range(1):
    # opt.model = "ViT"if rt==0 else 'ViT_mask_' + str(rt)
    target_model, shadow_model = load_model(opt, config)
    if opt.mia_type == "ft_nn":
        attack = attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, opt.device)

        attack.parse_dataset()
        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs
        train_acc, test_acc, precision, recall= attack.train(epoch_train)  # train 100 epoch
        print("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}".format(test_acc, precision, recall))
        dir = "csv_res_adp" if opt.adaptive else "csv_res"
        change_csv("{}/{}/acc.csv".format(dir, opt.dataset), test_acc)
        change_csv("{}/{}/pre.csv".format(dir, opt.dataset), precision)
        change_csv("{}/{}/rec.csv".format(dir, opt.dataset), recall)


    elif opt.mia_type == "ft_metric":
        attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                      shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, opt.device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = attack.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance
        opt.mia_type = "ft_conf"
        print("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}".format(test_tuple1[0], test_tuple1[1], test_tuple1[2]))
        dir = "csv_res_adp" if opt.adaptive else "csv_res"
        change_csv("{}/{}/acc.csv".format(dir, opt.dataset), test_tuple1[0])
        change_csv("{}/{}/pre.csv".format(dir, opt.dataset), test_tuple1[1])
        change_csv("{}/{}/rec.csv".format(dir, opt.dataset), test_tuple1[2])
        opt.mia_type = "ft_entr"
        print("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}".format(test_tuple2[0], test_tuple2[1], test_tuple2[2]))
        dir = "csv_res_adp" if opt.adaptive else "csv_res"
        change_csv("{}/{}/acc.csv".format(dir, opt.dataset), test_tuple2[0])
        change_csv("{}/{}/pre.csv".format(dir, opt.dataset), test_tuple2[1])
        change_csv("{}/{}/rec.csv".format(dir, opt.dataset), test_tuple2[2])
        # with open("log/model/exp_attack/mia_update.txt", "a") as wf:
        #     res0 = [target_train_acc, target_test_acc,
        #             shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0], test_tuple0[1], test_tuple0[2], test_tuple0[3], test_tuple0[4]]
        #     res1 = [target_train_acc, target_test_acc,
        #             shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0], test_tuple1[1], test_tuple1[2], test_tuple1[3], test_tuple1[4]]
        #     res2 = [target_train_acc, target_test_acc,
        #             shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0], test_tuple2[1], test_tuple2[2], test_tuple2[3], test_tuple2[4]]
        #     res3 = [target_train_acc, target_test_acc,
        #             shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0], test_tuple3[1], test_tuple3[2], test_tuple3[3], test_tuple3[4]]
        #     res4 = [target_test_acc, test_tuple0[0], test_tuple1[0], test_tuple2[0], test_tuple3[0]]
        #     write_time(wf)
        #     write_config(wf, opt)
        #     write_res(wf, "Metric-corr", res0)
        #     write_res(wf, "Metric-conf", res1)
        #     write_res(wf, "Metric-entr", res2)
        #     write_res(wf, "Metric-ment", res3)
        #     write_copy(wf, res4)
        #     write_spilt(wf)
        # print("Finish")
    elif opt.mia_type == "audit":
        atk_loader, atk_size = gain_data(target_model, shadow_model, target_loader, target_size, shadow_loader, shadow_size, not opt.maskPE, config, device)
        epoch_acc, best_acc = audit_data(atk_loader, atk_size, config, device)
        atk_loader, atk_size = gain_data(target_model, shadow_model, target_loader, target_size, shadow_loader, shadow_size, opt.maskPE, config, device)
        epoch_acc_pe, best_acc_pe = audit_data(atk_loader, atk_size, config, device)
        res = [epoch_acc, best_acc, epoch_acc_pe, best_acc_pe]
        with open("log/model/exp_attack/mia_update.txt", "a") as wf:
            write_config(wf, opt)
            write_res(wf, "audit", res)
            write_spilt(wf)

        print("Finish")
    elif opt.mia_type == "conf":
        target_model.to(device)
        min, max = conf(target_model, target_loader['val'], device)
        res = [min, max]
        with open("log/model/exp_attack/mia_update.txt", "a") as wf:
            write_config(wf, opt)
            write_res(wf, "confidence mean/var", res)
            write_spilt(wf)

        print("Finish")
    elif opt.mia_type == "label-only":
        attack = AttackLabelOnly(opt, target_train_loader, target_test_loader,
                                 shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        attack.searchThreshold(num_samples=500)
        test_tuple = attack.test(num_samples=500)
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc, threshold = attack.original_performance

        #!!!!!!!!!!!we replace train acc with the threshold!!!!!!
        res = [epoch_train, target_train_acc, target_test_acc, shadow_train_acc,
               shadow_test_acc, threshold, test_tuple[0]]
        with open("log/model/exp_attack/mia_update.txt", "a") as wf:
            write_res(opt, wf, "Label-only", res)


        print("Finish")
