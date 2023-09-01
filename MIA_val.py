import datetime
import torch
from Testtool import gain_data, audit_data, conf, PositionalEncoding
from trainer import get_dp_model
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
    parser.add_argument('--maskPE', action = "store_true",
                        help='whether use PE')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1,
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
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--dataset', type=str, default='cinic',
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
    parser.add_argument('--single_label_dataset', type=list, default=["cifar10", "cifar100", "STL10", "ImageNet10", "ImageNet", "cinic10", "Fmnist", "cinic"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")
    parser.add_argument('--mia_type', type=str, default="nn-based",
                        help="nn-based, lebel-only, metric-based")
    parser.add_argument('--select_posteriors', type=int, default=10,
                        help='how many posteriors we select, if -1, we remains the original setting')

    parser.add_argument('--mia_defense', type=str,
                        default="None", help='None or memGuard')

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
        "Fmnist": 10,
        "ImageNet": 10,
        "cinic": 10
    }
    opt.n_class = dataset_class_dict[opt.dataset]
    # opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt



def write_res(wf, attack_name, res):
    # line = "%s,%s,%s,%d," % (
    #     opt.dataset, opt.model, opt.no_PE, opt.shf_dim)

    line = "%s:\t" % attack_name

    line += ",".join(["%.4f" % (row) for row in res])
    line += "\n"
    wf.write(line)

def write_spilt(wf):
    wf.write('--------------------------------------------------------------------------------------------------------------------\n')

def write_time(wf):
    wf.write(str(datetime.datetime.now())[:19]+'\n')

def write_config(wf, opt):
    line = "%s, %s, use_PE: %s acc/prec/recall/fi/auc\n" % (
        opt.dataset, opt.model, not opt.maskPE)
    wf.write(line)

# def wr(acc):
#     with open("log/model/exp_attack/res.txt", "a") as wf:
#         wf.write(acc)


opt = parse_option()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_dict = { 'ViT': {
                'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'Fmnist': "config/fashion-mnist/",
                'ImageNet': "config/ImageNet-split/",
                'cinic': "config/cinic-split/",
                },
                'Swin': {
                'cifar10': "config/Swin-cifar10/",
                'cifar100': "config/Swin-cifar100/",
                'ImageNet10': "config/Swin-ImageNet10/",
                'ImageNet100': "config/Swin-ImageNet100/"
                }
}

# torch.random.manual_seed(1001)
config_path = config_dict['Swin'][opt.dataset] if 'Swin' in opt.model else config_dict['ViT'][opt.dataset]


config = MyConfig.MyConfig(path=config_path)
# config.set_subkey('learning','DP', False)
config.set_subkey('learning','DDP', False)
# config.set_subkey('path', 'data_path', config.path.data_root + 'img0/')
target_loader, target_size = get_loader(opt.dataset, config, is_target=True)
shadow_loader, shadow_size = get_loader(opt.dataset, config, is_target=False)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = target_loader['train'], target_loader['val'], shadow_loader['train'], shadow_loader['val']

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
        path = config.path.model_path + opt.model + '.pth'
        target_model = ViT_ape.load_VIT(path + '.pth', config)
        shadow_model = ViT_ape.load_VIT(path + '_shadow.pth', config)
    elif 'mask' in opt.model:
        path = config.path.model_path + opt.model[:8]
        target_model = ViT_mask.load_VIT(path + opt.model[8:] + '.pth', config)
        shadow_model = ViT_mask.load_VIT(path + '_shadow' + opt.model[8:] + '.pth', config)
    else :
        opt.model = 'cinic_0'
        path = config.path.model_path + opt.model
        target_model = ViT.load_VIT(path + '.pth', config)
        shadow_model = ViT.load_VIT(path + '_shadow.pth', config)

    if opt.maskPE:
        target_model.PE = False
        shadow_model.PE = False
    # if config.learning.DP:
    #     target_model = torch.nn.DataParallel(target_model, device_ids=[0, 1])
    #     shadow_model = torch.nn.DataParallel(shadow_model, device_ids=[0, 1])
    return target_model, shadow_model


# if opt.no_PE == False:
#     target_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth')
#     shadow_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE_shadow.pth')
# else:
#     target_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth')
#     shadow_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE_shadow.pth')


ratio = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]
pos = PositionalEncoding(192)
attack_model = MLP_CE(selected_posteriors=opt.select_posteriors)
os.makedirs("log/model/exp_attack/", exist_ok=True)
# attack_model = [MLP_CE(opt.select_posteriors) for i in range(len(ratio))]
for i in range(1):

    target_model, shadow_model = load_model(opt, config)
    if opt.mia_type == "nn-based":
        path = config.path.model_path + 'cinic_1'
        # target_model_2 = ViT.load_VIT(path + '.pth', config)
        # shadow_model_2 = ViT.load_VIT(path + '_shadow.pth', config)
        # target_model_2.to_patch_embedding = target_model.to_patch_embedding
        # shadow_model_2.to_patch_embedding = shadow_model.to_patch_embedding
        target_model.pos_embedding.data = pos.pe[0,:65, :]
        shadow_model.pos_embedding.data = pos.pe[0,:65, :]

        attack_model = torch.nn.DataParallel(attack_model, device_ids=[0, 1])
        attack = attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        attack.parse_dataset()
        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs
        train_acc, test_acc, precision, recall, f1, AUC = attack.train(epoch_train)  # train 100 epoch
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance
        # attack.attack_model = None
        # attack_model = None
        with open("log/model/exp_attack/mia_update.txt", "a") as wf:
            res = [epoch_train, target_train_acc, target_test_acc,
                   shadow_train_acc, shadow_test_acc, train_acc, test_acc, precision, recall, f1, AUC]
            write_time(wf)
            write_config(wf, opt)
            write_res(wf, "NN-ATK-based", res)
            write_spilt(wf)
        print("Finish")
        torch.cuda.empty_cache()


    elif opt.mia_type == "metric-based":
        attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                      shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = attack.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance

        with open("log/model/exp_attack/mia_update.txt", "a") as wf:
            res0 = [target_train_acc, target_test_acc,
                    shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0], test_tuple0[1], test_tuple0[2], test_tuple0[3], test_tuple0[4]]
            res1 = [target_train_acc, target_test_acc,
                    shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0], test_tuple1[1], test_tuple1[2], test_tuple1[3], test_tuple1[4]]
            res2 = [target_train_acc, target_test_acc,
                    shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0], test_tuple2[1], test_tuple2[2], test_tuple2[3], test_tuple2[4]]
            res3 = [target_train_acc, target_test_acc,
                    shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0], test_tuple3[1], test_tuple3[2], test_tuple3[3], test_tuple3[4]]
            write_time(wf)
            write_config(wf, opt)
            write_res(wf, "Metric-corr", res0)
            write_res(wf, "Metric-conf", res1)
            write_res(wf, "Metric-entr", res2)
            write_res(wf, "Metric-ment", res3)
            write_spilt(wf)
        print("Finish")
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
            write_res(wf, "Label-only", res)


        print("Finish")

