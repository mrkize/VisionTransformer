import argparse

from PIL import Image

import tim
from torchvision import datasets, transforms

from dataloader import get_loader, set2loader
from trainer import mask_train_model, predict
import torch
from utils import MyConfig
# from Testtool import test_mask_model_imgshuff, test_mask_model, predict_cmp, Privacy_laekage
import os


def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--ordinary_train', action="store_true", help='whether use mask')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--dataset', type=str, default='ImageNet100', help='dataset and config')
    parser.add_argument('--model_type', type=str, default='ape', help='model name')
    parser.add_argument('--ape', action="store_true", help='use ape')
    parser.add_argument('--init', action="store_true", help='init model or finetune')
    parser.add_argument('--pe_aug', type=str, default='scale', help='if fill')
    parser.add_argument('--img_aug', type=str, default='orain', help='if fill')
    parser.add_argument('--nums', type=int, default=0, help='if fill')
    parser.add_argument('--set', type=int, default=1, help='if fill')
    parser.add_argument('--mix_up', action="store_true", help='use Mixup')
    parser.add_argument('--n_class', type=int, default=10, help='')
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--defence", type=str, default='', help="defence method")
    opt = parser.parse_args()
    return opt


def masked_train(model_type):
    data_loader, data_size = get_loader(opt.dataset, config, is_target=True, set=opt.set)
    mask_train_model(model_type, opt, config, data_loader, data_size)
    # data_loader, data_size = get_loader(opt.dataset, config, is_target=False)
    # mask_train_model(model_type, opt, config, data_loader, data_size, is_target=False)


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
def get_model_name(opt):
    pad = '_ape' if opt.ape else ''
    pad0 = 'init_' if opt.init else ''
    if opt.img_aug == "pbf" or opt.img_aug == "orain":
        opt.model_type = '{}{}_{}{}'.format(pad0,opt.img_aug,opt.pe_aug,pad)
    else:
        opt.model_type = '{}{}{}'.format(pad0,opt.img_aug,pad)


def change_config(config):
    config.set_subkey('patch', 'image_size', 32)
    config.set_subkey('patch', 'patch_size', 4)
    config.set_subkey('patch', 'num_patches', 64)
    config.set_subkey('patch', 'embed_dim', 192)
    config.set_subkey('learning', 'learning_rate', 0.001)
    config.set_subkey('learning', 'epochs', 100)
    config.set_subkey('learning', 'warmup_epoch', 10)
    config.set_subkey('learning', 'val_epoch', False)
    config.set_subkey('mask', 'warmup_epoch', 20)
# def val():
#     path = config.path.model_path + opt.model_type
#     model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
#     model.load_state_dict(torch.load(path + '.pth'))
#     data_loader, data_size = get_loader(opt.dataset, config, is_target=False)
#     model1, _ = train1(opt.model_type + '_val', data_loader, data_size, opt, config, model.pos_embed.data)


def train():
    for i in range(config.mask.start,config.mask.stop,config.mask.step):
        config.set_subkey('patch', 'num_masking_patches', i)
        config.set_subkey('mask', 'mask_ratio', i/config.patch.num_patches)
        opt.mask_ratio = i/config.patch.num_patches
        masked_train(opt.model_type)


def get_img(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, 1

model_name = ["orain_mask_0.000.pth", "orain_mask_exchg_0.000.pth", "orain_mask_0.138.pth", "orain_mask_0.276.pth", "orain_mask_0.551.pth", "orain_mask_1.000.pth"]
model_name_cifar = ["orain_zero_0.000.pth", "orain_zero_0.276.pth"]
model_name_cifar100 = ["ViT_mask_0.051.pth", "ViT_mask_0.153.pth", "ViT_mask_0.255.pth", "ViT_mask_0.357.pth", "ViT_mask_0.459.pth", "ViT_mask_0.561.pth", "ViT_mask_0.663.pth", "ViT_mask_0.765.pth", "ViT_mask_0.867.pth", "ViT_mask_0.969.pth"]
model_name_cifar100_shadow = ["0.051", "0.153", "0.255", "0.357", "0.459", "0.561", "0.663", "0.765", "0.867", "0.969"]
def pre(split):
    data_loader, data_size = get_loader(opt.dataset, config, is_target=True, set=opt.set)
    data_loader, data_size = data_loader[split], data_size[split]
    # data_loader, data_size = get_img("./examples/0_test_1.JPEG")
    model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
    entropy_list = []
    for modeln in model_name:
        # if modeln == "orain_mask_1.000.pth":
        #     model.train_method = "no_pos"
        pad = "_shadow"
        pad = ""
        model.load_state_dict(torch.load(config.path.model_path + modeln))
        # model.load_state_dict(torch.load(config.path.model_path + "ViT_mask_{}{}.pth".format(modeln,pad)))
        print(modeln)
        _, entropy = predict(model, data_loader, data_size, opt.device)
        entropy_list.append(entropy)
    return entropy_list


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


config_dict = {'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'Fmnist': "config/fashion-mnist/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'cifar10-8': "config/cifar10-8*8/",
                'ImageNet10-8': "config/ImageNet10-8*8/"
                }


opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]
opt.mix_up = False
# torch.random.manual_seed(1001)
# config.set_subkey('learning', 'epochs', opt.epochs)
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)

# config.set_subkey('learning', 'DP', False)
# config.set_subkey('learning', 'DDP', False)
# config.set_subkey('learning', 'batch_size', 128)
# config.set_subkey('mask', 'warmup_epoch', 0)
# opt.nums = 108

opt.device = gpu_init(opt)
get_model_name(opt)
if opt.dataset == 'ImageNet100' and opt.set == 2:
    opt.model_type = opt.model_type + '_exchg'
# change_config(config)
config.set_subkey('patch', 'num_masking_patches', opt.nums)
config.set_subkey('mask', 'mask_ratio', opt.nums / config.patch.num_patches)
opt.mask_ratio = opt.nums / config.patch.num_patches
masked_train(opt.model_type)
# entropy_list_val = pre("val")
# entropy_list_train = pre("train")
# print("entropy_list_val: ", entropy_list_val)
# print("entropy_list_train: ", entropy_list_train)
# print("entropy_gap: ", torch.tensor(entropy_list_val) - torch.tensor(entropy_list_train))