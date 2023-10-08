import argparse
import os.path
import sys

import timm
import torch
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import cv2
import tim
import dataloader
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from dataloader import get_loader
from utils import MyConfig
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image', type=str, default='0_test_1.JPEG',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def write_img(img, mask, path):
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite(path, mask)

model_name = ["orain_mask_0.000.pth", "orain_mask_exchg_0.000.pth", "orain_mask_0.138.pth", "orain_mask_0.276.pth", "orain_mask_0.551.pth", "orain_mask_1.000.pth"]
test_dir = "../data/ImageNet100/test/"
save_path = "./attn_heat/"
plot_number = 1
for modn in model_name:
    args = get_args()
    print(modn)
    args.image_path = './examples/' + args.image
    # model = timm.create_model('vit_small_patch16_224', num_classes=1000, attn_drop_rate=0.1)
    model = tim.create_model('vit_small_patch16_224', num_classes=100)
    model_path = "./Network/VIT_Model_ImageNet100/" + modn
    model.load_state_dict(torch.load(model_path))
    # model.PE = False
    # model.train()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # img = Image.open(args.image_path)
    # img = img.resize((224, 224))
    # input_tensor = transform(img).unsqueeze(0)

    # config = MyConfig.MyConfig(path="config/ImageNet100/")
    # config.set_subkey('learning', 'DDP', False)
    # config.set_subkey('learning', 'batch_size', 128)
    # Transform = transforms.Compose([transforms.Resize([224, 224]),
    #                                 # transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # set1 = datasets.ImageFolder(test_dir, Transform)

    for i in range(3002,6000,100):
        # for q in range(i,i+100):
        #     img = Image.open(set1.imgs[q][0])
        #     if len(img.mode) ==3:
        #         break
        #
        # img = img.resize((224, 224))
        # input_tensor = transform(img).unsqueeze(0)
        img = Image.open(args.image_path)
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)
        mean = 0.0
        std = 0.1
        noise = torch.randn_like([]) * std + mean

        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)

        name = "/{}_{:.3f}_{}.png".format(modn, args.discard_ratio, args.head_fusion)
        # for i in range(input_tensor.shape[0]):
        path = "{}img{}".format(save_path,i)
        if not os.path.exists(path):
            os.makedirs(path)
        write_img(img, mask, "1.png")




