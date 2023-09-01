import argparse
import sys

import timm
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import tim
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
    parser.add_argument('--discard_ratio', type=float, default=0.0,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #     print("Using GPU")
    # else:
    #     print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

model_name = ["orain_mask_0.000.pth", "orain_maskexg_0.000.pth", "orain_mask_0.138.pth", "orain_mask_0.276.pth", "orain_mask_0.551.pth", "orain_mask_1.000.pth"]
for modn in model_name:
    args = get_args()
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
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    # config = MyConfig.MyConfig(path="config/ImageNet100/")
    # config.set_subkey('learning', 'DDP', False)
    # data_loader, data_size = get_loader("ImageNet100", config, is_target=True, set=1)
    # for input_tensor, label in data_loader['train']:
    #     break
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        # print("Doing Attention Rollout")
        print(modn)
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
            discard_ratio=args.discard_ratio)
        mask, metric = attention_rollout(input_tensor)
        name = "{}_attention_rollout_{:.3f}_{}.png".format(args.image, args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "{}_grad_rollout_{}_{:.3f}_{}.png".format(args.image, args.category_index,
            args.discard_ratio, args.head_fusion)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    dst_dir = './attn_heat/'
    # cv2.imshow("Input Image", np_img)
    # cv2.imshow(name, mask)
    cv2.imwrite(dst_dir+args.image, np_img)
    cv2.imwrite('pre_1_'+name, mask)
    cv2.waitKey(-1)