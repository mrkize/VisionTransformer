import argparse
import os.path
import sys
import matplotlib.pyplot as plt
import timm
import torch
# import torchvision.transforms.v2.functional
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import cv2
import tim
import torch.nn.functional as F
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
    parser.add_argument('--start', type=int, default=0,
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--addnoise', action='store_true', default=False,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args


def add_gaussian_noise_to_patches(image_tensor, patch_size=16, num_patches=196, mean=0, stddev=0.25):
    batch_size, channels, height, width = image_tensor.size()
    # input_tensor = image_tensor.unsqueeze(0)
    unfolded = torch.nn.functional.unfold(input_tensor, patch_size, stride=patch_size)
    # unfolded = unfolded.view(batch_size, channels, -1, patch_size, patch_size)
    selected_patches = torch.randperm(unfolded.size(2))[:num_patches]
    noise = torch.randn(batch_size, unfolded.shape[1], num_patches) * stddev + mean
    noise = noise.cuda()
    unfolded = unfolded.cuda()
    unfolded[:, :, selected_patches] += noise

    unfolded = unfolded.clamp(0, 1)
    # image_tensor = unfolded.view(batch_size, channels*patch_size*patch_size, -1)
    image_tensor = torch.nn.functional.fold(unfolded, (height, width), patch_size, stride=patch_size)
    return image_tensor


def add_gaussian_noise(image, mean=0, stddev=20):
    image_array = np.array(image)
    noise = np.random.normal(mean, stddev, image_array.shape).astype(np.uint8)
    noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image


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


def visualize_probabilities(probabilities, path):
    plt.imshow(probabilities, cmap='Reds', interpolation='nearest')
    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            plt.text(j, i, f'{probabilities[i, j]:.2f}', ha='center', va='center')
    plt.xticks(np.arange(probabilities.shape[1]), np.arange(1, probabilities.shape[1] + 1))
    plt.yticks(np.arange(probabilities.shape[0]), np.arange(1, probabilities.shape[0] + 1))
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title("attn heat map")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


model_name = ["orain_mask_0.000.pth", "orain_mask_exchg_0.000.pth", "orain_mask_0.138.pth", "orain_mask_0.276.pth", "orain_mask_0.551.pth", "orain_mask_1.000.pth"]
test_dir = "../data/ImageNet100/test/"
save_path = "./attn_heat_soft/"
plot_number = 1
args = get_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# img = Image.open(args.image_path)
# img = img.resize((224, 224))
# input_tensor = transform(img).unsqueeze(0)

config = MyConfig.MyConfig(path="config/ImageNet100/")
config.set_subkey('learning', 'DDP', False)
config.set_subkey('learning', 'batch_size', 128)
Transform = transforms.Compose([transforms.Resize([224, 224]),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
set1 = datasets.ImageFolder(test_dir, Transform)

for i in range(args.start+2,args.start+10,100):
    for q in range(i,i+100):
        img = Image.open(set1.imgs[q][0])
        if len(img.mode) ==3:
            break
    img = img.resize((224, 224))
    input_tensor = transforms.ToTensor()(img).unsqueeze(0)
    if args.addnoise:
        input_tensor = add_gaussian_noise_to_patches(input_tensor)
        img = transforms.ToPILImage()(input_tensor[0])


    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    path = "{}img{}{}".format(save_path, q, "_noise" if args.addnoise else "")
    if not os.path.exists(path):
        os.makedirs(path)
    img.save(path+"/img.png")
    model = tim.create_model('vit_small_patch16_224', num_classes=100)
    attn = []
    attn_noise = []

    for modn in model_name:
        # print(modn)
        args.image_path = './examples/' + args.image
        # model = timm.create_model('vit_small_patch16_224', num_classes=1000, attn_drop_rate=0.1)
        model_path = "./Network/VIT_Model_ImageNet100/" + modn
        model.load_state_dict(torch.load(model_path))
        if modn == "orain_mask_1.000.pth":
            model.train_method = "no_pos"
        # model.PE = False
        # model.train()

        if args.use_cuda:
            model = model.cuda()
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        attn.append(mask)


    input_tensor = add_gaussian_noise_to_patches(input_tensor)
    img = transforms.ToPILImage()(input_tensor[0])

    for modn in model_name:
        args.image_path = './examples/' + args.image
        # model = timm.create_model('vit_small_patch16_224', num_classes=1000, attn_drop_rate=0.1)
        model_path = "./Network/VIT_Model_ImageNet100/" + modn
        model.load_state_dict(torch.load(model_path))
        if modn == "orain_mask_1.000.pth":
            model.train_method = "no_pos"
        # model.PE = False
        # model.train()

        if args.use_cuda:
            model = model.cuda()
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                                                discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        attn_noise.append(mask)

    for i, modn in enumerate(model_name):
        entropy = torch.nn.functional.cross_entropy(attn[i], attn_noise[i])
        cos = torch.cosine_similarity(attn[i], attn_noise[i],dim=0)
        print("{}\ncos:{}\nent:{}".format(modn, cos, entropy))
        # print("{}\ncos:{}".format(modn, cos))




        # name = "/{}_{:.3f}_{}.png".format(modn, args.discard_ratio, args.head_fusion)

        # if not os.path.exists(path):
        #     os.makedirs(path)
        # visualize_probabilities(mask, path+name)
        # write_img(img, mask, path+name)




