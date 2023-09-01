import numpy as np
import torch
import copy

def imgshuffle(img, alpha, seed=101):
    img_size = 32
    patch_size = 8
    img2patchs =torch.nn.Unfold((patch_size,patch_size), stride=patch_size)
    patches_to_im = torch.nn.Fold(
        output_size=(img_size, img_size),
        kernel_size=(patch_size, patch_size),
        stride=patch_size
    )
    num_patchs = (img_size//patch_size)*(img_size//patch_size)
    shu_num = int((img_size//patch_size)*(img_size//patch_size)*alpha)

    idx = np.arange(num_patchs)
    np.random.seed(seed)
    idx_part = np.random.choice(idx, shu_num)
    np.random.seed(seed)
    idx_part_shu = np.random.permutation(idx_part)
    to_patches = img2patchs(img)
    to_patches_copy = copy.deepcopy(to_patches)
    to_patches[:, :, idx_part] = to_patches_copy[:, :, idx_part_shu]
    to_images = patches_to_im(to_patches)
    return to_images