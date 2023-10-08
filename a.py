import os

import matplotlib.pyplot as plt
import cv2


img_name = ["0.000", "0.138", "0.276", "0.551", "1.000", "exchg_0.000"]
png_path = "./attn_heat_soft/"
save_path = "./0apng_mix_soft/"
for subfloder in os.scandir(png_path):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(40, 40))
    image = cv2.imread('{}/img.png'.format(subfloder.path))
    axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('orain')
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    for i, name in enumerate(img_name):
        image = cv2.imread('{}/orain_mask_{}.pth_0.900_max.png'.format(subfloder.path,name))
        axes[i//3 + 1, i%3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i//3 + 1, i%3].set_title(name)
    plt.tight_layout()
    plt.savefig('{}{}.png'.format(save_path, subfloder.name))
    plt.close(fig)
