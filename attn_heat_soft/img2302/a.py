import matplotlib.pyplot as plt
from PIL import Image

# 加载两张图片
image1 = Image.open('orain_mask_0.000.pth_0.900_max.png')
image2 = Image.open('orain_mask_1.000.pth_0.900_max.png')

# 创建一个包含两个子图的画布
fig, axs = plt.subplots(1, 2)

# 在第一个子图中绘制第一张图片
axs[0].imshow(image1)
axs[0].axis('off')
axs[0].set_title("0.000")
# 在第二个子图中绘制第二张图片
axs[1].imshow(image2)
axs[1].axis('off')
axs[1].set_title("1.000g")
# 调整子图之间的间距
plt.subplots_adjust(wspace=0.1)

# 保存合并后的图片
plt.savefig('merged_image.png', bbox_inches='tight')