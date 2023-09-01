from PIL import Image

# 打开图片
img = Image.open('0_test_1.JPEG')

# 水平翻转
flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

# 保存翻转后的图片
flipped_img.save('0_test_11.JPEG')