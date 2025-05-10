import os
import cv2
import numpy as np
import random
from glob import glob
from PIL import Image

# 定义CutMix数据增强函数
def cutmix_augment(image: np.array, label: np.array, A_images: list, A_labels: list) -> (np.array, np.array):
    # 选择另一张图像和标签
    idx = random.choice(range(len(A_images)))
    img1 = Image.open(A_images[idx]).convert('L')
    label1 = Image.open(A_labels[idx]).convert('L')

    # 将图像和标签转换为numpy数组
    img1 = np.array(img1)
    label1 = np.array(label1)

    # 生成随机的裁剪区域
    h, w = image.shape[:2]
    x1, y1 = random.randint(0, w - 100), random.randint(0, h - 100)
    x2, y2 = x1 + 100, y1 + 100

    # 裁剪图像和标签
    img_patch = img1[y1:y2, x1:x2]
    label_patch = label1[y1:y2, x1:x2]

    # 将裁剪区域粘贴到原图像和标签
    image[y1:y2, x1:x2] = img_patch
    label[y1:y2, x1:x2] = label_patch

    return image, label

# 定义文件夹路径
image_dir = 'lwh_dataset/images/training'
label_dir = 'lwh_dataset/annotations/training'
output_image_dir = './cutmix_img'
output_label_dir = './cutmix_labels'

# 创建输出文件夹
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 读取图像和标签文件列表
image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))
label_files = sorted(glob(os.path.join(label_dir, '*.png')))

# 检查图像和标签文件数量是否相同
if len(image_files) != len(label_files):
    raise ValueError("The number of image files and label files must be the same")

# 将图像和标签文件列表合并成一个元组列表
pair_files = list(zip(image_files, label_files))

# 确保生成的组合数量是原始数据的两倍
num_combinations = len(image_files) // 2

# 进行数据增强
for i in range(num_combinations):
    # 随机选择两张图像和标签
    selected_pair = random.choice(pair_files)
    image = cv2.imread(selected_pair[0], cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(selected_pair[1], cv2.IMREAD_GRAYSCALE)

    # 使用CutMix增强
    cutmix_image, cutmix_label = cutmix_augment(image, label, image_files, label_files)

    # 保存结果
    output_image_file = os.path.join(output_image_dir, f'cutmix_{i}.png')
    output_label_file = os.path.join(output_label_dir, f'cutmix_{i}.png')

    cv2.imwrite(output_image_file, cutmix_image)
    cv2.imwrite(output_label_file, cutmix_label)

print("数据增强完成，结果已保存到指定文件夹。")