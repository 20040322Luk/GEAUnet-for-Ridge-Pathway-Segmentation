import os
import cv2
import numpy as np
import random
from glob import glob


# 定义椒盐噪声数据增强函数
def add_salt_pepper_noise(image, amount):
    """
    为输入图像覆盖一层椒盐噪声
    :param image: 原始图像 (NumPy 数组)
    :param amount: 添加椒盐噪声的密度, 接近0将只有少量噪声, 接近1则几乎满了
    :return: 添加了椒盐噪声的图样 (NumPy 数组)
    """
    result = np.copy(image)

    # 添加椒（黑色）噪声
    num_salt = int(amount * image.size * 0.5)
    indices_salt = np.random.choice(image.size, num_salt, replace=False)
    result.flat[indices_salt] = 0  # 椒（黑色噪声）的值

    # 添加盐（白色）噪声
    num_pepper = int(amount * image.size * 0.5)
    indices_pepper = np.random.choice(image.size, num_pepper, replace=False)
    result.flat[indices_pepper] = 255  # 盐（白色噪声）的值

    return result


# 定义文件夹路径
image_dir = 'b_and_a_test/original_img'
label_dir = 'b_and_a_test/original_label'
output_image_dir = 'b_and_a_test/noise_img'
output_label_dir = 'b_and_a_test/noise_labels'

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

# 定义噪声密度
amount = 0.08

# 生成的噪声数据数量
num_combinations = len(image_files)

# 进行数据增强
for i in range(num_combinations):
    # 读取原始图像和标签
    image_path, label_path = pair_files[i]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if image is None or label is None:
        print(f"无法读取文件: {image_path} 或 {label_path}")
        continue

    # 添加椒盐噪声到图像
    noisy_image = add_salt_pepper_noise(image, amount)

    # 保存增强后的图像和原始标签
    output_image_file = os.path.join(output_image_dir, f'noisy_{i}.png')
    output_label_file = os.path.join(output_label_dir, f'noisy_{i}.png')

    cv2.imwrite(output_image_file, noisy_image)
    cv2.imwrite(output_label_file, label)

print("数据增强完成，结果已保存到指定文件夹。")