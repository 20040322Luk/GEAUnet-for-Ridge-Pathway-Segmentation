import os
import numpy as np
from PIL import Image


def grayscale_images_in_directory(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录中所有文件
    files = os.listdir(input_dir)

    # 遍历所有文件
    for file in files:
        if file.lower().endswith('.png'):
            # 构建输入文件的完整路径
            input_file_path = os.path.join(input_dir, file)

            # 打开PNG图像
            png_image = Image.open(input_file_path)

            # 将图像转换为灰度图像
            grayscale_image = png_image.convert('L')

            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_dir, file)

            # 保存灰度图像
            grayscale_image.save(output_file_path)

            print(f"已将 {input_file_path} 转换为灰度图像并保存为 {output_file_path}。")


# 示例用法
input_directory = './strawberry_dataset/test/annotations_prepped_test'  # 替换为你的输入目录路径
output_directory = './strawberry_dataset/test/gray_gt_img'  # 替换为你的输出目录路径

grayscale_images_in_directory(input_directory, output_directory)
