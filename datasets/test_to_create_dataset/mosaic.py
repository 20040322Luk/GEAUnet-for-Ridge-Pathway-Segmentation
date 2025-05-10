import os
import cv2
import numpy as np
import random
from glob import glob

# 定义Mosaic数据增强函数
def build_mosaics(image1: np.array, image2: np.array, image3: np.array, image4: np.array,
                  label1: np.array, label2: np.array, label3: np.array, label4: np.array,
                  type: str = "diag") -> (np.array, np.array):
    if image1.shape != image2.shape or image1.shape != image3.shape or image1.shape != image4.shape:
        raise ValueError("All images should have the same shape")

    if label1.shape != image1.shape or label2.shape != image2.shape or label3.shape != image3.shape or label4.shape != image4.shape:
        raise ValueError("Images and labels should have the same shape")

    h, w = image1.shape[:2]

    # 生成新的图像和标签
    mosaic_image = np.zeros_like(image1)
    mosaic_label = np.zeros_like(label1)

    if type == "diag":
        # 以对角线方式拼接
        mosaic_image[:h // 2, :w // 2] = image1[:h // 2, :w // 2]
        mosaic_image[:h // 2, w // 2:] = image2[:h // 2, w // 2:]
        mosaic_image[h // 2:, :w // 2] = image3[h // 2:, :w // 2]
        mosaic_image[h // 2:, w // 2:] = image4[h // 2:, w // 2:]

        mosaic_label[:h // 2, :w // 2] = label1[:h // 2, :w // 2]
        mosaic_label[:h // 2, w // 2:] = label2[:h // 2, w // 2:]
        mosaic_label[h // 2:, :w // 2] = label3[h // 2:, :w // 2]
        mosaic_label[h // 2:, w // 2:] = label4[h // 2:, w // 2:]

    elif type == "upper_lower":
        # 以上下方式拼接
        mosaic_image[:h // 2, :] = image1[:h // 2, :]
        mosaic_image[h // 2:, :] = image2[h // 2:, :]

        mosaic_label[:h // 2, :] = label1[:h // 2, :]
        mosaic_label[h // 2:, :] = label2[h // 2:, :]

    elif type == "left_right":
        # 以左右方式拼接
        mosaic_image[:, :w // 2] = image1[:, :w // 2]
        mosaic_image[:, w // 2:] = image2[:, w // 2:]

        mosaic_label[:, :w // 2] = label1[:, :w // 2]
        mosaic_label[:, w // 2:] = label2[:, w // 2:]

    elif type == "random":
        # 以随机方式拼接
        pairs = [(image1, label1), (image2, label2), (image3, label3), (image4, label4)]
        random.shuffle(pairs)

        mosaic_image[:h // 2, :w // 2] = pairs[0][0][:h // 2, :w // 2]
        mosaic_image[:h // 2, w // 2:] = pairs[1][0][:h // 2, w // 2:]
        mosaic_image[h // 2:, :w // 2] = pairs[2][0][h // 2:, :w // 2]
        mosaic_image[h // 2:, w // 2:] = pairs[3][0][h // 2:, w // 2:]

        mosaic_label[:h // 2, :w // 2] = pairs[0][1][:h // 2, :w // 2]
        mosaic_label[:h // 2, w // 2:] = pairs[1][1][:h // 2, w // 2:]
        mosaic_label[h // 2:, :w // 2] = pairs[2][1][h // 2:, :w // 2]
        mosaic_label[h // 2:, w // 2:] = pairs[3][1][h // 2:, w // 2:]

    else:
        raise ValueError("Invalid type. Supported types: 'diag', 'upper_lower', 'left_right', 'random'")

    return mosaic_image, mosaic_label


# 定义文件夹路径
image_dir = 'b_and_a_test/original_img'
label_dir = 'b_and_a_test/original_label'
output_image_dir = 'b_and_a_test/mosaic_img'
output_label_dir = 'b_and_a_test/mosaic_labels'

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

# 定义增强类型
types = ["diag", "upper_lower", "left_right", "random"]

# 确保生成的组合数量是原始数据的两倍
num_combinations = len(image_files) // 2

# 进行数据增强
for i in range(num_combinations):
    # 随机选择四张图像和标签
    selected_pairs = random.sample(pair_files, 4)

    image1 = cv2.imread(selected_pairs[0][0], cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(selected_pairs[1][0], cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(selected_pairs[2][0], cv2.IMREAD_GRAYSCALE)
    image4 = cv2.imread(selected_pairs[3][0], cv2.IMREAD_GRAYSCALE)

    label1 = cv2.imread(selected_pairs[0][1], cv2.IMREAD_GRAYSCALE)
    label2 = cv2.imread(selected_pairs[1][1], cv2.IMREAD_GRAYSCALE)
    label3 = cv2.imread(selected_pairs[2][1], cv2.IMREAD_GRAYSCALE)
    label4 = cv2.imread(selected_pairs[3][1], cv2.IMREAD_GRAYSCALE)

    # 遍历不同的增强类型
    for type in types:
        mosaic_image, mosaic_label = build_mosaics(image1, image2, image3, image4, label1, label2, label3, label4,
                                                   type=type)

        # 保存结果
        output_image_file = os.path.join(output_image_dir, f'mosaic_{i}_{type}.png')
        output_label_file = os.path.join(output_label_dir, f'mosaic_{i}_{type}.png')

        cv2.imwrite(output_image_file, mosaic_image)
        cv2.imwrite(output_label_file, mosaic_label)

print("数据增强完成，结果已保存到指定文件夹。")