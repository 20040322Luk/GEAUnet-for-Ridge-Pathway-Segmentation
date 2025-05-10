import os
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob

# 定义文件夹路径
output_image_dir = './cutmix_img'     # 增强后的图像目录
output_label_dir = './cutmix_labels'      # 增强后的标签目录

# 读取文件列表
output_image_files = sorted(glob(os.path.join(output_image_dir, '*.png')))
output_label_files = sorted(glob(os.path.join(output_label_dir, '*.png')))

# 确保文件数量匹配
if len(output_image_files) != len(output_label_files):
    raise ValueError("增强后的图像文件和标签文件数量不匹配")

# 可视化比较增强后的图像和标签
def visualize_augmented_image_and_label(image_file, label_file):
    # 读取图片
    image = cv2.imread(image_file)
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

    # 转换为RGB格式以便显示（因为OpenCV读取的图像是BGR格式）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建颜色映射
    cmap = plt.cm.get_cmap('viridis', 4)  # 4个标签值（0, 1, 2, 3）

    # 创建画布
    plt.figure(figsize=(10, 5))

    # 显示增强后的图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Augmented Image')
    plt.axis('off')

    # 显示增强后的标签
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap=cmap, vmin=0, vmax=3)
    plt.title('Augmented Label')
    plt.axis('off')

    # 添加颜色条
    plt.colorbar(ticks=[0, 1, 2, 3])

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# 随机选择一张增强后的图像和标签进行可视化
def visualize_random_comparison():
    idx = random.randint(0, len(output_image_files) - 1)
    image_file = output_image_files[idx]
    label_file = output_label_files[idx]
    visualize_augmented_image_and_label(image_file, label_file)

# 进行可视化检查
for i in range(10):
    visualize_random_comparison()

print("检查完成。关闭图像窗口以选择新的图像进行比较。")