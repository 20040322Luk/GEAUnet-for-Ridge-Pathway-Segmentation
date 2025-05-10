import os
import numpy as np


def load_npy_file(file_path):
    """加载npy文件"""
    return np.load(file_path)


def count_and_modify_pixels(data, target_value=3, percentage=0.005):
    """统计目标值并修改指定比例的目标值"""
    # 获取所有目标值的索引
    target_indices = np.where(data == target_value)

    # 计算目标值的数量
    num_targets = len(target_indices[0])

    # 计算需要修改的数量
    num_to_modify = int(num_targets * percentage)

    # 随机选择需要修改的索引
    indices_to_modify = np.random.choice(num_targets, num_to_modify, replace=False)

    # 修改选中的索引
    for idx in indices_to_modify:
        data[target_indices[0][idx], target_indices[1][idx]] = 0

    return data


def save_npy_file(data, file_path):
    """保存修改后的数据到npy文件"""
    np.save(file_path, data)


def process_npy_files_in_directory(input_directory, output_directory):
    """处理目录中的所有npy文件，并将修改后的文件保存到输出目录"""
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 获取目录中的所有npy文件
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]

    for npy_file in npy_files:
        input_file_path = os.path.join(input_directory, npy_file)
        output_file_path = os.path.join(output_directory, npy_file)

        # 加载数据
        data = load_npy_file(input_file_path)

        # 统计并修改像素值
        modified_data = count_and_modify_pixels(data)

        # 保存修改后的数据
        save_npy_file(modified_data, output_file_path)
        print(f"Processed and saved: {output_file_path}")


if __name__ == "__main__":
    input_directory = "./results/processed_baseline"
    output_directory = "./results/processed_baseline"  # 输出目录路径
    process_npy_files_in_directory(input_directory, output_directory)
