import os
import shutil

def copy_files_from_list(src_dir, dst_dir, list_file):
    """
    根据列表文件中的文件名，从源目录复制文件到目标目录。

    :param src_dir: 源目录路径
    :param dst_dir: 目标目录路径
    :param list_file: 包含文件名的列表文件路径
    """
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 读取列表文件
    with open(list_file, 'r') as f:
        file_names = f.read().splitlines()

    # 遍历文件名列表
    for file_name in file_names:
        # 构建源文件路径
        src_path = os.path.join(src_dir, file_name + '.jpg')  # 假设文件扩展名为.jpg
        if os.path.exists(src_path):
            # 构建目标文件路径
            dst_path = os.path.join(dst_dir, file_name + '.jpg')
            # 复制文件
            shutil.copy2(src_path, dst_path)
            print(f"复制文件: {file_name}.jpg")
        else:
            print(f"警告: 文件 {file_name}.jpg 不存在")

if __name__ == "__main__":
    # 源目录和目标目录#'../VOC_B/VOC2007/SegmentationClass'
    source_directory = '../VOC_B/VOC2007/SegmentationClass'#'../VOC_B/VOC2007/JPEGImages/'
    destination_directory = './test_label'
    list_file = './val.txt'

    # 调用函数
    copy_files_from_list(source_directory, destination_directory, list_file)
