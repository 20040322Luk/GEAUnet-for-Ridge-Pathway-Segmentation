import os
import glob

# 定义要删除文件的目录
directory = 'finaltest/VOC2007/JPEGImages'

# 使用glob查找所有.png文件
png_files = glob.glob(os.path.join(directory, '*.png'))

# 删除所有找到的.png文件
for file_path in png_files:
    try:
        os.remove(file_path)
        print(f"已删除文件: {file_path}")
    except Exception as e:
        print(f"无法删除文件 {file_path}: {e}")

print("所有.png文件已删除。")