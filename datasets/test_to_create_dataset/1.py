import os


def main(directory, output_file):
    # 获取目录下的所有文件名
    files = os.listdir(directory)

    # 去除文件后缀并保存到列表中
    file_names = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(directory, f))]

    # 将文件名写入输出文件
    with open(output_file, 'w') as f:
        for name in file_names:
            f.write(name + '\n')


if __name__ == "__main__":
    # 指定目录和输出文件路径
    directory = 'cutmix_img'
    output_file = 'train.txt'

    # 调用主函数
    main(directory, output_file)