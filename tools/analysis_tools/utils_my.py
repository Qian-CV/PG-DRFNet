import pickle
import numpy as np
import glob
import os
import random


def readpkl(path):
    """

    Args:
        path: pkl的路径

    Returns:
        文件中的内容

    """
    # path = '/media/ubuntu/CE425F4D425F3983/datasets/dota_1024/test_split/test1024.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
    f = open(path, 'rb')
    data = pickle.load(f)
    a = np.array(data)
    # print(data)
    # print(type(a))
    # print(a[1, :])
    print(data)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rm_txt_line(path, line_num):
    files = glob.glob(path + '/*.txt')
    for file in files:
        f = open(file, mode='r')
        lines = f.readlines()
        for num in range(line_num):
            del lines[0]
        file_new = open(file, mode='w')
        for line in lines:
            file_new.write(line)
        f.close()

    print(file)


def split_dataset(img_path, dst_path):
    file_list = list(map(lambda x: x[:-4], os.listdir(img_path)))
    N = len(file_list)
    train_idx = random.sample(range(N), int(0.8 * N))
    val_idx = np.setdiff1d(range(N), train_idx)
    train_list = [file_list[idx] + '\n' for idx in train_idx]
    val_list = [file_list[idx] + '\n' for idx in val_idx]
    with open(os.path.join(dst_path, 'train_from_trainval.txt'), 'w') as f:
        # for name in train_list:
        f.writelines(train_list)
    with open(os.path.join(dst_path, 'val_from_trainval.txt'), 'w') as f:
        # for name in val_list:
        f.writelines(val_list)


if __name__ == '__main__':
    img_path = '/media/ubuntu/nvidia/dataset/DOTA-2/split_ss_dota/trainval/images/'
    dst_path = '/media/ubuntu/nvidia/dataset/DOTA-2/split_ss_dota/trainval/'
    split_dataset(img_path, dst_path)
    # rm_txt_line('/media/ubuntu/CE425F4D425F3983/datasets/DOTA/train/labelTxt', 2)
