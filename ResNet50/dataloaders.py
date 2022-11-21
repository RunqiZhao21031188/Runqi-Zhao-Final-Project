# coding:utf8

import torch
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


# 将path下所有的图像序列文件夹路径和标签整理成一个列表 Collate all the image sequence folder paths and labels under Path into a list
def list_data_roots(path):
    image_seq_roots = []
    for class_name in os.listdir(path):  # 取得是文件夹或者文件名字It's the name of the folder or file
        # print("1",class_name) # 这里的class_name是文件夹的名字 class_name here is the name of the folder
        class_root = os.path.join(path, class_name)
        temp_label = int(class_name)  # 读的是文件夹的名字，文件夹的名字是从001开始的。这里想让标签从000开始，所以减去1 It reads the name of the folder, and the folder name starts with 001. We want the label to start at 000, so we subtract 1
        for img_seq in os.listdir(class_root):
            img_seq_root = os.path.join(class_root, img_seq)  # 图像路径文件夹，这个路径下存放的就是图像了 Image path folder, this path is stored in the image
            item = (img_seq_root, temp_label)
            image_seq_roots.append(item)
    return image_seq_roots  # （图像路径文件夹及标签构成元组，再构成列表）(Image path folder and label form a tuple, and then form a list)


# 图片加载函数Image loading function
def img_loader(img_path):
    pic_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(img_path)
    image = image.convert('RGB')
    image = image.resize((256, 256), resample=Image.LANCZOS)
    image = pic_transform(image)
    # image = image.unsqueeze(0)
    return image


class foot_dataset(data.Dataset):
    # data_path:训练/测试数据所在目录
    # loader:图片加载函数
    # transform:加载后的图片转换处理
    # target_transform:加载后的标签转换处理
    def __init__(self, data_path, loader=img_loader, transform=None, target_transform=None):
        data_roots = list_data_roots(data_path)  # （图像路径文件夹及标签构成元组后，再构成列表）(Image path folder and label form a tuple, then form a list)
        if len(data_roots) == 0:
            print("Error: there is no seq in ", data_roots)
        self.root = data_path
        self.loader = loader
        self.data_roots = data_roots
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        temp_root, temp_label = self.data_roots[index]  # 相当于图像文件夹的路径，及标签Equivalent to the path to the image folder, and the label
        img_seq_temp = self.loader(temp_root)  # 将给图像文件夹的路径，给上面的形参The path to the image folder will be given to the above parameter
        if self.transform is not None:
            img_seq_temp = self.transform(img_seq_temp)
        if self.target_transform is not None:
            temp_label = self.target_transform(temp_label)
        return img_seq_temp, temp_label  # (frame_num,channel,224,224) 和 标签号

    def __len__(self):
        return len(self.data_roots)


if __name__ == "__main__":
    data_path = r"./datasets/train"
    train_data = foot_dataset(data_path)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    for step, (data, label) in enumerate(train_loader):
        print(data.size())
        print(label)
        print(" ")