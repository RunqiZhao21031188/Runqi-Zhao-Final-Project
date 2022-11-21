"""
    制作数据集
"""
import shutil
import random
import cv2
import csv
import os
import glob
import numpy as np
import pandas as pd


def makeTrain(base_path, save_path, label_path):
    """
    制作训练集
    Args:
        base_path:
        save_path:
        label_path: 原始数据的标签文件
    Returns:
    """
    img_path_list = glob.glob(os.path.join(base_path, "*", "*"))

    # 读取csv，获取人员id与图像名的对应列表
    id_img = np.array(pd.read_csv(label_path, usecols=[0, 2]))
    people_id_list = id_img[:, 0]
    image_id_list = id_img[:, 1]

    new_id = -1 # 重新标签，不按照csv里的id，按照0-99999的顺序
    last_id = -9999999999
    for img_path in img_path_list:
        img_name = img_path.split('\\')[-1].split(".")[0]
        # 如果图像名最后一个字符是X，去掉X
        if img_name[-1] == "X":
            img_name = img_name[0:-1]
        # 查找当前图像名在csv中的索引
        idx = np.where(image_id_list == img_name)
        # 如果没找到
        if len(idx) == 0:
            print("Not find %s" % img_name)
            exit()
        # 当前图像对应的原始id
        cur_id = people_id_list[idx[0][0]]
        if cur_id != last_id:
            new_id = new_id + 1
        # 转存图像
        cur_id_path = os.path.join(save_path, str(new_id))
        if not os.path.exists(cur_id_path):
            os.makedirs(cur_id_path)
        shutil.copy(img_path, os.path.join(cur_id_path, img_path.split('\\')[-1]))
        last_id = cur_id
        print(img_path)


def makeTest(train_path, test_path, test_pro=0.1):
    """
    制作测试集，每一个类别按照类别中图像总数目的比例分给测试集
    Args:
        train_path:
        test_path:
        test_pro:  测试集图像比例
    Returns:
    """
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    id_list = os.listdir(train_path)

    for id_ in id_list:
        train_id_path = os.path.join(train_path, id_)
        # 当前训练集id下图像数目
        cur_id_train_nums = len(os.listdir(train_id_path))
        # 计算测试集当前id的图像数目(保证每一类最少有一张)
        # 如果当前id下图像数目过少,测试集只取1张
        if cur_id_train_nums <= 3:
            cur_id_test_nums = 1
        else:
            cur_id_test_nums = cur_id_train_nums * test_pro + 1
            img_list = os.listdir(train_id_path)
            # 打乱
            random.shuffle(img_list)
            for i in range(len(img_list)):
                if i >= cur_id_test_nums:
                    break
                img_path = os.path.join(train_id_path, img_list[i])
                test_id_path = os.path.join(test_path, id_)
                if not os.path.exists(test_id_path):
                    os.makedirs(test_id_path)
                shutil.move(img_path, os.path.join(test_id_path, img_list[i]))
                print(img_list[i])


if __name__ == "__main__":
    # makeTrain(base_path=r"F:\原始数据\实物鞋\image", save_path="./datasets/or1W/train",
    #           label_path="F:\原始数据\实物鞋\label.csv")

    makeTest("./datasets/or1W/train", "./datasets/or1W/test", 0.1)