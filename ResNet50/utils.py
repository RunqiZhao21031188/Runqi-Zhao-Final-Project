import glob
import cv2
import shutil
import torch
from torch import nn, optim
from model import resnet50
from dataloaders import *
from PIL import ImageSequence


# 将图像放置在方形黑色背景的中心
def padding_img(img, bk_shape):
    """
    :param img: 单枚足迹图像
    :param bk_shape: 背景的大小
    :return:
    """
    img_w = img.shape[1]
    img_h = img.shape[0]
    max_ = max(img_w, img_h)
    up_ = int((max_ - img_h) / 2)
    down_ = max_ - img_h - up_
    left_ = int((max_ - img_w) / 2)
    right_ = max_ - img_w - left_
    paddings = cv2.copyMakeBorder(img, up_, down_, left_, right_, cv2.BORDER_CONSTANT, 0)
    paddings = cv2.resize(paddings, bk_shape, cv2.INTER_AREA)
    return paddings

# 读取图像并预处理
def img2tensor(img, shape_=(256, 256)):
    """
    读取图像到RGB通道，自动填充后再resize至指定尺寸
    Args:
        img_path:
        shape_:
    Returns:
    """
    img_temp = img.copy()
    pic_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
    img_temp = padding_img(img_temp, shape_)

    img_temp = pic_transform(img_temp)
    img_temp = img_temp.unsqueeze(0)
    return img_temp

# get label
def get_label():
    return  {"0": "丑牛 Ox", "1": "亥猪 Pig", "2": "午马 Horse", "3": "卯兔 Rabbit", "4": "取消", "5": "子鼠 Rat", "6": "寅虎 Tiger",
             "7": "巳蛇 Snack", "8": "戌狗 Dog", "9": "未羊 Goat", "10": "申猴 Monkey", "11": "空白動作", "12": "辰龙 Dragon", "13": "酉鸡 Rooster"}


# 解析gif
def read_gif(gif_path, save_base="./background"):
    print("解析gif中...In the parsing gif")
    if os.path.exists(save_base):
        shutil.rmtree(save_base)
    os.makedirs(save_base)

    img = Image.open(gif_path)
    i = 0
    for frame in ImageSequence.Iterator(img):
        frame.save(os.path.join(save_base, "%d.png" % i))
        i += 1
    print("gif解析完成Gif parsing complete")

if __name__ == '__main__':
    gif_path_list = glob.glob("./background/gif/*")
    save_base = "./background/frames"

    for gif_path in gif_path_list:
        print(gif_path)
        gif_name = gif_path.split("\\")[-1].split(".")[0]
        cur_save = os.path.join(save_base, gif_name)
        read_gif(gif_path, cur_save)