"""
    读取与预训练模型，制作底库
"""
import glob
import cv2
import torch
import xlsxwriter
from torch import nn, optim
from model import resnet50
from dataloaders import *
from PIL import Image


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
def img2tensor(img_path, shape_=(224, 224)):
    """
    读取图像到RGB通道，自动填充后再resize至指定尺寸
    Args:
        img_path:
        shape_:
    Returns:
    """
    pic_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = padding_img(img, shape_)
    img = pic_transform(img)
    img = img.unsqueeze(0)
    return img


# 对地库图像进行特征提取并录入excel
def img2txt(base_path, gallery_path, model_path, device='cuda'):
    """
    对地库图像进行特征提取并录入excel
    Args:
        base_path:
        gallery_path:
        model_path:
        device
    Returns:
    """
    workbook = xlsxwriter.Workbook(gallery_path)  # 新建excel表

    worksheet = workbook.add_worksheet('sheet1')

    # 加载模型
    device = torch.device(device)
    # model = resnet50(pretrained=False)
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.eval().to(device)

    # 图像预处理提取特征存入表格
    img_list = glob.glob(base_path)
    for img_path in img_list:
        print(img_path)
        img = img2tensor(img_path)
        img = img.to(device)
        image = img.to(device)
        logits, _ = model(image)
        feature = torch.flatten(logits).cpu().detach().numpy()
        # 先写入图像路径
        worksheet.write(img_list.index(img_path), 0, img_path)
        # 再写入512维特征
        for i in range(len(feature)):
            worksheet.write(img_list.index(img_path), i+1, str(feature[i]))
    workbook.close()


if __name__ == "__main__":
    img2txt(base_path=r"./datasets/onlyshoe/*", gallery_path='./gallery.xlsx', model_path="./model/best.pth")