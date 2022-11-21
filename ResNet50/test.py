"""
    检索排序
"""
import shutil

import torch
import xlsxwriter
import  pandas  as pd
from model import resnet50
from makegallery import *


# 计算距离
def calculate_distance(ver1, ver2):
    """
    计算向量之间距离
    Args:
        ver1:
        ver2:
    Returns:
    """
    return np.linalg.norm(ver1 - ver2) # 欧氏距离
    # return np.dot(ver1,ver2)/(np.linalg.norm(ver1)*(np.linalg.norm(ver2))) # 余弦距离


# 检索
def search(img_path, gallery_path, model_path, save_path='./results', rankN=3, device='cuda'):
    """
    Args:
        img_path:
        gallery_path:
        model_path:
        save_path: 保存相似度排名靠前的图
        rankN: 保存排名相似度前5的图像
        device:
    Returns:
    """
    # 加载模型
    device = torch.device(device)
    # model = resnet50(pretrained=False)
    # model.load_state_dict(torch.load(model_path))
    # model.eval().to(device)
    model = torch.load(model_path)
    model.eval().to(device)

    # 读取图像并预处理
    img = img2tensor(img_path)
    img = img.to(device)
    image = img.to(device)

    # 过模型
    logits, _ = model(image)
    feature = torch.flatten(logits).cpu().detach().numpy()

    # 读取底库
    df = pd.read_excel(gallery_path)  # 读取xlsx中第一个sheet
    height, width = df.shape
    result = {} # 图像路径: 得分
    for i in range(height):
        print(i)
        temp_feature  = []
        temp_img_path = df.iloc[i, 0]
        for j in range(width):  # 遍历的实际下标，即excel第一行
            if j != 0:
                temp_feature.append(float(df.iloc[i, j]))
        # 计算特征相似度
        temp_feature = np.array(temp_feature)
        fraction = calculate_distance(feature, temp_feature)
        result[temp_img_path] = float(fraction)

    # 排序
    result = sorted(result.items(), key=lambda x: x[1], reverse=False)

    # 取出前n个图像保存
    for i in range(rankN):
        img_path = result[i][0]
        # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_name = str(i) + '_' + img_path.split("\\")[-1]
        save_img_path = os.path.join(save_path, img_name)
        shutil.copy(img_path, save_img_path)
        # cv2.imencode('.png', img)[1].tofile(save_img_path)
        # cv2.imwrite(save_img_path, img)


search(img_path=r"./datasets/query/1635406221(1).jpg",  gallery_path='./gallery.xlsx', model_path="./model/best.pth")