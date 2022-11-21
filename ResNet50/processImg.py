"""
Reference    https://zhuanlan.zhihu.com/p/38425733?utm_source=wechat_session&utm_medium=social&utm_oi=941353197001019392
#     Code  https://github.com/Arthurzhangsheng/NARUTO_game https://github.com/LuXuanqing/tutorial-image-recognition
"""
import os
import glob
import cv2
import random
import shutil
import queue
import threading
import numpy as np


def img_queue():
    """
    构建图像路径的队列
    :return:
    """
    global q
    base_path = "./datasets/or1W/test"
    id_list = os.listdir(base_path)

    for id_ in  id_list:
        id_path = os.path.join(base_path, id_)
        q.put(id_path)


def work():
    global q

    save_path = "./datasets/or1W_rotate/test"

    each_num = int(1000/5)

    for i in range(each_num):
        if q.qsize() == 0:
            break
        path1 = q.get()
        randomRotate(path1, save_path)


def func():
    work()


def randomRotate(id_path, save_path, rotate_nums=3):
    """
    对图像进行随机旋转
    Args:
        id_path: id级目录
        save_path:
        rotate_nums:
    Returns:
    """
    id_ = id_path.split("\\")[-1]
    img_path_list = glob.glob(os.path.join(id_path, "*"))
    for img_path in img_path_list:
        img_name = img_path.split("\\")[-1]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        cur_img_save_path = os.path.join(save_path, id_)
        if not os.path.exists(cur_img_save_path):
            os.makedirs(cur_img_save_path)
        # 转存未旋转的图
        shutil.copy(img_path, os.path.join(cur_img_save_path, img_name))
        # 产生随机数组，不重复
        angel_list = random.sample(range(-90, 90), rotate_nums)
        # 开始随机旋转
        for random_angel in angel_list:
            M = cv2.getRotationMatrix2D(center, random_angel, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            new_img_name = img_name.split(".")[0] + "_" + str(random_angel) + ".jpg"
            cv2.imencode('.jpg', rotated)[1].tofile(
                os.path.join(cur_img_save_path, new_img_name))
            cv2.imwrite(os.path.join(cur_img_save_path, new_img_name), rotated)
        print(img_path)


if __name__ == "__main__":
    # 初始化栈
    q = queue.LifoQueue()

    img_queue()
    threads_list = []

    # 创建4个线程
    for i in range(5):
        thread1 = threading.Thread(target=func)
        threads_list.append(thread1)
    for i in threads_list:
        i.start()
    for i in threads_list:
        i.join()
