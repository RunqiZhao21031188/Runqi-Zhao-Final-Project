import os
import cv2
import time
import glob
import shutil
import numpy as np
from PIL import Image
from PIL import ImageSequence


# 提示reminder
def ops_fuc(gif_path, video_path):
    if gif_path is None and video_path is None:
        print("请输入视频路径或gif路径!!!Please enter the video path or gif path")
        exit()
    elif gif_path is not None and video_path is not None:
        print("视频路径与gif路径只需指定一个!!!You only need to specify one video path and one gif path")
        exit()
    elif gif_path is None:
        read_video(video_path)
    else:
        read_gif(gif_path)


# 解析gif Parsing Gif
def read_gif(gif_path, save_base="./background/img"):
    print("解析gif中...In the parsing Gif")
    if  os.path.exists(save_base):
        shutil.rmtree(save_base)
    os.makedirs(save_base)

    img = Image.open(gif_path)
    i = 0
    for frame in ImageSequence.Iterator(img):
        frame.save(os.path.join(save_base, "%d.png"% i))
        i += 1
    print("gif解析完成finsh")


# 读取视频 Read the video
def read_video(video_path, save_base="./background/img", w=960, h=720):
    print("解析视频中...In the parsing video")
    if  os.path.exists(save_base):
        shutil.rmtree(save_base)
    os.makedirs(save_base)

    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    capture.set(cv2.CAP_PROP_FPS, 30)
    resize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    i = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(save_base, "%d.png"% i), frame)
        i = i + 1
    print("视频解析完成Video parsing completed")


    # 合成背景与视频Synthesize background with video
def fus_img(frame, bk_img):
    left_   = int((bk_img.shape[1] - frame.shape[1]) / 2)
    right_  = left_ + frame.shape[1]
    top_    = int((bk_img.shape[0] - frame.shape[0]) / 2)
    bottom_ = top_ + frame.shape[0]

    result = bk_img.copy()
    result[top_:bottom_,left_: right_ , :] = frame

    return result


# 使用摄像头Use camera
def video_demo(bk_path="./background/img/*", w=960, h=720):
    bk_path_list = glob.glob(bk_path)
    bk_path_list.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))

    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    capture.set(cv2.CAP_PROP_FPS, 30)
    resize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    i = 0
    while (True):
        ret, frame = capture.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (int(w/2), int(h/2)))
        cur_bk = cv2.imread(bk_path_list[i])
        cur_bk = cv2.resize(cur_bk, (w, h))

        cur_fus = fus_img(frame, cur_bk)

        cv2.imshow("video", cur_fus)
        c = cv2.waitKey(50)

        if i + 1 == len(bk_path_list):
            i = 0
        else:
            i = i + 1

        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ =='__main__':

    # 指定GIF路径或者视频路径 Specify the GIF path or video path
    gif_path   = None
    video_path = r"./background/video/1.mp4"

    ops_fuc(gif_path, video_path)
    # video_demo()

