# coding=utf-8
import cv2
import os
import glob
import math
import shutil
import numpy as np
import pygame, time
from utils import *
from PIL import Image
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageFont


# 图像中加文字Text in image
def add_text(img, text):
    frame = img.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    # 在图片上添加文字（支持中文）
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("simhei.ttf", 40, encoding="utf-8")
    draw.text((50, 80), text, (255, 0, 0), font=font)
    cv2img2 = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2img2


# 合成背景与视频Synthesize background with video
def fus_img(frame, bk_img):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    bk_img = cv2.cvtColor(bk_img, cv2.COLOR_BGR2RGB)
    bk_img = Image.fromarray(bk_img)

    result = Image.blend(frame, bk_img, 0.4)

    result = np.array(result, dtype=np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def playBGM():
    bgm_path = r'./background/voice/all/1.mp3'
    pygame.mixer.init()
    pygame.mixer.music.load(bgm_path)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.Channel(0).play(pygame.mixer.Sound(bgm_path))
    main()


def playSound(sound_path):
    bgm_path = sound_path
    pygame.mixer.Channel(1).play(pygame.mixer.Sound(bgm_path))
    time.sleep(1)
    pygame.mixer.Channel(1).stop()


def main():
    # 加载模型Loading model
    model_path = "./best/best.pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # gif gallery
    bk_path_base = "./background/frames"

    # w = 1280
    # h = 720
    cap = cv2.VideoCapture(0)
    w = int(cap.get(3))
    h = int(cap.get(4))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # resize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    begin_frames_list = glob.glob("./background/begin_frames/*.png")
    begin_frames_list.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))

    frame_num = 1
    none_num  = 0 # 无动作帧数No action frames
    while True:
        _, frame_img = cap.read()
        # frame_img = cv2.flip(frame_img, 1)

        # 预处理当前帧Preprocess the current frame
        img = img2tensor(frame_img)
        label = get_label()

        # 过模型
        logits, _ = model(img)
        action = label[str(logits.argmax().item())]
        # # print(action)
        # action = "鸡 a"

        # 每100帧识别一次It is identified every 100 frames
        if frame_num % 60 != 0:
            print("等待识别")
            cv2.imshow('webcam', frame_img)
            # 按Q关闭窗口 press Q quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 停止动作终止程序Stop action terminates the procedure
            if  "取消" in action:
                print(action)
                cv2.imshow('webcam', frame_img)
                break
            # 无动作无特效No action, no special effects
            elif "空白" in action:
                print(action)
                if none_num >= 5:
                    for begin_frames_path in begin_frames_list:
                        cur_begin_frame = cv2.imdecode(np.fromfile(begin_frames_path, dtype=np.uint8), 1)
                        cv2.imshow('webcam', cur_begin_frame)
                        cv2.waitKey(10)
                    none_num = 0
                else:
                    none_num = none_num + 1
                    cv2.imshow('webcam', frame_img)

                # 按Q关闭窗口
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                pass
            # 对应不同生肖不同特效Corresponding to different zodiac different effects
            elif "鸡" in action :
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "鸡", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/鸡/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "狗" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "狗", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/狗/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif  "龙" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "龙", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/龙/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "兔" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "兔", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/兔/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "猪" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "猪", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/猪/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "马" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "马", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/马/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "牛" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "牛", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/牛/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "羊" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "羊", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/羊/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "鼠" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "鼠", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/鼠/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "猴" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "猴", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/猴/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "蛇" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "蛇", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/蛇/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            elif "虎" in action:
                cur_gif_path = glob.glob(os.path.join(bk_path_base, "虎", "*"))
                cur_gif_path.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
                cur_sound_path = "./background/voice/虎/1.mp3"
                playSound(cur_sound_path)
                for gif_path in cur_gif_path:
                    cur_bk = cv2.imdecode(np.fromfile(gif_path, dtype=np.uint8), 1)
                    cur_bk = cv2.resize(cur_bk, (w, h))
                    cur_text = action.split(" ")[0] + "\n" + action.split(" ")[1]
                    cur_bk = add_text(cur_bk, cur_text)
                    cur_img = fus_img(frame_img, cur_bk)
                    cv2.imshow('webcam', cur_img)
                    cv2.waitKey(30)
            else:
                # show image
                cv2.imshow('webcam', frame_img)
        frame_num = frame_num + 1
        # 按Q关闭窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    playBGM()
    # main()
