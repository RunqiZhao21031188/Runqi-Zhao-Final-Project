from utils import *


if __name__ =='__main__':
    model_path = "./best/best.pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))

    model.eval()

    # 读取图像并预处理 Read the image and preprocess it
    img = cv2.imread("./2.jpg")
    img = img2tensor(img)
    label = get_label()

    # 过模型 processing models
    logits, _ = model(img)
    predict = label[str(logits.argmax().item())]
    print(predict)

