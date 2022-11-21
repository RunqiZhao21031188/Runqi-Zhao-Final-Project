# Run Sequence
1.open 'ProcessImg.py'. Convert the video of the data set into pictures.
2.open 'main.py' to train model.
3.open 'video.py' to run the full project for interaction, have fun!

# Processing
The thrust goal of the project is to elevate people's interest in traditional Chinese culture, which necessarily requires the use of new and innovative approaches. As such, I used machine learning as a tool to explore a new form of interaction.

First of all, the code of this project processed the data and converted the data set of video files into a code recognizable format, i.e., images. Then ResNet was used for training, and after the successful completion of training, Video.py was applied to interact with the camera, namely to play the corresponding Chinese zodiac animation text and music after recognizing the corresponding image.

In the dataset, I created 14 types of files according to the twelve zodiac categories (in addition to "cancel" and "blank"). Each file is a multiple 2-3 minute video of the corresponding zodiac hand shadow. To ensure the accuracy of the recognition, the camera used to shoot the dataset should be the same as the final device camera (same resolution and aspect ratio).

Data processing phase: videos of the dataset were converted into images and split into a training set, validation set, and test set in the ratio of 60%, 20%, and 20%, and then the processed images were stretched and mirrored to expand the data volume of the training set and enhance the prediction accuracy of the model.

The model was trained using ResNet with a Torch architecture.Microsoft put forward this model in 2015. Compared with other CNN models, this structure is deeper and the overall performance is improved. Based on the short-circuit connection path part, no other parameters are introduced to increase the computational complexity. Therefore, even if the number of model layers is increased, the training speed of the model can be accelerated and the training effect of the model can be improved(Song Yang,2022).

Effect phase: The interaction logic was designed and feedback effects were produced.
Testing phase: I invited testers to participate in the interaction of the project according to the prompts, and the recognition was more accurate and all received positive feedback.

# Reference

https://github.com/Arthurzhangsheng/NARUTO_game  

https://github.com/LuXuanqing/tutorial-image-recognition

https://github.com/tornadomeet/ResNet

https://zhuanlan.zhihu.com/p/38425733?utm_source=wechat_session&utm_medium=social&utm_oi=941353197001019392
