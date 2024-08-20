# Naruto Hand Sign Classification

This is a project based on the anime Naruto by Masashi Kishimoto. In the show, the ninjas weave various hand-signs combinations as a way to use their techniques referred to as jutsus. This project attempts to use Deep Learning and Computer Vision techniques to identify what handsign an individual is making. 


## Live Demo Classification

The trained model performed great for 10 of the classes while classes Dog and Serpent are the machines weakpoints as it is not as consistent in classifying those two. See below for a demonstration of the live classification.

https://user-images.githubusercontent.com/41130598/174506955-9ea1c002-effd-4982-9365-2dba0703b6a7.mp4


## Models

We experimented on MobileNetV2, ResNet50, VGG16, and InceptionV3 architectures due to their prominence in image classification tasks. We froze all but their last few layers to take advantage of their feature extraction capabilities they learned from being trained on the imagenet dataset, effectively using transfer-learning for our image-classifcation task. Out of the four models/architectures, VGG16 performed the best at an accuracy of **93.60%** on the mannually curated static test dataset as well as a minimum accuracy of **83.33%** for the live demo.

Architecture of the VGG16 model is shown below.

**![image](https://user-images.githubusercontent.com/74623611/174507226-39e2be13-ff11-4697-b0c2-f2b2a9123474.png)**

See links below for details of the four models.
1. [MobileNetV2](https://keras.io/api/applications/mobilenet/)
2. [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
3. [VGG16](https://keras.io/api/applications/vgg/)
4. [InceptionV3](https://keras.io/api/applications/inceptionv3/)

## Data

The data used in this project was manually collected and augmented by recording a video from a webcam utilizing OpenCV, following a number of data-augmentations such as flips and rotations to increase dataset size.


## Packages Used
The requirements for the project are as follows: <br>
1. Python=3.7.7
2. Tensorflow=2.8.0
3. OpenCV=4.6.0.66
4. Numpy=1.21.5
5. Pillow=9.1.0
6. Mediapipe=0.8.10

## Contributers

Contributers of the project are listed below. Click the hyperlink to follow up on more projects.

1. [Saad Hossain](https://github.com/sdhossain)
2. [Jaeyoung Kang](https://github.com/j97kang)
3. [Yazan Masoud](https://github.com/yaxan)
4. [Michael Frew](https://github.com/mooshifrew)
