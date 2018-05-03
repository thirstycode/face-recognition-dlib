# Face Recognition From Drone (Dlib)
[![built with Python3](https://img.shields.io/badge/built%20with-Python3-red.svg)](https://www.python.org/)      &nbsp;&nbsp;   [![built with Dlib](https://img.shields.io/badge/build%20with-dlib-green.svg)](http://dlib.net/) 
<br><br>
Recognizes people from video using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. This is revised model of [face_recognition](https://github.com/ageitgey/face_recognition) which works on live video or live stream from drone. [Dlib](http://dlib.net/)'s accuracy is much more than openCV recognition models. (Almost 98% accurate)

### Installation:
```bash
1. git clone https://github.com/thirstycode/face-recognition-drone-dlib/
2. cd face-recognition-drone-dlib
3. pip3 install -r requirements.txt
```
4. Install dlib
  - For Windows Users : [Dlib for windows](https://www.learnopencv.com/install-dlib-on-windows/)
  - For MACOS : [Dlib for macos](https://www.learnopencv.com/install-dlib-on-macos/)
  - For Linux/Ubuntu : [Dlib for ubuntu](https://www.learnopencv.com/install-dlib-on-ubuntu/)
  ```bash
5. pip3 install dlib
6. pip3 install face_recognition
```
### Note:
  - There are two models included in here. 1. Normal model 2. Knn model
    - Normal model : It is normal model that use 1 image of user to train the face recognition.
    - Knn model (Works on Euclidean distance) : Uses as many as images to train the model , so that it can be better in accuracy. This model is 2-4 times slower than normal model.
  - There is status of every person's data . Vip status persons will be shown in green & Blacklisted in red & undetected in white
### How to add training data ? :
  - Normal Model : Add new folder in 'train-images' folder and name it as s3 , s4 etc etc (it depends on how much person's data you have added). Add only 1 image of person in respective folder and add name.txt having name and status.txt having status i.e vip/blacklisted.
  - Knn model : Add new folder in 'knn-train' folder , named as the person's name. Add as much as person's photo in respective folder and make status.txt
#### Execute It:
```bash
For normal model
1. python3 webcam.py
2. python3 drone.py
```
```bash
For knn model
1. python3 webcam_knn.py
2. python3 drone_knn.py
```
```bash
To check on image with normal model
Make sure open image_test and give path to your image
1. python3 image_test.py
```
#### Adjustments for best results:
  - Normal model : Edit variables in config.py (All the working of respective variables are given itself in config.py) 
  - Knn model : Edit variables in config_knn.py (All the working of respective variables are given itself in config_knn.py) 

#### Live stream :
  - Insert live stream url in config.py/config_knn.py to use drone.py
