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
    - 1. Normal model : It is normal model that use 1 image of user to train the face recognition.
    - 2. Knn model (Works on Euclidean distance) : Uses as many as images to train the model , so that it can be best accurate. This model is 2-4 times slower than normal model.
