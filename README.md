# Hand Gesture Recognition using Leap Motion Controller with LSTM

#### 2016025432 박찬현, 2016025987 조민지 with CVLAB, Hanyang Univ.

## Abstract

Leap motion controller(이하 leap motion)는 손의 정보를 제공한다. 우리는 leap motion 의 API를 통해 얻은 데이터에 LSTM을 적용하여  dynamic gesture를 인식한다.



## Requirement

python 2.7

tensorflow 1.10.0 (for python 2.7)

numpy

keyboard

leapmotion sdk 3.2.1 (Windows)



## Install

```shell
$pip install python2.7
$pip install tensorflow-1.10.0-cp27-cp27m-win_amd64.whl
$pip install numpy
$pip install keyboard
```

leap motion sdk : https://www.leapmotion.com/



## Run

#### for train

```shell
$python train.py
```

#### for test

```shell
$python test.py
```



## Result

gif upload

## Folder

```
.
├── checkpoint // train.py를 실행한 모델이 저장됨
|   ├── checkpoint 
|   ├── model.ckpt-100.data-00000-of-00001
|   ├── model.ckpt-100.index
|   └── model.ckpt-100.meta
├── data* // train 데이터
|   ├── down
|   ├── left
|   ├── pew
|   ├── right
|   └── up
├── lib // leap motion Windows SDK
├── minji_test // validation 데이터
|   ├── down
|   ├── left
|   ├── pew
|   ├── right
|   └── up
├── test_data // validation 데이터
├── keyboard_callback.py // keyboard API
├── tensorflow-1.10.0-cp27-cp27m-win_amd64.whl
├── test.py
└── train.py
```





