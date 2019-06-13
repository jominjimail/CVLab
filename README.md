### Hand Gesture Recognition using Leap Motion Controller with LSTM

##### 2016025432 박찬현, 2016025987 조민지 with CVLAB, Hanyang Univ.

#### Abstract

Leap motion controller(이하 leap motion)는 손의 정보를 제공한다. 우리는 leap motion 의 API를 통해 얻은 데이터에 LSTM을 적용하여  dynamic gesture를 인식한다.



#### Requirement

tensorflow version : 1.13.1

numpy

leap motion sdk : 2.3.1



#### Run

$python train.py



#### Folder

LeapTrainer.js - master 를 수정하여 데이터를 .csv로 저장

data* : train을 위해 저장된 데이터들

test_data : validation을 위해 저장된 데이터들

lasm* : 참고한 lstm 코드

data_process.py : 데이터 전처리를 위한 코드



