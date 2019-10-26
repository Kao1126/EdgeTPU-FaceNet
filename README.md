# EdgeTPU-FaceNet
Implement SSD and FaceNet on Edge TPU Accelerator.



You can use 'q' to quit and 'w' to add new class.

## Requirement 
tensorflow - 1.15.0

Ubuntu - 16.04

Edge TPU compilier - 2.0.267

USB Camera

## Demo

**about 25 FPS**

**https://youtu.be/I9F_GT_quFs**

![image](https://github.com/Kao1126/EdgeTPU-FaceNet/blob/master/sample.JPG)

## Usage

#### 1. Download SSD and FaceNet weight

(1). Download [ssd weight](https://drive.google.com/open?id=198woIHpHlhePd0F3ADIXnt5G2bDkEuig)
   and put in weight/SSD
   
(2). Download [facenet weight](https://drive.google.com/open?id=1LZF3Z2Z6mM_gHueMfTKOtxjiiaeLgexV)
   and put in weight/FacaNet
   
Both weights have already been compiled and quantized.
 
#### 2. Run demo.py
####
    $ git clone https://github.com/Kao1126/EdgeTPU-FaceNet.git
    $ cd EdgeTPU-FaceNet
    $ python3 demo.py
   
## Valitading on FLW
1. Create lfw folder
####
    $ mkdir lfw
2. Download LFW datasets and put in lfw
####
    $ python3 validate_lfw.py

## Reference
 - coral:
  https://coral.withgoogle.com/docs/edgetpu/models-intro/

 - tensorflow:
   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize
   
 - face net:
   https://github.com/LeslieZhoa/tensorflow-facenet
   
