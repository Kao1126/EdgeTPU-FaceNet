# EdgeTPU-FaceNet
Implement SSD and FaceNet on Edge TPU Accelerator.

You can use 'q' to quit and 'w' to add new class.

## Requirement 
tensorflow - 1.15.0
Ubuntu - 16.04
Edge TPU compilier - 2.0.267
USB Camera

## Usage

#### 1. Download SSD and FaceNet weight

Download ssd weight from https://drive.google.com/open?id=198woIHpHlhePd0F3ADIXnt5G2bDkEuig
   and put in weight/SSD/
   
Download facenet weight from https://drive.google.com/open?id=1LZF3Z2Z6mM_gHueMfTKOtxjiiaeLgexV
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
1. coral:
  https://coral.withgoogle.com/docs/edgetpu/models-intro/

2. tensorflow:
   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize
   
3. face net:
   https://github.com/LeslieZhoa/tensorflow-facenet
   
