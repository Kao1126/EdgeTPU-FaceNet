# EdgeTPU-FaceNet
Implement SSD and FaceNet on Edge TPU Accelerator
You can use 'q' to quit and 'w' to add new class.

## Requirement 
tensorflow - 1.15.0
Ubuntu - 16.04
Edge TPU compilier - 2.0.267
USB Camera

## Usage

1. Download SSD and FaceNet weight
  (1). download ssd weight from https://drive.google.com/open?id=198woIHpHlhePd0F3ADIXnt5G2bDkEuig
   and put in weight/SSD/
  (2). download facenet weight from https://drive.google.com/open?id=1LZF3Z2Z6mM_gHueMfTKOtxjiiaeLgexV
   and put in weight/FacaNet
   Both weights have already been compiled and quantized.
 
2. Run 
   ##
   $ git clone https://github.com/Kao1126/EdgeTPU-FaceNet.git
   $ cd EdgeTPU-FaceNet
   $ python3 demo.py
   ##

## Valitading on FLW
  ##
  $ mkdir lfw
  ##
  Download LFW datasets and put in lfw
  ##
  $ python3 validate_lfw.py
  ##
