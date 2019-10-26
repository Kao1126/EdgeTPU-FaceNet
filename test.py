import argparse
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image
import numpy as np


from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image


def takeSecond(elem):
  return elem[0]


def Tpu_FaceRecognize(engine, face_img):

  faces = []
  for face in face_img:
    img = np.asarray(face).flatten()
    result = engine.ClassifyWithInputTensor(img, top_k=200, threshold=-0.5)
    result.sort(key=takeSecond)

    np_result = []
    for i in range(0, len(result)):
      np_result.append(result[i][1])

    faces.append(np_result)
  np_face = np.array(faces)

  return np_face

