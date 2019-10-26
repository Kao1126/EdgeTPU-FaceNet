
# coding: utf-8

# In[1]:
from PIL import Image
from tensorflow.python.platform import gfile
from test import Tpu_FaceRecognize
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import re
from utils import *
import config
import cv2
import h5py
from config import Embedding_book
# In[2]:

def Create_embeddings(face_engine):
    
    img_arr, class_arr = align_face()
    embs = Tpu_FaceRecognize(face_engine, img_arr)

    f = h5py.File(Embedding_book,'w')
    class_arr=[i.encode() for i in class_arr]
    f.create_dataset('class_name',data=class_arr)
    f.create_dataset('embeddings',data=embs)
    f.close()



# In[3]:


def align_face(path='pictures/'):

    img_paths=os.listdir(path)
    class_names=[a.split('.')[0] for a in img_paths]
    img_paths=[os.path.join(path,p) for p in img_paths]
    scaled_arr=[]
    class_names_arr=[]
    
    for image_path,class_name in zip(img_paths,class_names):

        img = cv2.imread(image_path)
        scaled = cv2.resize(img,(160, 160),interpolation=cv2.INTER_LINEAR)

        scaled = Image.fromarray(cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB))
        scaled = np.asarray(img)

        scaled_arr.append(scaled)
        class_names_arr.append(class_name)
    

    scaled_arr=np.asarray(scaled_arr)
    class_names_arr=np.asarray(class_names_arr)
    print("scaled_arr", scaled_arr.shape)
    print('class_names_arr', class_names_arr)
    return scaled_arr,class_names_arr


