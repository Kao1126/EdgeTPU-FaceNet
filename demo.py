import cv2
import h5py
import time
import platform
import numpy
import subprocess
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from embeddings import  Create_embeddings
import numpy as np
from test import  Tpu_FaceRecognize
from config import*


def crop_image(ans, frame):
    Images_cropped = []
    for i in range(0, len(ans)):
        img_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        BBC = ans[i].bounding_box                   # bounding_box_coordinate

        x = int(BBC[0][0])
        y = int(BBC[0][1])
        w = int(BBC[1][0] - BBC[0][0])
        h = int(BBC[1][1] - BBC[0][1])

        img_crop = img_crop[y:y+h, x:x+w]

        img_crop = cv2.resize(img_crop, (160, 160))

        Images_cropped.append(img_crop)

    return Images_cropped

def read_embedding(path=Embedding_book):

    try:
        f=h5py.File(path,'r')
    except OSError:
        face_engine  = ClassificationEngine(FaceNet_weight)
        Create_embeddings(face_engine)
        f=h5py.File(path, 'r')

    class_arr=f['class_name'][:]
    class_arr=[k.decode() for k in class_arr]
    emb_arr=f['embeddings'][:]

    return class_arr, emb_arr




def main():

    load_time = time.time()

  # Initialize engine.
    engine = DetectionEngine(Model_weight)
    labels = None

  # Face recognize engine
    face_engine  = ClassificationEngine(FaceNet_weight)
  # read embedding
    class_arr, emb_arr = read_embedding(Embedding_book)
    
    l =  time.time() - load_time

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:

            cap = cv2.VideoCapture(0)

            while(True):
                t1 = cv2.getTickCount()
                print('Load_model: {:.2f} sec'.format(l))

                ret, frame = cap.read()

                img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)

                    
                # Run inference.
                ans = engine.DetectWithImage(
                    img,
                    threshold=0.05,
                    keep_aspect_ratio=False,
                    relative_coord=False,
                    top_k=10)
                
                img = numpy.asarray(img)
            # Display result.
                if ans:
                    crop_img = crop_image(ans, frame)

                    if cv2.waitKey(1) == ord('a'):
                 
                        for k in range(0, len(crop_img)):
                            new_class_name = input('Please input your name of class:')
                            new_save = cv2.cvtColor(crop_img[k], cv2.COLOR_BGR2RGB)
                            cv2.imwrite('pictures/' + str(new_class_name) + '.jpg', new_save)

                        Create_embeddings(face_engine)
                        class_arr, emb_arr = read_embedding('embedding_book/embeddings.h5')

                    embs = Tpu_FaceRecognize(face_engine, crop_img)


                    face_num = len(ans)
                    face_class = ['Others']*face_num

                    for i in range(face_num):
                        diff = np.mean(np.square(embs[i]-emb_arr), axis=1)
                        min_diff = min(diff)

                        if min_diff < THRED:

                            index = np.argmin(diff)
                            face_class[i] = class_arr[index]

                    print('Face_class:', face_class)
                    print('Classes:', class_arr)

                    for count, obj in enumerate(ans):
                        print('-----------------------------------------')
                        if labels:
                            print(labels[obj.label_id])
                        print('Score = ', obj.score)
                        box = obj.bounding_box.flatten().tolist()

                        # Draw a rectangle and label
                        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                        cv2.putText(img, '{}'.format(face_class[count]), (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN,
                                    1, (255, 0, 0), 1, cv2.LINE_AA)

                t2 = cv2.getTickCount()
                t = (t2-t1)/cv2.getTickFrequency()
                fps = 1.0/t
                cv2.putText(img, 'fps: {:.2f}'.format(fps), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(img, 'A: Add new class', (5, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, 'Q: Quit', (5, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
                img_ = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 

                cv2.imshow('frame', img_)

                if cv2.waitKey(1) == ord('q'):
                    break
        
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()