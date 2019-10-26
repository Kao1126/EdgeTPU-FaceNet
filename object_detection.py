import cv2
import argparse
import platform
import numpy
import subprocess
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      help='Path of the detection model, it must be a SSD model with postprocessing operator.',
      required=True)
  parser.add_argument('--label', help='Path of the labels file.')
  parser.add_argument('--output', help='File path of the output image.')
  parser.add_argument(
      '--keep_aspect_ratio',
      dest='keep_aspect_ratio',
      action='store_true',
      help=(
          'keep the image aspect ratio when down-sampling the image by adding '
          'black pixel padding (zeros) on bottom or right. '
          'By default the image is resized and reshaped without cropping. This '
          'option should be the same as what is applied on input images during '
          'model training. Otherwise the accuracy may be affected and the '
          'bounding box of detection result may be stretched.'))
  parser.set_defaults(keep_aspect_ratio=False)
  args = parser.parse_args()

  if not args.output:
    output_name = 'object_detection_result.jpg'
  else:
    output_name = args.output

  # Initialize engine.
  engine = DetectionEngine(args.model)
  labels = dataset_utils.ReadLabelFile(args.label) if args.label else None

  # Open image.
 # img = Image.open(args.input)
  #draw = ImageDraw.Draw(img)



  cap = cv2.VideoCapture(0)

  while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    
#    img = Image.open(im)
    img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)


    # Run inference.
    ans = engine.DetectWithImage(
        img,
        threshold=0.05,
        keep_aspect_ratio=args.keep_aspect_ratio,
        relative_coord=False,
        top_k=10)

    # Display result.
    if ans:
      for obj in ans:
        print('-----------------------------------------')
        if labels:
          print(labels[obj.label_id])
        print('score = ', obj.score)
        box = obj.bounding_box.flatten().tolist()
        print('box = ', box)
        # Draw a rectangle.
        draw.rectangle(box, outline='red')

    img_ = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)  

    cv2.imshow('frame', img_)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # 釋放攝影機
  cap.release()

  # 關閉所有 OpenCV 視窗
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
