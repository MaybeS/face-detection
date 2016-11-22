import cv2
from os import listdir
def draw(imgname, boxes):
  path = "/".join(imgname.split('/')[:-1])
  name = imgname.split('/')[-1]
  img = cv2.imread(path + '/' + name, 1)
  for box in boxes:
    box = [int(b) for b in box]
    cv2.rectangle(img, 
              (box[0], box[2]), (box[1], box[3]), 
              (121,255,0), 3)
  cv2.imwrite('./results/' + name, img)