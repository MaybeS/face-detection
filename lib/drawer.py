import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

def fix(x,c):
	return max(min(x,c),0)

def load_img(imPath):
  img = cv2.imread(imPath)
  h, w, _ = img.shape
  return img, h, w

def draw_double(imPath, toPath, boxes, boxms, color=(121,255,0)):
  img, h, w = load_img(imPath)
  for box in boxes:
    left = int((box['x'] - box['w'] / 2.))
    right = int((box['x'] + box['w'] / 2.))
    top = int((box['y'] - box['h'] / 2.))
    bot = int((box['y'] + box['h'] / 2.))

    cv2.rectangle(img, 
              (left, top), (right, bot), 
              color, 3)
  for box in boxms:
    left = int((box['x'] - box['w'] / 2.))
    right = int((box['x'] + box['w'] / 2.))
    top = int((box['y'] - box['h'] / 2.))
    bot = int((box['y'] + box['h'] / 2.))

    cv2.rectangle(img, 
              (left, top), (right, bot), 
              (0,121,255), 3)

  cv2.imwrite(toPath, img)

def draw(imPath, toPath, boxes, color=(121,255,0)):
  img, h, w = load_img(imPath)
  for box in boxes:
    left = int((box['x'] - box['w'] / 2.))
    right = int((box['x'] + box['w'] / 2.))
    top = int((box['y'] - box['h'] / 2.))
    bot = int((box['y'] + box['h'] / 2.))

    cv2.rectangle(img, 
              (left, top), (right, bot), 
              color, 3)
  cv2.imwrite(toPath, img)

def crop(imPath, allobj = None):
	im = cv2.imread(imPath)
	h, w, _ = im.shape

	im_ = cv2.resize(im, (448, 448))
	image_array = np.array(im_) / 255.
	image_array = image_array * 2. - 1.
	image_array = np.expand_dims(image_array, 0)

	return image_array, w, h