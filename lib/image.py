import cv2
import numpy as np

def imread(path):
    return cv2.imread(path)

def imresize(image, w=448, h=448):
    return cv2.resize(image, (w, h))

def imrefine(image):
    return np.expand_dims((np.array(image) / 255.) * 2. - 1., 0)

def imwrite(path, image):
    cv2.imwrite(path, image)

def draw_rectangle(image, lt, rb, color=(255,0,0), bold=3):
    cv2.rectangle(image, lt, rb, color, bold)
