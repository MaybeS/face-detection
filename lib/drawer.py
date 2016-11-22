import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

def fix(x,c):
	return max(min(x,c),0)

def draw(imgname, boxes):
  path = "/".join(imgname.split('/')[:-1])
  name = imgname.split('/')[-1]
  img = cv2.imread(path + '/' + name, 1)
  h, w, _ = img.shape
  for box in boxes:
    left = int((box['x'] - box['w'] / 2.) * w)
    right = int((box['x'] + box['w'] / 2.) * w)
    top = int((box['y'] - box['h'] / 2.) * h)
    bot = int((box['y'] + box['h'] / 2.) * h)

    cv2.rectangle(img, 
              (left, top), (right, bot), 
              (121,255,0), 3)
  cv2.imwrite('./results/' + name, img)

def crop(imPath, allobj = None):
	im = cv2.imread(imPath)
	if allobj is not None:
		h, w, _ = im.shape
		scale = np.random.uniform()/3. + 1.
		max_offx = (scale-1.) * w
		max_offy = (scale-1.) * h
		offx = int(np.random.uniform() * max_offx)
		offy = int(np.random.uniform() * max_offy)
		im = cv2.resize(im, (0,0), fx = scale, fy = scale)
		im = im[offy : (offy + h), offx : (offx + w)]

		newobj = []
		for obj in allobj:
			obj[1] = float(obj[1])
			obj[2] = float(obj[2])
			obj[3] = float(obj[3])
			obj[4] = float(obj[4])
			obj[1] = int(obj[1]*scale-offx)
			obj[3] = int(obj[3]*scale-offx)
			obj[2] = int(obj[2]*scale-offy)
			obj[4] = int(obj[4]*scale-offy)
			obj[1] = fix(obj[1], w)
			obj[3] = fix(obj[3], w)
			obj[2] = fix(obj[2], h)
			obj[4] = fix(obj[4], h)
		allobj = newobj

	im_ = cv2.resize(im, (448, 448))
	image_array = np.array(im_) / 255.
	image_array = image_array * 2. - 1.
	image_array = np.expand_dims(image_array, 0) # 1, height, width, 3

	if allobj is not None:
		return image_array, allobj
	else:
		return image_array