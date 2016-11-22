from .box import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

samplehold = 0.15

def fix(x,c):
	return max(min(x,c),0)

def filter(w, h, box):
	if box[3] - box[1] < w / 33:
		return False
	if box[4] - box[2] < h / 33:
		return False
	return True

def prob(box):
	return box.probs[np.argmax(box.probs)]

def expand(box):
	box.w *= 1.2
	box.h *= 1.2
	return box

from math import sqrt
def intersec(boxa, boxb):
	dx = abs(boxa.x - boxb.x)
	dy = abs(boxa.y - boxb.y)
	return (max(boxa.w, boxb.w) / 2) > dx and (max(boxa.h, boxb.h) / 2) > dy

def merge(boxes, w, h):
	left = w
	right = 0
	top = h
	bot = 0

	for box in boxes:
		box = expand(box)
		l = int((box.x - box.w/2.) * w)
		r = int((box.x + box.w/2.) * w)
		t = int((box.y - box.h/2.) * h)
		b = int((box.y + box.h/2.) * h)
		left = min(l, left)
		right = max(r, right)
		top = min(t, top)
		bot = max(b, bot)

	left = int(max(0, left))
	right = int(min(w, right))
	top = int(max(0, top))
	bot = int(min(h, bot))
	return left, right, top, bot


# def merge(boxes, w, h):
# 	origin = boxes[0]
# 	left = int((origin.x - origin.w/2.) * w)
# 	right = int((origin.x + origin.w/2.) * w)
# 	top = int((origin.y - origin.h/2.) * h)
# 	bot = int((origin.y + origin.h/2.) * h)


# 	for box in boxes[1:]:
# 		l = int((box.x - box.w/2.) * w)
# 		r = int((box.x + box.w/2.) * w)
# 		t = int((box.y - box.h/2.) * h)
# 		b = int((box.y + box.h/2.) * h)

# 		rate = ((prob(origin) - prob(box)) / 1)
# 		left -= (left - l) * rate
# 		right -= (right - r) * rate
# 		top -= (top - t) * rate
# 		bot -= (bot - b) * rate

# 	left = int(max(0, left))
# 	right = int(min(w, right))
# 	top = int(max(0, top))
# 	bot = int(min(h, bot))
# 	return left, right, top, bot

from .union import Union
def union_find(boxs):
	ln = len(boxs)
	union = Union(ln)
	for i in range(ln):
		for j in range(i + 1, ln):
			if intersec(boxs[i], boxs[j]) and prob(boxs[i]) > samplehold and prob(boxs[j]) > samplehold:
				union.union(i, j, prob(boxs[i]), prob(boxs[j]))

	unions = [ prob(boxs[i]) > samplehold and [union._parent(i)] or None for i in range(ln) ]
	for idx in range(len(unions)):
		if unions[idx] != None and unions[idx][0] != idx:
			unions[unions[idx][0]].append(idx)
			unions[idx] = None

	return unions

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
		#---------------
		# (x,y) --> (scale*x, scale*y)
		# (scale*x - offx, scale*y - offy)
		#--------------
		newobj = []
		for obj in allobj:
			obj[1] = float(obj[1])
			obj[2] = float(obj[2])
			obj[3] = float(obj[3])
			obj[4] = float(obj[4])
			if not filter(w, h, obj):
				continue	
			obj[1] = int(obj[1]*scale-offx)
			obj[3] = int(obj[3]*scale-offx)
			obj[2] = int(obj[2]*scale-offy)
			obj[4] = int(obj[4]*scale-offy)
			obj[1] = fix(obj[1], w)
			obj[3] = fix(obj[3], w)
			obj[2] = fix(obj[2], h)
			obj[4] = fix(obj[4], h)
		if len(newobj) > 12:
			allobj = None
		allobj = newobj

	# return im
	im_ = cv2.resize(im, (448, 448))
	image_array = np.array(im_)
	image_array = image_array / 255.
	image_array = image_array * 2. - 1.
	image_array = np.expand_dims(image_array, 0) # 1, height, width, 3

	if allobj is not None:
		return image_array, allobj
	else:
		return image_array
	
def to_color(indx, base):
	base2 = base * base
	b = indx / base2
	r = (indx % base2) / base
	g = (indx % base2) % base
	return (b * 127, r * 127, g * 127)

def draw_predictions(predictions, 
	img_path, flip, threshold,
	C, S, labels, colors, mergebox):

	B = 2
	boxes = []
	SS        =  S * S # number of grid cells
	prob_size = SS * C # class probabilities
	conf_size = SS * B # confidences for each grid cell
	probs = predictions[0 : prob_size]
	confs = predictions[prob_size : (prob_size + conf_size)]
	cords = predictions[(prob_size + conf_size) : ]
	probs = probs.reshape([SS, C])
	confs = confs.reshape([SS, B])
	cords = cords.reshape([SS, B, 4])

	predict = []

	for grid in range(SS):
		for b in range(B):
			new_box   = BoundBox(C)
			new_box.c =  confs[grid, b]
			new_box.x = (cords[grid, b, 0] + grid %  S) / S
			new_box.y = (cords[grid, b, 1] + grid // S) / S
			new_box.w =  cords[grid, b, 2] ** 2
			new_box.h =  cords[grid, b, 3] ** 2
			new_box.id = '{}-{}'.format(grid, b)
			for c in range(C):
				new_box.probs[c] = new_box.c * probs[grid, c]
			boxes.append(new_box)

	# non max suppress boxes
	if True:
		for c in range(C):
			for i in range(len(boxes)): boxes[i].class_num = c
			boxes = sorted(boxes, cmp=prob_compare)
			for i in range(len(boxes)):
				boxi = boxes[i]
				if boxi.probs[c] == 0: continue
				for j in range(i + 1, len(boxes)):
					boxj = boxes[j]
					boxij = box_intersection(boxi, boxj)
					boxja = boxj.w * boxj.h
					apart = boxij / boxja
					if apart >= .5:
						if boxi.probs[c] > boxj.probs[c]:
							boxes[j].probs[c] = 0.
						else:
						    boxes[i].probs[c] = 0.

	imgcv = cv2.imread(img_path)
	if flip: imgcv = cv2.flip(imgcv, 1)
	h, w, _ = imgcv.shape

	thick = int((h+w)/300)
	unions = union_find(boxes)

	if not mergebox:
		for b in boxes:
			max_indx = np.argmax(b.probs)
			max_prob = b.probs[max_indx]
			label = 'object' * int(C < 2)
			label += labels[max_indx] * int(C > 1)
			#if (max_prob > threshold):
			if (max_prob > 0.1):
				left  = int ((b.x - b.w/2.) * w)
				right = int ((b.x + b.w/2.) * w)
				top   = int ((b.y - b.h/2.) * h)
				bot   = int ((b.y + b.h/2.) * h)
				if left  < 0    :  left = 0
				if right > w - 1: right = w - 1
				if top   < 0    :   top = 0
				if bot   > h - 1:   bot = h - 1
				predict.append([left, right, top, bot])
	else:
		for s in unions:
			if s:
				max_prob = prob(boxes[s[0]])
				if max_prob > 0.35:
					label = 'object' * int(C < 2)
					left, right, top, bot = merge([boxes[each] for each in s], w, h)
					predict.append([left, right, top, bot])

	return predict
