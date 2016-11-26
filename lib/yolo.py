import numpy as np
import tensorflow as tf
from time import time
import os

def cfg_yielder(model, undiscovered = True):
    # Step 1: parsing cfg file
    with open(model + '/yolo-face.cfg', 'r') as f:
        lines = f.readlines()

    s = [] # contains layers' info
    S = int() # the number of grid cell
    add = dict()
    for line in lines:
        line = line.strip()
        if 'side' in line:
            S = int(line.split('=')[1].strip())
        if '[' in line:
            if add != {}:
                s += [add]
            add = dict()
        else:
            try:
                i = float(line.split('=')[1].strip())
                if i == int(i): i = int(i)
                add[line.split('=')[0]] = i
            except:
                try:
                    if line.split('=')[1] == 'leaky' and 'output' in add:
                        add[line.split('=')[0]] = line.split('=')[1]
                except:
                    pass
    yield S

    # Step 2: investigate the weight file
    weightf = model + '/yolo-face.weights'
    if undiscovered:
        allbytes = os.path.getsize(weightf)
        allbytes /= 4 # each float is 4 byte
        allbytes -= 4 # the first 4 bytes are darknet specifications
        last_convo = int() 
        for i, d in enumerate(s):
            if len(d) == 4:
                last_convo = i # the index of last convolution layer
        flag = False
        channel = 3 # initial number of channel in the tensor volume
        out = int() 
        for i, d in enumerate(s):
            # for each iteration in this loop
            # allbytes will be gradually subtracted
            # by the size of the corresponding layer (d)
            # except for the 1st dense layer
            # it should be what remains after subtracting
            # all other layers
            if len(d) == 4:
                allbytes -= d['size'] ** 2 * channel * d['filters']
                allbytes -= d['filters']
                channel = d['filters']
            elif 'output' in d: # this is a dense layer
                if flag is False: # this is the first dense layer
                    out = out1 = d['output'] # output unit of the 1st dense layer
                    flag = True # mark that the 1st dense layer is passed
                    continue # don't do anything with the 1st dense layer
                allbytes -= out * d['output']
                allbytes -= d['output']
                out = d['output']
        allbytes -= out1 # substract the bias
        if allbytes <= 0:
                message = "Error: yolo-{}.cfg suggests a bigger size"
                message += " than yolo-{}.weights actually is"
                assert allbytes > 0
        # allbytes is now = I * out1
        # where I is the input size of the 1st dense layer
        # I is also the volume of the last convolution layer
        # I = size * size * channel
        size = (np.sqrt(allbytes/out1/channel)) 
        size = int(size)
        n = last_convo + 1
        while 'output' not in s[n]:
            size *= s[n].get('size',1)
            n += 1
    else:
        last_convo = None
        size = None

    # Step 3: Yielding config
    w = 448
    h = 448
    c = 3
    l = w * h * c
    flat = False
    yield ['CROP']
    for i, d in enumerate(s):
        #print w, h, c, l
        flag = False
        if len(d) == 4:
            mult = (d['size'] == 3) 
            mult *= (d['stride'] != 2) + 1.
            if d['size'] == 1: d['pad'] = 0
            new = (w + mult * d['pad'] - d['size'])
            new /= d['stride']
            new = int(np.floor(new + 1.))
            if i == last_convo:
                d['pad'] = -size
                new = size
            yield ['conv', d['size'], c, d['filters'], 
                    h, w, d['stride'], d['pad']]    
            w = h = new
            c = d['filters']
            l = w * h * c
            #print w, h, c
        if len(d) == 2:
            if 'output' not in d:
                yield ['pool', d['size'], 0, 
                    0, 0, 0, d['stride'], 0]
                new = (w * 1.0 - d['size'])/d['stride'] + 1
                new = int(np.floor(new))
                w = h = new
                l = w * h * c
            else:
                if not flat:
                    flat = True
                    yield ['FLATTEN']
                yield ['conn', 0, 0,
                0, 0, 0, l, d['output']]
                l = d['output']
                if 'activation' in d:
                    yield ['LEAKY']
        if len(d) == 1:
            if 'output' not in d:
                yield ['DROPOUT']
            else:
                if not flat:
                    flat = True
                    yield ['FLATTEN']
                yield ['conn', 0, 0,
                0, 0, 0, l, d['output']]
                l = d['output']

class layer:
    def __init__(self, type, size = 0, c = 0, n = 0, h = 0, w = 0):
        self.type = type
        self.size = size
        self.c, self.n = (c, n) 
        self.h, self.w = (h, w)

class maxpool_layer(layer):
    def __init__(self, size, c, n, h, w, stride, pad):
        layer.__init__(self, 'MAXPOOL', size, c, n, h, w)
        self.stride = stride
        self.pad = pad

class convolu_layer(layer):
    def __init__(self, size, c, n, h, w, stride, pad):
        layer.__init__(self, 'CONVOLUTIONAL', size, c, n, h, w)
        self.stride = stride
        self.pad = pad

class connect_layer(layer):
    def __init__(self, size, c, n, h, w, 
        input_size, output_size):
        layer.__init__(self, 'CONNECTED', size, c, n, h, w)
        self.output_size = output_size
        self.input_size = input_size

class YOLO(object):

    layers = []
    S = int()
    model = str()

    def __init__(self, model):
        pick = ['face']
        self.labels = pick
        self.model = model
        self.layers = []
        self.build(model)
        self.layer_number = len(self.layers)
        weight_file = model +  '/yolo-face.weights'
        start = time()
        self.loadWeights(weight_file)
        stop = time()

    def build(self, model):
        cfg = model.split('-')[0]
        layers = cfg_yielder(cfg)
        for i, info in enumerate(layers):
            if i == 0: 
                self.S = info
                continue
            if len(info) == 1: new = layer(type = info[0])
            if info[0] == 'conv': new = convolu_layer(*info[1:])
            if info[0] == 'pool': new = maxpool_layer(*info[1:])
            if info[0] == 'conn': new = connect_layer(*info[1:])
            self.layers.append(new)

    def loadWeights(self, weight_path):
        self.startwith = np.array(
            np.memmap(weight_path, mode = 'r',
                offset = 0, shape = (),
                dtype = '(4)i4,'))
        offset = 16
        chunkMB = 1000
        chunk = int(chunkMB * 2**18) 
        
        for i in range(self.layer_number):
            l = self.layers[i]
            if l.type == "CONVOLUTIONAL":
                weight_number = l.n * l.c * l.size * l.size
                l.biases = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(l.n))
                offset += 4 * l.n
                l.weights = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(weight_number))
                offset += 4 * weight_number

            elif l.type == "CONNECTED":
                bias_number = l.output_size
                weight_number = l.output_size * l.input_size
                l.biases = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(bias_number))
                offset += bias_number * 4
            
                chunks  = [chunk] * int(weight_number / chunk) 
                chunks += [weight_number % chunk]
                l.weights = np.array([], dtype = np.float32)
                for c in chunks:
                    l.weights = np.concatenate((l.weights,
                        np.memmap(weight_path, mode = 'r',
                        offset = offset, shape = (),
                        dtype = '({})float32,'.format(c))))
                    offset += c * 4

        for i in range(self.layer_number):
            l = self.layers[i]
            
            if l.type == 'CONVOLUTIONAL':
                weight_array = l.weights
                weight_array = np.reshape(weight_array,
                	[l.n, l.c, l.size, l.size])
                weight_array = weight_array.transpose([2,3,1,0])
                l.weights = weight_array

            if l.type == 'CONNECTED':
                weight_array = l.weights
                weight_array = np.reshape(weight_array,
                	[l.input_size, l.output_size])
                l.weights = weight_array
