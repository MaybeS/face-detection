from math import sqrt
def inter(first, second):
    dist = sqrt((first['x'] - second['x']) ** 2 + (first['y'] - second['y']) ** 2)
    wide = (sqrt(first['w'] ** 2 + first['h'] ** 2) + sqrt(second['w'] ** 2 + second['h'] ** 2))/4.
    return dist < wide

def combine(boxes, ww, hh, threshold):
    ln = len(boxes)
    union = Union(ln)
    for i in range(ln):
        for j in range(i + 1, ln):
            if i != j and inter(boxes[i], boxes[j]):
                union.join(i, j, boxes[union._parent(i)]['p'], boxes[union._parent(j)]['p'])
    
    uboxes = [ list() for _ in range(ln) ]
    for i in range(ln):
        uboxes[union._parent(i)].append(i)

    nboxes = list()
    for mboxes in uboxes:
        if mboxes:
            l,r,t,b,p = ww,0,hh,0,0
            for i in mboxes:
                box = boxes[i]
                left = box['x'] - box['w']/2.
                right = box['x'] + box['w']/2.
                top = box['y'] - box['h']/2.
                bot = box['y'] + box['h']/2.
                prob = box['p']
                if l > left: l = left
                if r < right: r = right
                if t > top: t = top
                if b < bot: b = bot
                if p < prob: p = prob
            if p > threshold:
                w = r - l
                h = b - t
                x = r - w/2.
                y = b - h/2.
                nboxes.append({'x':x, 'y':y, 'w':w,'h':h, 'p':p })
    return nboxes

class Union:
    def __init__(self, n):
        self._id = range(n)

    def _parent(self, i):
        j = i
        while (j != self._id[j]):
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def find(self, p, q):
        return self._parent(p) == self._parent(q)

    def join(self, p, q, pp, qq):
        first = p if pp > qq else q
        second = q if pp > qq else p
        self._id[self._parent(second)] = self._parent(first)
