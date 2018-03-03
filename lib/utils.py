def merge(predictions):
    def _condition(i, j):
        distance = lambda ix, iy, jx, jy: ((jx-ix)**2+(jy-iy)**2)**(1/2)
        d = distance((i[2]-i[0])/2+i[0], (i[3]-i[1])/2+i[1], (j[2]-j[0])/2+j[0], (j[3]-j[1])/2+j[1])
        di = distance(*i[:4])
        dj = distance(*j[:4])
        return d < (di*(i[4]/(i[4] + j[4])) + dj*(j[4])/(i[4] + j[4]))
    def _concatenate(i, j):
        return [ii*(i[4]/(i[4]+j[4])) + jj*(j[4]/(i[4]+j[4])) for ii, jj in zip(i[:4], j[:4])] + [(i[4]+j[4])/2]

    for idx, i in enumerate(predictions):
        while True:
            flag = False
            for jdx, j in enumerate(predictions):
                if j == i: continue
                if _condition(i, j):
                    predictions[idx] = _concatenate(i, j)
                    i = predictions[idx]
                    del predictions[jdx]
                    flag = True
                    break
            if not flag: break
    return predictions