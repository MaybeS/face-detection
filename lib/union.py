class Union:
    def __init__(self, n):
        self._id = list(range(n))
        self._sz = [1] * n

    def _parent(self, i):
        j = i
        while (j != self._id[j]):
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def find(self, p, q):
        return self._parent(p) == self._parent(q)

    def union(self, p, q, pp, qq):
        if pp < qq:
            self._id[self._parent(p)] = self._parent(q)
        else:
            self._id[self._parent(q)] = self._parent(p)
