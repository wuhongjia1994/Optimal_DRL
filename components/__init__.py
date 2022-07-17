import numpy as np
from collections import deque
import itertools


class Deque(deque):
    """
    A modified deque, can use slice or list as index.
    """
    def __init__(self, iterable=(), maxlen=None):
        super(Deque, self).__init__(iterable, maxlen)

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(Deque, self).__getitem__(item)
        elif isinstance(item, slice):
            item_indices = item.indices(len(self))  # start, step, step
            return Deque(itertools.islice(self, *item_indices))
        elif isinstance(item, (list, np.ndarray)):
            new_deque = Deque(maxlen=len(list))
            for index in item:
                assert isinstance(index, int)
                new_deque.append(self[index])
            return new_deque
    def to_numpy(self):
        return np.asarray(self)

def hashable(item):
    try:
        hash(item)
    except TypeError:
        return False
    return True