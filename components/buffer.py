import numpy as np
from components import hashable


class Batch(dict):
    def __init__(self):
        super(Batch, self).__init__()

    def __getitem__(self, item):
        if hashable(item) and item in self.keys():
            return super(Batch, self).__getitem__(item)
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_batch = Batch()
            for it in item:
                new_batch[it] = self[it]
            return new_batch
        elif isinstance(item, (int, slice, list, np.ndarray)):
            new_batch = Batch()
            for k, v in self.items():
                new_batch[k] = v[item]
            return new_batch
        else:
            raise KeyError

    def shuffle(self, chunk=1):
        if chunk is None:
            return
        lengths = [super(Batch, self).__getitem__(item).shape[0] for item in self.keys()]
        if len(lengths) == 0:
            return
        assert np.mean(lengths) - lengths[0] == 0
        assert lengths[0] % chunk == 0
        size = lengths[0]
        perm = np.arange(0, size, chunk)
        np.random.shuffle(perm)
        if chunk is None or chunk == 1:
            indices = perm
        else:
            indices = []
            for start in perm:
                end = np.min([start + chunk, size])
                indices.append(np.arange(start, end))
            indices = np.concatenate(indices)
        for item in self.keys():
            self[item] = self[item][indices]

    def __repr__(self):
        return "Batch: data_keys:{}".format(list(self.keys()))


class Buffer(Batch):

    def __init__(self, buffer_size):
        super(Buffer, self).__init__()
        self.buffer_size = buffer_size
        self.counter = {}

    def store(self, data):
        # there might be a bad transistion, but this is a simple implementation.
        for k, v in data.items():
            if k not in self.keys():
                assert not isinstance(k, int) and hashable(k)
                try:
                    self[k] = np.zeros((self.buffer_size,) + v.shape)
                except AttributeError:
                    self[k] = np.zeros((self.buffer_size,))
                self.counter[k] = 0
            index = self.counter[k] % self.buffer_size
            self[k][index] = v
            self.counter[k] += 1

    def uniform_sample(self, batch_size):
        max_slice_start = self.current_size - batch_size
        start = np.random.randint(0, max_slice_start + 1)
        return self[start: start + batch_size]

    def last_batch(self, batch_size):
        return self[-batch_size:]

    def can_sample(self, sample_size):
        return self.current_size >= sample_size

    @property
    def current_size(self):
        if len(self.keys()) != 0:
            sizes = np.array([size for size in self.counter.values()])
            assert not np.any(sizes - sizes.mean())
            return min([sizes[0], self.buffer_size])
        return 0

    def clear(self):
        self.counter.clear()
        super(Batch, self).clear()

    def __repr__(self):
        return "Buffer: buffer size={}, current size={}, data_keys:{}".format(self.buffer_size,
                                                                              self.current_size,
                                                                              list(self.keys()))


if __name__ == '__main__':
    # demo
    buffer = Buffer(buffer_size=1024)
    for i in range(1024):
        print(i)
        obs = np.random.rand(4, 11, 700)
        state = np.random.rand(4, 300)
        actions = np.random.randint(0, 5, [4, 18])
        neglogp = np.random.rand(4, 11)
        avail_actions = np.random.randint(0, 5, [4, 10, 18])
        rewards = np.random.rand()
        dones = np.array([False, False, False, False])
        d = {'obs': obs, 'state': state, 'actions': actions,
             'neglogp': neglogp, 'avail_actions': avail_actions, 'rewards': rewards, 'dones': dones}
        buffer.store(d)
    batch1 = buffer.last_batch(256)
    batch2 = buffer.uniform_sample(256)
    batch3 = batch2[:128]

    #
    ba = Batch()
    a = np.arange(64)[:,None].repeat(10,axis=1)
    b = np.arange(128,192)[:,None].repeat(10,axis=1)
    ba['a'] = a
    ba['b'] = b
    ba.shuffle(4)
