import numpy as np
import pickle

from . import iterators


def stats_batchwise(dataset, batch_size=1024):
    mean = np.zeros(dataset.feature_shape, dtype=np.float32)
    mean_xs = np.zeros_like(mean, dtype=np.float32)

    for x, _ in iterators.iterate_batches(dataset, batch_size, expand=False):
        corr_fact = float(x.shape[0]) / batch_size
        mean += x.mean(axis=0) * corr_fact
        mean_xs += (x ** 2).mean(axis=0) * corr_fact

    corr_fact = float(batch_size) / dataset.n_data
    mean *= corr_fact
    mean_xs *= corr_fact
    std = np.sqrt(mean_xs - mean ** 2)

    return mean, std


def max_batchwise(dataset, batch_size=1024):
    max_val = np.zeros(dataset.feature_shape, dtype=np.float32)

    for x, _ in iterators.iterate_batches(dataset, batch_size, expand=True):
        max_val = np.maximum(max_val, np.abs(x).max(axis=0))

    return max_val


class ZeroMeanUnitVar(object):

    def __init__(self, mean=0., std_dev=1.):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, data):
        return (data - self.mean) / self.std_dev

    def train(self, dataset, batch_size=4096):
        self.mean, self.std_dev = stats_batchwise(dataset, batch_size)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.mean, self.std_dev = pickle.load(f)

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump((self.mean, self.std_dev), f)


class MaxNorm(object):

    def __init__(self, max_val=1.):
        self.max_val = max_val

    def __call__(self, data):
        return data / self.max_val

    def train(self, dataset, batch_size=4096):
        self.max_val = np.max(max_batchwise(dataset, batch_size))

    def load(self, filename):
        with open(filename, 'r') as f:
            self.max_val = pickle.load(f)

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self.max_val, f)


class PcaWhitening(object):

    def __init__(self):
        self.pca = None

    def __call__(self, data):
        if self.pca is not None:
            # flatten features, pca only works on 1d arrays
            data_shape = data.shape
            data_flat = data.reshape((data_shape[0], -1))
            whitened_data = self.pca.transform(data_flat)
            # get back original shape
            return whitened_data.reshape(data_shape).astype(np.float32)
        else:
            return data

    def train(self, dataset, batch_size=4096):
        from sklearn.decomposition import IncrementalPCA as IPCA
        data = dataset[:][0]  # ignore the labels
        data_shape = data.shape
        print data_shape
        data_flat = data.reshape((data_shape[0], -1))  # flatten features
        self.pca = IPCA(whiten=True,
                        batch_size=batch_size,
                        n_components=data_flat.shape[1])
        self.pca.fit(data_flat)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.pca.set_params(pickle.load(f))

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self.pca.get_params(deep=True), f)
