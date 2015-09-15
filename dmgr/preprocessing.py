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


class DataWhitener(object):

    def __init__(self, mean=0., std_dev=1.):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, data):
        return (data - self.mean) / self.std_dev

    def train(self, dataset, batch_size=1024):
        self.mean, self.std_dev = stats_batchwise(dataset, batch_size)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.mean, self.std_dev = pickle.load(f)

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump((self.mean, self.std_dev), f)
