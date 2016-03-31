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

    def __init__(self, n_train_vectors=None, n_components=None):
        """
        Data whitening using PCA
        :param n_train_vectors: max number of feature vectors used for
                                training. 'None' means 'use all vectors'
        """
        self.pca = None
        self.n_train_vectors = n_train_vectors
        self.n_components = n_components

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
        from sklearn.decomposition import PCA

        # select a random subset of the data if self.n_train_vectors is not
        # None
        if self.n_train_vectors is not None:
            sel_data = list(np.random.choice(dataset.n_data,
                                             size=self.n_train_vectors,
                                             replace=False))
        else:
            sel_data = slice(None)
        data = dataset[sel_data][0]  # ignore the labels
        data_flat = data.reshape((data.shape[0], -1))  # flatten features
        self.pca = PCA(whiten=True, n_components=self.n_components)
        self.pca.fit(data_flat)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.pca.set_params(pickle.load(f))

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self.pca.get_params(deep=True), f)


class ZcaWhitening(object):

    def __init__(self, regularisation, n_train_vectors=None):
        """
        Data whitening using ZCA
        :param regularisation:  regularisation parameter (surprise!)
        :param n_train_vectors: max number of feature vectors used for
                                training. 'None' means 'use all vectors'
        """
        self.mean = None
        self.components = None
        self.regularisation = regularisation
        self.n_train_vectors = n_train_vectors

    def __call__(self, data):

        if self.components is None:
            return data

        orig_shape = data.shape
        data = np.reshape(data, (data.shape[0], -1))
        return np.dot(data - self.mean, self.components.T).reshape(orig_shape)

    def train(self, dataset, batch_size=4096):
        if self.n_train_vectors is not None:
            sel_data = list(np.random.choice(dataset.n_data,
                                             size=self.n_train_vectors,
                                             replace=False))
        else:
            sel_data = slice(None)

        from scipy import linalg
        from sklearn.utils import as_float_array

        x = dataset[sel_data]
        x = np.reshape(x, (x.shape[0], -1))
        x = as_float_array(x, copy=True)

        # center data
        self.mean = np.mean(x, axis=0)
        x -= self.mean

        # compute covariance matrix
        sigma = np.dot(x.T, x) / (x.shape[1] - 1)

        # compute svd
        u, s, _ = linalg.svd(sigma)

        # compute whitening transform
        tmp = np.dot(u, np.diag(1.0 / np.sqrt(s + self.regularisation)))
        self.components = np.dot(tmp, u.T)

