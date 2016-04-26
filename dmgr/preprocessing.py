import pickle

import numpy as np

from .iterators import iterate_batches


def stats_batchwise(data_source, batch_size=1024):
    """
    Compute mean and standard deviation (diagonal) of a data source batch-wise.
    This means not all the data-set needs to be loaded into memory.

    Parameters
    ----------
    data_source : :class:DataSource
        Data source to compute the mean and std. deviation for.
    batch_size : int
        Batch size for the computation. Smaller values are slower, larger
        values need more memory

    Returns
    -------
    float, float
        Mean and standard deviation of the data

    """
    mean = np.zeros(data_source.dshape, dtype=np.float32)
    mean_xs = np.zeros_like(mean, dtype=np.float32)

    for x, _ in iterate_batches(data_source, batch_size, expand=False):
        corr_fact = float(x.shape[0]) / batch_size
        mean += x.mean(axis=0) * corr_fact
        mean_xs += (x ** 2).mean(axis=0) * corr_fact

    corr_fact = float(batch_size) / data_source.n_data
    mean *= corr_fact
    mean_xs *= corr_fact
    std = np.sqrt(mean_xs - mean ** 2)

    return mean, std


def max_batchwise(data_source, batch_size=1024):
    """
    Compute the point-wise maximum of data from a data-source batch-wise.
    This means not all the data needs to be loaded into memory

    Parameters
    ----------
    data_source : :class:DataSource
        Data source to compute the max for.
    batch_size : int
        Batch size for the computation. Smaller values are slower, larger
        values need more memory

    Returns
    -------
    float
        Point-wise maximum of the data
    """
    max_val = np.zeros(data_source.dshape, dtype=np.float32)

    for x, _ in iterate_batches(data_source, batch_size, expand=True):
        max_val = np.maximum(max_val, np.abs(x).max(axis=0))

    return max_val


class ZeroMeanUnitVar(object):
    """
    Normalises data such that each dimension has a zero mean and unit
    variance.

    Parameters
    ----------
    mean : float
        Initial mean shift
    std_dev : float
        Initial std. dev. norm
    """

    def __init__(self, mean=0., std_dev=1.):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, data):
        """Normalise :param:data"""
        return (data - self.mean) / self.std_dev

    def train(self, data_source, batch_size=4096):
        """
        Determine mean shift and std. dev. normalisation factor

        Parameters
        ----------
        data_source : :class:DataSource
            Data from which to estimate mean and std. dev.
        batch_size : int
            Batch size used to compute mean and std. dev.
            (see :func:stats_batchwise)
        """
        self.mean, self.std_dev = stats_batchwise(data_source, batch_size)

    def load(self, filename):
        """
        Load mean shift and std. dev. normalisation factor from a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file containing the parameters
        """
        with open(filename, 'r') as f:
            self.mean, self.std_dev = pickle.load(f)

    def save(self, filename):
        """
        Save mean shift and std. dev normalisation factor to a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to store the parameters to
        """
        with open(filename, 'w') as f:
            pickle.dump((self.mean, self.std_dev), f)


class MaxNorm(object):
    """
    Normalises the data such that the maximum value of each dimension is 1.

    Parameters
    ----------
    max_val : float
        Default scaling factor
    """

    def __init__(self, max_val=1.):
        self.max_val = max_val

    def __call__(self, data):
        """Normalise :param:data"""
        return data / self.max_val

    def train(self, data_source, batch_size=4096):
        """
        Determine max normalisation factors from data.

        Parameters
        ----------
        data_source : :class:DataSource
            Data from which to estimate maximum value for each dimension
        batch_size : int
            Batch size used to estimate the maximum value
            (see :func:max_batchwise)
        """
        self.max_val = np.max(max_batchwise(data_source, batch_size))

    def load(self, filename):
        """
        Load normalisation factors from a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file containing the parameters
        """
        with open(filename, 'r') as f:
            self.max_val = pickle.load(f)

    def save(self, filename):
        """
        Save normalisation factors to a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to store the parameters to
        """
        with open(filename, 'w') as f:
            pickle.dump(self.max_val, f)


class PcaWhitening(object):
    """
    Whitens the data using principal component analysis.

    To speed up training the transformation, you can specify how many data
    points from the training data source to use.

    Parameters
    ----------
    n_train_vectors : int or None
        Number of data points to use when training the PCA. If `None`, use all
        values.
    n_components : int or None
        Number of components to use in PCA. If `None`, use all components,
        resulting in no data reduction.
    """

    def __init__(self, n_train_vectors=None, n_components=None):
        from sklearn.decomposition import PCA
        self.pca = PCA(whiten=True, n_components=self.n_components)
        self.fit = False
        self.n_train_vectors = n_train_vectors
        self.n_components = n_components

    def __call__(self, data):
        """Project the :param:data using the PCA projection."""
        if self.fit:
            # flatten features, pca only works on 1d arrays
            data_shape = data.shape
            data_flat = data.reshape((data_shape[0], -1))
            whitened_data = self.pca.transform(data_flat)
            # get back original shape
            return whitened_data.reshape(data_shape).astype(np.float32)
        else:
            return data

    def train(self, data_source, batch_size=4096):
        """
        Fit the PCA projection to data.

        Parameters
        ----------
        data_source : :class:DataSource
            Data to use for fitting the projection
        batch_size : int
            Not used here.
        """

        # select a random subset of the data if self.n_train_vectors is not
        # None
        if self.n_train_vectors is not None:
            sel_data = list(np.random.choice(data_source.n_data,
                                             size=self.n_train_vectors,
                                             replace=False))
        else:
            sel_data = slice(None)
        data = data_source[sel_data][0]  # ignore the labels
        data_flat = data.reshape((data.shape[0], -1))  # flatten features
        self.pca.fit(data_flat)
        self.fit = True

    def load(self, filename):
        """
        Load the PCA projection parameters from a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file containing the projection parameters

        """
        with open(filename, 'r') as f:
            self.pca.set_params(pickle.load(f))
            self.fit = True

    def save(self, filename):
        """
        Save the PCA projection parameters to a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to store the parameters to
        """
        with open(filename, 'w') as f:
            pickle.dump(self.pca.get_params(deep=True), f)


class ZcaWhitening(object):
    """
    Whitens the data using ZCA whitening.

    Compared to PCA whitening, this transformation decorrelates *locally*,
    making it is useful for convolutional neural networks.

    For details on this method, see

    Alex Krizhevsky. "Learning Multiple Layers of Features from Tiny Images."
    Technical Report 2009.
    http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

    Parameters
    ----------
    n_train_vectors : int or None
        Number of data points to use when training the ZCA. If `None`, use all
        values.
    regularisation : float
        Regularisation parameter for fitting the ZCA.
    """

    def __init__(self,  n_train_vectors=None, regularisation=10**-5):
        self.mean = None
        self.components = None
        self.regularisation = regularisation
        self.n_train_vectors = n_train_vectors

    def __call__(self, data):
        """Project the :param:data using the ZCA projection."""
        if self.components is None:
            return data

        orig_shape = data.shape
        data = np.reshape(data, (data.shape[0], -1))
        return np.dot(data - self.mean, self.components.T).reshape(orig_shape)

    def train(self, data_source, batch_size=4096):
        """
        Fit the ZCA projection to data.

        Parameters
        ----------
        data_source : :class:DataSource
            Data to use for fitting the projection
        batch_size : int
            Not used here.
        """
        if self.n_train_vectors is not None:
            sel_data = list(np.random.choice(data_source.n_data,
                                             size=self.n_train_vectors,
                                             replace=False))
        else:
            sel_data = slice(None)

        from scipy import linalg
        from sklearn.utils import as_float_array

        x = data_source[sel_data]
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

    def load(self, filename):
        """
        Load the ZCA projection parameters from a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file containing the projection parameters.
        """
        with open(filename, 'r') as f:
            self.components, self.mean = pickle.load(f)

    def save(self, filename):
        """
        Save the ZCA projection parameters to a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to store the parameters to
        """
        with open(filename, 'w') as f:
            pickle.dump((self.components, self.mean), f)
