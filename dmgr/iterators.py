import collections
import functools
import random

import numpy as np


def compose(*functions):
    """Compose a list of function to one."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions,
                            lambda x: x)


def threaded(generator, num_cached=10):
    """
    Lets a generator run in a seperate thread and fill
    a queue of results.

    Parameters
    ----------
    generator : Generator
        Generator to compute in a separate thread
    num_cached : int
        Number of cached results

    Returns
    -------
    Generator
        A generator that yields items from the result
        queue
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    def consumer():
        # run as consumer
        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()

    return consumer()


def iterate_batches(data_source, batch_size, randomise=False, expand=True):
    """
    Generates mini-batches from a data source.

    Parameters
    ----------
    data_source : :class:DataSource
        Data source to generate mini-batches from
    batch_size : int
        Number of data points and targets in each mini-batch
    randomise : bool
        Indicates whether to randomize the items in each mini-batch
        or not.
    expand : bool
        Indicates whether to fill up the last mini-batch with
        random data points if there is not enough data available.

    Yields
    ------
    tuple of numpy arrays
        mini-batch of data and targets

    """

    idxs = range(data_source.n_data)

    if randomise:
        random.shuffle(idxs)

    start_idx = 0
    while start_idx < len(data_source):
        batch_idxs = idxs[start_idx:start_idx + batch_size]

        # last batch could be too small
        if len(batch_idxs) < batch_size and expand:
            # fill up with random indices not yet in the set
            n_missing = batch_size - len(batch_idxs)
            batch_idxs += random.sample(idxs[:start_idx], n_missing)

        start_idx += batch_size
        yield data_source[batch_idxs]


def _chunks_to_arrays(data_chunks, target_chunks, max_len):
    """
    Concatenates chunks of data and targets into a single array.

    This array has a pre-defined "length". If a chunk is shorter than this
    length, it is padded with the last valid value and corresponding elements
    are masked in the corresponding mask array.

    Parameters
    ----------
    data_chunks : list of numpy arrays
        Data chunks to concatenate
    target_chunks : list of numpy arrays
        Target chunks to concatenate
    max_len : int
        Length if the concatenated array

    Returns
    -------
    tuple of numpy arrays
        Concatenated data, target, and mask arrays

    """
    # create the arrays to store data, targets, and mask
    feature_shape = data_chunks[0].shape[1:]
    target_shape = target_chunks[0].shape[1:]
    data = np.zeros(
        (len(data_chunks), max_len) + feature_shape,
        dtype=data_chunks[0].dtype)
    targets = np.zeros(
        (len(target_chunks), max_len) + target_shape,
        dtype=target_chunks[0].dtype)
    mask = np.zeros(
        (len(data_chunks), max_len),
        dtype=np.float32
    )

    for i in range(len(data_chunks)):
        dlen = len(data_chunks[i])
        data[i, :dlen] = data_chunks[i]
        targets[i, :dlen] = target_chunks[i]
        mask[i, :dlen] = 1.
        # Repeat last valid value of data and targets throughout the whole
        # masked area. This is consistent with the semantics of Lasagne's RNN
        # implementation, which repeats the previous output value at every
        # masked element. Also, Spaghetti (CRF Library) requires it to be this
        # way.
        data[i, dlen:] = data[i, dlen - 1]
        targets[i, dlen:] = targets[i, dlen - 1]

    return data, targets, mask


def iterate_sequences(aggregated_data_source, batch_size, randomise=False,
                      expand=True, max_seq_len=None):
    """
    Generates mini batches of sequences by iterating datasource-wise over an
    aggregated data source. The order of data taken from a single data source
    is not randomised, while the order of data sources can be randomised.

    This generator generates mini batches of :param:batch_size sub-sequences
    of length :param:max_seq_len. Each :class:DataSource contained in the
    aggragated data source is considered a sequence. If too long, sequences
    are broken into several sub-sequences in a mini batch.

    Parameters
    ----------
    aggregated_data_source : :class:AggregatedDataSource
        Aggregated data source to generate mini-batches from
    batch_size : int
        Number of (sub-)sequences per mini batch
    randomise : bool
        Indicates whether to randomise the order of data sources
    expand : bool
        Indicates whether to fill the last mini batch with sequences from a
        random data source if there is not enough data available
        Maximum length of each sequence in a data source
    max_seq_len : int or None
        Maximum sequence length of each sub-sequence in the mini batch. If
        None, the maximum length is determined to be the longest data source
        in the mini-batch. Note that this might result in different
        sequence lengths in each mini batch.

    Yields
    ------
    tuple of numpy arrays
        mini batch of sub-sequences with data, target, and mask arrays

    """

    n_ds = aggregated_data_source.n_datasources
    ds_idxs = range(n_ds)

    if randomise:
        random.shuffle(ds_idxs)

    data_chunks = []
    target_chunks = []
    max_len = max_seq_len or 0

    for ds_idx in ds_idxs:
        ds = aggregated_data_source.datasource(ds_idx)
        # we chunk the data according to sequence_length
        for d, t in iterate_batches(ds, max_seq_len or ds.n_data,
                                    randomise=False, expand=False):
            data_chunks.append(d)
            target_chunks.append(t)
            max_len = max(max_len, len(d))

            if len(data_chunks) == batch_size:
                yield _chunks_to_arrays(data_chunks, target_chunks, max_len)
                data_chunks = []
                target_chunks = []
                max_len = max_seq_len or 0

    # after we processed all data sources, there might be some chunks left.
    while expand and len(data_chunks) < batch_size:
        # add more sequences until we fill it up
        # get a random data source
        ds_idx = random.sample(ds_idxs, 1)[0]
        ds = aggregated_data_source.datasource(ds_idx)
        for d, t in iterate_batches(ds, max_seq_len or ds.n_data,
                                    randomise=False, expand=False):
            data_chunks.append(d)
            target_chunks.append(t)
            max_len = max(max_len, len(d))

            if len(data_chunks) == batch_size:
                # we filled it!
                break

    if len(data_chunks) > 0:
        yield _chunks_to_arrays(data_chunks, target_chunks, max_len)


def iterate_batches_probabilistic(data_source, batch_size, distribution):
    """
    Iterate mini-batches, selecting data from the :class:`DataSource` based
    on data selection probabilities.

    Parameters
    ----------
    data_source : :class:`DataSource`
        Data source to iterate over
    batch_size : int
        Number of elements per mini-batch
    distribution : np.ndarray
        Probabilities to select an element from the data_source

    Yields
    ------
    tuple of two np.ndarray
        Data and targets of a mini-batch
    """

    cum_dist = distribution.cumsum()

    n_sampled = 0
    while n_sampled < len(data_source):
        batch_idxs = np.searchsorted(cum_dist, np.random.sample(batch_size))
        n_sampled += batch_size
        yield data_source[batch_idxs]


class BatchIterator:
    """
    Iterates over mini batches of a data source.

    Parameters
    ----------
    data_source : :class:DataSource
        Data source to generate mini-batches from
    batch_size : int
        Number of data points and targets in each mini-batch
    randomise : bool
        Indicates whether to randomize the items in each mini-batch
        or not.
    expand : bool
        Indicates whether to fill up the last mini-batch with
        random data points if there is not enough data available.

    Yields
    ------
    tuple of numpy arrays
        mini-batch of data and targets
    """

    def __init__(self, data_source, batch_size, randomise=False, expand=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.randomise = randomise
        self.expand = expand

    def __iter__(self):
        """Returns the mini batch generator."""
        return iterate_batches(self.data_source, self.batch_size,
                               self.randomise, self.expand)


class SequenceIterator:
    """
    Iterates over mini batches of sequences from an aggregated data source.

    Each mini batch contains :param:batch_size sub-sequences of length
    :param:max_seq_len. Each :class:DataSource contained in the
    aggregated data source is considered a sequence. If too long, it is
    broken into several sub-sequences in a mini batch.

    Parameters
    ----------
    data_source : :class:AggregatedDataSource
        Aggregated data source to generate mini-batches from
    batch_size : int
        Number of (sub-)sequences per mini batch
    randomise : bool
        Indicates whether to randomise the order of data sources
    expand : bool
        Indicates whether to fill the last mini batch with sequences from a
        random data source if there is not enough data available
        Maximum length of each sequence in a data source
    max_seq_len : int or None
        Maximum sequence length of each sub-sequence in the mini batch. If
        None, the maximum length is determined to be the longest data source
        in the mini-batch. Note that this might result in different
        sequence lengths in each mini batch.

    Yields
    ------
    tuple of numpy arrays
        mini batch of sub-sequences with data, target, and mask arrays
    """

    def __init__(self, data_source, batch_size, randomise=False, expand=True,
                 max_seq_len=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.randomise = randomise
        self.expand = expand
        self.max_seq_len = max_seq_len

    def __iter__(self):
        """Returns the sequence mini batch generator."""
        return iterate_sequences(self.data_source, self.batch_size,
                                 self.randomise, self.expand, self.max_seq_len)


class UniformClassIterator:
    """
    Iterates over mini batches of a data source, stratifying the target
    distribution to uniform. This means that each mini-batch should contain
    a similar number of instances from each class, even if their distribution
    in the dataset is different. This is achieved through sampling with
    replacement.

    Parameters
    ----------
    data_source : :class:DataSource
        Data source to generate mini-batches from
    batch_size : int
        Number of data points and targets in each mini-batch

    Yields
    ------
    tuple of numpy arrays
        mini batch of data and targets
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        freqs = collections.Counter(
            tuple(data_source[i][1]) for i in range(data_source.n_data)
        )

        true_dist = {t: float(f) / data_source.n_data
                     for t, f in freqs.iteritems()}

        ideal_dist = {t: 1. / len(true_dist) for t in freqs}

        select_prob = {t: ideal_dist[t] / (true_dist[t] * data_source.n_data)
                       for t in freqs}

        self.dist = np.array([select_prob[tuple(data_source[i][1])]
                              for i in range(data_source.n_data)])

    def __iter__(self):
        """Returns the mini batch generator."""
        return iterate_batches_probabilistic(self.data_source,
                                             self.batch_size, self.dist)


class AugmentedIterator:
    """
    Augments (i.e. changes) data and targets of an existing batch iterator
    using a number of augmentation functions.

    Parameters
    ----------
    batch_iterator : Iterator
        Batch iterator to augment
    *augment_fns : callables
        Augmentation functions. They have to accept the values the
        :param:batch_iterator returns, and themselves return similar values.

    Yields
    ------
    tuple of numpy arrays
        Augmented mini-batch of data and targets
    """

    def __init__(self, batch_iterator, *augment_fns):
        self.batch_iterator = batch_iterator
        self.augment = compose(*augment_fns)

    def __iter__(self):
        return self.augment(self.batch_iterator.__iter__())
