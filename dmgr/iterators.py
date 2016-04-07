import collections
import random
import numpy as np


def threaded(generator, num_cached=10):
    """
    Threaded generator
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


def iterate_batches(data_source, batch_size, shuffle=False, expand=True,
                    add_time_dim=False):

    idxs = range(data_source.n_data)

    if shuffle:
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

        if add_time_dim:
            d, t = data_source[batch_idxs]
            new_dshape = (d.shape[0], 1) + d.shape[1:]
            new_tshape = (t.shape[0], 1) + t.shape[1:]
            yield d.reshape(new_dshape), t.reshape(new_tshape)

        else:
            yield data_source[batch_idxs]


def _chunks_to_arrays(data_chunks, target_chunks, max_len):
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

    return data, mask, targets


def iterate_datasources(aggregated_data_source, batch_size, shuffle=False,
                        expand=True, max_seq_len=None):
    """
    Iterates datasource-wise over an aggragated datasource.
    :param aggregated_data_source: AggragatedDatasource object
    :param batch_size:             number of sequences per batch
    :param shuffle:                shuffle datasources
    :param expand:                 fill up last batch
    :param max_seq_len:        maximum sequence length. cuts data from one
                                   data source into chucks of max this length.
                                   If None, use total length of each data source
    :return:                       data of data sources, mask
    """

    n_ds = aggregated_data_source.n_datasources
    ds_idxs = range(n_ds)

    if shuffle:
        random.shuffle(ds_idxs)

    data_chunks = []
    target_chunks = []
    max_len = max_seq_len or 0

    for ds_idx in ds_idxs:
        ds = aggregated_data_source.get_datasource(ds_idx)
        # we chunk the data according to sequence_length
        for d, t in iterate_batches(ds, max_seq_len or ds.n_data,
                                    shuffle=False, expand=False):
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
        ds = aggregated_data_source.get_datasource(ds_idx)
        for d, t in iterate_batches(ds, max_seq_len or ds.n_data,
                                    shuffle=False, expand=False):
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

    def __init__(self, data_source, batch_size, shuffle=False, expand=True,
                 add_time_dim=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.expand = expand
        self.add_time_dim = add_time_dim

    def __iter__(self):
        return iterate_batches(self.data_source, self.batch_size, self.shuffle,
                               self.expand, self.add_time_dim)


class DatasourceIterator:

    def __init__(self, data_source, batch_size, shuffle=False, expand=True,
                 max_seq_len=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.expand = expand
        self.max_seq_len = max_seq_len

    def __iter__(self):
        return iterate_datasources(self.data_source, self.batch_size,
                                   self.shuffle, self.expand, self.max_seq_len)


class ClassBalancedIterator:

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
        return iterate_batches_probabilistic(self.data_source,
                                             self.batch_size, self.dist)
