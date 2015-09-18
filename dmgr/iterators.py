import random
import numpy as np


def threaded(generator, num_cached=10):
    """
    Threaded generator
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
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

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def iterate_batches(data_source, batch_size, shuffle=False, expand=True):
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
        yield data_source[batch_idxs]


def iterate_datasources(aggregated_data_source, batch_size, shuffle=False,
                        expand=True, sequence_length=None):
    """
    Iterates datasource-wise over an aggragated datasource.
    :param aggregated_data_source: AggragatedDatasource object
    :param batch_size:             number of datasources per batch
    :param shuffle:                shuffle datasources
    :param expand:                 fill up last batch
    :param sequence_length:        maximum sequence length. cuts data from one
                                   data source into chucks of max this length
    :return:                       data of data sources, mask
    """

    n_ds = aggregated_data_source.n_datasources
    idxs = range(n_ds)

    if shuffle:
        random.shuffle(idxs)

    start_idx = 0
    while start_idx < n_ds:
        batch_idxs = idxs[start_idx:start_idx + batch_size]

        # last batch could be too small
        if len(batch_idxs) < batch_size and expand:
            # fill up with random indices not yet in the set
            n_missing = batch_size - len(batch_idxs)
            batch_idxs += random.sample(idxs[:start_idx], n_missing)

        start_idx += batch_size

        data_chunks = []
        target_chunks = []
        max_len = 0

        # for each data source in the batch
        for batch_i in batch_idxs:
            ds = aggregated_data_source.get_datasource(batch_i)

            # we chunk the data according to sequence_length
            for d, t in iterate_batches(ds, sequence_length or ds.n_data,
                                        shuffle=False, expand=False):
                data_chunks.append(d)
                target_chunks.append(t)
                max_len = max(max_len, len(d))

        # create the arrays to store data, targets, and mask
        data = np.zeros(
            (len(data_chunks), max_len) + aggregated_data_source.feature_shape,
            dtype=data_chunks[0].dtype)
        targets = np.zeros(
            (len(target_chunks), max_len) + aggregated_data_source.target_shape,
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

        yield data, mask, targets
