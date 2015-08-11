import random


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
        # i = 0
        for item in generator:
            # print('Producing item {}'.format(i))
            # i += 1
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    # i = 0
    while item is not end_marker:
        # print('Yielding item {}'.format(i))
        # i += 1
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


