import random


def iterate_batches(data_source, batch_size, shuffle=False, expand=True):
    idxs = range(data_source.n_data)

    if shuffle:
        random.shuffle(idxs)

    start_idx = 0
    while start_idx < data_source.n_data:
        batch_idxs = idxs[start_idx:start_idx + batch_size]

        # last batch could be too small
        if len(batch_idxs) < batch_size and expand:
            # fill up with random indices not yet in the set
            n_missing = batch_size - len(batch_idxs)
            batch_idxs += random.sample(idxs[:start_idx], n_missing)

        start_idx += batch_size
        yield data_source[batch_idxs]
