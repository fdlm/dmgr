from itertools import izip, groupby
from tempfile import TemporaryFile
import os
import numpy as np


class DataSource(object):

    def __init__(self, data, targets):
        assert data.shape[0] == targets.shape[0]
        self._data = data
        self._targets = targets

        if self._data.ndim == 1:
            self._data = self._data[:, np.newaxis]

        if self._targets.ndim == 1:
            self._targets = self._targets[:, np.newaxis]

    @classmethod
    def from_files(cls, data_file, target_file, memory_mapped=False):
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap))

    def save(self, data_file, target_file):
        np.save(data_file, self._data)
        np.save(target_file, self._targets)

    def __getitem__(self, item):
        return self._data[item], self._targets[item]

    @property
    def n_data(self):
        return self._data.shape[0]

    def __len__(self):
        return self.n_data

    @property
    def feature_shape(self):
        return self._data.shape[1:]

    @property
    def target_shape(self):
        return self._targets.shape[1:]

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ttype(self):
        return self._targets.dtype


class AggregatedDataSource(object):

    def __init__(self, data_sources, use_perc=1.0):
        assert len(data_sources) > 0, 'Need at least one data source'
        assert all(x.feature_shape == data_sources[0].feature_shape
                   for x in data_sources), \
            'Data sources dimensionality has to be equal'
        assert all(x.target_shape == data_sources[0].target_shape
                   for x in data_sources), \
            'Data sources target dimensionality has to be equal'

        self._data_sources = data_sources
        self._ds_ends = np.array(
            [0] + [int(len(d) * use_perc) for d in data_sources]).cumsum()

    @classmethod
    def from_files(cls, data_files, target_files, memory_mapped=False,
                   data_source_type=DataSource, use_perc=1.0, **kwargs):
        return cls(
            [data_source_type.from_files(d, t, memory_mapped=memory_mapped,
                                         **kwargs)
             for d, t in izip(data_files, target_files)], use_perc=use_perc
        )

    def save(self, data_file, target_file):

        with TemporaryFile() as data_tmp, TemporaryFile() as target_temp:
            data_shape = (self.n_data,) + self.feature_shape
            df = np.memmap(data_tmp, shape=data_shape, dtype=self.dtype)
            target_shape = (self.n_data,) + self.target_shape
            tf = np.memmap(target_temp, shape=target_shape, dtype=self.ttype)

            for i in range(self.n_data):
                d, t = self[i]
                df[i] = d
                tf[i] = t

            np.save(data_file, df)
            np.save(target_file, tf)

    def _to_ds_idx(self, idx):
        ds_idx = self._ds_ends.searchsorted(idx, side='right') - 1
        d_idx = idx - self._ds_ends[ds_idx]
        return ds_idx, d_idx

    def __getitem__(self, item):
        """
        only list and int supported at the moment...
        """
        if isinstance(item, int):
            ds_idx, d_idx = self._to_ds_idx(item)
            return self._data_sources[ds_idx][d_idx]

        elif isinstance(item, list):
            item.sort()
            ds_idxs, d_idxs = self._to_ds_idx(item)
            data_list = []
            target_list = []

            for ds_idx, d_idx_iter in groupby(enumerate(d_idxs),
                                              lambda i: ds_idxs[i[0]]):
                d_idx = [di[1] for di in d_idx_iter]
                d, t = self._data_sources[ds_idx][d_idx]
                data_list.append(d)
                target_list.append(t)

            return np.vstack(data_list), np.vstack(target_list)
        else:
            raise TypeError('Only list and int indices supported')

    @property
    def n_data(self):
        return sum(ds.n_data for ds in self._data_sources)

    def __len__(self):
        return self.n_data

    @property
    def feature_shape(self):
        return self._data_sources[0].feature_shape

    @property
    def target_shape(self):
        return self._data_sources[0].target_shape

    @property
    def dtype(self):
        return self._data_sources[0].dtype

    @property
    def ttype(self):
        return self._data_sources[0].ttype


# taken from: http://www.scipy.org/Cookbook/SegmentAxis
def segment_axis(signal, frame_size, hop_size=1, axis=None, end='cut',
                 end_value=0):
    """
    Generate a new array that chops the given array along the given axis into
    (overlapping) frames.

    :param signal:     signal [numpy array]
    :param frame_size: size of each frame in samples [int]
    :param hop_size:   hop size in samples between adjacent frames [int]
    :param axis:       axis to operate on; if None, act on the flattened array
    :param end:        what to do with the last frame, if the array is not
                       evenly divisible into pieces; possible values:
                       'cut'  simply discard the extra values
                       'wrap' copy values from the beginning of the array
                       'pad'  pad with a constant value
    :param end_value:  value to use for end='pad'
    :return:           2D array with overlapping frames

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    The returned array is always of type np.ndarray.

    Example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    """
    # make sure that both frame_size and hop_size are integers
    frame_size = int(frame_size)
    hop_size = int(hop_size)
    if axis is None:
        signal = np.ravel(signal)  # may copy
        axis = 0
    if axis != 0:
        raise ValueError('please check if the resulting array is correct.')

    length = signal.shape[axis]

    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")

    if length < frame_size or (length - frame_size) % hop_size:
        if length > frame_size:
            round_up = (frame_size + (1 + (length - frame_size) // hop_size) *
                        hop_size)
            round_down = (frame_size + ((length - frame_size) // hop_size) *
                          hop_size)
        else:
            round_up = frame_size
            round_down = 0
        assert round_down < length < round_up
        assert round_up == round_down + hop_size or (round_up == frame_size and
                                                     round_down == 0)
        signal = signal.swapaxes(-1, axis)

        if end == 'cut':
            signal = signal[..., :round_down]
        elif end in ['pad', 'wrap']:
            # need to copy
            s = list(signal.shape)
            s[-1] = round_up
            y = np.empty(s, dtype=signal.dtype)
            y[..., :length] = signal
            if end == 'pad':
                y[..., length:] = end_value
            elif end == 'wrap':
                y[..., length:] = signal[..., :round_up - length]
            signal = y

        signal = signal.swapaxes(-1, axis)

    length = signal.shape[axis]
    if length == 0:
        raise ValueError("Not enough data points to segment array in 'cut' "
                         "mode; try end='pad' or end='wrap'")
    assert length >= frame_size
    assert (length - frame_size) % hop_size == 0
    n = 1 + (length - frame_size) // hop_size
    s = signal.strides[axis]
    new_shape = (signal.shape[:axis] + (n, frame_size) +
                 signal.shape[axis + 1:])
    new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                   signal.strides[axis + 1:])

    try:
        # noinspection PyArgumentList
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)
    except TypeError:
        import warnings
        warnings.warn("Problem with ndarray creation forces copy.")
        signal = signal.copy()
        # shape doesn't change but strides does
        new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                       signal.strides[axis + 1:])
        # noinspection PyArgumentList
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)


class ContextDataSource(object):

    def __init__(self, data, targets, context_size):

        if data.ndim == 1:
            data = data[:, np.newaxis]
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        frame_size = 1 + 2 * context_size
        self.context_size = context_size
        self._data = segment_axis(data, frame_size, axis=0)
        self._targets = targets
        self._n_data = data.shape[0]

        filler = np.zeros_like(data[0])

        self._begin_data = np.array(
            [np.vstack([filler] * (context_size - i) +
                       [data[0:i + context_size + 1]])
             for i in range(context_size)]
        )

        self._end_data = np.array(
            [np.vstack([data[data.shape[0] - context_size - i - 1:]] +
                       [filler] * (context_size - i))
             for i in range(context_size)[::-1]]
        )

        assert (self._n_data == self._data.shape[0] +
                self._begin_data.shape[0] + self._end_data.shape[0])

    @classmethod
    def from_files(cls, data_file, target_file, context_size,
                   memory_mapped=False):
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap), context_size)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < self.context_size:
                return self._begin_data[item], self._targets[item]
            elif item >= self.n_data - self.context_size:
                data_item = item - self.n_data + self.context_size
                return self._end_data[data_item], self._targets[item]
            else:
                return self._data[item - self.context_size], self._targets[item]
        elif isinstance(item, list):
            item = np.array(item)
            item.sort()

            gd_begin = np.searchsorted(item, self.context_size)
            gd_end = np.searchsorted(item, self.n_data - self.context_size - 1,
                                     side='right')

            d = []
            t = []

            if gd_begin < gd_end:
                idxs = item[gd_begin:gd_end]
                d.append(self._data[idxs - self.context_size])
                t.append(self._targets[idxs])

            if gd_begin > 0:
                idxs = item[:gd_begin]
                d.append(self._begin_data[idxs])
                t.append(self._targets[idxs])

            if gd_end < item.shape[0]:
                idxs = item[gd_end:]
                d.append(self._end_data[idxs - self.n_data + self.context_size])
                t.append(self._targets[idxs])

            return np.vstack(d), np.vstack(t)

        else:
            raise TypeError('Only list and int indices supported')

    @property
    def n_data(self):
        return self._n_data

    def __len__(self):
        return self.n_data

    @property
    def feature_shape(self):
        return self._data.shape[1:]

    @property
    def target_shape(self):
        return self._targets.shape[1:]

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ttype(self):
        return self._targets.dtype
