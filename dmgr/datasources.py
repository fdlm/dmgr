from itertools import izip
import numpy as np


class DataSource(object):

    def __init__(self, data, targets):
        assert data.shape[0] == targets.shape[0]
        self.data = data
        self.targets = targets

    @classmethod
    def from_files(cls, data_file, target_file, memory_mapped=False):
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap))

    def save(self, data_file, target_file):
        np.save(data_file, self.data)
        np.save(target_file, self.targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_data(self):
        return self.shape[0]

    @property
    def data_shape(self):
        return self.shape[1:]

    def __len__(self):
        return self.n_data


class AggregatedDataSource(object):

    def __init__(self, data_sources, use_perc=1.0):
        # TODO: try to make this nicer....
        self.data = [d[i][0]
                     for d in data_sources
                     for i in range(int(len(d) * use_perc))]
        self.targets = [d[i][1]
                        for d in data_sources
                        for i in range(int(len(d) * use_perc))]

    @classmethod
    def from_files(cls, data_files, target_files, memory_mapped=False,
                   data_source_type=DataSource, use_perc=1.0, **kwargs):
        return cls(
            [data_source_type.from_files(d, t, memory_mapped=memory_mapped,
                                         **kwargs)
             for d, t in izip(data_files, target_files)], use_perc=use_perc
        )

    def save(self, data_file, target_file):
        data_shape = (self.n_data,) + self.data[0].shape
        df = np.memmap(data_file, mode='w',
                       shape=data_shape, dtype=self.data[0].dtype)
        target_shape = (self.n_data,) + self.targets[0].shape
        tf = np.memmap(target_file, mode='w',
                       shape=target_shape, dtype=self.targets[0].dtype)

        for i in range(self.n_data):
            df[i] = self.data[i]
            tf[i] = self.targets[i]

    def __getitem__(self, item):
        if isinstance(item, list):
            return (np.vstack([[self.data[i]] for i in item]),
                    np.vstack([self.targets[i] for i in item]))
        if isinstance(item, int):
            return self.data[item], self.targets[item]
        return np.vstack(self.data[item]), np.vstack(self.targets[item])

    @property
    def shape(self):
        return (self.n_data,) + self.data_shape

    @property
    def n_data(self):
        return len(self.data)

    @property
    def data_shape(self):
        return self.data[0].shape if self.n_data > 0 else (0,)

    def __len__(self):
        return self.n_data


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
        frame_size = 1 + 2 * context_size
        self.context_size = context_size
        self.data = segment_axis(data, frame_size, axis=0)
        self.targets = targets

        filler = np.zeros_like(data[0])

        self.begin_data = np.array(
            [np.vstack([filler] * (context_size - i) +
                       [data[0:i + context_size + 1]])
             for i in range(context_size)]
        )

        self.end_data = np.array(
            [np.vstack([data[data.shape[0] - context_size - i - 1:]] +
                       [filler] * (context_size - i))
             for i in range(context_size)[::-1]]
        )

    @classmethod
    def from_files(cls, data_file, target_file, context_size,
                   memory_mapped=False):
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap), context_size)

    def __getitem__(self, item):
        assert type(item) == int, 'currently only int is supported as index'

        # wraparound
        if item < 0:
            item = self.n_data - item

        if item < self.context_size:
            return self.begin_data[item], self.targets[item]
        elif item >= self.n_data - self.context_size:
            item -= self.n_data - self.context_size
            return self.end_data[item], self.targets[item]
        else:
            return self.data[item - self.context_size], self.targets[item]

    @property
    def shape(self):
        sh = self.data.shape
        return (sh[0] + 2 * self.context_size,) + sh[1:]

    @property
    def n_data(self):
        return self.shape[0]

    @property
    def data_shape(self):
        return self.shape[1:]

    def __len__(self):
        return self.n_data


