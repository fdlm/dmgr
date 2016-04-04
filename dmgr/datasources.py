from os.path import basename, splitext
from itertools import izip, groupby
from tempfile import TemporaryFile
import numpy as np
from .iterators import iterate_batches


class DataSource(object):
    """
    A :class:DataSource is a collection of data and corresponding targets.
    Data can be provided raw or pre-processed. Data and targets
    can be sliced using the `start`, `stop`, and `step` parameters.

    Parameters
    ----------
    data : np.ndarray or np.memmap
        The data array.
    targets : np.ndarray or np.memmap
        The target array
    start : int or None
        Start index for data and target slicing.
    stop : int or None
        Stop index for data and target slicing.
    stop : int or None
        Step size for data and target slicing.
    preprocessors: list or None
        List of pre-processing functions. These functions should take only
        one parameter (the data) and return the pre-processed data. They
        will be called in the order given in the list.
    name : string
        Name of the :class:DataSource. Will be used when converting to
        a string.
    """

    def __init__(self, data, targets, start=None, stop=None, step=None,
                 preprocessors=None, name=None):

        if data.shape[0] != targets.shape[0]:
            raise ValueError(
                'Number of data ({}) is not equals '
                'number of targets ({})'.format(data.shape[0],
                                                targets.shape[0]))

        self.name = name
        self._data = data[start:stop:step]
        self._targets = targets[start:stop:step]

        self._preprocessors = preprocessors or []

        if self._data.ndim == 1:
            self._data = self._data[:, np.newaxis]

        if self._targets.ndim == 1:
            self._targets = self._targets[:, np.newaxis]

    @classmethod
    def from_files(cls, data_file, target_file, memory_mapped=True,
                   *args, **kwargs):
        """
        Create a :class:DataSource from NumPy binary files.

        Parameters
        ----------
        data_file : file-like object or string
            The file to read. File-like objects must support the
            ``seek()`` and ``read()`` methods. Pickled files require that the
            file-like object support the ``readline()`` method as well.
        target_file : file-like object or string
            The file to read. File-like objects must support the
            ``seek()`` and ``read()`` methods. Pickled files require that the
            file-like object support the ``readline()`` method as well.
        memory_mapped : bool
            Should data and targets be read memory-mapped or loaded into
            memory.
        *args :
            Arguments passed to the init method of :class:DataSource.
        **kwargs :
            Keyword arguments passed to the init method of :class:DataSource.

        Returns
        -------
        DataSource
            DataSource with data and targets from the given files
        """
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap), *args, **kwargs)

    def _process(self, data):
        """Process the data using all preprocessors."""
        for pp in self._preprocessors:
            data = pp(data)

        return data

    def __getitem__(self, idx):
        """
        Return pre-processed data and targets for given index.

        Parameters
        ----------
        idx : int, slice, list
            Index of data and targets to return.

        Returns
        -------
        tuple
            Pre-processed data and targets for given index.

        """
        return self._process(self._data[idx]), self._targets[idx]

    @property
    def n_data(self):
        """int: number of data points (equals number of targets)"""
        return self._data.shape[0]

    def __len__(self):
        """int: number of data points (equals number of targets)"""
        return self.n_data

    @property
    def dshape(self):
        """tuple: shape of a data point"""
        return self._data.shape[1:]

    @property
    def tshape(self):
        """tuple: shape of a target"""
        return self._targets.shape[1:]

    @property
    def dtype(self):
        """type: dtype of the data"""
        return self._data.dtype

    @property
    def ttype(self):
        """type: dtype of the targets"""
        return self._targets.dtype

    def __str__(self):
        """str: a string representation of the :class:DataSource"""
        return '{}: N={}  dshape={}  tshape={}'.format(
            self.__class__.__name__,
            self.n_data, self.dshape, self.tshape)


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


class ContextDataSource(DataSource):
    # TODO: Add Docstring!

    def __init__(self, data, targets, context_size,
                 start=None, stop=None, step=None, preprocessors=None,
                 name=None, keep_order=False):

        # step is taken care of in another way within this class. we thus
        # pass 'None' to the parent
        super(ContextDataSource, self).__init__(
            data, targets, start=start, stop=stop, step=None,
            preprocessors=preprocessors, name=name
        )

        self.keep_order = keep_order

        self.step = step or 1

        self.context_size = context_size
        self._data = segment_axis(self._data, 1 + 2 * context_size, axis=0)
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
    def from_files(cls, data_file, target_file, memory_mapped=True,
                   *args, **kwargs):
        # TODO: Add Docstring!
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap),
                   *args, **kwargs)

    def __getitem__(self, item):
        # TODO: Add Docstring!
        if isinstance(item, int):
            item *= self.step
            if item < self.context_size:
                return (self._process(self._begin_data[item]),
                        self._targets[item])
            elif item >= self._n_data - self.context_size:
                data_item = item - self._n_data + self.context_size
                return (self._process(self._end_data[data_item]),
                        self._targets[item])
            else:
                return (self._process(self._data[item - self.context_size]),
                        self._targets[item])

        elif isinstance(item, list):

            item = np.array(item) * self.step

            if self.keep_order:
                # first sort the indices to be retrieved so we can get all the
                # padded data and segmented data in one command
                sort_idxs = item.argsort()
                # remember how to un-sort the indices so we can restore the
                # correct ordering in the end
                revert_idxs = sort_idxs.argsort()
                item = item[sort_idxs]
            else:
                item.sort()

            gd_begin = np.searchsorted(item, self.context_size)
            gd_end = np.searchsorted(item, self._n_data - self.context_size - 1,
                                     side='right')

            d = []
            t = []

            # 0 padded begin data
            if gd_begin > 0:
                idxs = item[:gd_begin]
                d.append(self._process(self._begin_data[idxs]))
                t.append(self._targets[idxs])

            # segmented data
            if gd_begin < gd_end:
                idxs = item[gd_begin:gd_end]
                d.append(self._process(self._data[idxs - self.context_size]))
                t.append(self._targets[idxs])

            # 0-padded end data
            if gd_end < item.shape[0]:
                idxs = item[gd_end:]
                d.append(self._process(self._end_data[idxs - self._n_data +
                                                      self.context_size]))
                t.append(self._targets[idxs])

            data = np.vstack(d)
            targets = np.vstack(t)

            if self.keep_order:
                data = data[revert_idxs]
                targets = targets[revert_idxs]

            return data, targets

        elif isinstance(item, slice):
            return self[range(item.start or 0, item.stop or self.n_data,
                              item.step or 1)]

        else:
            raise TypeError('Index type {} not supported!'.format(type(item)))

    @property
    def n_data(self):
        # TODO: Add Docstring
        return self._n_data / self.step

    @property
    def dshape(self):
        # TODO: Add Docstring
        return self._data.shape[1:]

    @property
    def tshape(self):
        # TODO: Add Docstring
        return self._targets.shape[1:]


class AggregatedDataSource(object):
    # TODO: Add Docstring

    def __init__(self, data_sources, keep_order=False):
        assert len(data_sources) > 0, 'Need at least one data source'
        assert all(x.dshape == data_sources[0].dshape
                   for x in data_sources), \
            'Data sources dimensionality has to be equal'
        assert all(x.tshape == data_sources[0].tshape
                   for x in data_sources), \
            'Data sources target dimensionality has to be equal'

        self._data_sources = data_sources
        self._ds_ends = np.array([0] + [len(d) for d in data_sources]).cumsum()
        self.keep_order = keep_order

    @classmethod
    def from_files(cls, data_files, target_files, memory_mapped=False,
                   data_source_type=DataSource, names=None, **kwargs):
        # TODO: Add Docstring

        if not names:
            names = [basename(d).split('.')[0] for d in data_files]

        return cls(
            [data_source_type.from_files(d, t, memory_mapped=memory_mapped,
                                         name=n, **kwargs)
             for d, t, n in izip(data_files, target_files, names)]
        )

    def save(self, data_file, target_file):
        # TODO: Add Docstring

        with TemporaryFile() as data_tmp, TemporaryFile() as target_temp:
            data_shape = (self.n_data,) + self.dshape
            df = np.memmap(data_tmp, shape=data_shape, dtype=self.dtype)
            target_shape = (self.n_data,) + self.tshape
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
        # TODO: Add Docstring
        if isinstance(item, int):
            ds_idx, d_idx = self._to_ds_idx(item)
            return self._data_sources[ds_idx][d_idx]

        elif isinstance(item, list):
            if self.keep_order:
                item = np.array(item)
                sort_idxs = item.argsort()
                item = item[sort_idxs]
                revert_idxs = sort_idxs.argsort()
            else:
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

            data = np.vstack(data_list)
            targets = np.vstack(target_list)

            if self.keep_order:
                data = data[revert_idxs]
                targets = targets[revert_idxs]

            return data, targets

        elif isinstance(item, slice):
            return self[range(item.start or 0, item.stop or self.n_data,
                              item.step or 1)]
        else:
            raise TypeError('Index type {} not supported!'.format(type(item)))

    def get_datasource(self, idx):
        """
        Gets a single DataSource
        :param idx: index of the datasource
        :return: datasource
        """
        return self._data_sources[idx]

    @property
    def n_datasources(self):
        # TODO: Add Docstring
        return len(self._data_sources)

    @property
    def n_data(self):
        # TODO: Add Docstring
        return sum(ds.n_data for ds in self._data_sources)

    def __len__(self):
        # TODO: Add Docstring
        return self.n_data

    @property
    def dshape(self):
        # TODO: Add Docstring
        return self._data_sources[0].dshape

    @property
    def tshape(self):
        # TODO: Add Docstring
        return self._data_sources[0].tshape

    @property
    def dtype(self):
        # TODO: Add Docstring
        return self._data_sources[0].dtype

    @property
    def ttype(self):
        # TODO: Add Docstring
        return self._data_sources[0].ttype

    def __str__(self):
        # TODO: Add Docstring
        return '{}: N={}  dshape={}  tshape={}'.format(
            self.__class__.__name__,
            self.n_data, self.dshape, self.tshape)


def get_datasources(files, preprocessors=None, cached=False, cache_dir=None,
                    **kwargs):
    """
    This function creates datasources with given preprocessors given
    a files dictionary. The dictionary looks as follows:

    {'train': {'feat': [train feature files],
               'targ': [train targets files]}
     'val': {'feat': [validation feature files],
             'targ': [validation target files]},
     'test': {'feat': [test feature files],
             'targ': [test target files]}
    }

    The preprocessors are trained on the training data.

    :param files:         file dictionary with the aforementioned format
    :param preprocessors: list of preprocessors to be applied to the data
    :param cached:        cache datasources
    :param cache_dir:     where to cache datasources
    :param kwargs:        additional arguments to be passed to
                          AggregatedDataSource.from_files
    :return:              tuple of train data source, validation data source
                          and test data source
    """
    train_set = AggregatedDataSource.from_files(
        files['train']['feat'], files['train']['targ'], memory_mapped=True,
        preprocessors=preprocessors,
        **kwargs
    )

    val_set = AggregatedDataSource.from_files(
        files['val']['feat'], files['val']['targ'], memory_mapped=True,
        preprocessors=preprocessors,
        **kwargs
    )

    test_set = AggregatedDataSource.from_files(
        files['test']['feat'], files['test']['targ'], memory_mapped=True,
        preprocessors=preprocessors,
        **kwargs
    )

    if preprocessors is not None:
        for p in preprocessors:
            p.train(train_set)

    if cached:
        train_set = cache_aggregated_datasource(train_set, cache_dir=cache_dir)
        test_set = cache_aggregated_datasource(test_set, cache_dir=cache_dir)
        val_set = cache_aggregated_datasource(val_set, cache_dir=cache_dir)

    return train_set, val_set, test_set


def cache_datasource(datasource, batch_size=8192, cache_dir=None):
    # TODO: Add Docstring
    from tempfile import NamedTemporaryFile

    f_cache = NamedTemporaryFile(suffix='.npz', dir=cache_dir)
    t_cache = NamedTemporaryFile(suffix='.npz', dir=cache_dir)
    f_shape = (datasource.n_data,) + datasource.dshape
    t_shape = (datasource.n_data,) + datasource.tshape

    features = np.memmap(f_cache, dtype=datasource.dtype, shape=f_shape)
    targets = np.memmap(t_cache, dtype=datasource.ttype, shape=t_shape)

    i = 0
    for f, t in iterate_batches(datasource, batch_size=batch_size,
                                shuffle=False, expand=False):
        features[i:i + f.shape[0]] = f
        targets[i:i + t.shape[0]] = t
        i += f.shape[0]

    return DataSource(features, targets, name=datasource.name)


def cache_aggregated_datasource(agg_datasource, batch_size=8192,
                                cache_dir=None):
    # TODO: Add Docstring
    datasources = [agg_datasource.get_datasource(i)
                   for i in range(agg_datasource.n_datasources)]
    cached_ds = [cache_datasource(ds, batch_size, cache_dir)
                 for ds in datasources]
    return AggregatedDataSource(cached_ds,
                                keep_order=agg_datasource.keep_order)
