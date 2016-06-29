from itertools import izip, groupby
from os.path import basename

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
        Name of the :class:DataSource.
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
            :class:DataSource with data and targets from the given files
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
        idx : int, slice, list, np.ndarray
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

    Parameters
    ----------
    signal : numpy array
        Signal.
    frame_size : int
        Size of each frame [samples].
    hop_size : int
        Hop size between adjacent frames [samples].
    axis : int, optional
        Axis to operate on; if 'None', operate on the flattened array.
    end : {'cut', 'wrap', 'pad'}, optional
        What to do with the last frame, if the array is not evenly divisible
        into pieces; possible values:

        - 'cut'
          simply discard the extra values,
        - 'wrap'
          copy values from the beginning of the array,
        - 'pad'
          pad with a constant value.

    end_value : float, optional
        Value used to pad if `end` is 'pad'.

    Returns
    -------
    numpy array, shape (num_frames, frame_size)
        Array with overlapping frames

    Notes
    -----
    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    The returned array is always of type np.ndarray.

    Examples
    --------
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
    """
    A :class:ContextDataSource is a collection of data and corresponding
    targets. Data can be provided raw or pre-processed. Data and targets
    can be sliced using the `start`, `stop`, and `step` parameters.
    Additionally, data is provided with a context window. This assumes some
    relationship between consecutive data points.

    Parameters
    ----------
    data : np.ndarray or np.memmap
        The data array.
    targets : np.ndarray or np.memmap
        The target array
    context_size : int
        Context size to add to the data point. This is the number of data
        points concatenated before and after the indexed datapoint. E.g., if
        context_size is 2, the returned data will contain 5 data points.
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
        Name of the :class:ContextDataSource.

    Notes
    -----
    Slicing refers to the central data points in the returned contexts. This
    means that the data points concatenated to a data point are not subject
    to the step parameter.

    """

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
        """
        Create a :class:ContextDataSource from NumPy binary files.

        Parameters
        ----------
        data_file : file-like object or string
            The data file to read. File-like objects must support the
            ``seek()`` and ``read()`` methods. Pickled files require that the
            file-like object support the ``readline()`` method as well.
        target_file : file-like object or string
            The target file to read. File-like objects must support the
            ``seek()`` and ``read()`` methods. Pickled files require that the
            file-like object support the ``readline()`` method as well.
        memory_mapped : bool
            Should data and targets be read memory-mapped or loaded into
            memory.
        *args :
            Arguments passed to the init method of :class:ContextDataSource.
        **kwargs :
            Keyword arguments passed to the init method of
            :class:ContextDataSource.

        Returns
        -------
        ContextDataSource
            :class:ContextDataSource with data and targets from the given files
        """
        mmap = 'r+' if memory_mapped else None
        return cls(np.load(data_file, mmap_mode=mmap),
                   np.load(target_file, mmap_mode=mmap),
                   *args, **kwargs)

    def __getitem__(self, idx):
        """
        Return pre-processed data and targets for given index.

        Parameters
        ----------
        idx : int, slice, list, np.ndarray
            Index of data and targets to return.

        Returns
        -------
        tuple
            Pre-processed data and targets for given index.

        """
        if isinstance(idx, int):
            idx *= self.step
            if idx < self.context_size:
                return (self._process(self._begin_data[idx]),
                        self._targets[idx])
            elif idx >= self._n_data - self.context_size:
                data_item = idx - self._n_data + self.context_size
                return (self._process(self._end_data[data_item]),
                        self._targets[idx])
            else:
                return (self._process(self._data[idx - self.context_size]),
                        self._targets[idx])

        elif isinstance(idx, list):

            idx = np.array(idx) * self.step

            if self.keep_order:
                # first sort the indices to be retrieved so we can get all the
                # padded data and segmented data in one command
                sort_idxs = idx.argsort()
                # remember how to un-sort the indices so we can restore the
                # correct ordering in the end
                revert_idxs = sort_idxs.argsort()
                idx = idx[sort_idxs]
            else:
                idx.sort()

            gd_begin = np.searchsorted(idx, self.context_size)
            gd_end = np.searchsorted(idx, self._n_data - self.context_size - 1,
                                     side='right')

            d = []
            t = []

            # 0 padded begin data
            if gd_begin > 0:
                idxs = idx[:gd_begin]
                d.append(self._process(self._begin_data[idxs]))
                t.append(self._targets[idxs])

            # segmented data
            if gd_begin < gd_end:
                idxs = idx[gd_begin:gd_end]
                d.append(self._process(self._data[idxs - self.context_size]))
                t.append(self._targets[idxs])

            # 0-padded end data
            if gd_end < idx.shape[0]:
                idxs = idx[gd_end:]
                d.append(self._process(self._end_data[idxs - self._n_data +
                                                      self.context_size]))
                t.append(self._targets[idxs])

            data = np.vstack(d)
            targets = np.vstack(t)

            if self.keep_order:
                data = data[revert_idxs]
                targets = targets[revert_idxs]

            return data, targets

        elif isinstance(idx, slice):
            return self[range(idx.start or 0, idx.stop or self.n_data,
                              idx.step or 1)]

        elif isinstance(idx, np.ndarray):
            return self[list(idx)]

        else:
            raise TypeError('Index type {} not supported!'.format(type(idx)))

    @property
    def n_data(self):
        """int: number of data points (equals number of targets)"""
        return self._n_data / self.step


class AggregatedDataSource(object):
    """
    A :class:AggregatedDataSource is a collection of :class:DataSource objects.
    It provides a convenient way to access data and targets from multiple
    data sources.

    Parameters
    ----------
    data_sources : list of :class:DataSource or similar
        List of :class:DataSource objects to aggregate. Data sources must have
        the same data and target shapes and types.

    keep_order : bool
        If True, preserves the original order of data points when indexing
        multiple data points (e.g. using a slice or a list of indices).
        If False, no order is guaranteed.

    Raises
    ------
    ValueError
        If data sources do not have the same format and type, or if no
        data sources are passed.
    """

    def __init__(self, data_sources, keep_order=False):
        if len(data_sources) == 0:
            raise ValueError('Need at least one data source')
        if not all(x.dshape == data_sources[0].dshape for x in data_sources):
            raise ValueError('Data sources dimensionality has to be equal')
        if not all(x.tshape == data_sources[0].tshape for x in data_sources):
            raise ValueError(
                'Data sources target dimensionality has to be equal')
        if not all(x.dtype == data_sources[0].dtype for x in data_sources):
            raise ValueError('Data sources data type has to be equal')
        if not all(x.ttype == data_sources[0].ttype for x in data_sources):
            raise ValueError('Data sources target type has to be equal')

        self._data_sources = data_sources
        self._ds_ends = np.array([0] + [len(d) for d in data_sources]).cumsum()
        self.keep_order = keep_order

    @classmethod
    def from_files(cls, data_files, target_files, memory_mapped=False,
                   data_source_type=DataSource, names=None, **kwargs):
        """
        Create a :class:AggregatedDataSource from multiple NumPy binary
        files.

        Parameters
        ----------
        data_files : list of file-like objects or strings
            Data files to read. For each element in this list a new
            :class:`DataSource` will be created.
        target_files : list of file-like objects or strings
            Target files to read. Each file in this list must correspond to the
            data file in :param:`data_files` at the same index.
        memory_mapped : bool
            Should data and targets be read memory-mapped or loaded into
            memory.
        data_source_type : type
            Datasource type. By default a standard :class:`DataSource`, but
            you can use other types, e.g. :class:`ContextDataSource`.
        names : list of strings or None
            Names of the datasources. Has to be of same length as
            :param:`data_files`. If None, file names without extensions will
            be used.
        **kwargs :
            Keyword arguments passed to the `from_files` classmethod of each
            data source created.

        Returns
        -------
        AggregatedDataSource
            :class:AggregatedDataSource of data sources created given the
            file lists.

        """
        if not names:
            names = [basename(d).split('.')[0] for d in data_files]

        if len(data_files) != len(target_files):
            raise ValueError('Need same number of data files ({}) and '
                             'target files ({})'.format(len(data_files),
                                                        len(target_files)))

        if len(names) != len(data_files):
            raise ValueError('Need same number of names ({}) and '
                             'data files ({})'.format(len(names),
                                                      len(data_files)))

        return cls(
            [data_source_type.from_files(d, t, memory_mapped=memory_mapped,
                                         name=n, **kwargs)
             for d, t, n in izip(data_files, target_files, names)]
        )

    def _to_ds_idx(self, idx):
        """Computes the datasource index for a given element index"""
        ds_idx = self._ds_ends.searchsorted(idx, side='right') - 1
        d_idx = idx - self._ds_ends[ds_idx]
        return ds_idx, d_idx

    def __getitem__(self, idx):
        """
        Return pre-processed data and targets for given index.

        Parameters
        ----------
        idx : int, slice, list, np.ndarray
            Index of data and targets to return.

        Returns
        -------
        tuple
            Pre-processed data and targets for given index. If
            :param:`keep_order` was set to False, and :param:`idx` is a list,
            slice or np.ndarray, the order of the data might not correspond to
            the order given in :param:`idx`.
        """
        if isinstance(idx, int):
            ds_idx, d_idx = self._to_ds_idx(idx)
            return self._data_sources[ds_idx][d_idx]

        elif isinstance(idx, list):
            if self.keep_order:
                idx = np.array(idx)
                sort_idxs = idx.argsort()
                idx = idx[sort_idxs]
                revert_idxs = sort_idxs.argsort()
            else:
                idx.sort()

            ds_idxs, d_idxs = self._to_ds_idx(idx)
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

        elif isinstance(idx, slice):
            return self[range(idx.start or 0, idx.stop or self.n_data,
                              idx.step or 1)]

        elif isinstance(idx, np.ndarray):
            return self[list(idx)]

        else:
            raise TypeError('Index type {} not supported!'.format(type(idx)))

    def datasource(self, idx):
        """
        Get a single DataSource.

        Parameters
        ----------
        idx : int
            Index of the data source

        Returns
        -------
        :class:`DataSource`
        """
        return self._data_sources[idx]

    @property
    def n_datasources(self):
        """int: number of data sources"""
        return len(self._data_sources)

    @property
    def n_data(self):
        """int: number of data points (equals number of targets)"""
        return sum(ds.n_data for ds in self._data_sources)

    def __len__(self):
        """int: number of data points (equals number of targets)"""
        return self.n_data

    @property
    def dshape(self):
        """tuple: shape of a data point"""
        return self._data_sources[0].dshape

    @property
    def tshape(self):
        """tuple: shape of a target"""
        return self._data_sources[0].tshape

    @property
    def dtype(self):
        """type: dtype of the data"""
        return self._data_sources[0].dtype

    @property
    def ttype(self):
        """type: dtype of the targets"""
        return self._data_sources[0].ttype

    def __str__(self):
        """str: a string representation of the :class:DataSource"""
        return '{}: N={}  dshape={}  tshape={}'.format(
            self.__class__.__name__,
            self.n_data, self.dshape, self.tshape)


def get_datasources(files, preprocessors=None, cached=False, cache_dir=None,
                    **kwargs):
    """
    This function creates pre-processed datasources given a list of
    file dictionares. A file dictionary contains two keys: 'feat' and 'targ',
    where each refers to a list of feature and target files respectively.
    The preprocessors are trained using the datasource defined by the first
    dictionary

    Parameters
    ----------
    files : list of dict
        file dictionaries with the aforementioned format
    preprocessors : list of callables or None
        preprocessors to be applied to the data
    cached : bool
        cache datasources or not
    cache_dir : string
        directory where to cache datasources
    **kwargs :
        additional arguments to be passed to AggregatedDataSource.from_files

    Returns
    -------
    list of :class:AggregatedDataSource
        :class:AggregatedDataSource instances for each entry in :param:files
    """

    datasources = [
        AggregatedDataSource.from_files(
            f['feat'], f['targ'], memory_mapped=True,
            preprocessors=preprocessors, **kwargs
        )
        for f in files
    ]

    # train on first datasource
    if preprocessors is not None:
        for p in preprocessors:
            p.train(datasources[0])

    if cached:
        datasources = [cache_aggregated_datasource(ds, cache_dir=cache_dir)
                       for ds in datasources]

    return datasources


def cache_datasource(datasource, batch_size=8192, cache_dir=None):
    """
    Caches a :class:DataSource into a temporary file. This function can be
    used to pre-compute all pre-processing steps and avoid generating
    the context when using a :class:ContextDataSource

    Parameters
    ----------
    datasource : :class:DataSource
        Data source to be cached
    batch_size : int
        Pre-compute the datasource using this batch size (useful when low on
        memory)
    cache_dir : str
        Directory to put the temporary files to

    Returns
    -------
    :class:DataSource
        Cached data source

    """
    from tempfile import NamedTemporaryFile

    f_cache = NamedTemporaryFile(suffix='.npz', dir=cache_dir)
    t_cache = NamedTemporaryFile(suffix='.npz', dir=cache_dir)
    f_shape = (datasource.n_data,) + datasource.dshape
    t_shape = (datasource.n_data,) + datasource.tshape

    features = np.memmap(f_cache, dtype=datasource.dtype, shape=f_shape)
    targets = np.memmap(t_cache, dtype=datasource.ttype, shape=t_shape)

    i = 0
    for f, t in iterate_batches(datasource, batch_size=batch_size,
                                randomise=False, expand=False):
        features[i:i + f.shape[0]] = f
        targets[i:i + t.shape[0]] = t
        i += f.shape[0]

    ds = DataSource(features, targets, name=datasource.name)
    ds.cache_files = [f_cache, t_cache]
    return ds


def cache_aggregated_datasource(agg_datasource, batch_size=8192,
                                cache_dir=None):
    """
    Cache an :class:AggregatedDataSource. This is done by caching all data
    sources that it comprises.

    Parameters
    ----------
    agg_datasource : :class:AggregatedDataSource
        Aggregated data source to cache.
    batch_size : int
        Pre-compute the datasource using this batch size (useful when low on
        memory)
    cache_dir : str
        Directory to put the temporary files to

    Returns
    -------
    :class:AggregatedDataSource
        Cached aggregated data source

    """
    datasources = [agg_datasource.datasource(i)
                   for i in range(agg_datasource.n_datasources)]
    cached_ds = [cache_datasource(ds, batch_size, cache_dir)
                 for ds in datasources]
    return AggregatedDataSource(cached_ds,
                                keep_order=agg_datasource.keep_order)
