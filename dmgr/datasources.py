from itertools import izip
import numpy as np


class DataSource(object):

    def __init__(self, data, targets):
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
    def n_data(self):
        return self.data.shape[0]

    @property
    def data_shape(self):
        return self.data.shape[1:]

    def __len__(self):
        return self.n_data


class AggregatedDataSource(object):

    def __init__(self, data_sources):
        self.data = [d.data[i]
                     for d in data_sources
                     for i in range(len(d))]
        self.targets = [d.targets[i]
                        for d in data_sources
                        for i in range(len(d))]

    @classmethod
    def from_files(cls, data_files, target_files, memory_mapped=False):
        return cls([DataSource.from_files(d, t, memory_mapped)
                    for d, t in izip(data_files, target_files)])

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
            return (np.vstack([self.data[i] for i in item]),
                    np.vstack([self.targets[i] for i in item]))
        return np.vstack(self.data[item]), np.vstack(self.targets[item])

    @property
    def n_data(self):
        return len(self.data)

    @property
    def data_shape(self):
        return self.data[0].shape if self.n_data > 0 else (0,)

    def __len__(self):
        return self.n_data
