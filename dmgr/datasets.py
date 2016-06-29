import numpy as np

from . import files


def name_callable(func, name):
    """
    Attaches a name to a callable.

    Parameters
    ----------
    func : callable
        The callable to attach the name to
    name : str
        Name to attach to the callable

    Returns
    -------
    Function with `name` attribute

    """
    def fn(*args, **kwargs):
        return func(*args, **kwargs)
    fn.name = name
    return fn


class Dataset(object):
    """
    The Dataset class prepares and manages features and targets for a data set.
    It assumes that a data set consists of "source files", containing the
    raw input data, and "ground truth files", containing some sort of ground
    truth for these files. The source files and ground truth files reside
    in the same directory (or in a sub-directory of this directory) and have
    a different file extension. From these files, features and targets
    can be computed. The features and targets will be cached.

    Parameters
    ----------
    data_dir : str
        Base directory of the data set
    feature_cache_dir : str
        Directory where to store cached features
    split_defs : list of str
        Files containing the fold split definitions. To not use any
        split definition, set to [], None, or False.
    source_ext : str
        File extension of the source files (e.g. '.wav')
    gt_ext : str
        File extension of the ground truth files (e.g. '.beats')
    compute_features : callable with `name` attribute
        Callable that takes a source file as parameter and returns the computed
        features. If no `name` attribute is provided, the `__name__` attribute
        will be used. See also :func:`name_callable`.
    compute_targets : callable with `name` attribute
        Callable that takes a ground truth file and the number of targets
        as parameters and returns the computed targets. If no `name` attribute
        is provided, the `__name__` attribute will be used.
        See also :func:`name_callable`.
    """

    def __init__(self, data_dir, feature_cache_dir, split_defs,
                 source_ext, gt_ext, compute_features, compute_targets):

        src_files = files.expand(data_dir, '*' + source_ext)
        gt_files = files.expand(data_dir, '*' + gt_ext)
        gt_files = files.match_files(src_files, source_ext, gt_files, gt_ext)

        feat_files, target_files = files.prepare(
            src_files, gt_files, feature_cache_dir,
            compute_feat=compute_features,
            compute_targets=compute_targets,
        )

        self.feature_files = feat_files
        self.target_files = target_files
        self.gt_files = gt_files

        self.split_defs = split_defs

    def all_files(self):
        """
        Return all feature and target files without split

        Returns
        -------
        List of dict
            Dictionaries containing all feature files (key: 'feat') and
            all target files (key: 'targ')
        """
        return {'feat': self.feature_files,
                'targ': self.target_files}

    def predefined_split(self, *split_defs):
        """
        For each split_def, creates a file dictionary with features and target
        files matching the splits. A split is a text file containing one
        file name per line. All feature and target files not belonging to any
        of the given splits are also returned.

        Parameters
        ----------
        *split_defs : str
            One or more files defining the splits.

        Returns
        -------
        List of dict
            Dictionaries containing feature files (key: 'feat') and target
            files (key: 'targ') for each split definition in `*split_defs`.
            The first dictionary contains feature and target files not
            belonging to any split defined in `*split_defs`
        """

        def match_targets(feat):
            return files.match_files(feat, files.FEAT_EXT,
                                     self.target_files, files.TARGET_EXT)

        feature_splits = files.predefined_split(
            self.feature_files, files.FEAT_EXT, *split_defs)
        return [{'feat': feat,
                 'targ': match_targets(feat)}
                for feat in feature_splits]

    def fold_split(self, *folds):
        """
        Creates a dictionary containing feature and target files for each
        given fold. Folds are defined by the :param:split_defs parameter
        when creating a :class:Dataset instance.

        Parameters
        ----------
        folds : int
            Fold IDs to return the feature and target files for

        Returns
        -------
        list of dict
            Dictionaries containing feature files (key: 'feat') and target
            files (key: 'targ') for each fold in `*folds`.
            The first dictionary contains feature and target files not
            belonging to any fold defined in `*folds`

        Raises
        ------
        RuntimeError
            If no fold splits were defined upon constructing the Dataset

        """
        if not self.split_defs:
            raise RuntimeError('No cross-validation folds defined!')

        return self.predefined_split(*[self.split_defs[f] for f in folds])

    def random_split(self, distribution=None, random=np.random.RandomState()):
        """
        Creates a dictionary containing feature and target files, split
        randomly according to the given :param:distribution.

        Parameters
        ----------
        distribution : ndarray-like
            Distribution defining the splits. If it sums to less than 1.0,
            another bin that catches the remaining probability will be created
            and added as first bin.
        random : np.random.RandomState
            Random state. Default: np.random.RandomState()

        Returns
        -------
        list of dict
            Dictionaries containing feature files (key: 'feat') and target
            files (key: 'targ') for each bin defined in :param:`distribution`.
            Number of files for each bin corresponds to its probability, e.g.
            for a distribution `[0.6, 0.4]`, the first bin will contain 60%
            of files, and the second bin 40%.

        Raises
        ------
        ValueError
            If distribution sums to more than 1.0

        """
        distribution = np.array(distribution)

        if sum(distribution) > 1.0:
            raise ValueError('Distribution sums to > 1.0.')
        elif sum(distribution) < 1.0:
            distribution = np.hstack((1.0 - sum(distribution), distribution))

        ffiles = list(self.feature_files)
        random.shuffle(ffiles)
        idxs = [None] + list(distribution.cumsum() * len(ffiles))

        splits = []
        for begin, end in zip(idxs[:-1], idxs[1:]):
            feat = ffiles[begin:end]
            targ = files.match_files(feat, self.target_files,
                                     files.FEAT_EXT, files.TARGET_EXT)
            splits.append({'feat': feat, 'targ': targ})

        return splits


