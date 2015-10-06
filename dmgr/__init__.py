from . import (datasources, iterators, files, preprocessing)
import numpy as np


class Dataset:
    """
    Class for easier dataset loading
    """

    def __init__(self, data_dir, feature_cache_dir, split_defs,
                 source_ext, gt_ext, compute_features, compute_targets, fps):
        """
        Initialises the dataset class
        :param data_dir:          dataset base directory
        :param feature_cache_dir: directory where to store cached features
        :param split_defs:        files containing the fold split definitions
        :param source_ext:        file extension of source files (e.g. '.wav')
        :param gt_ext:            file extension of ground truth files
                                  (e.g. '.beats')
        :param compute_features:  function that computes the features given
                                  a source file
        :param compute_targets:   function that computes the targets given
                                  a target file, number of frames and frames
                                  per second
        :param fps:               frames per second
        """

        src_files = files.expand(data_dir, '*' + source_ext)
        gt_files = files.expand(data_dir, '*' + gt_ext)
        gt_files = files.match_files(src_files, gt_files, source_ext, gt_ext)

        feat_files, target_files = files.prepare(
            src_files, gt_files, feature_cache_dir,
            compute_feat=compute_features,
            compute_targets=compute_targets,
            fps=fps
        )

        self.feature_files = feat_files
        self.target_files = target_files
        self.gt_files = gt_files

        self.split_defs = split_defs

    def get_fold_split(self, val_fold=0, test_fold=1):
        """
        Creates a file dictionary (as used by get_preprocessed_datasource),
        where train, validation, and test folds are pre-defined in split
        files.
        :param val_fold:  index of validation fold
        :param test_fold: index of test fold
        :return: file dictionary
        """
        if not self.split_defs:
            raise RuntimeError('No cross-validation folds defined!')

        train_feat, val_feat, test_feat = \
            files.predefined_train_val_test_split(
                self.feature_files,
                self.split_defs[val_fold],
                self.split_defs[test_fold],
                match_suffix=files.FEAT_EXT
            )

        train_targ = files.match_files(train_feat, self.target_files,
                                       files.FEAT_EXT,
                                       files.TARGET_EXT)
        val_targ = files.match_files(val_feat, self.target_files,
                                     files.FEAT_EXT,
                                     files.TARGET_EXT)
        test_targ = files.match_files(test_feat, self.target_files,
                                      files.FEAT_EXT,
                                      files.TARGET_EXT)

        return {'train': {'feat': train_feat,
                          'targ': train_targ},
                'val': {'feat': val_feat,
                        'targ': val_targ},
                'test': {'feat': test_feat,
                         'targ': test_targ}}

    def get_rand_split(self, val_perc=0.2, test_perc=0.2,
                       random=np.random.RandomState(seed=0)):
        """
        Creates a file dictionary (as used by get_preprocessed_datasource),
        where train, validation, and test folds are created randomly.
        :param val_perc:  percentage of files to be used for validation
        :param test_perc: percentage of files to be used for testing
        :param random:    random state
        :return:          file dictionary
        """
        indices = np.arange(len(self.feature_files))
        random.shuffle(indices)
        n_test_files = int(len(indices) * test_perc)
        n_val_files = int(len(indices) * val_perc)

        test_feat = self.feature_files[:n_test_files]
        val_feat = self.feature_files[n_test_files:n_test_files + n_val_files]
        train_feat = self.feature_files[n_test_files + n_val_files:]

        train_targ = files.match_files(train_feat, self.target_files,
                                       files.FEAT_EXT,
                                       files.TARGET_EXT)
        val_targ = files.match_files(val_feat, self.target_files,
                                     files.FEAT_EXT,
                                     files.TARGET_EXT)
        test_targ = files.match_files(test_feat, self.target_files,
                                      files.FEAT_EXT,
                                      files.TARGET_EXT)

        return {'train': {'feat': train_feat,
                          'targ': train_targ},
                'val': {'feat': val_feat,
                        'targ': val_targ},
                'test': {'feat': test_feat,
                         'targ': test_targ}}
