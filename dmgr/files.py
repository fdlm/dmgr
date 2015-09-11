import os
import random
import fnmatch

import numpy as np

from madmom.utils import match_file, search_files

AUDIO_EXT = '.flac'
FEAT_EXT = '.features.npy'
TARGET_EXT = '.targets.npy'
FPS = 100


def find(directory, pattern):
    """
    Generator that recursively finds all files in a directory that match the
    pattern
    :param directory: directory to search for files in
    :param pattern:   file pattern (e.g. '*.npy')
    :return:          filenames
    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def expand(files_and_dirs, pattern):
    """
    Goes through list of files and directories, and expands directories
    to all files contained in it matching the pattern.
    :param files_and_dirs: list of files and directories
    :param pattern:        file pattern to look for in directories
    :return:               list of filenames
    """
    if not isinstance(files_and_dirs, list):
        files_and_dirs = [files_and_dirs]

    def expand_dirs(d):
        if os.path.isdir(d):
            return list(find(d, pattern))
        else:
            return [d]

    # search directories recursively, or just take the file
    return sum(map(expand_dirs, files_and_dirs), [])


def prepare(files, compute_feat, compute_targets,
            feat_ext=FEAT_EXT, target_ext=TARGET_EXT, fps=FPS):
    """
    Prepares features and ground truth for audio files. This function assumes
    that audio files, feature files and target files will be stored in the
    same directory. If the corresponding feature or target file already exists,
    it will not be computed again.

    :param files:            list of audio files
    :param compute_feat:     function that takes an audio file and computes the
                             features for it
    :param compute_targets:  function that takes the audio file name without
                             extension ('foo/bar/baz.flac' -> 'foo/bar/baz')
                             and computes the ground truth for it. If 'None',
                             no targets will be computed
    :param feat_ext:         feature file extension
    :param target_ext:       target file extension.
    :param fps:              frames per second
    :return:                 list of feature files, and if targets were
                             computed, a list of target files
    """

    feat_files = []
    target_files = []

    for f in files:
        neutral_file = os.path.splitext(f)[0]
        feat_file = neutral_file + feat_ext

        feat = None
        if not os.path.exists(feat_file):
            feat = compute_feat(f)
            np.save(feat_file, feat)

        if compute_targets is not None:
            target_file = neutral_file + target_ext
            if not os.path.exists(target_file):
                if feat is None:
                    feat = np.load(feat_file)
                    num_frames = feat.shape[0]
                else:
                    num_frames = feat.num_frames

                targets = compute_targets(neutral_file, num_frames, fps)
                if targets is None:
                    print('No ground truth for ' + neutral_file)
                    continue
                np.save(target_file, targets)

            target_files.append(target_file)

        feat_files.append(feat_file)

    if compute_targets is not None:
        return feat_files, target_files
    else:
        return feat_files


def match_feature_and_target(files, feat_ext=FEAT_EXT, target_ext=TARGET_EXT):
    """
    Extracts from a list of file names the feature and matching target files
    :param files:       list of file names
    :param feat_ext:    extension of feature files
    :param target_ext:  extension of target files
    :return:            a list of feature files and a list of matching target
                        files
    """
    feat_files = search_files(files, suffix=feat_ext)
    target_files = []

    for ff in feat_files:
        matches = match_file(ff, files, feat_ext, target_ext)
        if len(matches) > 1:
            raise SystemExit('Multiple target files for {}!'.format(ff))
        elif len(matches) == 0:
            raise SystemExit('No target files for {}!'.format(ff))
        else:
            target_files.append(matches[0])

    return feat_files, target_files


def random_split(files, split_perc=0.5):
    """
    Splits files randomly into two sets
    :param files:      list of files
    :param split_perc: percentage of files to be contained in first set
    :return:           two lists of files split
    """
    random.shuffle(files)

    split_perc = split_perc
    n_train = int(len(files) * split_perc)

    train_files = files[:n_train]
    val_files = files[n_train:]

    return train_files, val_files


def split(split_files, files, match_suffix=AUDIO_EXT):
    """
    Splits files based on a predefined list of file names
    :param split_files:  names of files to be contained in split (without ext)
    :param files:        list of files (with extension)
    :param match_suffix: file extension in 'files' list
    :return:
    """
    return [match_file(fd, files, match_suffix=match_suffix)[0]
            for fd in split_files]


def predefined_train_val_test_split(files,
                                    train_def, val_def=None, test_def=None,
                                    match_suffix=AUDIO_EXT):
    """
    Splits a list of files into train, validation and test set based on
    file lists defined in text files. These text files contain one file name
    per line without file extension. If no file defining the validation
    set or test set are given, the function will return None for the
    respective set

    :param files:        total list of all files
    :param train_def:    file containing the train set definition
    :param val_def:      file containing the validation set definition
    :param test_def:     file containing the test set definition
    :param match_suffix: suffix of the files in the total file list
    :return:             list of files for the train set, validation set
                         and test set respectively (tuple)
    """

    with open(train_def, 'r') as f:
        train_files = split(f.read().splitlines(), files, match_suffix)

    if val_def:
        with open(val_def, 'r') as f:
            val_files = split(f.read().splitlines(), files, match_suffix)
    else:
        val_files = None

    if test_def:
        with open(test_def, 'r') as f:
            test_files = split(f.read().splitlines(), files, match_suffix)
    else:
        test_files = None

    return train_files, val_files, test_files

