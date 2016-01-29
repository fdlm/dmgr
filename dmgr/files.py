import itertools as it
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
    :param pattern:        file pattern (or list of file patterns) to look for
                           in directories.
    :return:               list of filenames
    """
    if not isinstance(files_and_dirs, list):
        files_and_dirs = [files_and_dirs]

    if not isinstance(pattern, list):
        pattern = [pattern]

    files = []
    for p in pattern:
        def expand_dirs(d):
            if os.path.isdir(d):
                return list(find(d, p))
            else:
                return [d]

        files += sum(map(expand_dirs, files_and_dirs), [])

    return files


def prepare(source_files, ground_truth_files, dest_dir,
            compute_feat, compute_targets,
            feat_ext=FEAT_EXT, target_ext=TARGET_EXT):
    """
    Prepares features and targets for audio files. This function assumes that
    source and ground truth files are given in separate lists with matching
    order (meaning that the first ground truth file matches the first
    source file). Feature and target files will be named after the source
    file with the given extensions into the given destination directory.
    If the corresponding feature or target file already exists,
    it will not be computed again.

    :param source_files:     list of source files (e.g. list of audio files)
    :param ground_truth_files: list of ground truth files (e.g. list of files
                             with beat annotations). If 'None', no targets will
                             be computed.
    :param dest_dir:         directory where to store feature and target files
    :param compute_feat:     function that takes an audio file and computes the
                             features for it
    :param compute_targets:  function that takes the audio file name without
                             extension ('foo/bar/baz.flac' -> 'foo/bar/baz')
                             and computes the ground truth for it.
    :param feat_ext:         feature file extension
    :param target_ext:       target file extension.
    :return:                 list of feature files, and if targets were
                             computed, a list of target files
    """

    feat_files = []
    target_files = []

    if not ground_truth_files:
        ground_truth_files = it.repeat(None)
        target_files = None

    feat_cache_path = os.path.join(dest_dir, compute_feat.name)
    if not os.path.exists(feat_cache_path):
        os.makedirs(feat_cache_path)

    for sf, gtf in zip(source_files, ground_truth_files):
        neutral_file = os.path.splitext(os.path.basename(sf))[0]
        feat_file = os.path.join(feat_cache_path, neutral_file + feat_ext)

        feat = None
        if not os.path.exists(feat_file):
            feat = compute_feat(sf)
            np.save(feat_file, feat)

        if gtf is not None:
            target_file = os.path.join(dest_dir, neutral_file + target_ext)
            if not os.path.exists(target_file):
                if feat is None:
                    feat = np.load(feat_file)

                targets = compute_targets(gtf, feat.shape[0], compute_feat.fps)
                np.save(target_file, targets)

            target_files.append(target_file)

        feat_files.append(feat_file)

    if target_files is not None:
        return feat_files, target_files
    else:
        return feat_files


def match_files_single(files, first_ext, second_ext):
    first_files = search_files(files, suffix=first_ext)
    second_files = search_files(files, suffix=second_ext)
    return match_files(first_files, second_files, first_ext, second_ext)


def match_files(first_files, second_files, first_ext, second_ext):
    """
    Matches files from one list to files from a second list with same name
    but different extension.
    :param first_files: list of file names
    :param second_files: list of file names
    :param first_ext:   file extension of files to match
    :param second_ext:  file extension of second files to match
    :return:            list of files from second_files that match the ones
                        in the first_files list
    """
    matched_second_files = []

    for ff in first_files:
        matches = match_file(ff, second_files, first_ext, second_ext)
        if len(matches) > 1:
            raise SystemExit('Multiple matching files for {}!'.format(ff))
        elif len(matches) == 0:
            raise SystemExit('No matching files for {}!'.format(ff))
        else:
            matched_second_files.append(matches[0])

    return matched_second_files


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


def predefined_train_val_test_split(files, val_def=None, test_def=None,
                                    match_suffix=AUDIO_EXT):
    """
    Splits a list of files into train, validation and test set based on
    file lists defined in text files. These text files contain one file name
    per line without file extension. If no file defining the validation
    set or test set are given, the function will return None for the
    respective set. All files that are not defined as belonging to the
    validation set or the test set will be automatically put into the train set.

    :param files:        total list of all files
    :param val_def:      file containing the validation set definition
    :param test_def:     file containing the test set definition
    :param match_suffix: suffix of the files in the total file list
    :return:             list of files for the train set, validation set
                         and test set respectively (tuple)
    """

    train_files = set(files)

    if val_def:
        with open(val_def, 'r') as f:
            val_files = split(f.read().splitlines(), files, match_suffix)
        train_files.difference_update(set(val_files))
    else:
        val_files = None

    if test_def:
        with open(test_def, 'r') as f:
            test_files = split(f.read().splitlines(), files, match_suffix)
        train_files.difference_update(set(test_files))
    else:
        test_files = None

    return list(train_files), val_files, test_files

