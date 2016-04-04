import itertools as it
import os
import random
import fnmatch

import numpy as np

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

    def get_name(c):
        return c.name if 'name' in c.__dict__ else c.__name__

    feat_cache_path = os.path.join(dest_dir, get_name(compute_feat))
    if not os.path.exists(feat_cache_path):
        os.makedirs(feat_cache_path)

    target_cache_path = os.path.join(dest_dir, get_name(compute_targets))
    if not os.path.exists(target_cache_path):
        os.makedirs(target_cache_path)

    for sf, gtf in zip(source_files, ground_truth_files):
        neutral_file = os.path.splitext(os.path.basename(sf))[0]
        feat_file = os.path.join(feat_cache_path, neutral_file + feat_ext)

        feat = None
        if not os.path.exists(feat_file):
            feat = compute_feat(sf)
            np.save(feat_file, feat)

        if gtf is not None:
            target_file = os.path.join(target_cache_path,
                                       neutral_file + target_ext)
            if not os.path.exists(target_file):
                if feat is None:
                    feat = np.load(feat_file)

                targets = compute_targets(gtf, feat.shape[0])
                np.save(target_file, targets)

            target_files.append(target_file)

        feat_files.append(feat_file)

    if target_files is not None:
        return feat_files, target_files
    else:
        return feat_files


def match_files(files, ext, matching_files, matching_ext):
    """
    Finds matching files in two lists, ignoring extensions.
    :param files:  list of file names to find matches for
    :param ext:    file extension of files to find matches for
    :param matching_files: list of file names to matches
    :param matching_ext:   file extension of second files to match
    :return:               list of files from second_files that match the ones
                           in the first_files list
    """
    matched_second_files = []

    def strip_ext(fn, e):
        return fn[:-len(e)] if fn.endswith(e) else fn

    for f in files:
        if len(ext) > 0:
            f = strip_ext(f, ext)
        matches = fnmatch.filter(matching_files, f + matching_ext)
        if len(matches) > 1:
            raise SystemExit('Multiple matching '
                             'files for {}: {}'.format(f, matches))
        elif len(matches) == 0:
            raise SystemExit('No matching files for {}!'.format(f))
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
    files = list(files)
    random.shuffle(files)
    n_split = int(len(files) * split_perc)
    return files[:n_split], files[n_split:]


def predefined_split(files, ext, *split_defs):
    """
    Splits a list of files into seperate lists based on file lists defined in
    text files. These text files contain one file name per line, without file
    extension. All files not belonging to any of the given splits are also
    returned.

    Parameters
    ----------
    files : list of str
        List of all file names.
    ext : str
        File extension to be ignored in the list of files.
    *split_defs : str
        One or more files defining the splits.

    Returns
    -------
    list of list of str
        Lists of matching files as defined in `*split_defs`. The first list
        contains all files in the `files` list that do not belong to any
        split as defined in `*split_defs`.
    """

    train_files = set(files)

    splits = []
    for split_def in split_defs:
        with open(split_def, 'r') as f:
            splits.append(match_files(f.read().splitlines(), '', files, ext))
        train_files.difference_update(set(splits[-1]))

    return [list(train_files)] + splits

