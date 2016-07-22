import fnmatch
import itertools as it
import os
import random

import numpy as np

FEAT_EXT = '.features.npy'
TARGET_EXT = '.targets.npy'


def find(directory, pattern):
    """
    Generator that recursively finds all files in a directory that match the
    pattern

    Parameters
    ----------
    directory : str
        directory to search for files in
    pattern : str
       file pattern (e.g. '*.npy')

    Yields
    ------
    str
        found filenames

    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def expand(files_and_dirs, pattern):
    """
    Goes through list of files and directories, and expands directories
    to all files contained in it (and its sub-directories) matching the
    pattern.

    Parameters
    ----------
    files_and_dirs : list of str
        files and directories to traverse
    pattern : list of str, or str
        file pattern (or list of file patterns) to look for in directories

    Returns
    -------
    list of str
        list of filenames
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

    Parameters
    ----------
    source_files : list of str
        Source files (e.g. audio files)
    ground_truth_files : list of str or None
        Ground truth files (e.g. list of files with beat annotations). If
        'None', no targets will be computed.
    dest_dir : str
        Directory where to store feature and target files
    compute_feat : callable
        Takes an audio file and computes the features for it.
        If it has a 'name' attribute, the value of this attribute will be
        used as name of the sub-directory where computed features are stored.
        Else, the name of the callable will be used for that.
    compute_targets : callable
        Takes a ground truth file and computes the targets for it.
        If it has a 'name' attribute, the value of this attribute will be
        used as name of the sub-directory where computed targets are stored.
        Else, the name of the callable will be used for that.
    feat_ext : str
        Feature file extension.
    target_ext : str
        Target file extension.

    Returns
    -------
    tuple of lists
        list of feature files, and list of target files (latter possibly empty)
    """

    feat_files = []
    target_files = []

    ground_truth_files = ground_truth_files or it.repeat(None)

    def get_name(c):
        try:
            return c.name
        except AttributeError:
            return c.__name__

    feat_cache_path = os.path.join(dest_dir, get_name(compute_feat))
    if not os.path.exists(feat_cache_path):
        os.makedirs(feat_cache_path)

    target_cache_path = os.path.join(dest_dir, get_name(compute_targets))
    if not os.path.exists(target_cache_path):
        os.makedirs(target_cache_path)

    for sf, gtf in zip(source_files, ground_truth_files):
        neutral_file = os.path.splitext(os.path.basename(sf))[0]
        feat_file = os.path.abspath(
            os.path.join(feat_cache_path, neutral_file + feat_ext))

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

    return feat_files, target_files


def match_files(files, ext, match_files, match_ext):
    """
    Find matching files in two lists, ignoring extensions and directories.
    Concretely, for each file in :param:files, find a file in
    :param:match_files with the same name, ignoring file extensions as given
    by :param:ext and :param:match_ext, and directories.

    Parameters
    ----------
    files : list of str
        File names to find matches for
    ext : str
        File extension to ignore in :param:files
    match_files : list of str
        File names to search for matches in
    match_ext : str
        File extension to ignore in :param:match_files

    Returns
    -------
    list of str
        Files from :param:match_files that match the ones in
        :param:files

    Raises
    ------
    RuntimeError
        If no file or multiple files were found for a file in :param:files
    """
    matched_second_files = []

    def strip_ext(fn, e):
        return fn[:-len(e)] if fn.endswith(e) else fn

    for f in files:
        f = os.path.basename(f)
        if len(ext) > 0:
            f = strip_ext(f, ext)
        matches = fnmatch.filter(match_files, '*' + f + match_ext)
        if len(matches) > 1:
            raise RuntimeError('Multiple matching '
                               'files for {}: {}'.format(f, matches))
        elif len(matches) == 0:
            raise RuntimeError('No matching files for {}!'.format(f))
        else:
            matched_second_files.append(matches[0])

    return matched_second_files


def random_split(lst, p=0.5):
    """
    Randomly split a list into two lists

    Parameters
    ----------
    lst : list
        List to be split
    p : float
        Percentage of files to be contained in first list (<=1.0)

    Returns
    -------
    tuple of lists
        split lists

    Raises
    ------
    ValueError
        if p > 1.0
    """
    if p > 1.0:
        raise ValueError("Split percentage must be <= 1.0")
    lst = list(lst)
    random.shuffle(lst)
    n_split = int(len(lst) * p)
    return lst[:n_split], lst[n_split:]


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
