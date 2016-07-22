# dmgr - data manager

This library provides "data source" objects that provide training material
for my neural network experiments. It also contains functionality to load
audio datasets that come in the specific directory structure used at the
[Department of Computational Perception](http://www.cp.jku.at/), and thus by
me.

This library is **primarily for my personal use**, so it is tailored to my
development environment. Although I made some effort to make it more general,
documentation might be lacking, and there might be assumptions that do not
hold in general. I'm sorry.

You might want to look at the following repositories:

 - [nn](https://github.com/fdlm/nn): Helper functions for training
   [Lasagne](https://github.com/Lasagne/Lasagne) neural networks.
 - [chordrec](https://github.com/fdlm/chordrec): My experimental framework
   to train neural network-based models for chord recognition. This is the
   actual reason why the dmgr and nn libraries exist.

## Dataset structure

Each dataset has the following directory structure:

      +-- dataset_name/
           +-- audio/
                +-- *.flac
           +-- annotations/
                +-- beats/
                     +-- *.beats
                +-- chords/
                     +-- *.chords
                ...
           +-- splits/
                +-- 8-fold_cv_*.fold

Names of annotation and audio files need to correspond. `dmgr` does not require
these files to be in separate directories; it will search the whole dataset
directory recursively. However, the crossvalidation fold definitions have
to be in a `splits` sub-directory. Each `.fold` file contains a list of file
names (one per line, without file extension) that represent the *test files* of
this fold. The rest of the files can be used for training.
