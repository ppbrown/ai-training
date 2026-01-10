# AI-training

Tools for training certain AI models.

(Command-line script based)

I am primarily focused on txt2img models. I created this new program 
because I want to create a model with a currently unsupported architecture.
None of the existing programs had all the features I wanted, so
I wrote my own, instead of trying to mod one of those.

## Subdirs

The actual "train a model" scripts are under
[trainer](/trainer/)

But before you train a mode, you need a dataset to train it on.
So there are some relevant scripts to help with that under
[dataset_scripts](/dataset_scripts/)


# Features

The primary features I like about my own scripts are:

* Easy to understand and prune structure for tensor caching

* Easier-to-understand flow(for me, anyway) for the actual training

* Full training config gets copied alongside the resulting model


# Drawbacks

* Currently, only SD1.5 unet supported

* Currently, Only "diffusers" format supported
