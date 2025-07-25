# Training subdirectory

This directory holds my current attempts at a custom AI model training script,
along with associated utilities and cache creation tools.

These tools can also be used to train standard SD1.5. 

My main focus is using the CLI tools in an attempt to train a model that has
the following components:

* SDXL vae
* T5 xxl text encoder
* SD 1.5 unet

I'm training this in bf16 precision and 512x512 images, because with this setup,
I can run a native batch size of 64 on my rtx 4090, and get
1600 steps per hour. I can even just barely fit batch 64, accum 4

## Assumptions

The code is currently centered around SD1.5 training.
See the "train_sd.sh" script for the vanilla version,
or "train_t5.sh" for the fancier one.

In theory the core loop could be modified to work with SDXL.
More work would be required to fit it for for any other model.

Loosely speaking it uses the "diffusers" Pipeline methodology, at least in some places.

The training script is a work in progress. Not guaranteed to work correctly at this point!

## Data prep

To use the training stuff, you need to prepare a dataset.
Initially, it should be a directory, or directory tree, with a bunch of image files
(usually .jpg) and a set of matching .txt files which contain a caption for its jpg twin.

You will then need to generate cache files for them. See below.

## Cache generation

* image caching script (create_img_sdvae.py or create_img_sdxl.py)
* text caption caching script (create_t5cache_768.py, or create_clipl.py)


Note that some scripts expect to make use of the custom "diffusers pipeline" present in
huggingface model "opendiffusionai/stablediffusion_t5"
or "opendiffusionai/sdx_t5"

Sample usage;

    ./create_img_sdxl.py --model opendiffusionai/stablediffusion_t5 --data_root /data --custom

    # Note that this only pulls the vae from the pipeline. So if you are really sure you know
    # which vae to use, you may use one of the standard pipelines, and skip the --custom
    #  eg:
    # ./create_img_cache.py --model stabilityai/stable-diffusion-xl-base-1.0 --data_root /data 

    ./create_t5cache_768.py --model opendiffusionai/stablediffusion_t5 --data_root /data

    # The t5cache on the other hand, HAS to use one of our custom pipelines. Therefore,
    # the --custom is already implied and built-in

## Training

You can directly just call the backend "train_from_cached.py" if you like. 
However, you will probably prefer to use the convenience front-end wrapper.
It also effectively functions as a configuration save file.

Choose either train_sd.sh to train_t5.sh

Edit the variable assignments within, to match your preferences.
Then run the script.

It takes care of calling the back-end train_from_cached.py


## Benefits

Benefits of this method over larger programs:

* You can easily identify the cache files. 
* You can also easily choose to regenerate JUST img cache or text cache files
* Similarly, it is easy to selectively remove cache files associated with main ones.

## Comparing runs

When you have two seperate output directories with different settings...
as long as you sampled at the same step interval, you can use the 
[../dataset_scripts/compare_imgdirs.py](../dataset_scripts/compare_imgdirs.py)
tool to show the same sample images from each directory side-by-side


# Square image limitation

By default, these tools will expect to work with 512x512 resolution.

Resolution is controlled by the size of the generated image cache files.
That is why the back-end does not have any -resolution flag

In theory, if you use the --resolution tweaking in create_img_sdvae.py,
(and remove the CenterCrop call) 
you could also train on other sizes. But BEWARE!
You need to understand more about model training to do that correctly.

## Manual image ratio bucketing

I recommend you do not use mixed-aspect ratio training.

With that in mind, lets say you wanted to train on 512x768 resolution.
First you would adjust the image cache creation script to resize all images to strictly that size
before generating the latent image cache files.

Then in theory, you could just go ahead and run the train_xx script of your choice

## Model capacity limits

There is a theoretical upper limit on total amount of knowledge you can train.

If you want a model knowledable about many things, you must stick to one aspect ratio. 

This is because the more varients of size you train on for a particular subject, the more
you will displace knowledge about other things you are not training on.

This is why I am focusing on square training exclusively for my current projects

# ------------------------------------------------------------------------

# Tensorboard logging

These scripts output logging to tensorboard, in a "tensorboard" subdirectory.
To view, start tensorboard as a seperate program in that directory, etc.

When using tensorboard with other training programs, you may be used to 
seeing the typical "learning rate" and "loss" graphs.
This program additionally adds in "qk_grads_av" and "raw loss".

This is because when you are training from scratch, it is really important to make sure
that the "q/k gradients" arent doing crazy things like going to 0.
It is also sometimes nice, when SNR is enabled, to compare the default loss stats, vs the
"raw (non snr)" loss

