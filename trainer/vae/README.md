# vae subdirectory

Seperated out from top level, because this code is completely seperate.

[train_vae.py](train_vae.py) focuses exclusively on training the (SDXL) vae
to be better with your choice of datasets.

In my case that will be real-world images. This is an experiment to see
whether I can improve real-world rendering, at the expense of 
artistic based imagess

Note: If you are using a high-channel-count vae, and you really, really
care about pixel accuracy to original image... then you really, really
want to use --lpips_rawvgg

