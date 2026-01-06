# Training subdirectory

This directory holds tools to create and manage the cache files needed 
by the main training program


## Cache generation

* image caching script (create_img_sdvae.py or create_img_sdxl.py)
* text caption caching script (create_t5cache_768.py, or create_clipl.py)


Note that some scripts expect to make use of the custom "diffusers pipeline" present in
huggingface model "opendiffusionai/stablediffusion_t5"
or "opendiffusionai/sdx_t5"

Sample cache creation:

    ./create_img_sdxl.py --model opendiffusionai/stablediffusion_t5 --data_root /data --custom

    # Note that this only pulls the vae from the pipeline. So if you are really sure you know
    # which vae to use, you may use one of the standard pipelines, and skip the --custom
    #  eg:
    # ./create_img_cache.py --model stabilityai/stable-diffusion-xl-base-1.0 --data_root /data 

    ./create_t5cache_768.py --model opendiffusionai/stablediffusion_t5 --data_root /data

    # The t5cache on the other hand, HAS to use one of our custom pipelines. Therefore,
    # the --custom is already implied and built-in

