from tqdm.auto import tqdm
from diffusers import DiffusionPipeline
import torch



def collate_fn(examples):
    return {
        "img_cache": [e["img_cache"] for e in examples],
        "txt_cache": [e["txt_cache"] for e in examples],
    }


# PIPELINE_CODE_DIR is typicaly the dir of original model
def sample_img(args, seed, CHECKPOINT_DIR, PIPELINE_CODE_DIR):
    prompt = args.sample_prompt
    tqdm.write(f"Trying render of '{prompt}' using seed {seed} ..")
    pipe = DiffusionPipeline.from_pretrained(
        CHECKPOINT_DIR,
        custom_pipeline=PIPELINE_CODE_DIR,
        use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
        # torch_dtype=torch.bfloat16,
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_sequential_cpu_offload()

    # Make sure that prompt order doesnt change effective seed
    generator = [torch.Generator(device="cuda").manual_seed(seed)
                 for _ in range(len(prompt))]

    images = pipe(prompt, num_inference_steps=args.sampler_steps, generator=generator).images
    for ndx, image in enumerate(images):
        fname = f"sample-{seed}-{ndx}.png"
        outname = f"{CHECKPOINT_DIR}/{fname}"
        image.save(outname)
        print(f"Saved {outname}")


def log_unet_l2_norm(unet, tb_writer, step):
    """
    Util to log the overall average parameter size.
    Purpose is to determine if we maybe need weight decay or not.
    (If it grows significantly over time, we prob need it)
    """
    # Gather all parameters as a single vector
    params = [p.data.flatten() for p in unet.parameters() if p.requires_grad]
    all_params = torch.cat(params)
    l2_norm = torch.norm(all_params, p=2).item()
    tb_writer.add_scalar('unet/L2_norm', l2_norm, step)
