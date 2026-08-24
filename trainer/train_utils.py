import os

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


def sample_without_checkpoint(pipe, unet, accelerator, tstate, prompts, seed,
                              sampler_steps, out_dir, device):
    """
    Render sample images straight from the in-memory training pipeline -
    no save-to-disk/reload round trip needed.

    sample_img() above always reloads a fresh pipeline off disk. That one
    is a throwaway object, so it's free to call enable_sequential_cpu_offload()
    on it. This function instead borrows the LIVE pipe/unet, which makes
    pipe(...)'s own cleanup dangerous if --cpu_offload is active:
    every diffusers pipeline's __call__ ends by calling
    self.maybe_free_model_hooks(), which (whenever cpu-offload hooks are
    present) calls self.enable_model_cpu_offload() again. THAT re-run
    calls self.remove_all_hooks() - which restores unet.forward to the
    raw, pre-accelerator.prepare() forward it had captured back at
    trainer startup - and then rebuilds fresh offload hooks on top of
    THAT stale forward. The net effect: accelerator.prepare()'s bf16
    autocast wrapper silently vanishes from the live unet, and the very
    next training step dies with a bf16/fp32 dtype mismatch. (This is
    exactly what happened before this comment was added.) So we save the
    unet's current .forward right before handing it to the pipeline, and
    forcibly restore that exact callable afterward - whatever
    maybe_free_model_hooks() did internally to unet._hf_hook/_old_forward,
    the outer autocast wrapper we put back still wraps around it.

    We also swap in a config-cloned throwaway scheduler for the duration
    of the call. pipe(...) mutates scheduler state internally
    (set_timesteps() rewrites .timesteps/.sigmas/.step_index); train_core
    only ever reads static config off tstate.noise_sched (alphas_cumprod,
    config.num_train_timesteps, ...), so this isn't required for
    correctness today, but it costs nothing and means a future scheduler
    change can't reach back into training state through this path.
    """
    if not prompts:
        return

    os.makedirs(out_dir, exist_ok=True)
    tqdm.write(f"Trying in-memory sample of {prompts} using seed {seed} ..")

    unwrapped_unet = accelerator.unwrap_model(unet)
    was_training = unwrapped_unet.training

    pinned_unet = pipe.unet
    pinned_scheduler = pipe.scheduler
    pinned_safety_checker = getattr(pipe, "safety_checker", None)
    pinned_forward = unwrapped_unet.forward

    pipe.unet = unwrapped_unet
    pipe.scheduler = type(pinned_scheduler).from_config(pinned_scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    unwrapped_unet.eval()
    try:
        # Make sure that prompt order doesnt change effective seed
        generator = [torch.Generator(device=device).manual_seed(seed)
                     for _ in range(len(prompts))]
        with torch.no_grad():
            images = pipe(prompts, num_inference_steps=sampler_steps,
                         generator=generator).images
        for ndx, image in enumerate(images):
            fname = f"sample-{seed}-{ndx}.png"
            outname = os.path.join(out_dir, fname)
            image.save(outname)
            print(f"Saved {outname}")

        # Same bookkeeping checkpointandsave() does: record which cached
        # latents were consumed since the last time this was flushed
        # (whether that flush was a checkpoint save or a prior sample),
        # then reset the rolling list.
        savefile = os.path.join(out_dir, "latent_paths")
        with open(savefile, "w") as f:
            f.write('\n'.join(tstate.latent_paths) + '\n')
        print("Wrote", len(tstate.latent_paths), "loglines to", savefile)
        tstate.latent_paths = []
    finally:
        pipe.unet = pinned_unet
        pipe.scheduler = pinned_scheduler
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = pinned_safety_checker
        unwrapped_unet.forward = pinned_forward
        unwrapped_unet.to(device)
        if was_training:
            unwrapped_unet.train()


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
