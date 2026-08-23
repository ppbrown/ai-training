import os
import shutil

import torch
from tqdm.auto import tqdm

from train_state import TrainState
from train_utils import sample_img, log_unet_l2_norm


def save_train_state(path, optim, lr_sched, tstate: TrainState):
    """Save optimizer/scheduler/RNG/counters needed to resume with --continue_steps."""
    state = {
        "global_step": tstate.global_step,
        "batch_count": tstate.batch_count,
        "optimizer": optim.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if lr_sched is not None:
        state["scheduler"] = lr_sched.state_dict()
    if tstate.disc is not None:
        # Resuming with a freshly-initialized discriminator would hand the
        # UNet a garbage adversarial signal for however long the
        # discriminator takes to become competent again.
        state["disc"] = tstate.disc.state_dict()
        state["opt_d"] = tstate.opt_d.state_dict()
    torch.save(state, path)
    print(f"Saved training state: {path}")


def checkpointandsave(pipe, unet, accelerator, tstate: TrainState,
                      optim=None, lr_sched=None,
                      save_training_state=False, tag=None,
                      skip_sample=False):
    args = tstate.args

    if args.is_custom:
        custom_pipeline = args.pretrained_model
    else:
        custom_pipeline = None

    if tstate.global_step % args.gradient_accum != 0:
        print("INTERNAL ERROR: checkpointandsave() not called on clean step")
        return
    log_unet_l2_norm(unet, tstate.tb_writer, tstate.batch_count)

    if tag:
        ckpt_dir = os.path.join(args.output_dir, tag)
    else:
        ckpt_dir = os.path.join(args.output_dir,
                                f"checkpoint-{tstate.batch_count:05}")
    if os.path.exists(ckpt_dir):
        print(f"Checkpoint {ckpt_dir} already exists. Skipping redundant save")
        return
    pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
    pipe.unet = accelerator.unwrap_model(unet)



    print(f"Saving checkpoint to {ckpt_dir}")
    pipe.save_pretrained(ckpt_dir, safe_serialization=True)
    pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
    if args.sample_prompt is not None and not skip_sample:
        sample_img(args, args.seed, ckpt_dir,
                   custom_pipeline)
    if args.copy_config:
        savefile = os.path.join(args.output_dir, args.copy_config)
        if not os.path.exists(savefile):
            tqdm.write(f"Copying {args.copy_config} to {args.output_dir}")
            shutil.copy(args.copy_config, args.output_dir)

    savefile = os.path.join(ckpt_dir, "latent_paths")
    with open(savefile, "w") as f:
        f.write('\n'.join(tstate.latent_paths) + '\n')
        f.close()
    print("Wrote", len(tstate.latent_paths), "loglines to", savefile)
    tstate.latent_paths = []

    if save_training_state:
        save_train_state(os.path.join(ckpt_dir, "training_state.pt"),
                         optim, lr_sched, tstate)
