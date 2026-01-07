import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from train_state import TrainState
from train_utils import sample_img


def checkpointandsave(pipe, unet, accelerator, tstate: TrainState):
    args = tstate.args

    if args.is_custom:
        custom_pipeline = args.pretrained_model
    else:
        custom_pipeline = None

    if tstate.global_step % args.gradient_accum != 0:
        print("INTERNAL ERROR: checkpointandsave() not called on clean step")
        return

    ckpt_dir = os.path.join(args.output_dir,
                            f"checkpoint-{tstate.batch_count:05}")
    if os.path.exists(ckpt_dir):
        print(f"Checkpoint {ckpt_dir} already exists. Skipping redundant save")
        return
    pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
    pipe.unet = accelerator.unwrap_model(unet)

    # log_unet_l2_norm(pipe.unet, tstate.tb_writer, tstate.batch_count)

    print(f"Saving checkpoint to {ckpt_dir}")
    pipe.save_pretrained(ckpt_dir, safe_serialization=True)
    pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
    if args.sample_prompt is not None:
        sample_img(args, args.seed, ckpt_dir,
                   custom_pipeline)
    if args.copy_config:
        savefile = os.path.join(args.output_dir, args.copy_config)
        if not os.path.exists(savefile):
            tqdm.write(f"Copying {args.copy_config} to {args.output_dir}")
            shutil.copy(args.copy_config, args.output_dir)

            import yaml
            savefile = os.path.join(args.output_dir, "args.yaml")
            tqdm.write(f"Saving commandline to  {savefile}")
            Path(savefile).write_text(yaml.safe_dump(vars(args), sort_keys=True))

    savefile = os.path.join(ckpt_dir, "latent_paths")
    with open(savefile, "w") as f:
        f.write('\n'.join(tstate.latent_paths) + '\n')
        f.close()
    print("Wrote", len(tstate.latent_paths), "loglines to", savefile)
    tstate.latent_paths = []
