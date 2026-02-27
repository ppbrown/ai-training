

# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #
import argparse
from argparse import BooleanOptionalAction

def parse_args():
    p = argparse.ArgumentParser(epilog="Touch 'trigger.checkpoint' in the output_dir to dynamically trigger checkpoint save")
    p.add_argument("--fp32", action="store_true",
                   help="Override default mixed precision fp32/bf16, to force everything full fp32")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable cpu offload at pipe level")
    p.add_argument("--allow_tf32", action="store_true",
                   help="Speed optimization. (Possibly bad at extremely low LR?)")
    p.add_argument("--pretrained_model", required=True,  help="HF repo or local dir")
    p.add_argument("--is_custom", action="store_true",
                   help="Model provides a 'custom pipeline'")
    p.add_argument("--train_data_dir",  nargs="+", required=True, action="append",
                   help="Directory tree(s) containing *.jpg + *.txt.\nCan use more than once, but make sure same resolution for each")
    p.add_argument("--scheduler",      type=str, default="constant", help="Default=constant")
    p.add_argument("--scale_loss_with_accum", action="store_true", 
                   help="When accum >1, scale each microbatch loss /accum")
    p.add_argument("--scheduler_at_epoch", action="store_true", 
                   help="Only consult scheduler at epoch boundaries. Useful if less than 15 epochs")
    p.add_argument("--optimizer",      type=str, choices=["adamw","adamw8","opt_lion","py_lion", "d_lion"], default="adamw8", 
                   help="opt_lion is recommended over py_lion")
    p.add_argument("--num_cycles",     type=float, help="Typically only used with cosine decay")
    p.add_argument("--min_sigma",      type=float, default=1e-5, 
                   help="For FlowMatch. Default=1e-5. If you are using effective batchsize <256, consider a higher value like 2e-4")
    p.add_argument("--copy_config",    type=str, help="Config file to archive with training, if model load succeeds")
    p.add_argument("--output_dir",     required=True)

    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--warmup_steps",   type=str, default="0",
                   help="Measured in effective batchsize steps (b * a). Default=0")
    p.add_argument("--max_steps",      default=10_000, 
                   help="Maximum EFFECTIVE BATCHSIZE steps(b * accum) default=10_000. May use '2e' for whole epochs")
    p.add_argument("--save_steps",     type=int, help="Measured in effective batchsize(b * a)")
    p.add_argument("--save_start",     type=int, default=0, help="Dont start saving or samples until this step")
    p.add_argument("--save_on_epoch",  action="store_true")
    p.add_argument("--force_toklen",   type=int, 
                   help="Force token length to a single value, like 256. Use for T5 cache")

    p.add_argument("--sample_prompt", nargs="+", type=str, help="Prompt to use for a checkpoint sample image")
    p.add_argument("--sample_steps", type=int,
                   help="If you want to run the sampler but not save every time")
    p.add_argument("--sampler_steps", type=int, default=30, help="Steps for the sample process. NOT THE SAME AS sample_steps!! Default=30")
    p.add_argument("--seed",        type=int, default=90)
    p.add_argument("--txtcache_suffix", type=str, default=".txt_t5cache", help="Default=.txt_t5cache")
    p.add_argument("--imgcache_suffix", type=str, default=".img_sdvae", help="Default=.img_sdvae")
    p.add_argument("--force_txtcache", type=str, help="Force txt cache to be a single file. Useful for forcing cached null for unsupervised training")

    p.add_argument("--gradient_accum", type=int, default=1, help="Default=1")
    p.add_argument('--gradient_checkpointing', action='store_true',
                   help="Enable grad checkpointing in unet")
    p.add_argument( "--gradient_clip", type=float, default=1.0,
                        help="Max global grad norm. Set <=0 to disable gradient clipping.")
    p.add_argument( "--gradient_topk", type=float,
                        help="Optional gradient sparsification. " \
                                "Give the percent of largest ones that you want to keep. " \
                                "0.0 < topk < 1.0, but typically 0.3  ..."
                                " WARNING: SIGNIFICANT PERFORMANCE HIT!!")

    p.add_argument("--learning_rate",  type=float, default=1e-5, help="Default=1e-5")
    p.add_argument("--learning_rate_decay", type=float,
                   help="Subtract this every epoch, if schedler==constant")
    p.add_argument("--min_lr_ratio",   type=float, default=0.1, 
                   help="Actually a ratio, not hard number. Only used if 'min_lr' type schedulers are used")
    p.add_argument("--initial_d",  type=float, help="WOO! Hit the Downhill! (With d_lion)")
    p.add_argument("--rex_start_factor", type=float, default=1.0, help="Only used with REX Scheduler during warmup steps. Must be greater than 0. Default=1")
    p.add_argument("--rex_end_factor", action='store_const', const=1.0, default=1.0,
                   help='[read-only] fixed at 1.0; providing a value is an error')
                   #end factor is fixed at 1.0 to avoid odd LR jumps messing things up


    p.add_argument("--weight_decay",   type=float)

    p.add_argument("--vae_scaling_factor", type=float, help="Override vae scaling factor")
    p.add_argument("--text_scaling_factor", type=float, help="Override embedding scaling factor")

    p.add_argument("--noise_gamma",    type=float, default=5.0)
    p.add_argument("--betas",  type=float, nargs=2, metavar=("BETA1","BETA2"),
                   help="Typical LION default is '0.9, 0.99'." \
                   "For instability issues,  use 0.95 0.98")
    p.add_argument("--use_snr", action="store_true",
                   help="Use Min SNR noise adjustments")

    p.add_argument("--targetted_training", action="store_true",
                   help="Only train reset layers")
    p.add_argument("--reinit_crossattn", action="store_true",
                   help="Attempt to reset cross attention weights for text realign")
    p.add_argument("--reinit_crossattnout", action="store_true",
                   help="Attempt to reset just the 'out' cross attention weights")
    p.add_argument("--reinit_attention", action="store_true",
                   help="Attempt to reset ALL attention weights for text realign")
    p.add_argument("--reinit_qk", action="store_true",
                   help="Attempt to reset just qk weights for text realign")
    p.add_argument("--reinit_out", action="store_true",
                   help="Attempt to reset just out blocks")
    p.add_argument("--unfreeze_out", action="store_true",
                   help="Just make the out blocks trainable")
    p.add_argument("--reinit_in", action="store_true",
                   help="Attempt to reset just in blocks")
    p.add_argument("--unfreeze_in", action="store_true",
                   help="Just make the in blocks trainable")
    p.add_argument("--reinit_time", action="store_true",
                   help="Attempt to reset just noise schedule layer")
    p.add_argument("--unfreeze_time", action="store_true",
                   help="Attempt to unfreeze just noise schedule layer")
    p.add_argument("--unfreeze_up_blocks", type=int, nargs="+",
                   help="Just unfreeze, dont reinit. Give 1 or more space-seperated numbers ranged [0-3]. 3 is outer layer, fine detail")
    p.add_argument("--unfreeze_down_blocks", type=int, nargs="+",
                   help="Just unfreeze, dont reinit. Give 1 or more space-seperated numbers ranged [0-3]. 0 is outer layer, fine detail")
    p.add_argument("--unfreeze_mid_block", action="store_true",
                   help="Just unfreeze, dont reinit.")
    p.add_argument("--unfreeze_norms", action="store_true",
                   help="Just unfreeze, dont reinit.")
    p.add_argument("--reinit_unet", action="store_true",
                   help="Train from scratch unet (Do not use, this is broken)")
    p.add_argument("--unfreeze_attention", action="store_true",
                   help="Just unfreeze, dont reinit.")
    p.add_argument("--unfreeze_attn2", action="store_true",
                   help="Just unfreeze, dont reinit.")


    return p.parse_args()

# Give an easy fast short way to invoke -h
if __name__ == "__main__":
    parse_args()
    print("If you got here: you were supposed to invoke this with -h ! ")
