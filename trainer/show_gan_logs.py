#!/usr/bin/env python3

# alternative to show_logs.py that focuses on the GAN related stats
# prints one aligned step/value table per matching tag

import argparse

KEYS = ("disc", "vgg", "adv", "g_loss", "d_loss")


def main():
    ap = argparse.ArgumentParser(
        description="Show GAN related scalars (disc, vgg, adv, g_loss, d_loss) from a tensorboard log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("logpath", help="(required) tensorboard log dir or event file")
    ap.add_argument("--points", type=int, default=8, help="approx number of sampled points per tag")
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(args.logpath)
    ea.Reload()
    for tag in sorted(ea.Tags()["scalars"]):
        if not any(k in tag.lower() for k in KEYS):
            continue
        v = ea.Scalars(tag)
        if not v:
            continue
        stride = max(1, len(v) // args.points)
        pts = v[::stride]
        if pts[-1] is not v[-1]:
            pts.append(v[-1])
        steps = [str(p.step) for p in pts]
        vals = [f"{p.value:.4g}" for p in pts]
        w = max(len(s) for s in steps + vals)
        print(f"{tag}  ({len(v)} points)")
        print("  step  " + " ".join(s.rjust(w) for s in steps))
        print("  value " + " ".join(s.rjust(w) for s in vals))
        print()


if __name__ == "__main__":
    main()
