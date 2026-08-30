#!/usr/bin/env python

# show_logs.py
#
# Compact terminal summary of a tensorboard logfile written by
# train_from_cached.py. One line per checkpoint, so you can pick a
# checkpoint to keep without starting the tensorboard web UI.
#
# The trainer logs unet/L2_norm exactly once per checkpoint save
# (see train_checkpointandsave.py), at the same step number used for the
# "checkpoint-NNNNN" directory name. So those points define the rows here.
# Everything else is summarized over the window of steps leading up to
# each checkpoint.
#
# Usage:
#   show_logs.py tensorboard/myrun
#   show_logs.py tensorboard/myrun/events.out.tfevents.1234
#   show_logs.py tensorboard/myrun --every 500      # ignore ckpt marks
#   show_logs.py tensorboard/myrun --csv > runs.csv

import argparse
import bisect
import os
import sys

CKPT_TAG = "unet/L2_norm"
LOSS_TAG = "train/loss_raw"
SNR_TAG = "train/loss_snr"
GNORM_TAG = "train/grad_norm"
QK_TAG = "train/qk_grads_av"
LR_TAG = "train/learning_rate"


class Series:
    """Step-sorted scalar series with window lookups."""

    def __init__(self, points):
        points = sorted(points, key=lambda p: p[0])
        self.steps = [p[0] for p in points]
        self.vals = [p[1] for p in points]

    def __len__(self):
        return len(self.steps)

    def window(self, lo, hi):
        """Values for steps in the range (lo, hi]."""
        i = bisect.bisect_right(self.steps, lo)
        j = bisect.bisect_right(self.steps, hi)
        return self.vals[i:j]

    def at(self, step):
        """Most recent value at or before step."""
        j = bisect.bisect_right(self.steps, step)
        return self.vals[j - 1] if j > 0 else None


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def stddev(vals):
    if len(vals) < 2:
        return None
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def pct_change(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def resolve_logfile(path):
    """A dir means 'use the newest event file in it'."""
    if not os.path.isdir(path):
        return path

    names = [n for n in os.listdir(path)
             if os.path.isfile(os.path.join(path, n))]
    events = [n for n in names if "tfevents" in n]
    if not events:
        events = names
    if not events:
        print("ERROR: no event files in dir", path)
        sys.exit(1)

    events.sort(key=lambda n: os.path.getmtime(os.path.join(path, n)))
    return os.path.join(path, events[-1])


def load_scalars(path):
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        print("ERROR: need the 'tensorboard' module to read event files.")
        print("       pip install tensorboard")
        sys.exit(1)

    # size_guidance 0 means "keep every point", no downsampling
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()

    series = {}
    for tag in acc.Tags()["scalars"]:
        series[tag] = Series([(e.step, e.value) for e in acc.Scalars(tag)])
    return series


def checkpoint_steps(series, every):
    """Step numbers to summarize at."""
    if every:
        last = 0
        for s in series.values():
            if len(s):
                last = max(last, s.steps[-1])
        return list(range(every, last + 1, every))

    ckpt = series.get(CKPT_TAG)
    if ckpt is None or not len(ckpt):
        return []
    # A restarted run can log the same step twice; keep unique and ordered
    seen, steps = set(), []
    for s in ckpt.steps:
        if s not in seen:
            seen.add(s)
            steps.append(s)
    return steps


def build_rows(series, steps, tail_frac):
    loss = series.get(LOSS_TAG)
    snr = series.get(SNR_TAG)
    gnorm = series.get(GNORM_TAG)
    qk = series.get(QK_TAG)
    lr = series.get(LR_TAG)
    l2 = series.get(CKPT_TAG)

    rows = []
    prev_step = -1
    prev_tail = None
    prev_l2 = None

    for step in steps:
        row = {"ckpt": f"checkpoint-{step:05}", "step": step}

        lvals = loss.window(prev_step, step) if loss else []
        row["n"] = len(lvals)
        row["loss"] = mean(lvals)
        row["min"] = min(lvals) if lvals else None
        row["std"] = stddev(lvals)

        # Tail mean: the end of the window is what the saved weights
        # actually reflect, the early part is stale.
        ntail = max(1, int(len(lvals) * tail_frac))
        row["tail"] = mean(lvals[-ntail:]) if lvals else None
        row["d%"] = pct_change(row["tail"], prev_tail)
        prev_tail = row["tail"] if row["tail"] is not None else prev_tail

        row["snr"] = mean(snr.window(prev_step, step)) if snr else None
        row["gnorm"] = mean(gnorm.window(prev_step, step)) if gnorm else None
        row["qk"] = mean(qk.window(prev_step, step)) if qk else None
        row["lr"] = lr.at(step) if lr else None

        row["L2"] = l2.at(step) if l2 else None
        row["dL2%"] = pct_change(row["L2"], prev_l2)
        prev_l2 = row["L2"] if row["L2"] is not None else prev_l2

        rows.append(row)
        prev_step = step

    return rows


# (key, header, format). Columns with no data at all are dropped.
COLUMNS = [
    ("step", "step", "{}"),
    ("loss", "loss", "{:.4f}"),
    ("tail", "tail", "{:.4f}"),
    ("d%", "d%", "{:+.1f}"),
    ("std", "std", "{:.4f}"),
    ("snr", "snrloss", "{:.4f}"),
    ("gnorm", "gnorm", "{:.3g}"),
    ("qk", "qkgrad", "{:.2e}"),
    ("lr", "lr", "{:.2e}"),
    ("L2", "L2", "{:.3f}"),
]


def render(val, fmt):
    return "-" if val is None else fmt.format(val)


def print_table(rows, best_step):
    cols = [c for c in COLUMNS
            if any(r.get(c[0]) is not None for r in rows)]

    header = ["", *[c[1] for c in cols]]
    body = []
    for r in rows:
        mark = "*" if r["step"] == best_step else " "
        body.append([mark, *[render(r.get(k), f) for k, _, f in cols]])

    widths = [max(len(row[i]) for row in [header] + body)
              for i in range(len(header))]

    def line(cells):
        return "  ".join(c.rjust(w) for c, w in zip(cells, widths))

    print(line(header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for cells in body:
        print(line(cells))


def print_csv(rows):
    cols = [c for c in COLUMNS
            if any(r.get(c[0]) is not None for r in rows)]
    print(",".join(c[1] for c in cols))
    for r in rows:
        out = []
        for key, _, _ in cols:
            v = r.get(key)
            out.append("" if v is None else str(v))
        print(",".join(out))


def main():
    ap = argparse.ArgumentParser(
        description="Compact per-checkpoint summary of a tensorboard log.")
    ap.add_argument("logfile",
                    help="tensorboard event file, or a dir "
                         "(newest event file in it gets used)")
    ap.add_argument("--every", type=int, default=None,
                    help="summarize every N steps instead of "
                         "using checkpoint marks")
    ap.add_argument("--tail-frac", type=float, default=0.25,
                    help="fraction at the end of each window used for the "
                         "'tail' loss (default 0.25)")
    ap.add_argument("--csv", action="store_true",
                    help="emit csv instead of an aligned table")
    ap.add_argument("--tags", action="store_true",
                    help="just list the scalar tags in the file, and exit")
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        print("ERROR: no such file or dir:", args.logfile)
        sys.exit(1)

    logfile = resolve_logfile(args.logfile)

    series = load_scalars(logfile)
    if not series:
        print("ERROR: no scalars found in", logfile)
        sys.exit(1)

    if args.tags:
        for tag in sorted(series):
            s = series[tag]
            print(f"{tag:24} {len(s):7} points  "
                  f"steps {s.steps[0]}..{s.steps[-1]}")
        return

    steps = checkpoint_steps(series, args.every)
    if not steps:
        print(f"No '{CKPT_TAG}' points found, so no checkpoint marks.")
        print("Use --every N to summarize on a fixed step interval instead.")
        sys.exit(1)

    rows = build_rows(series, steps, args.tail_frac)

    # Best = lowest tail loss. That is the raw MSE nearest the save point,
    # which is the number comparable across runs (loss_snr is reweighted).
    scored = [r for r in rows if r["tail"] is not None]
    best = min(scored, key=lambda r: r["tail"]) if scored else None
    best_step = best["step"] if best else None

    if args.csv:
        print_csv(rows)
        return

    loss = series.get(LOSS_TAG)
    print(f"{logfile}: {len(steps)} checkpoints, "
          f"steps {steps[0]}..{steps[-1]}"
          + (f", {len(loss)} logged train steps" if loss else ""))
    print()
    print_table(rows, best_step)
    print()

    if best:
        print(f"best (lowest tail loss): {best['ckpt']}  "
              f"tail={best['tail']:.4f}")
        last = rows[-1]
        if last["tail"] is not None and last["step"] != best_step:
            print(f"  vs last: {last['ckpt']}  tail={last['tail']:.4f}  "
                  f"({pct_change(last['tail'], best['tail']):+.1f}%)")

    l2vals = [r["L2"] for r in rows if r["L2"] is not None]
    if len(l2vals) > 1:
        drift = pct_change(l2vals[-1], l2vals[0])
        note = "  (large growth suggests you want weight decay)" \
            if drift is not None and drift > 5.0 else ""
        print(f"unet L2 norm drift over run: {drift:+.2f}%{note}")

    print()
    print("cols: loss/min/std over steps since previous checkpoint")
    print(f"   tail=mean of last{args.tail_frac:.0%} of that window")
    print("   d%=tail vs previous checkpoint tail")
    print("   std=loss volatility since last checkpoint (not how low it is, how noisy it is)")
    print("   gnorm/qkgrad=mean grad norms")
    print("   L2=unet weight size at checkpoint - growth signals overfitting/decay needed")


if __name__ == "__main__":
    main()
