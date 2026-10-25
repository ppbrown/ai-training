#!/bin/env python

# alternative to show_logs.py that focuses on the GAN related stats
# not particularly human readable

import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1]); ea.Reload()
for tag in ea.Tags()["scalars"]:
    if any(k in tag.lower() for k in ("disc", "vgg", "adv", "g_loss", "d_loss")):
        v = ea.Scalars(tag)
        pts = [v[i] for i in range(0, len(v), max(1, len(v)//8))] + [v[-1]]
        print(tag, " ".join(f"{p.step}:{p.value:.4g}" for p in pts))
