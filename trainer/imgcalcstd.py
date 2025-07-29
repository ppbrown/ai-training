#!/bin/env python

# Give this a list of images files on stdin
#
# It will printout the "mean" and std deviation for each of them.
# So give samples using the same seed and prompt, to see if things
# are improving?

summaries = "" \
"| ----------------------- | ----------- | --------- | ----------------------- |\n" \
"| Situation               | Mean        | Stddev    | Interpretation          |\n" \
"| ----------------------- | ----------- | --------- | ----------------------- |\n" \
"| Healthy, diverse images | 0.4-0.6     | 0.15-0.35 | Normal                  |\n" \
"| White-out   collapse    | ~ 1.0       | < 0.05    | Model outputs all white |\n" \
"|  Black-out  collapse    | ~ 0.0       | < 0.05    | Model outputs all black |\n" \
"| Collapse to gray        |  ~0.5       | < 0.05    | All outputs nearly gray |\n" \
"| Oversaturated/noisy     |     extreme | > 0.4+    | Too much color/noise    |\n"



import sys
from PIL import Image
import numpy as np

#print a column header
print(f"{'filename':40s} {'mean':8s} {'stddev':8s}")

for line in sys.stdin:
    fname = line.strip()
    if not fname:
        continue
    try:
        img = Image.open(fname).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        mean = arr.mean()
        std = arr.std()
        print(f"{fname:40s} {mean:8.4f} {std:8.4f}")
    except Exception as e:
        print(f"{fname:40s} ERROR: {e}")

print(summaries)
