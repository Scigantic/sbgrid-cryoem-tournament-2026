#!/usr/bin/env python3
# Sum the frames of a few raw movies and save small JPEGs, so the specimen
# can be looked at without a GPU node (no motion correction; a frame sum is
# enough to see what is on the grid). About a minute per movie, login node.
#   python3 -I ~/pipeline/preview.py ~/results/chunk/pick.txt 2
import os
import sys

import numpy as np
import tifffile
from PIL import Image

lst = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.expanduser('~/results/chunk/preview')
os.makedirs(out, exist_ok=True)
paths = [l.strip() for l in open(lst) if l.strip()][:n]

gain = None
gp = os.path.expanduser('~/results/chunk/gain_src.tiff')
if os.path.exists(gp):
    gain = tifffile.imread(gp).astype(np.float32)


def save(img, name, b):
    h, w = img.shape
    h -= h % b
    w -= w % b
    m = img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    lo, hi = np.percentile(m, [0.5, 99.5])
    g = np.clip((m - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(g).save(os.path.join(out, name), quality=85)


for i, p in enumerate(paths):
    s = None
    nf = 0
    with tifffile.TiffFile(p) as t:
        for pg in t.pages:
            a = pg.asarray().astype(np.float32)
            s = a if s is None else s + a
            nf += 1
    note = 'no gain'
    if gain is not None and gain.shape == s.shape:
        s = s * gain
        note = 'gain applied'
    save(s, 'mic%d_bin8.jpg' % i, 8)
    cy, cx = s.shape[0] // 2, s.shape[1] // 2
    c = s[cy - 1024:cy + 1024, cx - 1024:cx + 1024]
    save(c, 'mic%d_center_bin2.jpg' % i, 2)
    msg = 'movie %d: %d frames, %.1f counts/px/frame, %s, 2 jpgs'
    print(msg % (i, nf, s.mean() / nf, note))
print('wrote to', out)
