#!/usr/bin/env python3
# A 2D class average next to its power spectrum, for a few classes. A helical
# rod shows its repeat as a line of spots (layer line) at right angles to the
# rod axis. Login node, a few seconds, no GPU.
#   python3 -I ~/pipeline/ps.py \
#     ~/results/chunk/Class2D/run_it012_classes.mrcs 39 23 17 5
import os
import sys

import numpy as np
from PIL import Image

path = sys.argv[1]
ks = [int(a) for a in sys.argv[2:]]
out = os.path.expanduser('~/results/chunk/preview')
os.makedirs(out, exist_ok=True)
PAD = 256

with open(path, 'rb') as f:
    hdr = f.read(1024)
    nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
    cella = float(np.frombuffer(hdr[40:44], np.float32)[0])
    ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
    f.seek(1024 + ext)
    dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
    data = np.fromfile(f, dt, nx * ny * nz).reshape(nz, ny, nx)
apix = cella / nx if cella > 0 else 0.0
print('box %d px, %d classes, %.4f A/px' % (nx, nz, apix))
if apix > 0:
    print('a repeat of d A sits %.0f/d px from the centre of the spectrum'
          % (PAD * apix))


def to8(a):
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def up(a, s):
    im = Image.fromarray(a)
    return im.resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST)


for k in ks:
    img = data[k - 1].astype(np.float32)
    pad = np.zeros((PAD, PAD), np.float32)
    o = (PAD - nx) // 2
    pad[o:o + ny, o:o + nx] = img - img.mean()
    ps = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(pad))))
    left = up(to8(img), PAD // nx)
    right = up(to8(ps), 1)
    cv = Image.new('L', (left.width + right.width, PAD))
    cv.paste(left, (0, 0))
    cv.paste(right, (left.width, 0))
    cv.save(os.path.join(out, 'ps_%02d.jpg' % k), quality=90)
    print('class %d written' % k)
print('wrote to', out)
