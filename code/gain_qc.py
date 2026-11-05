#!/usr/bin/env python3
# Which gain flip does RELION's motion correction need? A wrongly oriented gain leaves
# per-pixel noise that a correct one removes, so of the same movies corrected with each
# candidate flip, the right one has the least power near Nyquist (relative to the mean
# squared). Frame alignment smears the fixed pattern by the drift, which weakens but does
# not remove the excess, so the margin is printed too.
#   python3 -I gain_qc.py DIR0 DIR1 DIR2 ...   (each holds the same micrographs, *.mrc)
# Prints one line per directory and, last, the index of the best one.
import glob
import os
import struct
import sys

import numpy as np


def read_mrc(path):
    with open(path, 'rb') as f:
        h = f.read(1024)
        nx, ny, nz, mode = struct.unpack('<4i', h[:16])
        ext = struct.unpack('<i', h[92:96])[0]
        f.seek(1024 + ext)
        dt = {2: '<f4', 12: '<f2', 1: '<i2', 6: '<u2', 0: 'i1'}[mode]
        a = np.frombuffer(f.read(nx * ny * np.dtype(dt).itemsize), dtype=dt)
    return a.reshape(ny, nx).astype(np.float64)


def hf_ratio(a):
    a = a / a.mean() - 1.0
    p = np.abs(np.fft.rfft2(a)) ** 2
    ny, nx = a.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    # the gain error is white, so compare absolute power (relative to the mean) where the
    # specimen contributes least; a ratio to lower frequencies would cancel it
    return p[(r > 0.40) & (r < 0.5)].mean() / a.size


dirs = sys.argv[1:]
scores = []
for d in dirs:
    mics = sorted(m for m in glob.glob(os.path.join(d, '**', '*.mrc'), recursive=True)
                  if not m.endswith(('_PS.mrc', '_noDW.mrc')))
    if not mics:
        print('%s: no micrographs' % d)
        scores.append(np.inf)
        continue
    v = [hf_ratio(read_mrc(m)) for m in mics]
    scores.append(float(np.median(v)))
    print('%s: %d micrographs, relative power near Nyquist %.5f' % (d, len(v), scores[-1]))
best = int(np.argmin(scores))
srt = sorted(scores)
margin = (srt[1] - srt[0]) / srt[0] if len(srt) > 1 and np.isfinite(srt[1]) else 0.0
print('best %d (margin %.1f%%)' % (best, 100 * margin))
