#!/usr/bin/env python3
# Montage of extracted particles, to see what the 2D classification was fed.
#   python3 -I montage.py ExtractRods/particles.star out.jpg [N] [stride]
# Takes every `stride`-th particle, N of them, six per row, 2x upscaled.
# Prints only numbers (box, mean/std of the tiles), no file names.
import os
import sys

import numpy as np
from PIL import Image

star, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 48
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 97
root = os.path.dirname(os.path.dirname(os.path.abspath(star)))

# last loop block of the star (data_particles), column _rlnImageName
labels, rows, inloop = [], [], False
for line in open(star):
    s = line.strip()
    if s == 'loop_':
        inloop, labels, rows = True, [], []
        continue
    if not inloop:
        continue
    if s.startswith('_'):
        labels.append(s.split()[0])
    elif s and not s.startswith('#'):
        rows.append(s.split())
    elif rows:
        inloop = False
ii = labels.index('_rlnImageName')
names = [r[ii] for r in rows][::stride][:N]


def slice_of(name):
    k, path = name.split('@')
    k = int(k) - 1
    with open(os.path.join(root, path), 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
        ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
        assert mode == 2, mode
        f.seek(1024 + ext + k * nx * ny * 4)
        return np.frombuffer(f.read(nx * ny * 4), np.float32).reshape(ny, nx)


tiles = [slice_of(n) for n in names]
nx = tiles[0].shape[1]
per_row = 6
sheet = Image.new('L', (nx * per_row, nx * ((len(tiles) + per_row - 1) // per_row)))
means, stds, maxs = [], [], []
for j, t in enumerate(tiles):
    means.append(float(t.mean())); stds.append(float(t.std())); maxs.append(float(np.abs(t).max()))
    lo, hi = np.percentile(t, [0.5, 99.5])
    g = np.clip((t - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    sheet.paste(Image.fromarray(g), (nx * (j % per_row), nx * (j // per_row)))
sheet = sheet.resize((sheet.width * 2, sheet.height * 2))
sheet.save(out, 'JPEG', quality=88)
print('%d particles of %d, box %d px' % (len(tiles), len(rows), nx))
print('tile mean: median %.2f, range %.2f..%.2f' % (np.median(means), min(means), max(means)))
print('tile std: median %.2f, range %.2f..%.2f' % (np.median(stds), min(stds), max(stds)))
print('tile |max|: median %.1f, 90th pct %.1f, max %.1f' % tuple(np.percentile(maxs, [50, 90, 100])))
