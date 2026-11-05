#!/usr/bin/env python3
# Draw picked coordinates on a binned micrograph so the picks can be judged by eye.
#   python3 -I overlay.py micrograph.mrc coords.star out.jpg [bin]
# coords.star: any RELION coordinate file with _rlnCoordinateX/_rlnCoordinateY.
import sys

import numpy as np
from PIL import Image, ImageDraw

mrc, star, out = sys.argv[1:4]
b = int(sys.argv[4]) if len(sys.argv) > 4 else 8

with open(mrc, 'rb') as f:
    hdr = f.read(1024)
    nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
    ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
    f.seek(1024 + ext)
    dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
    img = np.fromfile(f, dt, nx * ny).reshape(ny, nx).astype(np.float32)

h, w = ny - ny % b, nx - nx % b
m = img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
lo, hi = np.percentile(m, [0.5, 99.5])
g = np.clip((m - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
im = Image.fromarray(g).convert('RGB')
d = ImageDraw.Draw(im)

labels, n = [], 0
inloop = False
for line in open(star):
    s = line.strip()
    if s == 'loop_':
        inloop, labels = True, []
        continue
    if not inloop:
        continue
    if s.startswith('_'):
        labels.append(s.split()[0])
    elif s:
        v = s.split()
        x = float(v[labels.index('_rlnCoordinateX')]) / b
        y = float(v[labels.index('_rlnCoordinateY')]) / b
        d.rectangle([x - 2, y - 2, x + 2, y + 2], outline=(255, 40, 40))
        n += 1
im.save(out, quality=88)
print('%d coordinates drawn on %dx%d (bin %d)' % (n, im.width, im.height, b))
