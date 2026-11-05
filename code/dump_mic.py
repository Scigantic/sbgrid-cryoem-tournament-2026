#!/usr/bin/env python3
# Save a few corrected micrographs as binned float32 numpy arrays with neutral
# names, so a rod detector can be developed off the cluster.
#   python3 -I ~/pipeline/dump_mic.py 4 0 1      (bin 4, micrographs 0 and 1)
import os
import sys

import numpy as np

b = int(sys.argv[1])
idx = [int(a) for a in sys.argv[2:]]
root = os.path.expanduser('~/results/chunk')
star = os.path.join(root, 'CtfFind/micrographs_ctf.star')
mics = [l.split()[0] for l in open(star) if 'MotionCorr/Movies/' in l]
out = os.path.expanduser('~/results/sweep/look')
os.makedirs(out, exist_ok=True)
for i in idx:
    p = os.path.join(root, mics[i])
    with open(p, 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
        ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
        f.seek(1024 + ext)
        dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
        img = np.fromfile(f, dt, nx * ny).reshape(ny, nx).astype(np.float32)
    h, w = ny - ny % b, nx - nx % b
    m = img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    name = 'mic%d_bin%d.npy' % (i, b)
    np.save(os.path.join(out, name), m.astype(np.float32))
    print('%s: %dx%d, mean %.2f, std %.2f' % (name, m.shape[1], m.shape[0],
                                              m.mean(), m.std()))
print('wrote to', out)
