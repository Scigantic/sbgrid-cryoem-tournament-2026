#!/usr/bin/env python3
# Local resolution along the rod and out from its axis. The axis and centre come from the map
# (as in measure_classes.py); the local-resolution map is summarised inside the mask in 10 A
# slabs along the axis and in 10 A shells out from it. Resolution that worsens steadily toward
# the ends says the rod bends (multi-body refinement can recover that); a flat profile says
# it does not. Numbers only.
#   python3 -I locres_profile.py map.mrc locres.mrc mask.mrc APIX
import sys

import numpy as np

NSIG = 2.0


def read(p):
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        nx, ny, nz = (int(v) for v in h[:3])
        f.seek(1024 + int(h[23]))
        return np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)


m, lr, mk, apix = read(sys.argv[1]), read(sys.argv[2]), read(sys.argv[3]), float(sys.argv[4])
thr = m.mean() + NSIG * m.std()
z, y, x = np.nonzero(m > thr)
w = m[z, y, x] - thr
xyz = np.stack([x, y, z], axis=1).astype(np.float64) * apix
c = (xyz * w[:, None]).sum(0) / w.sum()
d = xyz - c
axis = np.linalg.eigh((d * w[:, None]).T @ d / w.sum())[1][:, -1]

z, y, x = np.nonzero(mk > 0.5)
p = np.stack([x, y, z], axis=1).astype(np.float64) * apix - c
t = p @ axis
r = np.linalg.norm(p - np.outer(t, axis), axis=1)
res = lr[z, y, x]
print('along the axis (slab centre A: median local res, voxels):')
for s in range(int(np.floor(t.min() / 10)) * 10, int(np.ceil(t.max())) + 1, 10):
    sel = (t >= s) & (t < s + 10)
    if sel.sum() > 200:
        print('  %+5d  %.2f A  %d' % (s + 5, np.median(res[sel]), sel.sum()))
print('out from the axis (shell A: median local res, voxels):')
for s in range(0, int(r.max()) + 1, 10):
    sel = (r >= s) & (r < s + 10)
    if sel.sum() > 200:
        print('  %3d-%-3d  %.2f A  %d' % (s, s + 10, np.median(res[sel]), sel.sum()))
