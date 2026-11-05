#!/usr/bin/env python3
# Is the rod pseudo-symmetric end to end? Rotates the map 180 deg about axes perpendicular to
# the rod (every 10 deg around it, through the density centroid), slides along the rod axis,
# and reports the best correlation with the unrotated map inside its density. Baselines: the
# same map shifted by one bead along the axis, and 180 deg about the rod axis itself. A flip
# correlation near the half-map level would mean flipped particles can align almost as well
# as correct ones. Bins by 2 first to keep it light. Numbers only.
#   python3 -I flip_test.py map.mrc [halfmap_for_reference.mrc]
import sys

import numpy as np
from scipy import ndimage


def read(p):
    with open(p, 'rb') as f:
        h = f.read(1024)
        i = np.frombuffer(h, dtype=np.int32)
        fl = np.frombuffer(h, dtype=np.float32)
        nx, ny, nz = (int(v) for v in i[:3])
        apix = float(fl[10]) / int(i[7])
        f.seek(1024 + int(i[23]))
        a = np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)
    return a.astype(np.float64), apix


def bin2(a):
    n = a.shape[0] // 2 * 2
    a = a[:n, :n, :n]
    return a.reshape(n // 2, 2, n // 2, 2, n // 2, 2).mean(axis=(1, 3, 5))


m, apix = read(sys.argv[1])
if m.shape[0] > 200:
    m, apix = bin2(m), apix * 2
m = ndimage.gaussian_filter(m, 4.0 / apix)             # ~8 A low-pass, reduces noise
n = m.shape[0]
sel = m > np.quantile(m, 0.985)
mask = ndimage.binary_dilation(sel, iterations=2)
idx = np.array(np.nonzero(sel), dtype=np.float64).T     # z, y, x
w = m[sel]
c = (idx * w[:, None]).sum(0) / w.sum()
cov = np.cov((idx - c).T, aweights=w)
ev, evec = np.linalg.eigh(cov)
a = evec[:, -1]
u = np.cross(a, [1.0, 0, 0])
if np.linalg.norm(u) < 0.1:
    u = np.cross(a, [0, 1.0, 0])
u /= np.linalg.norm(u)
v = np.cross(a, u)
ref = m[mask] - m[mask].mean()


def corr(R, shift):
    # output voxel x samples input at R (x - c) + c + shift
    off = c - R @ c + shift
    out = ndimage.affine_transform(m, R, offset=off, order=1, mode='constant')
    t = out[mask] - out[mask].mean()
    return float(t @ ref / np.linalg.norm(t) / np.linalg.norm(ref))


I = np.eye(3)
print('pixel %.3f A, box %d, rod length (4 sd) %.0f A' % (apix, n, 4 * np.sqrt(ev[-1]) * apix))
steps = np.arange(-45, 46, 3) / apix
best_shift = max((corr(I, s * a), s * apix) for s in np.arange(30, 45, 1) / apix)
print('baseline, same map shifted one bead along the axis: corr %.3f at %.0f A' % best_shift)
Rax = 2 * np.outer(a, a) - I
b = max((corr(Rax, s * a), s * apix) for s in steps)
print('180 deg about the rod axis: corr %.3f (axial shift %.0f A)' % b)
rows = []
for phi in range(0, 180, 10):
    p = np.cos(np.radians(phi)) * u + np.sin(np.radians(phi)) * v
    R = 2 * np.outer(p, p) - I
    rows.append(max((corr(R, s * a), phi, s * apix) for s in steps))
rows.sort(reverse=True)
print('end-to-end flip (180 deg about a perpendicular axis), best 5 of 18 axes:')
for r in rows[:5]:
    print('  corr %.3f  axis at %3d deg  axial shift %+.0f A' % r)
if len(sys.argv) > 2:
    h, ah = read(sys.argv[2])
    if h.shape[0] > 200:
        h = bin2(h)
    h = ndimage.gaussian_filter(h, 4.0 / apix)
    t = h[mask] - h[mask].mean()
    print('reference, half map vs full map at the same filter: corr %.3f' % (
        t @ ref / np.linalg.norm(t) / np.linalg.norm(ref)))
