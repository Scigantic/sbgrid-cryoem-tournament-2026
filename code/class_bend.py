#!/usr/bin/env python3
# Bending and similarity of 3D class maps that share one reference frame (classification
# without alignment). For each class: density centroids in 10 A slabs along the rod axis
# (axis from the mean map), lateral offset of each end from the line through the middle,
# and the bend angle between the two halves. Then pairwise correlation inside the mean
# map's density. Numbers only.
#   python3 -I class_bend.py map1.mrc map2.mrc ...
import sys

import numpy as np


def read(p):
    with open(p, 'rb') as f:
        h = f.read(1024)
        i = np.frombuffer(h, dtype=np.int32)
        fl = np.frombuffer(h, dtype=np.float32)
        nx, ny, nz = (int(v) for v in i[:3])
        apix = float(fl[10]) / int(i[7])
        f.seek(1024 + int(i[23]))
        a = np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)
    return a, apix


maps, apix = [], None
for p in sys.argv[1:]:
    a, apix = read(p)
    maps.append(a.astype(np.float64))
mean = np.mean(maps, axis=0)
n = mean.shape[0]
thr = np.quantile(mean, 0.985)
sel = mean > thr
zz, yy, xx = np.nonzero(sel)
pts = np.stack([xx, yy, zz], 1).astype(np.float64)
w = mean[sel]
c0 = (pts * w[:, None]).sum(0) / w.sum()
cov = np.cov((pts - c0).T, aweights=w)
ev, evec = np.linalg.eigh(cov)
axis = evec[:, -1]
print('pixel %.4f A, box %d, mean-map axis length (2 sd) %.0f A' % (
    apix, n, 4 * np.sqrt(ev[-1]) * apix))

grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij'), -1)
xyz = grid[..., ::-1].reshape(-1, 3).astype(np.float64)    # x, y, z per voxel (z,y,x order)
t = (xyz - c0) @ axis * apix                               # A along the axis
u = np.cross(axis, [0, 0, 1.0])
if np.linalg.norm(u) < 0.1:
    u = np.cross(axis, [0, 1.0, 0])
u /= np.linalg.norm(u)
v = np.cross(axis, u)
pu = (xyz - c0) @ u * apix
pv = (xyz - c0) @ v * apix
radial = np.sqrt(pu ** 2 + pv ** 2)
inrod = radial < 45
edges = np.arange(-100, 101, 10)


def profile(a):
    d = np.clip(a.reshape(-1), 0, None)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = inrod & (t >= lo) & (t < hi)
        s = d[m].sum()
        out.append((s, (d[m] * pu[m]).sum() / s if s > 0 else np.nan,
                    (d[m] * pv[m]).sum() / s if s > 0 else np.nan))
    return np.array(out)


mid = (edges[:-1] + edges[1:]) / 2
print('class  mass_by_slab(-95..95 A, % of max)')
rows = []
for k, a in enumerate(maps, 1):
    pr = profile(a)
    mass = pr[:, 0] / pr[:, 0].max()
    rows.append(pr)
    print('%5d  %s' % (k, ' '.join('%3d' % round(100 * m) for m in mass)))
print('class  bend_deg  end_offset_A(-75)  end_offset_A(+75)  direction_deg')
for k, pr in enumerate(rows, 1):
    ok = (pr[:, 0] / pr[:, 0].max() > 0.3) & np.isfinite(pr[:, 1])
    tm, cu, cv = mid[ok], pr[ok, 1], pr[ok, 2]
    lo, hi, ce = tm < -25, tm > 25, np.abs(tm) <= 25
    if lo.sum() < 2 or hi.sum() < 2 or ce.sum() < 1:
        print('%5d  too short' % k)
        continue
    pc = np.array([cu[ce].mean(), cv[ce].mean()])
    def fit(m):
        A = np.vstack([tm[m], np.ones(m.sum())]).T
        su = np.linalg.lstsq(A, cu[m], rcond=None)[0][0]
        sv = np.linalg.lstsq(A, cv[m], rcond=None)[0][0]
        return np.array([1.0, su, sv])
    d1, d2 = fit(lo | ce), fit(hi | ce)
    ang = np.degrees(np.arccos(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2)))
    e1 = np.array([cu[lo].mean(), cv[lo].mean()]) - pc
    e2 = np.array([cu[hi].mean(), cv[hi].mean()]) - pc
    s = e1 + e2
    print('%5d  %8.1f  %17.1f  %17.1f  %13.0f' % (
        k, ang, np.linalg.norm(e1), np.linalg.norm(e2), np.degrees(np.arctan2(s[1], s[0]))))
m = sel.reshape(-1)
vecs = [(a.reshape(-1)[m] - a.reshape(-1)[m].mean()) for a in maps]
print('pairwise correlation inside the mean map density:')
for i, a in enumerate(vecs, 1):
    print('%5d  %s' % (i, ' '.join('%.3f' % (a @ b / np.linalg.norm(a) / np.linalg.norm(b))
                                   for b in vecs)))
