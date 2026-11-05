#!/usr/bin/env python3
# Q5 classification figure. Login node, small maps only. Prints numbers only.
#   python3 -I q5_figure.py CLASS3D_DIR OUT.png
# Panel A: side-view projections of the 8 classes from 3D classification without
#          alignment (job 78), all in one frame (the rod axis of the largest class).
# Panel B: axial density profile of each class along that axis (end occupancy).
# Panel C: focused classification of the middle three beads (job 93) vs a random
#          subset of the same size (numbers from the job 93 log, typed below).
import glob
import os
import sys

import numpy as np
from scipy import ndimage

cdir, out = sys.argv[1], sys.argv[2]


def read_mrc(p):
    with open(p, 'rb') as f:
        h = f.read(1024)
        i = np.frombuffer(h, dtype=np.int32)
        fl = np.frombuffer(h, dtype=np.float32)
        nx, ny, nz = (int(v) for v in i[:3])
        f.seek(1024 + int(i[23]))
        a = np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)
    return a, float(fl[10]) / int(i[7])


def model_classes(p):
    lab, rows, blk = [], [], None
    for line in open(p):
        s = line.strip()
        if s.startswith('data_'):
            blk = s
            lab, rows = ([], []) if s == 'data_model_classes' else (lab, rows)
            continue
        if blk != 'data_model_classes':
            continue
        if s.startswith('_'):
            lab.append(s.split()[0])
        elif s and not s.startswith(('#', 'loop_')):
            rows.append(s.split())
    c = {k: j for j, k in enumerate(lab)}
    return [(float(r[c['_rlnClassDistribution']]), float(r[c['_rlnEstimatedResolution']]))
            for r in rows]


maps = sorted(glob.glob(os.path.join(cdir, 'run_it025_class0*.mrc')))
cls = model_classes(os.path.join(cdir, 'run_it025_model.star'))
assert len(maps) == len(cls) == 8, (len(maps), len(cls))
vols = []
for p in maps:
    v, apix = read_mrc(p)
    vols.append(v)
n = vols[0].shape[0]
order = sorted(range(8), key=lambda k: -cls[k][0])

# Frame: principal axes of the largest class (top 2% voxels, density weighted).
ref = vols[order[0]]
thr = np.percentile(ref, 98)
zz, yy, xx = np.nonzero(ref > thr)
w = ref[zz, yy, xx]
pts = np.stack([zz, yy, xx], 1).astype(float)
cen = (pts * w[:, None]).sum(0) / w.sum()
cov = np.cov((pts - cen).T, aweights=w)
ev, evec = np.linalg.eigh(cov)
R = evec[:, ::-1].T  # rows: long axis, middle, short (map z,y,x components)
print('frame: eigen sd (A) %s' % ' '.join('%.1f' % (np.sqrt(e) * apix) for e in ev[::-1]))


def to_frame(v):
    # out[a, b, c] samples v at cen + a*R0 + b*R1 + c*R2 (offsets from n/2)
    g = np.mgrid[0:n, 0:n, 0:n].reshape(3, -1).astype(float) - n / 2
    src = (R.T @ g) + cen[:, None]
    return ndimage.map_coordinates(v, src, order=1, cval=0.0).reshape(n, n, n)


fr = [to_frame(v) for v in vols]
# Side view: project along the short axis (axis 2); image rows = long axis.
proj = [f.sum(axis=2) for f in fr]
# Axial profile: positive density within 30 A of the axis, per 1 A slab along long axis.
b, c = np.mgrid[0:n, 0:n] - n / 2
cyl = (np.hypot(b, c) * apix) < 30
prof = [np.clip(f, 0, None)[:, cyl].sum(axis=1) for f in fr]
axis_A = (np.arange(n) - n / 2) * apix
# Each class scaled to its own middle (|x| < 40 A), so end occupancy compares fairly.
mid = np.abs(axis_A) < 40
prof = [p / p[mid].mean() for p in prof]


def length(p, frac=0.15):
    idx = np.nonzero(p > frac * p.max())[0]
    return (idx[-1] - idx[0]) * apix if len(idx) else float('nan')


def seg(p, a, b):
    return p[(axis_A >= a) & (axis_A <= b)].mean()


for k in order:
    print('class %d end/middle: %.2f at -90..-65, %.2f at 60..80, %.2f beyond +-100' % (
        k + 1, seg(prof[k], -90, -65), seg(prof[k], 60, 80),
        (seg(prof[k], -130, -100) + seg(prof[k], 100, 130)) / 2))
for k in order:
    print('class %d: %.1f%%, %.2f A, profile length %.0f A, proj min/max %.3g/%.3g' % (
        k + 1, 100 * cls[k][0], cls[k][1], length(prof[k]), proj[k].min(), proj[k].max()))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 9.2), dpi=150)
gs = fig.add_gridspec(3, 8, height_ratios=[1.45, 1.0, 0.9], hspace=0.55, wspace=0.08)
crop = int(round(130 / apix))  # +-130 A along the axis
lo, hi = n // 2 - crop, n // 2 + crop
wcrop = int(round(65 / apix))
for j, k in enumerate(order):
    ax = fig.add_subplot(gs[0, j])
    im = proj[k][lo:hi, n // 2 - wcrop:n // 2 + wcrop]
    vmax = np.percentile(np.concatenate([p.ravel() for p in proj]), 99.9)
    ax.imshow(im, cmap='gray', vmin=0, vmax=vmax, origin='lower', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('class %d\n%.1f%%, %.1f \u00c5' % (k + 1, 100 * cls[k][0], cls[k][1]),
                 fontsize=8)
    if j == 0:
        ax.text(-0.35, 1.25, 'A', transform=ax.transAxes, fontsize=13, weight='bold')
ax = fig.add_subplot(gs[1, :])
cm = plt.get_cmap('tab10')
for j, k in enumerate(order):
    ax.plot(axis_A, prof[k], color=cm(j), lw=1.3,
            label='class %d (%.1f%%)' % (k + 1, 100 * cls[k][0]))
ax.set_xlim(-130, 130)
ax.set_xlabel('position along the rod axis (\u00c5)')
ax.set_ylabel('density / middle')
ax.legend(fontsize=7, ncol=4, frameon=False)
ax.text(-0.06, 1.08, 'B', transform=ax.transAxes, fontsize=13, weight='bold')

ax = fig.add_subplot(gs[2, :4])
lab = ['class 1', 'class 2', 'class 3', 'class 4', 'random\n232,934']
n_p = [232934, 234414, 241818, 280172, 232934]
res = [4.617, 4.617, 4.617, 4.813, 4.617]
cols = ['#1f4e8c'] * 4 + ['#999999']
ax.bar(range(5), res, color=cols)
for x, (r, m) in enumerate(zip(res, n_p)):
    ax.text(x, r + 0.02, '%.2f \u00c5\n%dk' % (r, m // 1000), ha='center', fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7)
ax.set_ylim(4.3, 4.95)
ax.set_ylabel('FSC 0.143 (\u00c5)')
ax.set_title('focused 3D classification, middle three beads (K=4)', fontsize=8)
ax.text(-0.14, 1.08, 'C', transform=ax.transAxes, fontsize=13, weight='bold')
fig.savefig(out, bbox_inches='tight')
print('wrote figure')
