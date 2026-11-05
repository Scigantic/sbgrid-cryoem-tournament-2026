#!/usr/bin/env python3
# Length and bead count of each 3D class map of a rod. The rod axis is the principal axis of
# the density above a threshold (mean + NSIG sd of the box); the density is projected onto
# that axis in 1 A bins, the extent is where the smoothed profile exceeds 10% of its peak,
# and beads are the profile's local maxima above 25% of the peak at least MINSEP A apart.
# Prints one line per class with its share of the particles. Numbers only.
#   python3 -I measure_classes.py run_it025_model.star APIX class001.mrc [class002.mrc ...]
import sys

import numpy as np

NSIG, MINSEP = 2.0, 25.0


def read_mrc(path):
    with open(path, 'rb') as f:
        head = np.frombuffer(f.read(1024), dtype=np.int32)
        if head[3] != 2:
            sys.exit('%s: MRC mode %d, expected 2 (float32)' % (path, head[3]))
        nx, ny, nz = (int(v) for v in head[:3])
        f.seek(1024 + int(head[23]))
        return np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)


def shares(model):
    labels, rows, inblock, inloop = [], [], False, False
    for line in open(model):
        s = line.strip()
        if s.startswith('data_'):
            inblock, inloop = (s == 'data_model_classes'), False
        elif inblock and s == 'loop_':
            inloop, labels = True, []
        elif inblock and inloop and s.startswith('_'):
            labels.append(s.split()[0])
        elif inblock and inloop and s and not s.startswith('#'):
            rows.append(s.split())
    i = labels.index('_rlnClassDistribution')
    j = labels.index('_rlnEstimatedResolution')
    return [(float(r[i]), float(r[j])) for r in rows]


def measure(v, apix):
    thr = v.mean() + NSIG * v.std()
    z, y, x = np.nonzero(v > thr)
    w = v[z, y, x] - thr
    xyz = np.stack([x, y, z], axis=1).astype(np.float64) * apix
    c = (xyz * w[:, None]).sum(0) / w.sum()
    d = xyz - c
    cov = (d * w[:, None]).T @ d / w.sum()
    axis = np.linalg.eigh(cov)[1][:, -1]
    t = d @ axis
    lo, hi = np.floor(t.min()) - 5, np.ceil(t.max()) + 5
    prof, edges = np.histogram(t, bins=np.arange(lo, hi + 1, 1.0), weights=w)
    k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    prof = np.convolve(prof, k / k.sum(), mode='same')
    top = prof.max()
    on = np.nonzero(prof > 0.1 * top)[0]
    extent = (on[-1] - on[0]) * 1.0
    peaks = [i for i in range(1, len(prof) - 1)
             if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.25 * top]
    kept = []
    for i in sorted(peaks, key=lambda i: -prof[i]):
        if all(abs(i - j) >= MINSEP for j in kept):
            kept.append(i)
    kept.sort()
    gaps = np.diff(kept) if len(kept) > 1 else np.array([0.0])
    return extent, len(kept), float(np.median(gaps))


model, apix, maps = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
sh = shares(model)
for n, path in enumerate(maps):
    share, res = sh[n]
    if share <= 0:
        print('class %d: empty' % (n + 1))
        continue
    extent, beads, gap = measure(read_mrc(path), apix)
    print('class %d: %5.1f%% of particles, est %.1f A, extent %.0f A, %d beads, spacing %.0f A'
          % (n + 1, 100 * share, res, extent, beads, gap))
