#!/usr/bin/env python3
# Orientation census of refined rods. Is the rotation about the rod axis evenly sampled,
# or do the rods lie on the grid with a preferred face (which would leave a missing wedge
# around the rod axis and smear the map in one direction)?
#   python3 -I rodview.py MAP.mrc run_data.star
# Reads the map, finds the rod axis by PCA of the voxels above mean + 2 sd, converts each
# particle's rot/tilt into a viewing direction, and histograms (a) the angle between the
# view and the rod axis (90 = rod seen side-on) and (b) the azimuth of the view around the
# rod axis, folded to 180 since opposite views fill the same Fourier plane.
import struct
import sys

import numpy as np

mapf, starf = sys.argv[1], sys.argv[2]

with open(mapf, 'rb') as f:
    hdr = f.read(1024)
    nx, ny, nz, mode = struct.unpack('<4i', hdr[:16])
    assert mode == 2, 'expected mode 2 map'
    vol = np.frombuffer(f.read(nx * ny * nz * 4), dtype='<f4').reshape(nz, ny, nx).astype(np.float32)
thr = float(vol.mean() + 2 * vol.std())
zz, yy, xx = np.nonzero(vol > thr)
w = vol[zz, yy, xx]
pts = np.stack([xx, yy, zz], 1).astype(np.float64)
c = (pts * w[:, None]).sum(0) / w.sum()
d = pts - c
cov = (d * w[:, None]).T @ d / w.sum()
vals, vecs = np.linalg.eigh(cov)
a = vecs[:, np.argmax(vals)]
a /= np.linalg.norm(a)
print('rod axis in map frame (x, y, z): %.2f %.2f %.2f; voxels above threshold %d'
      % (a[0], a[1], a[2], len(w)))

# star: find the particles table and the column indices
cols, rows, table = {}, [], None
with open(starf) as f:
    for line in f:
        s = line.strip()
        if s.startswith('data_'):
            table = s
            cols = {}
            continue
        if table != 'data_particles':
            continue
        if s.startswith('_rln'):
            name, idx = s.split()[:2]
            cols[name[4:]] = int(idx.lstrip('#')) - 1
            continue
        if s and not s.startswith('loop_'):
            rows.append(s.split())
rot = np.array([float(r[cols['AngleRot']]) for r in rows])
tilt = np.array([float(r[cols['AngleTilt']]) for r in rows])
n = len(rows)
print('particles %d' % n)

tr, tt = np.radians(rot), np.radians(tilt)
v = np.stack([np.sin(tt) * np.cos(tr), np.sin(tt) * np.sin(tr), np.cos(tt)], 1)
# orthonormal basis around the rod axis
tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
b = np.cross(a, tmp); b /= np.linalg.norm(b)
cc = np.cross(a, b)
ang = np.degrees(np.arccos(np.clip(np.abs(v @ a), 0, 1)))      # 90 = side view
az = np.degrees(np.arctan2(v @ cc, v @ b)) % 180.0               # folded azimuth

h, _ = np.histogram(ang, bins=np.arange(0, 91, 10))
print('angle between view and rod axis (deg, 10 bins 0..90; 90 = side-on):')
print('  ' + ' '.join('%d' % x for x in h))
print('  median %.1f, 10th pct %.1f' % (np.median(ang), np.percentile(ang, 10)))
h2, _ = np.histogram(az, bins=np.arange(0, 181, 10))
print('azimuth of the view around the rod axis (deg mod 180, 18 bins of 10):')
print('  ' + ' '.join('%d' % x for x in h2))
print('  min/max bin %d / %d, ratio %.2f (1 = uniform)' % (h2.min(), h2.max(), h2.min() / max(h2.max(), 1)))

if 'MaxValueProbDistribution' in cols:
    mp = np.array([float(r[cols['MaxValueProbDistribution']]) for r in rows])
    print('max prob: quartiles %.2f %.2f %.2f' % tuple(np.percentile(mp, [25, 50, 75])))
if 'NrOfSignificantSamples' in cols:
    ns = np.array([float(r[cols['NrOfSignificantSamples']]) for r in rows])
    print('significant samples: quartiles %.0f %.0f %.0f' % tuple(np.percentile(ns, [25, 50, 75])))
if 'MicrographName' in cols:
    from collections import Counter
    cnt = Counter(r[cols['MicrographName']] for r in rows)
    print('micrographs %d, particles per micrograph median %d' % (len(cnt), int(np.median(list(cnt.values())))))
