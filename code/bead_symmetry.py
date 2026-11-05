#!/usr/bin/env python3
# Are the beads of the rod copies of one subunit, related by a fixed screw (rise + twist)?
# Finds the rod axis and the bead centres as measure_classes.py does, samples each bead on a
# cylindrical grid (r, phi, z) about the axis, and correlates every pair of neighbouring beads
# over all rotations about the axis (5 deg steps) and small shifts along it (1 A steps).
# A clear best rotation with a high correlation, repeated for every pair, says the beads
# are one subunit repeated by a screw; a flat correlation over rotation says the beads are
# too featureless (or too different) to tell. numpy only, light enough for the login node.
# With the two half maps as HALF1 HALF2 it also prints, for every bead, the correlation of that
# bead between the half maps: the ceiling an identical copy could reach at this noise level.
#   python3 -I bead_symmetry.py map.mrc APIX [HALF1 HALF2]
import sys

import numpy as np

path, apix, halves = sys.argv[1], float(sys.argv[2]), sys.argv[3:5]
NSIG, MINSEP, RMAX = 2.0, 25.0, 45.0


def read_mrc(p):
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        nx, ny, nz = (int(v) for v in h[:3])
        f.seek(1024 + int(h[23]))
        return np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)


v = read_mrc(path)
nz, ny, nx = v.shape
thr = v.mean() + NSIG * v.std()
z, y, x = np.nonzero(v > thr)
w = v[z, y, x] - thr
xyz = np.stack([x, y, z], axis=1).astype(np.float64) * apix
c = (xyz * w[:, None]).sum(0) / w.sum()
d = xyz - c
axis = np.linalg.eigh((d * w[:, None]).T @ d / w.sum())[1][:, -1]
t = d @ axis
lo = np.floor(t.min()) - 5
prof, _ = np.histogram(t, bins=np.arange(lo, np.ceil(t.max()) + 6, 1.0), weights=w)
k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
prof = np.convolve(prof, k / k.sum(), mode='same')
peaks = [i for i in range(1, len(prof) - 1)
         if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.25 * prof.max()]
kept = []
for i in sorted(peaks, key=lambda i: -prof[i]):
    if all(abs(i - j) >= MINSEP for j in kept):
        kept.append(i)
centres = [lo + i + 0.5 for i in sorted(kept)]
print('beads %d at %s A along the axis' % (len(centres), ' '.join('%.0f' % q for q in centres)))

u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
u /= np.linalg.norm(u)
vv = np.cross(axis, u)


def trilinear(pts, v=v):  # pts in A (x, y, z); zero outside the box
    g = pts / apix
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = np.zeros(len(g))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) \
                    * (f[:, 2] if dz else 1 - f[:, 2])
                out[ok] += wgt[ok] * v[iz[ok], iy[ok], ix[ok]]
    return out


rs = np.arange(1.0, RMAX, 1.5)
phis = np.deg2rad(np.arange(0, 360, 5))
spacing = np.median(np.diff(centres)) if len(centres) > 1 else 38.0
zs = np.arange(-spacing / 2, spacing / 2, 1.0)


def bead(t0, m=v):  # samples[z, r, phi] of the slab centred at t0 along the axis
    Z, R, P = np.meshgrid(zs + t0, rs, phis, indexing='ij')
    pts = c + Z[..., None] * axis + R[..., None] * (np.cos(P)[..., None] * u + np.sin(P)[..., None] * vv)
    return trilinear(pts.reshape(-1, 3), m).reshape(Z.shape)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


print('bead mean density: %s' % ' '.join('%.3f' % bead(q).mean() for q in centres))
for i in range(len(centres) - 1):
    a = bead(centres[i])
    best = (-2, 0, 0)
    allc = []
    for dz in range(-4, 5):
        b = bead(centres[i + 1] + dz)
        for s in range(len(phis)):
            r = corr(a, np.roll(b, s, axis=2))
            allc.append(r)
            if r > best[0]:
                best = (r, s * 5, dz)
    allc = np.array(allc)
    print('beads %d-%d: rise %.1f A, best twist %d deg, shift %+d A, corr %.3f '
          '(median over rotations %.3f, worst %.3f)'
          % (i + 1, i + 2, centres[i + 1] - centres[i] + best[2], best[1], best[2], best[0],
             np.median(allc), allc.min()))

if halves:
    h1, h2 = read_mrc(halves[0]), read_mrc(halves[1])
    print('same bead, half1 vs half2 (noise ceiling): %s'
          % ' '.join('%.3f' % corr(bead(q, h1), bead(q, h2)) for q in centres))
    print('neighbours across halves, no rotation or shift: %s' % ' '.join(
        '%.3f' % corr(bead(centres[i], h1), bead(centres[i + 1], h2)) for i in range(len(centres) - 1)))
