#!/usr/bin/env python3
"""Rotational symmetry of each bead about the rod axis.

If each bead is a ring of n identical subunits stacked along the rod, density in a
slab through that bead repeats every 360/n deg around the axis. For each bead the map
is resampled on a cylinder (r 3 to RMAX A, 2 deg steps) and correlated with itself
rotated by 360/n for n = 2..8, and the angular power at harmonic k is summed.
Prints numbers only.
Usage: rotsym.py map mask apix [rmax_A] [axis: long|short|mid] [bead centres along long axis, comma list]
With short or mid, the cylinder axis is that principal axis of the mask, placed through
each bead centre on the long axis, and the slab is +-HZ A along it.
"""
import sys

import mrcfile
import numpy as np
from scipy.ndimage import map_coordinates

mp, kp, apix = sys.argv[1], sys.argv[2], float(sys.argv[3])
RMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 32.0
AXIS = sys.argv[5] if len(sys.argv) > 5 else 'long'
CEN = [float(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else []
HZ = 20.0


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float32)


v, k = load(mp), load(kp)
idx = np.argwhere(k > 0.5).astype(np.float64)          # voxel (z, y, x)
w = k[k > 0.5].astype(np.float64)
c = (idx * w[:, None]).sum(0) / w.sum()
cov = np.cov((idx - c).T, aweights=w)
ev, evec = np.linalg.eigh(cov)
ax = evec[:, -1]                                        # long axis
e1 = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(ax, e1)
half_len = 2.2 * np.sqrt(ev[-1]) * apix
print('box %s, mask long axis half-length about %.0f A, widths %.0f %.0f A (2 sd)' % (
    v.shape, half_len, 2 * np.sqrt(ev[0]) * apix, 2 * np.sqrt(ev[1]) * apix))
if AXIS != 'long':
    long_ax = ax
    ax = evec[:, 0] if AXIS == 'short' else evec[:, 1]
    e1 = long_ax.copy()
    e2 = np.cross(ax, e1)
    vm = v * (k > 0.05)
    ph = np.deg2rad(np.arange(0, 360, 2.0))
    rs = np.arange(3.0, RMAX + 0.1, 1.0)
    zs = np.arange(-HZ, HZ + 0.1, 1.0)
    Z, R, P = np.meshgrid(zs, rs, ph, indexing='ij')
    print('axis %s; cylinder r 3 to %.0f A, slab +-%.0f A' % (AXIS, RMAX, HZ))
    for z0 in CEN:
        cc0 = c + z0 * long_ax / apix
        pts = (cc0[:, None, None, None]
               + (Z[None] * ax[:, None, None, None]
                  + R[None] * np.cos(P)[None] * e1[:, None, None, None]
                  + R[None] * np.sin(P)[None] * e2[:, None, None, None]) / apix)
        S = map_coordinates(vm, pts.reshape(3, -1), order=1).reshape(Z.shape)
        S = S - S.mean(axis=2, keepdims=True)
        wr = rs[None, :, None]
        cc = []
        for n in range(2, 9):
            b = np.roll(S, int(round(180 / n)), axis=2)
            cc.append((S * b * wr).sum() / np.sqrt((S * S * wr).sum() * (b * b * wr).sum()))
        F = np.abs(np.fft.rfft(S, axis=2)) ** 2 * wr
        pw = F.sum(axis=(0, 1))[1:13]; pw = pw / pw.sum()
        print('bead @%5.0f  CC(360/n) n=2..8: %s | power k=1..12: %s' % (
            z0, ' '.join('%.2f' % x for x in cc), ' '.join('%.2f' % x for x in pw)))
    sys.exit(0)

zs = np.arange(-half_len, half_len + 0.1, 1.0)
rs = np.arange(3.0, RMAX + 0.1, 1.0)
ph = np.deg2rad(np.arange(0, 360, 2.0))
Z, R, P = np.meshgrid(zs, rs, ph, indexing='ij')
pts = (c[:, None, None, None]
       + (Z[None] * ax[:, None, None, None]
          + R[None] * np.cos(P)[None] * e1[:, None, None, None]
          + R[None] * np.sin(P)[None] * e2[:, None, None, None]) / apix)
D = map_coordinates(v * (k > 0.05), pts.reshape(3, -1), order=1).reshape(Z.shape)

# axial profile: positive density per 1 A slab, then bead centres = local maxima
prof = np.clip(D, 0, None).sum(axis=(1, 2)) * 1.0
sm = np.convolve(prof, np.ones(9) / 9, mode='same')
peaks = [i for i in range(5, len(sm) - 5)
         if sm[i] == sm[i - 12 if i >= 12 else 0:i + 13].max() and sm[i] > 0.4 * sm.max()]
print('bead centres along axis (A):', ' '.join('%.0f' % zs[i] for i in peaks))

# radial profile overall
rp = np.clip(D, 0, None).mean(axis=(0, 2))
print('radial profile (r A: rel. density):', ' '.join(
    '%d:%.2f' % (r, x / rp.max()) for r, x in zip(rs[::3], rp[::3])))


def report(sl, name):
    S = D[sl]                                            # (z, r, phi)
    S = S - S.mean(axis=2, keepdims=True)                # remove the round part per ring
    wr = rs[None, :, None]                               # area weight
    cc = []
    for n in range(2, 9):
        sh = int(round(180 / n))                         # 360/n deg in 2-deg steps
        a, b = S, np.roll(S, sh, axis=2)
        cc.append((a * b * wr).sum() / np.sqrt((a * a * wr).sum() * (b * b * wr).sum()))
    F = np.abs(np.fft.rfft(S, axis=2)) ** 2 * wr
    pw = F.sum(axis=(0, 1))[1:13]
    pw = pw / pw.sum()
    print('%-14s CC(360/n) n=2..8: %s | angular power k=1..12: %s' % (
        name, ' '.join('%.2f' % x for x in cc), ' '.join('%.2f' % x for x in pw)))


report(slice(None), 'all')
for j, i in enumerate(peaks):
    report(slice(max(0, i - 14), i + 15), 'bead %d @%.0f' % (j + 1, zs[i]))
