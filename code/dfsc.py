#!/usr/bin/env python3
"""Directional (conical) FSC of map03, in the style of 3DFSC.

Masked half maps -> FFT -> FSC inside a 20 deg cone around each of 300 directions
over a hemisphere (Friedel symmetry covers the rest). Prints numbers only and
writes a small plot of resolution against angle from the rod axis.
Usage: dfsc.py half1 half2 mask apix outdir
"""
import os
import sys

import mrcfile
import numpy as np

h1p, h2p, mp, apix, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True)
CONE = np.deg2rad(20.0)
NDIR = 300
THR = 0.143


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float32)


m = load(mp)
f1 = np.fft.rfftn(load(h1p) * m).astype(np.complex64)
f2 = np.fft.rfftn(load(h2p) * m).astype(np.complex64)
n = m.shape[0]
assert m.shape == (n, n, n)

# rod axis = long principal axis of the mask, in (z, y, x) order like the arrays
idx = np.argwhere(m > 0.5).astype(np.float64)
idx -= idx.mean(0)
w, v = np.linalg.eigh(idx.T @ idx)
rod = v[:, np.argmax(w)]

kz = np.fft.fftfreq(n) * n
ky = np.fft.fftfreq(n) * n
kx = np.fft.rfftfreq(n) * n
Z, Y, X = np.meshgrid(kz, ky, kx, indexing='ij')
r = np.sqrt(Z * Z + Y * Y + X * X)
sel = (r >= 1) & (r < n // 2)
r = r[sel]
shell = np.rint(r).astype(np.int32)
u = np.stack([Z[sel], Y[sel], X[sel]]) / r
del Z, Y, X
a, b = f1[sel], f2[sel]
num = (a * np.conj(b)).real.astype(np.float32)
d1 = (np.abs(a) ** 2).astype(np.float32)
d2 = (np.abs(b) ** 2).astype(np.float32)
del a, b, f1, f2
nsh = n // 2


def fsc(keep=None):
    if keep is None:
        nn = np.bincount(shell, num, nsh)
        x1 = np.bincount(shell, d1, nsh)
        x2 = np.bincount(shell, d2, nsh)
    else:
        s = shell[keep]
        nn = np.bincount(s, num[keep], nsh)
        x1 = np.bincount(s, d1[keep], nsh)
        x2 = np.bincount(s, d2[keep], nsh)
    with np.errstate(invalid='ignore', divide='ignore'):
        return nn / np.sqrt(x1 * x2)


def crossing(c):
    for i in range(3, nsh):
        if not c[i] >= THR:  # also catches nan
            if i == 0 or not np.isfinite(c[i]):
                return n * apix / (i - 1)
            f = (c[i - 1] - THR) / (c[i - 1] - c[i])
            return n * apix / (i - 1 + f)
    return n * apix / (nsh - 1)


glob_res = crossing(fsc())
# Fibonacci hemisphere (z >= 0 in a frame we then use as (z, y, x))
k = np.arange(NDIR) + 0.5
cz = k / NDIR
phi = np.pi * (1 + 5 ** 0.5) * k
sz = np.sqrt(1 - cz * cz)
dirs = np.stack([cz, sz * np.cos(phi), sz * np.sin(phi)], 1)
cosc = np.cos(CONE)
res = np.empty(NDIR)
for i, d in enumerate(dirs):
    keep = np.abs(d @ u) >= cosc
    res[i] = crossing(fsc(keep))
ang = np.rad2deg(np.arccos(np.abs(dirs @ rod)))

q = np.percentile(res, [0, 5, 25, 50, 75, 95, 100])
print('box %d  apix %.4f  cone %.0f deg  directions %d' % (n, apix, np.rad2deg(CONE), NDIR))
print('global masked FSC 0.143: %.2f A' % glob_res)
print('directional 0.143 (min/5/25/50/75/95/max): ' + ' '.join('%.2f' % x for x in q))
print('best/worst ratio %.2f   spread (worst-best) %.2f A' % (q[-1] / q[0], q[-1] - q[0]))
for lo, hi in ((0, 30), (30, 60), (60, 90.01)):
    s = (ang >= lo) & (ang < hi)
    print('angle from rod axis %2d-%2d deg: n %3d  median %.2f  range %.2f-%.2f A' % (
        lo, int(hi), s.sum(), np.median(res[s]), res[s].min(), res[s].max()))
print('worst direction at %.0f deg from rod axis, best at %.0f deg' % (
    ang[np.argmax(res)], ang[np.argmin(res)]))
np.savetxt(os.path.join(out, 'dfsc_directions.txt'),
           np.column_stack([dirs, ang, res]), fmt='%.4f',
           header='dz dy dx angle_from_rod_axis_deg res_0.143_A')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
ax.scatter(ang, res, s=10, color='#2f6db5')
ax.axhline(glob_res, color='0.4', lw=0.8, ls='--')
ax.text(1, glob_res, ' global %.2f A' % glob_res, va='bottom', fontsize=8, color='0.3')
ax.set_xlabel('angle between cone axis and rod axis (deg)')
ax.set_ylabel('FSC 0.143 in 20 deg cone (A)')
ax.set_xlim(0, 90)
ax.set_title('map03 directional FSC, %d directions' % NDIR, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, 'dfsc.png'))
print('wrote', out)
