#!/usr/bin/env python3
# Focused-refinement reference and mask: moves the body mask's centre of mass to the box
# centre (RELION's centre is index n//2, the same point relion_particle_subtract
# --recenter_on_mask moves the particles to), applies the same shift to map*mask, and crops
# both to NEWBOX. Fourier shift, so sub-voxel centres are kept.
#   python3 -I recenter.py map.mrc bodymask.mrc APIX NEWBOX OUTDIR
import os
import sys

import numpy as np

mpath, kpath, apix, newbox, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), \
    int(sys.argv[4]), sys.argv[5]


def read(p):
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        nx, ny, nz = (int(v) for v in h[:3])
        f.seek(1024 + int(h[23]))
        return np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)


def write(p, a):
    n = a.shape[0]
    h = np.zeros(256, dtype=np.int32)
    h[0:3] = n
    h[3] = 2
    h[7:10] = n
    hf = h.view(np.float32)
    hf[10:13] = n * apix
    hf[13:16] = 90.0
    h[16:19] = (1, 2, 3)
    hf[19], hf[20], hf[21] = a.min(), a.max(), a.mean()
    h[52] = int.from_bytes(b'MAP ', 'little')
    h[53] = 0x00004444
    hf[54] = a.std()
    with open(p, 'wb') as f:
        f.write(h.tobytes())
        f.write(a.astype(np.float32).tobytes())


m, k = read(mpath), read(kpath)
n = m.shape[0]
if newbox > n or newbox % 2:
    sys.exit('NEWBOX must be even and <= %d' % n)
w = np.clip(k, 0, None)
idx = np.arange(n, dtype=np.float64)
com = np.array([(w.sum(axis=(1, 2)) * idx).sum(), (w.sum(axis=(0, 2)) * idx).sum(),
                (w.sum(axis=(0, 1)) * idx).sum()]) / w.sum()          # z, y, x
shift = n // 2 - com
print('mask centre of mass (x y z voxels): %.2f %.2f %.2f; shift %.2f %.2f %.2f A' % (
    com[2], com[1], com[0], *(shift[::-1] * apix)))
f = [np.fft.fftfreq(n)] * 3
ph = np.exp(-2j * np.pi * (shift[0] * f[0][:, None, None] + shift[1] * f[1][None, :, None]
                           + shift[2] * f[2][None, None, :]))


def move(a):
    return np.real(np.fft.ifftn(np.fft.fftn(a) * ph)).astype(np.float32)


lo = n // 2 - newbox // 2
crop = (slice(lo, lo + newbox),) * 3
km = np.clip(move(k)[crop], 0, 1)
kept = km.sum() / np.clip(k, 0, 1).sum()
print('mask kept inside the new box: %.1f%%' % (100 * kept))
if kept < 0.995:
    sys.exit('new box cuts the mask: use a bigger NEWBOX')
os.makedirs(outdir, exist_ok=True)
write(os.path.join(outdir, 'ref.mrc'), move(m * k)[crop])
write(os.path.join(outdir, 'mask.mrc'), km)
print('wrote ref.mrc and mask.mrc, box %d, %.1f A' % (newbox, newbox * apix))
