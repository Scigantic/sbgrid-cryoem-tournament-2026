#!/usr/bin/env python3
"""Size and volume of the density in a map at several thresholds. Numbers only.
For each threshold: enclosed volume, the mass it implies at 1.21 A^3/Da, and the
extent (2nd to 98th percentile of voxel positions) along the principal axes of the
thresholded density, plus how many connected pieces there are.
Usage: mapdims.py map mask apix
"""
import sys
import mrcfile
import numpy as np
from scipy import ndimage

mp, kp, apix = sys.argv[1], sys.argv[2], float(sys.argv[3])
rd = lambda p: np.asarray(mrcfile.open(p, permissive=True).data, dtype=np.float32)
v, k = rd(mp), rd(kp)
inside = v[k > 0.5]
outside = v[k < 0.01]
sd = outside.std() if outside.size > 1000 else v.std()
print('solvent sd %.4g, max inside mask %.1f sd' % (sd, inside.max() / sd))
for nsd in (3, 5, 8, 12):
    b = (v > nsd * sd) & (k > 0.05)
    lab, n = ndimage.label(b)
    sizes = np.bincount(lab.ravel())[1:]
    big = lab == (sizes.argmax() + 1)
    idx = np.argwhere(big).astype(np.float64) * apix
    c = idx.mean(0)
    w, vec = np.linalg.eigh(np.cov((idx - c).T))
    proj = (idx - c) @ vec
    ext = [np.percentile(proj[:, i], 98) - np.percentile(proj[:, i], 2) for i in (2, 1, 0)]
    vol = big.sum() * apix ** 3
    print('%2d sd: volume %8.0f A^3 = %4.0f kDa at 1.21 A^3/Da; extents %3.0f x %3.0f x %3.0f A; '
          'pieces %d, largest holds %.0f%%' % (nsd, vol, vol / 1.21 / 1000, *ext, n,
                                               100 * sizes.max() / sizes.sum()))
