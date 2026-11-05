#!/usr/bin/env python3
# Synthetic 3D references for sorting rods by length: n beads of a given diameter on a
# straight line with a fixed repeat, centred in the box, one volume per n. Written as
# MRC mode 2 with the pixel size in the header, plus a star file listing them for
# relion_refine --ref. Used with --ini_high 25 so only the bead count carries weight.
#   python3 -I make_rod_refs.py OUTDIR BOX_PX APIX BEAD_DIAM_A STEP_A 3,4,5,6,7
import os
import struct
import sys

import numpy as np

out, box, apix, diam, step = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
counts = [int(v) for v in sys.argv[6].split(',')]
os.makedirs(out, exist_ok=True)
sigma = diam / 2.0 / 2.0 / apix          # bead half width at ~2 sigma
zz, yy, xx = np.mgrid[0:box, 0:box, 0:box].astype(np.float32)
c = (box - 1) / 2.0
names = []
for n in counts:
    v = np.zeros((box, box, box), np.float32)
    for b in range(n):
        cz = c + (b - (n - 1) / 2.0) * step / apix    # rod along z
        v += np.exp(-((zz - cz) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))
    hdr = bytearray(1024)
    struct.pack_into('<4i', hdr, 0, box, box, box, 2)
    struct.pack_into('<3i', hdr, 28, box, box, box)
    struct.pack_into('<3f', hdr, 40, box * apix, box * apix, box * apix)
    struct.pack_into('<3f', hdr, 52, 90.0, 90.0, 90.0)
    struct.pack_into('<3i', hdr, 64, 1, 2, 3)
    struct.pack_into('<3f', hdr, 76, float(v.min()), float(v.max()), float(v.mean()))
    hdr[208:212] = b'MAP '
    hdr[212:216] = bytes([68, 65, 0, 0])
    path = os.path.join(out, 'rod%02d.mrc' % n)
    with open(path, 'wb') as f:
        f.write(bytes(hdr) + v.tobytes())
    names.append(path)
with open(os.path.join(out, 'refs.star'), 'w') as f:
    f.write('\ndata_\n\nloop_\n_rlnReferenceImage #1\n')
    for p in names:
        f.write(p + '\n')
print('wrote %d references (%s beads) at %d px, %.3f A/px, bead %.0f A, step %.0f A'
      % (len(names), sys.argv[6], box, apix, diam, step))
