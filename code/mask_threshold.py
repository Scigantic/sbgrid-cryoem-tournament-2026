#!/usr/bin/env python3
# Mask threshold heuristic for relion_mask_create: mean + 2 sd of the map's voxels.
#   python3 -I mask_threshold.py map.mrc
import sys
import numpy as np
with open(sys.argv[1], 'rb') as f:
    hdr = f.read(1024)
    nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
    ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
    assert mode == 2, mode
    f.seek(1024 + ext)
    v = np.fromfile(f, np.float32, nx * ny * nz)
print('%.5f' % (v.mean() + 2.0 * v.std()))
