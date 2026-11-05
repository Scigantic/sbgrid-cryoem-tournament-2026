#!/usr/bin/env python3
# Clip a float32 MRC mask to 0..1 in place. Rescaling a soft mask in Fourier space rings a
# little past both ends, and relion_refine and relion_postprocess reject masks outside 0..1.
# Prints the range before and after.
#   python3 -I clip_mask.py mask.mrc [mask2.mrc ...]
import sys

import numpy as np

for path in sys.argv[1:]:
    with open(path, 'r+b') as f:
        head = np.frombuffer(f.read(1024), dtype=np.int32)
        if head[3] != 2:
            sys.exit('%s: MRC mode %d, expected 2 (float32)' % (path, head[3]))
        n = int(head[0]) * int(head[1]) * int(head[2])
        v = np.memmap(f, dtype=np.float32, mode='r+', offset=1024 + int(head[23]), shape=(n,))
        lo, hi = float(v.min()), float(v.max())
        np.clip(v, 0, 1, out=v)
        v.flush()
        print('%s: %.4f..%.4f -> %.4f..%.4f' % (path.rsplit('/', 1)[-1], lo, hi, v.min(), v.max()))
