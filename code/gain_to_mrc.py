"""Convert a gain reference (TIFF or MRC) to a float32 MRC, optionally inverted.

usage: python3 -I gain_to_mrc.py IN OUT [--invert]

RELION cannot convert TIFF, and MotionCor3 divides by the gain it is given, so the
tournament gain (a multiplier, mean 1.0) is passed to it as 1/gain with --invert.
Needs numpy and tifffile (both exist on SBCloud).
"""
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
if src.lower().endswith((".tif", ".tiff")):
    import tifffile
    a = tifffile.imread(src)
else:
    import mrcfile
    with mrcfile.open(src) as m:
        a = m.data[0] if m.data.ndim == 3 else m.data
a = np.ascontiguousarray(a, dtype=np.float32)
if "--invert" in sys.argv:
    a = 1.0 / a
ny, nx = a.shape
h = np.zeros(256, dtype=np.int32)
h[0:4] = (nx, ny, 1, 2)
h[7:10] = (nx, ny, 1)
h[16:19] = (1, 2, 3)
h[22] = 1
f = h.view(np.float32)
f[10:13] = (nx, ny, 1)
f[13:16] = (90, 90, 90)
f[19:22] = (a.min(), a.max(), a.mean())
hb = bytearray(h.tobytes())
hb[208:212] = b"MAP "
hb[212:216] = bytes((0x44, 0x44, 0, 0))
with open(out, "wb") as o:
    o.write(bytes(hb) + a.tobytes())
print("%s -> %s %s mean %.3f%s" % (src.rsplit("/", 1)[-1], out.rsplit("/", 1)[-1],
      a.shape, a.mean(), " (inverted)" if "--invert" in sys.argv else ""))
