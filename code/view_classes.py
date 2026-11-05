"""Render the most populated 2D class averages as one JPEG and print their stats.

usage: python3 -I view_classes.py MODEL.star CLASSES.mrcs OUT.jpg [NTOP]

Needs only Pillow. Reads the MRC stack directly (float32, 1024-byte header).
Output order is by population, six per row, so class number != position; the printed
list gives the class number for each tile, left to right, top to bottom.
"""
import array
import struct
import sys
from PIL import Image

model_star, mrcs, out, ntop = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 24

text = open(model_star).read().split("data_model_classes")[1].split("data_model_class_1")[0].split("\n")
cols = [l.split()[0] for l in text if l.startswith("_rln")]
rows = [l.split() for l in text if l.strip() and not l.startswith(("_", "#", "loop_", "data_"))]
idist, ires = cols.index("_rlnClassDistribution"), cols.index("_rlnEstimatedResolution")
total_classes = len(rows)

raw = open(mrcs, "rb").read()
nx, ny, nz = struct.unpack("<3i", raw[:12])
assert nx == ny and nz >= total_classes, (nx, ny, nz, total_classes)
npix = nx * ny
data = array.array("f")
data.frombytes(raw[1024:1024 + nz * npix * 4])

stats = sorted(((float(r[idist]), k, r[ires]) for k, r in enumerate(rows, 1)), reverse=True)[:ntop]
per_row = 6
sheet = Image.new("L", (nx * per_row, nx * ((len(stats) + per_row - 1) // per_row)))
for j, (dist, k, res) in enumerate(stats):
    v = data[(k - 1) * npix:k * npix]
    lo, hi = min(v), max(v)
    s = 255.0 / (hi - lo + 1e-9)
    tile = Image.new("L", (nx, nx))
    tile.putdata([int((x - lo) * s) for x in v])
    sheet.paste(tile, (nx * (j % per_row), nx * (j // per_row)))
sheet = sheet.resize((sheet.width * 2, sheet.height * 2))
sheet.save(out, "JPEG", quality=85)
print("tiles, left to right then down: class number, share of particles, estimated resolution (A)")
for dist, k, res in stats:
    print("  class %3d  %5.1f%%  %s" % (k, 100 * dist, res[:5]))
print("wrote", out)
