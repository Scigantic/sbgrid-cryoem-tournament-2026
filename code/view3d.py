#!/usr/bin/env python3
# Render and measure the class volumes of a 3D classification or a multi-class initial
# model. For each class: share, resolution, rod length along its principal axis, width,
# and the number of beads on the axial profile. Sheet: one row per class, columns =
# projections along z, y, x and a projection with the rod axis horizontal.
#   python3 -I view3d.py Rod3D/Cls/run_it025_model.star Rod3D/Cls/classes.jpg [step_A]
# Prints numbers only (no file names).
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks

model, out = sys.argv[1], sys.argv[2]
step = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
root = os.path.dirname(os.path.dirname(os.path.abspath(model)))  # project root, two up from the run


def read_mrc(path):
    with open(path, 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
        xlen = float(np.frombuffer(hdr[40:44], np.float32)[0])
        ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
        assert mode == 2, mode
        f.seek(1024 + ext)
        v = np.fromfile(f, np.float32, nx * ny * nz).reshape(nz, ny, nx)
    return v, (xlen / nx if xlen > 0 else 1.0)


# model.star: data_model_classes loop
rows, labels, inblock, inloop = [], [], False, False
for line in open(model):
    s = line.strip()
    if s.startswith('data_'):
        inblock = s == 'data_model_classes'
        inloop = False
        continue
    if not inblock:
        continue
    if s == 'loop_':
        inloop = True
    elif inloop and s.startswith('_'):
        labels.append(s.split()[0])
    elif inloop and s and not s.startswith('#'):
        rows.append(s.split())
iref, idist = labels.index('_rlnReferenceImage'), labels.index('_rlnClassDistribution')
ires = labels.index('_rlnEstimatedResolution') if '_rlnEstimatedResolution' in labels else None

tiles = []
for j, r in enumerate(rows, 1):
    path = r[iref]
    if not os.path.isabs(path):
        for base in (os.getcwd(), root, os.path.dirname(os.path.abspath(model))):
            if os.path.exists(os.path.join(base, path)):
                path = os.path.join(base, path)
                break
    v, apix = read_mrc(path)
    n = v.shape[0]
    m = ndi.gaussian_filter(v, 1.0)
    thr = m.mean() + 3.0 * m.std()
    w = np.clip(m - thr, 0, None)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    tot = w.sum() + 1e-9
    c = np.array([(w * zz).sum(), (w * yy).sum(), (w * xx).sum()]) / tot
    d = np.stack([zz - c[0], yy - c[1], xx - c[2]]).reshape(3, -1)
    cov = (d * w.ravel()) @ d.T / tot
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, -1]                      # principal axis (z, y, x)
    proj = d.T @ ax
    sel = w.ravel() > 0
    length = (np.percentile(proj[sel], 99) - np.percentile(proj[sel], 1)) * apix if sel.any() else 0.0
    width = 4.0 * np.sqrt(evals[0]) * apix  # ~ full width of the density across the thinnest axis
    # axial profile: density within 25 A of the axis, binned along it
    perp = np.sqrt(np.maximum((d * d).sum(axis=0) - proj ** 2, 0))
    cyl = perp < 25.0 / apix
    bins = np.round(proj[cyl]).astype(int)
    prof = np.bincount(bins - bins.min(), weights=np.clip(m.ravel()[cyl], 0, None))
    prof = ndi.gaussian_filter1d(prof, 1.0)
    pk, _ = find_peaks(prof, distance=max(2, int(0.6 * step / apix)), prominence=0.1 * prof.max())
    share = 100.0 * float(r[idist])
    res = float(r[ires]) if ires is not None else 0.0
    print('class %2d: %5.1f%%  %6.1f A  rod length %4.0f A = %.1f repeats of %.0f A, width %3.0f A, %d beads'
          % (j, share, res, length, length / step, step, width, len(pk)))
    # projections: z, y, x and axis-horizontal
    az = np.degrees(np.arctan2(ax[1], ax[2]))          # rotate about z to put the axis in the xz plane
    r1 = ndi.rotate(v, az, axes=(1, 2), reshape=False, order=1)
    el = np.degrees(np.arctan2(ax[0], np.hypot(ax[1], ax[2])))
    r2 = ndi.rotate(r1, el, axes=(0, 2), reshape=False, order=1)   # then about y to make it horizontal
    tiles.append((j, share, [v.sum(axis=0), v.sum(axis=1), v.sum(axis=2), r2.sum(axis=0)]))

n = tiles[0][2][0].shape[0]
sheet = Image.new('L', (4 * n, n * len(tiles)))
dr = ImageDraw.Draw(sheet)
for row, (j, share, ims) in enumerate(tiles):
    for i, t in enumerate(ims):
        lo, hi = np.percentile(t, [0.5, 99.8])
        g = np.clip((t - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(g), (i * n, row * n))
    dr.text((2, row * n + 1), '%d %.0f%%' % (j, share), fill=255)
sheet = sheet.resize((sheet.width * 3, sheet.height * 3))
sheet.save(out, quality=88)
print('wrote', os.path.basename(out))
