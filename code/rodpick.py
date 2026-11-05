#!/usr/bin/env python3
# Rod picker for the tournament specimen (short rods ~50 A wide, 150-400 A long).
# RELION's template and LoG pickers do not find them, so: flatten the micrograph,
# run a bank of oriented line filters at 8x binning, threshold, label the
# connected blobs, keep the elongated ones, and report centre, direction and
# length of each rod. Writes RELION coordinate files for whole-rod extraction,
# a census of rod lengths in units of the repeat, and overlay JPEGs.
# Login node, no GPU, about 2 s per micrograph.
#   python3 -I rodpick.py --star CtfFind/micrographs_ctf.star --root ~/results/chunk \
#       --out RodsPick [--n 300] [--overlay 3]
#   python3 -I rodpick.py --npy mic0_bin4.npy --out test --overlay 1   (binned array)
import argparse
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import measure, morphology
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('--star')
ap.add_argument('--root', default='.')
ap.add_argument('--npy', nargs='*', default=[])
ap.add_argument('--npy_bin', type=int, default=4)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=100000)
ap.add_argument('--overlay', type=int, default=0)
ap.add_argument('--angpix', type=float, default=0.8837)
ap.add_argument('--bin', type=int, default=8)
ap.add_argument('--step', type=float, default=40.0, help='repeat along the rod, A')
ap.add_argument('--width', type=float, default=50.0, help='rod width, A')
ap.add_argument('--thresh', type=float, default=3.0, help='response threshold in sigma')
ap.add_argument('--min_len', type=float, default=70.0, help='shortest rod kept, A')
ap.add_argument('--max_width', type=float, default=100.0, help='widest blob kept, A')
ap.add_argument('--margin', type=float, default=260.0,
                help='drop rods closer than this to a micrograph edge, A (half the extraction box)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
apix = a.angpix * a.bin          # A per pixel at working (detection) binning
apix4 = a.angpix * 4             # lengths are measured on the 4x binned image


def read_mrc(path):
    with open(path, 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
        ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
        f.seek(1024 + ext)
        dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}[mode]
        return np.fromfile(f, dt, nx * ny).reshape(ny, nx).astype(np.float32)


def binned(img, b):
    h, w = img.shape[0] - img.shape[0] % b, img.shape[1] - img.shape[1] % b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def line_bank(width_px, n_ang=12):
    """Zero-mean, unit-norm oriented kernels: a ridge of the rod's width."""
    sw = width_px / 2.35                           # across-rod sigma
    sl = max(0.8 * width_px, 3.0)                   # along-rod sigma: a short segment,
    r = int(3 * sl)                                 # so rod ends and blobs do not smear
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    ks = []
    for k in range(n_ang):
        t = np.pi * k / n_ang
        u = x * np.cos(t) + y * np.sin(t)          # along
        v = -x * np.sin(t) + y * np.cos(t)         # across
        prof = np.exp(-v * v / (2 * sw * sw)) - 0.5 * np.exp(-v * v / (8 * sw * sw))
        kern = prof * np.exp(-u * u / (2 * sl * sl))
        kern -= kern.mean()
        kern /= np.sqrt((kern * kern).sum())
        ks.append(kern)
    return ks


def flatten(m, s=30):
    hp = m - ndi.gaussian_filter(m, s)
    sd = np.sqrt(ndi.gaussian_filter(hp * hp, s)) + 1e-6
    return -hp / sd                                 # protein positive


BANK = line_bank(a.width / apix)
ANG = [180.0 * k / len(BANK) for k in range(len(BANK))]


def profile_length(sm, cx, cy, t, max_half_px):
    """Extent of the rod along its axis from the smoothed image: walk out from
    the centre in both directions until the signal drops below a fraction of
    the central plateau. Returns (length_px, new centre x, new centre y)."""
    d = np.arange(-max_half_px, max_half_px + 1, dtype=np.float32)
    xs = cx + d * np.cos(t)
    ys = cy + d * np.sin(t)
    # average across the rod width: three parallel lines
    prof = np.zeros_like(d)
    for off in (-3.0, 0.0, 3.0):
        px = xs - off * np.sin(t)
        py = ys + off * np.cos(t)
        prof += ndi.map_coordinates(sm, [py, px], order=1, mode='nearest')
    prof /= 3.0
    c = len(d) // 2
    plateau = np.median(prof[c - 2:c + 3])
    if plateau <= 0:
        return 0.0, cx, cy
    lvl = 0.45 * plateau
    lo = c
    while lo > 0 and prof[lo - 1] > lvl:
        lo -= 1
    hi = c
    while hi < len(d) - 1 and prof[hi + 1] > lvl:
        hi += 1
    mid = 0.5 * (d[lo] + d[hi])
    return float(d[hi] - d[lo] + 1), cx + mid * np.cos(t), cy + mid * np.sin(t)


def pick(m4):
    """m4: micrograph binned 4x. Detection runs at 8x, lengths are measured at 4x.
    Returns the rods (coordinates in 8x pixels) and the smoothed 8x image."""
    m = binned(m4, 2)
    n4 = flatten(m4, 60)
    sm4 = ndi.gaussian_filter(np.clip(n4, -4, 4), 2.4)
    sm4 = sm4 / (1.4826 * np.median(np.abs(sm4 - np.median(sm4))))
    n = flatten(m)
    # ice blobs: big bright round things; zero them (with a margin) before filtering
    # so they cannot radiate streaks, and never pick inside them
    sm = ndi.gaussian_filter(n, 2)
    blob = sm > 3.5 * np.std(sm)
    blob = morphology.remove_small_objects(blob, max_size=20)
    blob = ndi.binary_dilation(blob, iterations=3)
    n = np.clip(n, -4, 4)
    n[blob] = 0
    resp = np.stack([fftconvolve(n, k, mode='same') for k in BANK])
    rmax = resp.max(axis=0)
    rarg = resp.argmax(axis=0)
    med = np.median(rmax)
    sig = 1.4826 * np.median(np.abs(rmax - med))
    mask = (rmax > med + a.thresh * sig) & ~blob
    lab = measure.label(mask, connectivity=2)
    sm1 = ndi.gaussian_filter(n, 1.2)
    sm1 = sm1 / (1.4826 * np.median(np.abs(sm1 - np.median(sm1))))
    rods = []
    for p in measure.regionprops(lab, intensity_image=rmax):
        if p.area < 4:
            continue
        cy, cx = p.centroid
        coords = p.coords.astype(np.float32)
        angs = np.deg2rad(np.array(ANG)[rarg[p.coords[:, 0], p.coords[:, 1]]])
        t = 0.5 * np.arctan2(np.sin(2 * angs).mean(), np.cos(2 * angs).mean())
        ax = np.array([np.cos(t), np.sin(t)])
        perp = -(coords[:, 1] - cx) * ax[1] + (coords[:, 0] - cy) * ax[0]
        width = (perp.max() - perp.min() + 1) * apix
        if width > a.max_width:
            continue
        lpx, cx2, cy2 = profile_length(sm4, 2 * cx + 0.5, 2 * cy + 0.5, t,
                                       int(300 / apix4))
        length = lpx * apix4
        cx2, cy2 = (cx2 - 0.5) / 2, (cy2 - 0.5) / 2
        if length < a.min_len:
            continue
        rods.append(dict(x=cx2, y=cy2, psi=-np.degrees(t), length=length,
                         width=width, fom=float(p.intensity_mean), area=int(p.area)))
    # rods near the edge: the extraction box would leave the micrograph, and the
    # flattening filter is unreliable there (the edge classes of the first 2D run)
    mpx = a.margin / apix
    H, W = n.shape
    rods = [r for r in rods if mpx <= r['x'] <= W - mpx and mpx <= r['y'] <= H - mpx]
    # a rod found twice (two response blobs on one rod) keeps the stronger one
    rods.sort(key=lambda r: -r['fom'])
    kept = []
    for r in rods:
        dup = False
        for k in kept:
            dx, dy = r['x'] - k['x'], r['y'] - k['y']
            if dx * dx + dy * dy < (0.5 * min(r['length'], k['length']) / apix) ** 2:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept, sm1


def write_coords(path, rods, b):
    with open(path, 'w') as f:
        f.write('\n# rodpick.py\n\ndata_\n\nloop_\n_rlnCoordinateX #1\n'
                '_rlnCoordinateY #2\n_rlnAnglePsi #3\n_rlnAutopickFigureOfMerit #4\n'
                '_rlnHelicalTrackLengthAngst #5\n_rlnClassNumber #6\n')
        for r in rods:
            f.write('%.1f %.1f %.2f %.4f %.0f %d\n' % (
                (r['x'] + 0.5) * b, (r['y'] + 0.5) * b, r['psi'], r['fom'],
                r['length'], int(round(r['length'] / a.step))))
        f.write('\n')


def overlay(n, rods, path):
    lo, hi = np.percentile(n, [0.5, 99.5])
    g = np.clip((n - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(g).convert('RGB')
    im = im.resize((im.width * 2, im.height * 2), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for r in rods:
        t = np.deg2rad(-r['psi'])
        hl = 0.5 * r['length'] / apix
        x0, y0 = 2 * (r['x'] - hl * np.cos(t)), 2 * (r['y'] - hl * np.sin(t))
        x1, y1 = 2 * (r['x'] + hl * np.cos(t)), 2 * (r['y'] + hl * np.sin(t))
        d.line([x0, y0, x1, y1], fill=(255, 40, 40), width=1)
    im.save(path, quality=88)


items = []                      # (name for coordinate file, micrograph path or array)
if a.star:
    mics = [l.split()[0] for l in open(a.star) if 'MotionCorr/Movies/' in l][:a.n]
    items = [(m, os.path.join(a.root, m)) for m in mics]
for p in a.npy:
    items.append((os.path.basename(p), p))

list_rows, all_len, nmic = [], [], 0
for i, (name, src) in enumerate(items):
    if src.endswith('.npy'):
        m = np.load(src).astype(np.float32)      # expected at 4x binning
    else:
        m = binned(read_mrc(src), 4)
    rods, n = pick(m)
    nmic += 1
    all_len += [r['length'] for r in rods]
    stem = os.path.splitext(os.path.basename(name))[0]
    cpath = os.path.join(a.out, stem + '_rods.star')
    write_coords(cpath, rods, a.bin)
    list_rows.append((name, cpath))
    if i < a.overlay:
        overlay(n, rods, os.path.join(a.out, 'overlay_mic%d.jpg' % i))
    if nmic % 25 == 0:
        print('%d micrographs, %d rods so far' % (nmic, len(all_len)), flush=True)

with open(os.path.join(a.out, 'rods.star'), 'w') as f:
    f.write('\n# rodpick.py\n\ndata_coordinate_files\n\nloop_\n'
            '_rlnMicrographName #1\n_rlnMicrographCoordinates #2\n')
    for name, cpath in list_rows:
        f.write('%s %s\n' % (name, cpath))
    f.write('\n')

L = np.array(all_len)
print('micrographs %d, rods %d, %.1f rods per micrograph' % (nmic, len(L), len(L) / max(1, nmic)))
if len(L):
    print('rod length: median %.0f A, 10th pct %.0f A, 90th pct %.0f A'
          % tuple(np.percentile(L, [50, 10, 90])))
    print('rods by number of %.0f A repeats (n: count, percent):' % a.step)
    units = np.round(L / a.step).astype(int)
    for u in range(max(1, units.min()), min(units.max(), 16) + 1):
        c = int((units == u).sum())
        print('  %2d: %6d  %5.1f%%' % (u, c, 100.0 * c / len(L)))
    c = int((units > 16).sum())
    print('  >16: %5d  %5.1f%%' % (c, 100.0 * c / len(L)))
    # fine histogram in 10 A bins, to see whether lengths cluster at multiples of the repeat
    h, e = np.histogram(L, bins=np.arange(a.min_len, min(L.max(), 600) + 10, 10))
    print('length histogram, 10 A bins from %.0f A: %s' % (a.min_len, ' '.join(str(int(v)) for v in h)))
