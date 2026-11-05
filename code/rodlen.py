#!/usr/bin/env python3
# Per-particle rod length from the 2D alignment. 2D classification does not sort rods by
# length (a 3-bead rod aligns happily onto the middle of a 5-bead class), so the census is
# measured on the particles themselves: each particle is brought into its class's frame
# with the refined psi and offsets, the rod is turned horizontal using the class average's
# own axis, and the intensity profile along the axis gives the length and the bead count.
# The sign conventions of the shifts and the rotation are not assumed: the four
# combinations are tried on a few hundred particles and the one whose aligned average
# matches RELION's class average is used (printed as a check).
#   python3 -I rodlen.py --data Class2DSel/run_it020_data.star \
#       --classes Class2DSel/run_it020_classes.mrcs --root . --out RodLen [--top 8]
# Prints numbers only. Writes RodLen/lengths.txt (image name, class, length A, beads),
# RodLen/check.jpg (RELION average | aligned average, per class).
import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--root', default='.')
ap.add_argument('--out', required=True)
ap.add_argument('--top', type=int, default=8, help='classes used, by population')
ap.add_argument('--step', type=float, default=40.0, help='repeat along the rod, A')
ap.add_argument('--band', type=int, default=5, help='half width of the profile band, px')
ap.add_argument('--frac', type=float, default=0.4, help='length threshold, fraction of peak')
ap.add_argument('--n', type=int, default=10 ** 9, help='particles per class (debug)')
a = parser_args = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def read_star_blocks(path):
    """All loop blocks: {block name: (labels, rows)}."""
    blocks, name, labels, rows, inloop = {}, '', [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith('data_'):
            if name:
                blocks[name] = (labels, rows)
            name, labels, rows, inloop = s, [], [], False
        elif s == 'loop_':
            inloop = True
        elif inloop and s.startswith('_'):
            labels.append(s.split()[0])
        elif inloop and s and not s.startswith('#'):
            rows.append(s.split())
    if name:
        blocks[name] = (labels, rows)
    return blocks


def read_mrcs(path):
    with open(path, 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz, mode = (int(v) for v in np.frombuffer(hdr[:16], np.int32))
        ext = int(np.frombuffer(hdr[92:96], np.int32)[0])
        assert mode == 2, mode
        f.seek(1024 + ext)
        return np.fromfile(f, np.float32, nx * ny * nz).reshape(nz, ny, nx)


blocks = read_star_blocks(a.data)
olab, orows = blocks['data_optics']
apix = float(orows[0][olab.index('_rlnImagePixelSize')])
plab, prows = blocks['data_particles']
ic = plab.index('_rlnClassNumber')
ipsi = plab.index('_rlnAnglePsi')
iox, ioy = plab.index('_rlnOriginXAngst'), plab.index('_rlnOriginYAngst')
iname = plab.index('_rlnImageName')
byclass = defaultdict(list)
for r in prows:
    byclass[int(r[ic])].append(r)
classes = read_mrcs(a.classes)
nx = classes.shape[1]
c0 = nx / 2.0
top = sorted(byclass, key=lambda k: -len(byclass[k]))[:a.top]
print('%d particles, box %d px at %.3f A/px; classes used: %s'
      % (len(prows), nx, apix, ' '.join('%d(%d)' % (k, len(byclass[k])) for k in top)))

stacks = {}


def particle(name):
    k, path = name.split('@')
    if path not in stacks:
        stacks[path] = read_mrcs(os.path.join(a.root, path))
    return stacks[path][int(k) - 1]


def align(img, psi, ox, oy, ssign, rsign):
    sh = ndi.shift(img, (ssign * oy / apix, ssign * ox / apix), order=1, mode='constant')
    return ndi.rotate(sh, rsign * psi, reshape=False, order=1, mode='constant')


def axis_of(avg):
    """Orientation (deg) and centroid of the bright rod in a class average."""
    m = ndi.gaussian_filter(avg, 1.0)
    thr = m.mean() + 2.0 * m.std()
    w = np.clip(m - thr, 0, None)
    yy, xx = np.mgrid[0:nx, 0:nx]
    tot = w.sum() + 1e-9
    cx, cy = (w * xx).sum() / tot, (w * yy).sum() / tot
    mu20 = (w * (xx - cx) ** 2).sum() / tot
    mu02 = (w * (yy - cy) ** 2).sum() / tot
    mu11 = (w * (xx - cx) * (yy - cy)).sum() / tot
    theta = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    return theta, cx, cy


def profile(img, theta, cx, cy):
    """Turn the rod horizontal about its centroid and sum a band of rows."""
    r = ndi.rotate(ndi.shift(img, (c0 - cy, c0 - cx), order=1), theta, reshape=False, order=1)
    lo, hi = int(round(c0)) - a.band, int(round(c0)) + a.band + 1
    return r[lo:hi].sum(axis=0)


def measure(p):
    base = np.percentile(p, 25)
    p = p - base
    ps = ndi.gaussian_filter1d(p, 0.5 * a.step / apix)   # smoother than the bead repeat
    peak = ps.max()
    if peak <= 0:
        return 0.0, 0, None
    above = ps > a.frac * peak
    i = int(np.argmax(ps))
    l = i
    while l > 0 and above[l - 1]:
        l -= 1
    h = i
    while h < len(ps) - 1 and above[h + 1]:
        h += 1
    length = (h - l + 1) * apix
    p1 = ndi.gaussian_filter1d(p, 1.0)
    pk, _ = find_peaks(p1[l:h + 1], distance=max(2, int(0.6 * a.step / apix)),
                       prominence=0.15 * p1.max())
    # the threshold extent includes the bead tails (about +0.3 repeat). On the class
    # averages the beads are resolved, so outermost bead centre to centre plus one repeat
    # is unbiased there; the difference calibrates the extent measure for the noisy
    # single particles, where end beads are often lost in the noise.
    peaklen = (pk[-1] - pk[0]) * apix + a.step if len(pk) >= 2 else None
    return length, len(pk), peaklen


# 1. pick the alignment convention on the biggest class
k0 = top[0]
avg0 = classes[k0 - 1]
best, bestc = None, -2
sample = byclass[k0][:: max(1, len(byclass[k0]) // 300)][:300]
for ssign in (1, -1):
    for rsign in (1, -1):
        acc = np.zeros((nx, nx), np.float32)
        for r in sample:
            acc += align(particle(r[iname]), float(r[ipsi]), float(r[iox]), float(r[ioy]), ssign, rsign)
        c = np.corrcoef(acc.ravel(), avg0.ravel())[0, 1]
        print('convention shift %+d rotate %+d: correlation with the class average %.3f' % (ssign, rsign, c))
        if c > bestc:
            best, bestc = (ssign, rsign), c
ssign, rsign = best
print('using shift %+d rotate %+d' % best)

# 2. class averages: axis, length, beads
axes, corr = {}, {}
for k in top:
    theta, cx, cy = axis_of(classes[k - 1])
    axes[k] = (theta, cx, cy)
    L, nb, Lp = measure(profile(classes[k - 1], theta, cx, cy))
    corr[k] = (L - Lp) if Lp else 0.3 * a.step
    print('class %3d: %6d particles, average rod %3.0f A = %.1f repeats, %d beads (extent %.0f A, correction %.0f A)'
          % (k, len(byclass[k]), Lp or L, (Lp or L) / a.step, nb, L, corr[k]))

# 3. per-particle lengths, with an aligned average per class as a check
hist_all = defaultdict(int)
tiles = []
with open(os.path.join(a.out, 'lengths.txt'), 'w') as f:
    f.write('# image class length_A beads\n')
    for k in top:
        theta, cx, cy = axes[k]
        acc = np.zeros((nx, nx), np.float32)
        hist, lens = defaultdict(int), []
        for r in byclass[k][:a.n]:
            al = align(particle(r[iname]), float(r[ipsi]), float(r[iox]), float(r[ioy]), ssign, rsign)
            acc += al
            L, nb, _ = measure(profile(al, theta, cx, cy))
            L = max(0.0, L - corr[k])
            u = int(round(L / a.step))
            hist[u] += 1
            hist_all[u] += 1
            lens.append(L)
            f.write('%s %d %.0f %d\n' % (r[iname], k, L, nb))
        lens = np.array(lens)
        print('class %3d particles: median %3.0f A, repeats ' % (k, np.median(lens))
              + ' '.join('%d:%d' % (u, hist[u]) for u in sorted(hist)))
        tiles.append((classes[k - 1], acc / max(1, len(lens))))
n = sum(hist_all.values())
print('all particles, rods by number of %.0f A repeats (n: count, percent):' % a.step)
for u in sorted(hist_all):
    print('  %2d: %6d  %5.1f%%' % (u, hist_all[u], 100.0 * hist_all[u] / n))

sheet = Image.new('L', (2 * nx, nx * len(tiles)))
for j, (relion_avg, mine) in enumerate(tiles):
    for i, t in enumerate((relion_avg, mine)):
        lo, hi = np.percentile(t, [0.5, 99.5])
        g = np.clip((t - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(g), (i * nx, j * nx))
sheet = sheet.resize((sheet.width * 3, sheet.height * 3))
sheet.save(os.path.join(a.out, 'check.jpg'), quality=88)
print('wrote %s/lengths.txt and %s/check.jpg' % (a.out, a.out))
