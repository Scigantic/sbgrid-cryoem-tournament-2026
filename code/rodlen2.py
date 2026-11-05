#!/usr/bin/env python3
# Per-particle bead count by template fitting. rodlen.py thresholded each particle's noisy
# axial profile and gave a spread of +-1.5 repeats per class, too wide to be trusted. Here
# each particle is aligned into its class frame (same convention check), turned horizontal
# with the class average's axis, and its band profile is correlated against a family of
# n-bead templates (Gaussian beads, step and width fitted on the class average) at every
# axial shift. The n with the best normalised correlation is the bead count; the margin
# over the runner-up says how sure. Two controls are printed: the same fit on a background
# band of each particle, off the rod (what noise alone yields), and a sheet of the
# aligned averages of the particles assigned to each n within each class (sorted.jpg):
# if the sorting is real, the n-bead average of a class shows n sharp beads.
#   python3 -I rodlen2.py --data run_it020_data.star --classes run_it020_classes.mrcs \
#       --root . --out RodLen2 [--top 8 --nmax 10 --band 5]
# Prints numbers only. Writes RodLen2/beads.txt (image name, class, n, ncc, margin),
# RodLen2/sorted.jpg.
import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--root', default='.')
ap.add_argument('--out', required=True)
ap.add_argument('--top', type=int, default=8, help='classes used, by population')
ap.add_argument('--nmax', type=int, default=10, help='largest bead count tried')
ap.add_argument('--band', type=int, default=5, help='half width of the profile band, px')
ap.add_argument('--off', type=int, default=18, help='background band offset from the rod, px')
ap.add_argument('--n', type=int, default=10 ** 9, help='particles per class (debug)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def read_star_blocks(path):
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


def horizontal(img, theta, cx, cy):
    return ndi.rotate(ndi.shift(img, (c0 - cy, c0 - cx), order=1), theta, reshape=False, order=1)


def band(r, off=0):
    lo, hi = int(round(c0)) + off - a.band, int(round(c0)) + off + a.band + 1
    p = r[lo:hi].sum(axis=0)
    return p - np.percentile(p, 25)


def ncc_rows(T):
    """Row-normalise a template bank for correlation."""
    T = T - T.mean(axis=1, keepdims=True)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)


def templates(step_px, sig_px, shifts):
    """(nmax, nshift, nx) bank of n-bead profiles, beads centred at c0 + shift."""
    x = np.arange(nx)
    T = np.zeros((a.nmax, len(shifts), nx), np.float32)
    for n in range(1, a.nmax + 1):
        for j, s in enumerate(shifts):
            for b in range(n):
                cb = c0 + s + (b - (n - 1) / 2.0) * step_px
                T[n - 1, j] += np.exp(-(x - cb) ** 2 / (2 * sig_px ** 2))
    return T


def fit(p, bank):
    """Best n (1-based), its ncc, margin over the best other n, shift index."""
    q = p - p.mean()
    q = q / (np.linalg.norm(q) + 1e-9)
    c = bank @ q                       # (nmax, nshift)
    per_n = c.max(axis=1)
    order = np.argsort(per_n)[::-1]
    n = int(order[0]) + 1
    return n, float(per_n[order[0]]), float(per_n[order[0]] - per_n[order[1]]), int(c[order[0]].argmax())


# 1. alignment convention on the biggest class
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

# 2. per class: axis, then bead step and width fitted on the class average
shifts = np.arange(-14, 15)
cls = {}
for k in top:
    theta, cx, cy = axis_of(classes[k - 1])
    pavg = band(horizontal(classes[k - 1], theta, cx, cy))
    bestfit = None
    for step_A in np.arange(34, 47, 1.0):
        for sig_A in np.arange(8, 21, 2.0):
            bank = ncc_rows(templates(step_A / apix, sig_A / apix, shifts).reshape(-1, nx)).reshape(a.nmax, len(shifts), nx)
            n, c, m, j = fit(pavg, bank)
            if bestfit is None or c > bestfit[0]:
                bestfit = (c, n, m, step_A, sig_A, bank)
    c, n, m, step_A, sig_A, bank = bestfit
    cls[k] = (theta, cx, cy, bank, step_A, sig_A)
    print('class %3d: %6d particles, average fits %d beads (ncc %.3f, margin %.3f), step %.0f A, bead sigma %.0f A'
          % (k, len(byclass[k]), n, c, m, step_A, sig_A))

# 3. per particle
hist_all, hist_sure, hist_null = defaultdict(int), defaultdict(int), defaultdict(int)
tiles = {}
snr = []
with open(os.path.join(a.out, 'beads.txt'), 'w') as f:
    f.write('# image class n ncc margin\n')
    for k in top:
        theta, cx, cy, bank, step_A, sig_A = cls[k]
        hist, hsure, hnull, margins = defaultdict(int), defaultdict(int), defaultdict(int), []
        acc = defaultdict(lambda: np.zeros((nx, nx), np.float32))
        cnt = defaultdict(int)
        for r in byclass[k][:a.n]:
            al = align(particle(r[iname]), float(r[ipsi]), float(r[iox]), float(r[ioy]), ssign, rsign)
            h = horizontal(al, theta, cx, cy)
            p = band(h)
            n, c, m, j = fit(p, bank)
            nn, cn, mn, jn = fit(band(h, a.off), bank)
            hist[n] += 1
            hist_all[n] += 1
            hnull[nn] += 1
            hist_null[nn] += 1
            if m >= 0.05:
                hsure[n] += 1
                hist_sure[n] += 1
            margins.append(m)
            snr.append(p.max() / (p[: a.band].std() + p[-a.band:].std() + 1e-9) * 2)
            # bring the particle to the template's shift so the per-n averages are sharp
            acc[n] += ndi.shift(h, (0, -shifts[j]), order=1)
            cnt[n] += 1
            f.write('%s %d %d %.3f %.3f\n' % (r[iname], k, n, c, m))
        margins = np.array(margins)
        print('class %3d particles: beads ' % k + ' '.join('%d:%d' % (u, hist[u]) for u in sorted(hist))
              + ' | margin>=0.05 ' + ' '.join('%d:%d' % (u, hsure[u]) for u in sorted(hsure))
              + ' | background band ' + ' '.join('%d:%d' % (u, hnull[u]) for u in sorted(hnull)))
        print('class %3d margin quartiles %.3f %.3f %.3f' % (k, *np.percentile(margins, [25, 50, 75])))
        tiles[k] = {n: (acc[n] / cnt[n], cnt[n]) for n in acc}
n = sum(hist_all.values())
ns = sum(hist_sure.values())
nn = sum(hist_null.values())
print('rod band peak over edge noise, median %.2f (a crude per-particle signal to noise)' % np.median(snr))
print('all particles, beads (n: count percent | margin>=0.05 count percent | background band count percent):')
for u in range(1, a.nmax + 1):
    print('  %2d: %6d %5.1f%% | %6d %5.1f%% | %6d %5.1f%%'
          % (u, hist_all[u], 100.0 * hist_all[u] / n, hist_sure[u], 100.0 * hist_sure[u] / max(1, ns),
             hist_null[u], 100.0 * hist_null[u] / max(1, nn)))

# 4. sorted sheet: rows = classes, columns = n; tile = aligned average of that class's n-bead particles
sheet = Image.new('L', (nx * (a.nmax + 1), nx * len(top)))
d = ImageDraw.Draw(sheet)
for j, k in enumerate(top):
    theta, cx, cy = cls[k][:3]
    col = [horizontal(classes[k - 1], theta, cx, cy)] + [tiles[k].get(u, (None, 0))[0] for u in range(1, a.nmax + 1)]
    for i, t in enumerate(col):
        if t is None:
            continue
        lo, hi = np.percentile(t, [0.5, 99.5])
        g = np.clip((t - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(g), (i * nx, j * nx))
        label = 'c%d' % k if i == 0 else '%d:%d' % (i, tiles[k][i][1])
        d.text((i * nx + 2, j * nx + 1), label, fill=255)
sheet = sheet.resize((sheet.width * 3, sheet.height * 3))
sheet.save(os.path.join(a.out, 'sorted.jpg'), quality=88)
print('wrote %s/beads.txt and %s/sorted.jpg' % (a.out, a.out))
