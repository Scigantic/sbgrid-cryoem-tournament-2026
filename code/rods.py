#!/usr/bin/env python3
# Rod census. RELION's helical autopicker writes one segment every --min_distance
# Angstroms along each rod it finds, tagged with the rod (tube) number. Collapse
# the segments to one coordinate per rod (its midpoint) with the rod's length and
# direction, write RELION coordinate files for whole-rod extraction, and print a
# length histogram in units of the repeat. Prints numbers only, no file names.
# Plain (non-helical) picks work too: picks closer than 1.6 repeats are joined into rods.
#   python3 -I rods.py AutoPickRods/autopick.star 40 RodsPick [angpix]
import os
import sys
from collections import defaultdict

import numpy as np


def read_star(path):
    """One loop block: return (labels, rows as list of str lists)."""
    labels, rows, inloop = [], [], False
    for line in open(path):
        s = line.strip()
        if s == 'loop_':
            inloop, labels, rows = True, [], []
            continue
        if not inloop:
            continue
        if s.startswith('_'):
            labels.append(s.split()[0])
        elif s and not s.startswith('#'):
            rows.append(s.split())
        elif rows:
            break
    return labels, rows


def write_star(path, labels, rows, block='data_'):
    with open(path, 'w') as f:
        f.write('\n# written by rods.py\n\n%s\n\nloop_\n' % block)
        for i, l in enumerate(labels):
            f.write('%s #%d\n' % (l, i + 1))
        for r in rows:
            f.write(' '.join(str(v) for v in r) + '\n')
        f.write('\n')


def cluster(xy, link):
    """Union-find: picks closer than `link` pixels belong to the same rod."""
    n = len(xy)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    ii, jj = np.nonzero(np.triu(d2 <= link * link, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(a) for a in range(n)]


coord_list, step, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
angpix = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8837
os.makedirs(outdir, exist_ok=True)
labels, rows = read_star(coord_list)
imic = labels.index('_rlnMicrographName')
icoord = labels.index('_rlnMicrographCoordinates')

lengths, out_rows, nseg, singles = [], [], 0, 0
for r in rows:
    mic, cfile = r[imic], r[icoord]
    if not os.path.exists(cfile):
        continue
    lab, segs = read_star(cfile)
    if not segs:
        continue
    ix, iy = lab.index('_rlnCoordinateX'), lab.index('_rlnCoordinateY')
    ifom = lab.index('_rlnAutopickFigureOfMerit')
    ip = None
    for name in ('_rlnAnglePsiPrior', '_rlnAnglePsi'):
        if name in lab:
            ip = lab.index(name)
    tubes = defaultdict(list)
    if '_rlnHelicalTubeID' in lab:
        it = lab.index('_rlnHelicalTubeID')
        ids = [s[it] for s in segs]
    else:
        # plain picks: join picks closer than 1.6 repeats into one rod
        xy = np.array([[float(s[ix]), float(s[iy])] for s in segs])
        ids = cluster(xy, 1.6 * step / angpix)
    for s, t in zip(segs, ids):
        psi = float(s[ip]) if ip is not None else 0.0
        tubes[t].append((float(s[ix]), float(s[iy]), psi, float(s[ifom])))
    nseg += len(segs)
    rod_rows = []
    for t, pts in tubes.items():
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        n = len(pts)
        if n > 1:
            # direction and extent from the picks themselves (principal axis)
            c = np.stack([xs - xs.mean(), ys - ys.mean()])
            w, v = np.linalg.eigh(c @ c.T)
            ax = v[:, np.argmax(w)]
            proj = ax[0] * c[0] + ax[1] * c[1]
            length = (proj.max() - proj.min()) * angpix + step
            psi = np.degrees(np.arctan2(-ax[1], ax[0]))
        else:
            length = step
            psi = pts[0][2]
            singles += 1
        fom = float(np.mean([p[3] for p in pts]))
        rod_rows.append(['%.1f' % xs.mean(), '%.1f' % ys.mean(),
                         '%.2f' % psi, '%.4f' % fom, '%.0f' % length, n])
        lengths.append(length)
    stem = os.path.splitext(os.path.basename(mic))[0]
    rfile = os.path.join(outdir, stem + '_rods.star')
    write_star(rfile, ['_rlnCoordinateX', '_rlnCoordinateY', '_rlnAnglePsi',
                       '_rlnAutopickFigureOfMerit',
                       '_rlnHelicalTrackLengthAngst', '_rlnClassNumber'],
               rod_rows)
    out_rows.append([mic, rfile])

write_star(os.path.join(outdir, 'rods.star'),
           ['_rlnMicrographName', '_rlnMicrographCoordinates'], out_rows,
           block='data_coordinate_files')
L = np.array(lengths)
print('micrographs %d, picks %d, rods %d (%d single picks), %.1f rods per micrograph'
      % (len(out_rows), nseg, len(L), singles, len(L) / max(1, len(out_rows))))
if len(L):
    print('rod length: median %.0f A, 10th pct %.0f A, 90th pct %.0f A'
          % tuple(np.percentile(L, [50, 10, 90])))
    print('rods by number of %.0f A repeats (n: count, percent):' % step)
    units = np.round(L / step).astype(int)
    for u in range(units.min(), min(units.max(), 20) + 1):
        c = int((units == u).sum())
        print('  %2d: %6d  %5.1f%%' % (u, c, 100.0 * c / len(L)))
    c = int((units > 20).sum())
    if c:
        print('  >20: %5d  %5.1f%%' % (c, 100.0 * c / len(L)))
