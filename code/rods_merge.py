#!/usr/bin/env python3
# Merge the rods.star files of parallel rodpick.py parts (OUTDIR/p*/rods.star) into
# OUTDIR/rods.star and print the same length census rodpick.py prints for one run.
#   python3 -I rods_merge.py OUTDIR STEP_A
import glob
import os
import sys

import numpy as np

out, step = sys.argv[1], float(sys.argv[2])
rows = []
for part in sorted(glob.glob(os.path.join(out, 'p*', 'rods.star'))):
    for line in open(part):
        s = line.split()
        if len(s) == 2 and s[1].endswith('_rods.star'):
            rows.append((s[0], s[1]))
with open(os.path.join(out, 'rods.star'), 'w') as f:
    f.write('\n# rodpick.py (merged)\n\ndata_coordinate_files\n\nloop_\n'
            '_rlnMicrographName #1\n_rlnMicrographCoordinates #2\n')
    for name, cpath in rows:
        f.write('%s %s\n' % (name, cpath))
    f.write('\n')

L = []
for _, cpath in rows:
    cols, body = [], []
    for line in open(cpath):
        s = line.strip()
        if s.startswith('_rln'):
            cols.append(s.split()[0])
        elif s and not s.startswith(('#', 'data_', 'loop_')):
            body.append(s.split())
    k = cols.index('_rlnHelicalTrackLengthAngst')
    L += [float(b[k]) for b in body]
L = np.array(L)
nmic = len(rows)
print('micrographs %d, rods %d, %.1f rods per micrograph' % (nmic, len(L), len(L) / max(1, nmic)))
if len(L):
    print('rod length: median %.0f A, 10th pct %.0f A, 90th pct %.0f A' % tuple(np.percentile(L, [50, 10, 90])))
    print('rods by number of %.0f A repeats (n: count, percent):' % step)
    units = np.round(L / step).astype(int)
    for u in range(max(1, units.min()), min(units.max(), 16) + 1):
        c = int((units == u).sum())
        print('  %2d: %6d  %5.1f%%' % (u, c, 100.0 * c / len(L)))
    c = int((units > 16).sum())
    print('  >16: %5d  %5.1f%%' % (c, 100.0 * c / len(L)))
