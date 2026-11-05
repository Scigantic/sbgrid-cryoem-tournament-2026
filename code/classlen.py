#!/usr/bin/env python3
# Picker length per 2D class: does rodpick's per-rod length track the class's bead
# count? Reads a RELION particles star that still carries _rlnHelicalTrackLengthAngst.
#   python3 -I classlen.py Class2DSel/run_it020_data.star 3,56,29,17,54,19,38,43 40
import sys
from collections import defaultdict
import numpy as np
lab, rows, inloop, blk = [], [], False, ''
for line in open(sys.argv[1]):
    s = line.strip()
    if s.startswith('data_'):
        blk, inloop, lab = s, False, []
    elif s == 'loop_':
        inloop = True
    elif blk == 'data_particles' and inloop and s.startswith('_'):
        lab.append(s.split()[0])
    elif blk == 'data_particles' and inloop and s and not s.startswith('#'):
        rows.append(s.split())
if '_rlnHelicalTrackLengthAngst' not in lab:
    print('no picker length column; columns are', len(lab))
    sys.exit(0)
il, ic = lab.index('_rlnHelicalTrackLengthAngst'), lab.index('_rlnClassNumber')
step = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
byc = defaultdict(list)
for r in rows:
    byc[int(r[ic])].append(float(r[il]))
print('%d particles; picker length by 2D class (A): n, p10 p25 p50 p75 p90, mean' % len(rows))
for k in [int(v) for v in sys.argv[2].split(',')]:
    v = np.array(byc.get(k, []))
    if len(v) == 0:
        continue
    q = np.percentile(v, [10, 25, 50, 75, 90])
    h = np.bincount(np.clip(np.round(v / step).astype(int), 0, 12), minlength=13)
    print('class %3d: %5d  %4.0f %4.0f %4.0f %4.0f %4.0f  mean %5.1f  repeats '
          % (k, len(v), *q, v.mean()) + ' '.join('%d:%d' % (i, h[i]) for i in range(13) if h[i]))
