#!/usr/bin/env python3
# Keep the particles whose picker length (_rlnHelicalTrackLengthAngst) lies in [lo, hi].
#   python3 -I select_by_length.py in.star out.star LO HI
import sys
src, dst, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
out, lab, blk, inloop, kept, total = [], [], '', False, 0, 0
for line in open(src):
    s = line.strip()
    if s.startswith('data_'):
        blk, inloop, lab = s, False, []
    elif s == 'loop_':
        inloop = True
    elif blk == 'data_particles' and inloop and s.startswith('_'):
        lab.append(s.split()[0])
    elif blk == 'data_particles' and inloop and s and not s.startswith('#'):
        total += 1
        v = float(s.split()[lab.index('_rlnHelicalTrackLengthAngst')])
        if not (lo <= v <= hi):
            continue
        kept += 1
    out.append(line)
open(dst, 'w').writelines(out)
print('kept %d of %d particles with picker length %.0f-%.0f A' % (kept, total, lo, hi))
