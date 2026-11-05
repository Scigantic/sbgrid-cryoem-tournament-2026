#!/usr/bin/env python3
# Keep the particles of the given 2D classes. Copies a RELION data star verbatim
# (optics block included) and drops the rows of the particles loop whose
# _rlnClassNumber is not in the list. Prints counts only.
#   python3 -I select_classes.py run_it012_data.star out.star 17,3,33
import sys

src, dst, keep = sys.argv[1], sys.argv[2], set(sys.argv[3].split(','))
out, labels, inloop, block = [], [], False, ''
kept = total = 0
for line in open(src):
    s = line.strip()
    if s.startswith('data_'):
        block, inloop, labels = s, False, []
    elif s == 'loop_':
        inloop, labels = True, []
    elif inloop and s.startswith('_'):
        labels.append(s.split()[0])
    elif inloop and s and not s.startswith('#') and block == 'data_particles':
        total += 1
        v = s.split()
        if v[labels.index('_rlnClassNumber')] not in keep:
            continue
        kept += 1
    out.append(line)
open(dst, 'w').writelines(out)
print('kept %d of %d particles from %d classes' % (kept, total, len(keep)))
