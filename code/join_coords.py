#!/usr/bin/env python3
# Keep the rows of a particle star whose particle (micrograph, x, y) also appears in a
# selection star made from another extraction of the same coordinates (for example the
# bin-4 2D classification's selected classes applied to the bin-2 stack).
#   python3 -I join_coords.py selected.star all.star out.star
import os
import sys

sel, allf, out = sys.argv[1:4]


def read(path):
    head, cols, rows, intable = [], [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith('data_particles'):
            intable = True
        if not intable:
            head.append(line)
            continue
        if s.startswith('_rln'):
            cols.append(s.split()[0]); head.append(line)
        elif s and not s.startswith(('data_', 'loop_', '#')):
            rows.append(line)
        else:
            if not rows:
                head.append(line)
    return head, cols, rows


def key(cols, line):
    s = line.split()
    m = os.path.basename(s[cols.index('_rlnMicrographName')])
    x = round(float(s[cols.index('_rlnCoordinateX')]))
    y = round(float(s[cols.index('_rlnCoordinateY')]))
    return (m, x, y)


_, scols, srows = read(sel)
keep = set(key(scols, r) for r in srows)
head, cols, rows = read(allf)
kept = [r for r in rows if key(cols, r) in keep]
with open(out, 'w') as f:
    f.writelines(head)
    f.writelines(kept)
    f.write('\n')
print('kept %d of %d (selection had %d)' % (len(kept), len(rows), len(srows)))
