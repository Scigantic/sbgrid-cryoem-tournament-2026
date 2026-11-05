#!/usr/bin/env python3
# Drop the particles of poor micrographs: CTF fit worse than MAXRES (A) or defocus above
# DFMAX (A, the larger of U and V), read from one or more micrographs_ctf.star files and
# matched by micrograph basename. The particle star is copied otherwise verbatim (optics
# block, orientations and all). Prints counts only.
#   python3 -I filter_mics.py in.star out.star MAXRES DFMAX ctf1.star [ctf2.star ...]
import os
import sys


def table(path, block):
    labels, rows, inblock, inloop = [], [], False, False
    for line in open(path):
        s = line.strip()
        if s.startswith('data_'):
            inblock, inloop = (s == block), False
        elif inblock and s == 'loop_':
            inloop, labels = True, []
        elif inblock and inloop and s.startswith('_'):
            labels.append(s.split()[0])
        elif inblock and inloop and s and not s.startswith('#'):
            rows.append(s.split())
    return labels, rows


src, dst, maxres, dfmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
good, nmic = set(), 0
for ctf in sys.argv[5:]:
    lab, rows = table(ctf, 'data_micrographs')
    im, ir = lab.index('_rlnMicrographName'), lab.index('_rlnCtfMaxResolution')
    iu, iv = lab.index('_rlnDefocusU'), lab.index('_rlnDefocusV')
    for r in rows:
        nmic += 1
        if float(r[ir]) <= maxres and max(float(r[iu]), float(r[iv])) <= dfmax:
            good.add(os.path.basename(r[im]))
print('micrographs kept %d of %d (CTF fit <= %g A, defocus <= %g A)' % (len(good), nmic, maxres, dfmax))

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
        if os.path.basename(s.split()[labels.index('_rlnMicrographName')]) not in good:
            continue
        kept += 1
    out.append(line)
open(dst, 'w').writelines(out)
print('particles kept %d of %d' % (kept, total))
