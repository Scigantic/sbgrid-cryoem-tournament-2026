#!/usr/bin/env python3
# Concatenate the particle stars of several chunks into one. Each chunk's image paths are
# relative to its own project, so PREFIX/ is put in front of the stack path in
# _rlnImageName (micrograph names already differ between chunks, the movies being
# disjoint). The optics block is the first star's; every other star must have the same
# optics labels and the same pixel and image sizes, and all particles go to group 1.
# Prints counts only.
#   python3 -I merge_stars.py out.star full_t00 A/particles.star full_t01 B/particles.star ...
import sys


def read(path):
    blocks, name, labels, rows, inloop = {}, None, [], [], False
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


out, pairs = sys.argv[1], sys.argv[2:]
if not pairs or len(pairs) % 2:
    sys.exit('usage: merge_stars.py out.star PREFIX STAR [PREFIX STAR ...]')
optics = plabels = None
rows = []
CHECK = ('_rlnImagePixelSize', '_rlnImageSize', '_rlnVoltage', '_rlnSphericalAberration')
for prefix, path in zip(pairs[0::2], pairs[1::2]):
    b = read(path)
    olab, orows = b['data_optics']
    lab, prow = b['data_particles']
    if len(orows) != 1:
        sys.exit('%s: %d optics groups, expected 1' % (prefix, len(orows)))
    if optics is None:
        optics, plabels = (olab, orows), lab
    else:
        if olab != optics[0] or lab != plabels:
            sys.exit('%s: labels differ from the first star' % prefix)
        for c in CHECK:
            if c in olab and orows[0][olab.index(c)] != optics[1][0][olab.index(c)]:
                sys.exit('%s: %s differs from the first star' % (prefix, c))
    im, og = lab.index('_rlnImageName'), lab.index('_rlnOpticsGroup')
    for r in prow:
        n, stack = r[im].split('@', 1)
        r[im] = '%s@%s/%s' % (n, prefix, stack)
        r[og] = '1'
        rows.append(r)
    print('%s: %d particles' % (prefix, len(prow)))

with open(out, 'w') as f:
    for name, (lab, rr) in (('data_optics', optics), ('data_particles', (plabels, rows))):
        f.write('\n# version 30001\n\n%s\n\nloop_\n' % name)
        f.writelines('%s #%d\n' % (l, i + 1) for i, l in enumerate(lab))
        f.writelines(' '.join(r) + '\n' for r in rr)
        f.write('\n')
print('merged %d particles from %d stars' % (len(rows), len(pairs) // 2))
