#!/usr/bin/env python3
# Give a refined particle STAR file new image names, matched on _rlnImageOriginalName.
# Used after re-running the same particle subtraction: the stacks are rebuilt on a new node,
# so the refined poses are kept and only _rlnImageName is pointed at the new stacks.
#   remap_poses.py REFINED.star NEW_SUBTRACTED.star OUT.star
import sys


def blocks(path):
    """Yield (block_name, labels, header_lines, rows) for each data block."""
    name, labels, head, rows, inloop = None, [], [], [], False
    out = []
    for line in open(path):
        s = line.strip()
        if s.startswith('data_'):
            if name is not None:
                out.append((name, labels, head, rows))
            name, labels, head, rows, inloop = s, [], [line], [], False
        elif s.startswith('loop_'):
            inloop = True
            head.append(line)
        elif inloop and s.startswith('_'):
            labels.append(s.split()[0])
            head.append(line)
        elif inloop and s and not s.startswith('#'):
            rows.append(s.split())
        elif not rows:
            head.append(line)
    if name is not None:
        out.append((name, labels, head, rows))
    return out


def particles(bl):
    for b in bl:
        if b[0].lower().startswith('data_particles'):
            return b
    sys.exit('no data_particles block')


ref_path, new_path, out_path = sys.argv[1:4]
ref_bl = blocks(ref_path)
_, rl, _, rrows = particles(ref_bl)
_, nl, _, nrows = particles(blocks(new_path))
ri, ro = rl.index('_rlnImageName'), rl.index('_rlnImageOriginalName')
ni, no = nl.index('_rlnImageName'), nl.index('_rlnImageOriginalName')
new = {r[no]: r[ni] for r in nrows}
if len(new) != len(nrows):
    sys.exit('duplicate original names in the new file')
miss = 0
for r in rrows:
    n = new.get(r[ro])
    if n is None:
        miss += 1
    else:
        r[ri] = n
print('refined %d, new %d, matched %d, missing %d' % (len(rrows), len(nrows), len(rrows) - miss, miss))
if miss or len(rrows) != len(nrows):
    sys.exit('mismatch, nothing written')
with open(out_path, 'w') as f:
    for name, labels, head, rows in ref_bl:
        f.writelines(head)
        for r in rows:
            f.write(' '.join(r) + '\n')
        f.write('\n')
