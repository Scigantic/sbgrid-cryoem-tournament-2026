#!/usr/bin/env python3
# Like remap_poses.py, but for particles re-subtracted into a different box size:
# particle rows (poses, half sets, CTF) come from the refined file with _rlnImageName
# pointed at the new stacks, and the optics block (image size) comes from the new file.
#   remap_box.py REFINED.star NEW_SUBTRACTED.star OUT.star
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
ref_bl, new_bl = blocks(ref_path), blocks(new_path)
_, rl, rhead, rrows = particles(ref_bl)
_, nl, _, nrows = particles(new_bl)


def optics(bl):
    for b in bl:
        if b[0].lower().startswith('data_optics'):
            return b
    sys.exit('no data_optics block')


ro_name, ro_labels, _, ro_rows = optics(ref_bl)
no_name, no_labels, no_head, no_rows = optics(new_bl)
gi, gr = ro_labels.index('_rlnOpticsGroupName'), no_labels.index('_rlnOpticsGroupName')
if [r[gi] for r in ro_rows] != [r[gr] for r in no_rows]:
    sys.exit('optics groups differ between the two files')
si = no_labels.index('_rlnImageSize')
print('optics groups %d, new image size %s' % (len(no_rows), sorted({r[si] for r in no_rows})))

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
    f.writelines(no_head)
    for r in no_rows:
        f.write(' '.join(r) + '\n')
    f.write('\n')
    f.writelines(rhead)
    for r in rrows:
        f.write(' '.join(r) + '\n')
    f.write('\n')
