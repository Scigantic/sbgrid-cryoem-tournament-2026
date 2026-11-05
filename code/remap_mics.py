#!/usr/bin/env python3
# Point a refined particle star and a CTF micrograph star at the micrographs written by
# RELION's own motion correction, so Bayesian polishing (which needs RELION's per-micrograph
# motion metadata) can run on particles that were picked and refined on MotionCor3 output.
# Micrographs are matched by stem: <stem>_DW.mrc or <stem>.mrc in the old name, the stem of
# the movie in the new corrected_micrographs.star. Particles and micrographs with no match
# are dropped and counted. CTF values are carried over unchanged (same movies, same optics).
#   python3 -I remap_mics.py corrected_micrographs.star micrographs_ctf.star run_data.star OUTDIR
import os
import sys


def stem(path):
    b = os.path.basename(path)
    for suf in ('_DW.mrc', '.mrc'):
        if b.endswith(suf):
            return b[:-len(suf)]
    return os.path.splitext(b)[0]


def blocks(path):
    # list of (name, header_lines, columns, rows); a block without a loop has no rows
    # comment lines (RELION's "# version") seen before a block are kept in its header
    out, cur, pending = [], None, []
    for line in open(path):
        s = line.strip()
        if s.startswith('data_'):
            cur = [s, pending + [line], [], []]
            pending = []
            out.append(cur)
            continue
        if s.startswith('#') and (cur is None or cur[3]):
            pending.append(line)
            continue
        if cur is None:
            continue
        if s.startswith('_rln'):
            cur[2].append(s.split()[0])
            cur[1].append(line)
        elif s and not s.startswith(('loop_', '#')):
            cur[3].append(line.split())
        elif not cur[3]:
            cur[1].append(line)
    return out


def write(path, bl, rows_by_name):
    with open(path, 'w') as f:
        for name, head, cols, rows in bl:
            f.writelines(head)
            for r in rows_by_name.get(name, rows):
                f.write(' '.join(r) + '\n')
            f.write('\n')


corr, ctf, data, outdir = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)

new = {}
for name, head, cols, rows in blocks(corr):
    if name == 'data_micrographs':
        k = cols.index('_rlnMicrographName')
        for r in rows:
            new[stem(r[k])] = r[k]
print('corrected micrographs: %d' % len(new))


def remap(path, block, outname):
    bl = blocks(path)
    done = {}
    for name, head, cols, rows in bl:
        if name != block:
            continue
        k = cols.index('_rlnMicrographName')
        keep = []
        for r in rows:
            s = stem(r[k])
            if s in new:
                r = list(r)
                r[k] = new[s]
                keep.append(r)
        done[name] = keep
        print('%s: kept %d of %d %s' % (outname, len(keep), len(rows), block[5:]))
    if not done:
        sys.exit('no %s block in %s' % (block, path))
    write(os.path.join(outdir, outname), bl, done)
    return len(next(iter(done.values())))


nm = remap(ctf, 'data_micrographs', 'micrographs_ctf.star')
np_ = remap(data, 'data_particles', 'particles.star')
if nm == 0 or np_ == 0:
    sys.exit('nothing matched: check the micrograph names')
