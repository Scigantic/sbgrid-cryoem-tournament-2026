#!/usr/bin/env python3
# Choose the 3D classes to keep: the class with the best estimated resolution, plus every
# class within SLACK A of it. Prints the class table and the choice, writes the kept
# particles to OUT and a random subset of the same size (fixed seed) of all classified
# particles to OUT_RANDOM, the control that tells selection apart from simply using fewer
# particles. Also writes one star per class (OUTDIR/class_NN.star) for the heterogeneity
# questions.
#   python3 -I select_3d.py run_it025_model.star run_it025_data.star SLACK OUT OUT_RANDOM OUTDIR
import os
import random
import sys

model, data, slack, out, outr, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), *sys.argv[4:7]


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


lab, rows = table(model, 'data_model_classes')
idist, ires = lab.index('_rlnClassDistribution'), lab.index('_rlnEstimatedResolution')
cls = [(k + 1, float(r[idist]), float(r[ires])) for k, r in enumerate(rows)]
best = min(c[2] for c in cls if c[1] > 0)
keep = {c[0] for c in cls if c[1] > 0 and c[2] <= best + slack}
for k, d, res in cls:
    print('class %d: %5.1f%% of particles, estimated %.2f A%s' % (k, 100 * d, res, '  kept' if k in keep else ''))

# split the data star into header (everything up to the particle rows) and particle rows
head, prow, labels, inloop, block = [], [], [], False, ''
for line in open(data):
    s = line.strip()
    if s.startswith('data_'):
        block, inloop, labels = s, False, []
    elif s == 'loop_':
        inloop, labels = True, []
    elif inloop and s.startswith('_'):
        labels.append(s.split()[0])
    elif inloop and s and not s.startswith('#') and block == 'data_particles':
        prow.append(line)
        continue
    if not prow:
        head.append(line)
icl = labels.index('_rlnClassNumber')
bycls = {}
for line in prow:
    bycls.setdefault(int(line.split()[icl]), []).append(line)


def write(path, rr):
    with open(path, 'w') as f:
        f.writelines(head)
        f.writelines(rr)
        f.write('\n')


sel = [l for k in sorted(keep) for l in bycls.get(k, [])]
write(out, sel)
rnd = random.Random(20261023).sample(prow, len(sel))
write(outr, rnd)
os.makedirs(outdir, exist_ok=True)
for k, rr in bycls.items():
    write(os.path.join(outdir, 'class_%02d.star' % k), rr)
print('kept %d of %d particles from classes %s; random control %d' %
      (len(sel), len(prow), ','.join(map(str, sorted(keep))), len(rnd)))
