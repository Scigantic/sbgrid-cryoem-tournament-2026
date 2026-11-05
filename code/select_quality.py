#!/usr/bin/env python3
# Quality selection of a refined particle star, the rehearsal's arm B (4.04 A against 4.92 A
# for a random subset of the same size): keep particles whose micrograph's CTF fit is at
# least as good as the KEEP_CTF quantile and whose mean defocus is under DFMAX, then the
# KEEP_CONF most confident of those by _rlnMaxValueProbDistribution. The half-set split
# (_rlnRandomSubset) is kept as it is. Prints counts only.
#   python3 -I select_quality.py run_data.star out.star KEEP_CTF KEEP_CONF DFMAX
import sys

src, out = sys.argv[1], sys.argv[2]
keep_ctf, keep_conf, dfmax = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

head, rows, labels, block, inloop = [], [], [], '', False
for line in open(src):
    s = line.strip()
    if s.startswith('data_'):
        block, inloop, labels = s, False, []
    elif s == 'loop_':
        inloop, labels = True, []
    elif inloop and s.startswith('_'):
        labels.append(s.split()[0])
    elif inloop and s and not s.startswith('#') and block == 'data_particles':
        rows.append(line)
        continue
    if not rows:
        head.append(line)

ix = {k: labels.index(k) for k in ('_rlnCtfMaxResolution', '_rlnDefocusU', '_rlnDefocusV',
                                   '_rlnMaxValueProbDistribution', '_rlnRandomSubset')}
f = [r.split() for r in rows]


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


ctf = [float(r[ix['_rlnCtfMaxResolution']]) for r in f]
cut_ctf = q(ctf, keep_ctf)
a = [i for i, r in enumerate(f) if ctf[i] <= cut_ctf and
     (float(r[ix['_rlnDefocusU']]) + float(r[ix['_rlnDefocusV']])) / 2 <= dfmax]
conf = [float(f[i][ix['_rlnMaxValueProbDistribution']]) for i in a]
cut_conf = q(conf, 1 - keep_conf)
b = [i for i, c in zip(a, conf) if c >= cut_conf]
with open(out, 'w') as o:
    o.writelines(head)
    o.writelines(rows[i] for i in b)
    o.write('\n')
halves = [sum(1 for i in b if f[i][ix['_rlnRandomSubset']] == h) for h in ('1', '2')]
print('all %d; CTF max res <= %.2f A and defocus <= %.0f A: %d; confidence >= %.4f: %d '
      '(half sets %d / %d)' % (len(f), cut_ctf, dfmax, len(a), cut_conf, len(b), *halves))
