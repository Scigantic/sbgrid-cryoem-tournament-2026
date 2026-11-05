#!/usr/bin/env python3
# Checks the package maps agree (box, pixel size, mode), reads the gold-standard FSC from
# postprocess.star, reports the 0.143 and 0.5 crossings, and draws OUT/ID_fsc.png.
# Numbers only on stdout.
#   python3 -I package_check.py OUT ID
import glob
import os
import sys

import numpy as np

out, ident = sys.argv[1], sys.argv[2]


def header(p):
    with open(p, 'rb') as f:
        h = f.read(1024)
    i = np.frombuffer(h, dtype=np.int32)
    fl = np.frombuffer(h, dtype=np.float32)
    nx, ny, nz, mode = (int(v) for v in i[:4])
    mx = int(i[7])
    return nx, ny, nz, mode, float(fl[10]) / mx


print('maps (box, mode, pixel A):')
seen = set()
for p in sorted(glob.glob(os.path.join(out, '*.mrc'))):
    nx, ny, nz, mode, apix = header(p)
    seen.add((nx, ny, nz, round(apix, 4)))
    print('  %-48s %dx%dx%d mode %d %.4f' % (os.path.basename(p), nx, ny, nz, mode, apix))
print('all maps agree' if len(seen) == 1 else 'MAPS DISAGREE: %s' % sorted(seen))


def blocks(p):
    data, name, labels, inloop = {}, None, [], False
    for line in open(p):
        s = line.strip()
        if s.startswith('data_'):
            name, labels, inloop = s, [], False
            data[name] = {'pairs': {}, 'rows': [], 'labels': labels}
        elif s == 'loop_':
            inloop = True
        elif s.startswith('_'):
            parts = s.split()
            if inloop:
                labels.append(parts[0])
            elif len(parts) > 1:
                data[name]['pairs'][parts[0]] = parts[1]
        elif s and not s.startswith('#') and inloop and name:
            data[name]['rows'].append(s.split())
    return data


b = blocks(os.path.join(out, ident + '_postprocess.star'))
gen = b.get('data_general', {}).get('pairs', {})
print('final resolution %s A, B-factor %s, randomize phases beyond %s A' % (
    gen.get('_rlnFinalResolution'), gen.get('_rlnBfactorUsedForSharpening'),
    gen.get('_rlnRandomiseFrom')))
fsc = b['data_fsc']
col = {k: i for i, k in enumerate(fsc['labels'])}
rows = np.array([[float(x) for x in r] for r in fsc['rows']])
res = rows[:, col['_rlnResolution']]          # 1/A
curves = [(k, rows[:, col[k]]) for k in (
    '_rlnFourierShellCorrelationCorrected', '_rlnFourierShellCorrelationMaskedMaps',
    '_rlnFourierShellCorrelationUnmaskedMaps',
    '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps') if k in col]


def crossing(y, t):
    for i in range(1, len(y)):
        if res[i] > 0 and y[i - 1] >= t > y[i]:
            f = res[i - 1] + (res[i] - res[i - 1]) * (y[i - 1] - t) / (y[i - 1] - y[i])
            return 1 / f
    return float('nan')


for k, y in curves:
    print('  %-62s 0.143 at %.2f A, 0.5 at %.2f A' % (k, crossing(y, 0.143), crossing(y, 0.5)))

png = os.path.join(out, ident + '_fsc.png')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'_rlnFourierShellCorrelationCorrected': 'corrected (reported)',
             '_rlnFourierShellCorrelationMaskedMaps': 'masked',
             '_rlnFourierShellCorrelationUnmaskedMaps': 'unmasked',
             '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps':
                 'phase-randomized masked'}
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for k, y in curves:
        ax.plot(res, y, lw=1.6 if k == '_rlnFourierShellCorrelationCorrected' else 1.0, label=names[k])
    ax.axhline(0.143, color='k', ls='--', lw=0.8)
    ax.axhline(0.5, color='grey', ls=':', lw=0.8)
    r = float(gen.get('_rlnFinalResolution', 'nan'))
    ax.axvline(1 / r, color='k', lw=0.6)
    ax.text(1 / r, 0.9, ' %.2f A at FSC 0.143' % r, fontsize=8)
    ticks = [20, 10, 7, 5, 4, 3.3, 3, 2.5, 2]
    ax.set_xticks([1 / t for t in ticks if 1 / t <= res.max()])
    ax.set_xticklabels([str(t) for t in ticks if 1 / t <= res.max()])
    ax.set_xlabel('resolution (A)')
    ax.set_ylabel('FSC')
    ax.set_ylim(-0.1, 1.02)
    ax.set_title('%s: gold-standard FSC, solvent mask, RELION 5 postprocess' % ident, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png)
    print('wrote fsc plot (matplotlib)')
except ImportError:
    print('matplotlib missing: no fsc plot')
