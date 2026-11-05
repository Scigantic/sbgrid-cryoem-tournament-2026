#!/usr/bin/env python3
# Body masks for multi-body refinement of the rod: one end bead, the middle beads, the other
# end bead. The rod axis and bead centres come from the map (as in bead_symmetry.py); each
# body is the solvent mask times a slab along the axis whose edges fall midway between beads,
# with a cosine edge SOFT A wide centred on the cut, so the bodies sum to the solvent mask.
# Writes OUTDIR/body1.mrc .. body3.mrc and OUTDIR/bodies.star, and prints the cuts.
#   python3 -I make_body_masks.py map.mrc solvent_mask.mrc APIX OUTDIR [SOFT]
import os
import sys

import numpy as np

mpath, spath, apix, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
soft = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
NSIG, MINSEP = 2.0, 25.0


def read(p):
    with open(p, 'rb') as f:
        raw = f.read(1024)
        h = np.frombuffer(raw, dtype=np.int32)
        nx, ny, nz = (int(v) for v in h[:3])
        ext = f.read(int(h[23]))
        a = np.fromfile(f, dtype=np.float32, count=nx * ny * nz).reshape(nz, ny, nx)
        return raw, ext, a


_, _, m = read(mpath)
shead, sext, sm = read(spath)
nz, ny, nx = m.shape
thr = m.mean() + NSIG * m.std()
z, y, x = np.nonzero(m > thr)
w = m[z, y, x] - thr
xyz = np.stack([x, y, z], axis=1).astype(np.float64) * apix
c = (xyz * w[:, None]).sum(0) / w.sum()
d = xyz - c
axis = np.linalg.eigh((d * w[:, None]).T @ d / w.sum())[1][:, -1]
t = d @ axis
lo = np.floor(t.min()) - 5
prof, _ = np.histogram(t, bins=np.arange(lo, np.ceil(t.max()) + 6, 1.0), weights=w)
k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
prof = np.convolve(prof, k / k.sum(), mode='same')
peaks = [i for i in range(1, len(prof) - 1)
         if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.25 * prof.max()]
kept = []
for i in sorted(peaks, key=lambda i: -prof[i]):
    if all(abs(i - j) >= MINSEP for j in kept):
        kept.append(i)
beads = [lo + i + 0.5 for i in sorted(kept)]
if len(beads) < 4:
    sys.exit('found %d beads, need at least 4' % len(beads))
cuts = [(beads[0] + beads[1]) / 2, (beads[-2] + beads[-1]) / 2]
print('beads at %s A; cuts at %.1f and %.1f A' % (' '.join('%.0f' % b for b in beads), *cuts))

# distance along the axis of every voxel, by broadcasting (no full coordinate grids)
tt = ((np.arange(nx) * apix - c[0]) * axis[0])[None, None, :] \
    + ((np.arange(ny) * apix - c[1]) * axis[1])[None, :, None] \
    + ((np.arange(nz) * apix - c[2]) * axis[2])[:, None, None]


def edge(s):  # 0 -> 1 across a cosine ramp, s = signed distance past the cut in units of soft
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s + 0.5, 0, 1))


slabs = [edge((cuts[0] - tt) / soft),
         edge((tt - cuts[0]) / soft) * edge((cuts[1] - tt) / soft),
         edge((tt - cuts[1]) / soft)]
os.makedirs(outdir, exist_ok=True)
for n, s in enumerate(slabs, 1):
    body = (sm * s).astype(np.float32)
    with open(os.path.join(outdir, 'body%d.mrc' % n), 'wb') as f:
        f.write(shead)
        f.write(sext)
        f.write(body.tobytes())
    print('body%d: %.1f%% of the solvent mask' % (n, 100 * body.sum() / sm.sum()))
with open(os.path.join(outdir, 'bodies.star'), 'w') as f:
    f.write('data_\nloop_\n_rlnBodyMaskName\n_rlnBodyRotateRelativeTo\n'
            '_rlnBodySigmaAngles\n_rlnBodySigmaOffsetAngst\n')
    for n, rel in ((1, 2), (2, 1), (3, 2)):
        f.write('%s/body%d.mrc %d 10 2\n' % (outdir, n, rel))
