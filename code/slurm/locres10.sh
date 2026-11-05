#!/usr/bin/env bash
#SBATCH -J locres10
#SBATCH -n 4
#SBATCH --no-requeue
# Local resolution of Refine10 (bin 1, 0.8837 A, 4.33 A overall) under Refine8's mask. The
# FSC of Refine8 and Refine10 is a plateau near 0.55 from 7 to 5.3 A, then falls: part of
# the rod is better ordered than the rest. This shows which part, and whether any of it is
# already near 3.3 A, before any money goes into a focused refinement. Numbers only.
#   sbatch -p ct03g-g4dn-xlarge -t 2:00:00 slurm/locres10.sh
set -euo pipefail
module load sbgrid/relion/5.0.1_cu12.2
cd "$HOME/results/final"
mkdir -p LocRes10
mpirun -np "${SLURM_NTASKS:-4}" relion_postprocess_mpi --locres \
  --i Refine10/run_half1_class001_unfil.mrc --mask Mask/mask.mrc --angpix 0.8837 \
  --locres_sampling 25 --adhoc_bfac -250 --o LocRes10/relion > LocRes10/log.txt 2>&1 \
  || { echo "locres failed:"; tail -n 5 LocRes10/log.txt; exit 3; }
python3 -I - LocRes10/relion_locres.mrc Mask/mask.mrc <<'EOF'
import sys
import numpy as np


def read(path):
    with open(path, 'rb') as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        f.seek(1024 + int(h[23]))
        return np.fromfile(f, dtype=np.float32, count=int(h[0]) * int(h[1]) * int(h[2]))


res, mask = read(sys.argv[1]), read(sys.argv[2])
v = res[mask > 0.5]
print('voxels in mask %d' % v.size)
print('local resolution percentiles 5/25/50/75/95: %s A' %
      ' '.join('%.2f' % p for p in np.percentile(v, [5, 25, 50, 75, 95])))
for t in (3.3, 3.5, 3.8, 4.0, 4.5):
    print('better than %.1f A: %.1f%% of the masked volume' % (t, 100 * (v < t).mean()))
EOF
