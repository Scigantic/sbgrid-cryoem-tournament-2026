#!/usr/bin/env bash
#SBATCH -J locres3
#SBATCH -p ct03g-g4dn-xlarge
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --no-requeue
#SBATCH -o slurm-%j.out
# RELION local resolution for map03 (Blush, iteration 10 half maps), same mask and
# B-factor as its postprocess. CPU only. Output goes straight into the submission folder.
set -euo pipefail
B=$HOME/results/blush
OUT=$HOME/submission/ct03
ID=ct03_map03_focused_blush
W=$B/Locres
mkdir -p "$W"
NP=$(( $(nproc) < 4 ? $(nproc) : 4 ))
echo "== locres $(date +%T) on $NP ranks"
mpirun -np "$NP" relion_postprocess_mpi --locres \
  --i "$B/Blush/run_it010_half1_class001_unfil.mrc" \
  --mask "$HOME/results/focus/Focus0/mask.mrc" --angpix 0.8837 \
  --adhoc_bfac -115.2 --o "$W/relion" > "$W/log.txt" 2>&1
echo "== locres done $(date +%T)"
cp "$W/relion_locres.mrc" "$OUT/${ID}_local_resolution.mrc"
cp "$W/relion_locres_filtered.mrc" "$OUT/${ID}_locres_filtered.mrc"
python3 - "$OUT/${ID}_local_resolution.mrc" "$HOME/results/focus/Focus0/mask.mrc" <<'EOF'
import sys, numpy as np
def rd(p):
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        f.seek(1024 + int(h[23]))
        return np.fromfile(f, dtype=np.float32).reshape(h[2], h[1], h[0])
lr, m = rd(sys.argv[1]), rd(sys.argv[2])
v = lr[m > 0.5]
print('RESULT locres percentiles 5/25/50/75/95: ' +
      ' / '.join('%.2f' % x for x in np.percentile(v, [5, 25, 50, 75, 95])) + ' A')
EOF
cd "$OUT" && rm -f SHA256SUMS && sha256sum -- * > ../S.new && mv ../S.new SHA256SUMS
echo "== package now $(ls | wc -l) files"
