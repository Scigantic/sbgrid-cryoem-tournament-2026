#!/usr/bin/env bash
#SBATCH --no-requeue
# Box-size test for map03. The focused particles were cut to a 256 px (226 A) box; at 4 A and
# 1.4-2 um defocus the CTF spreads signal 70-100 A outward, beyond that box along the rod.
# Same subtraction as final_focus2/final_blush, run twice: 256 px (control) and 338 px (the
# polished particle box, 299 A). Both get map03's own poses (Blush iteration 10, by original
# image name), are reconstructed into half maps with relion_reconstruct and post-processed
# under the same focused mask (padded for 338). No re-alignment: only the box differs.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 4:00:00 slurm/final_box.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/box} SCRATCH=${SCRATCH:-/scratch/box/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
FIN=${FIN:-$PERSIST_DIR/final}; FOC=${FOC:-$PERSIST_DIR/focus}; BLU=${BLU:-$PERSIST_DIR/blush}
BIGBOX=${BIGBOX:-338}; NP_SUB=${NP_SUB:-24}; J_CPU=${J_CPU:-2}
NP_REC=${NP_REC:-24}; NP_REC_BIG=${NP_REC_BIG:-16}; J_REC_BIG=${J_REC_BIG:-3}
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
mkdir -p "$KEPT"
(
  echo "--- job ${SLURM_JOB_ID:-?} on $(hostname) $(date -u +%F)"
  while true; do
    echo "$(date -u +%T)" \
      "$(free -m | awk '/^Mem/{print "mem_used_mb",$3,"avail_mb",$7}')" \
      "scratch_used $(df -BG --output=used "$PROJ" | tail -n 1 | tr -d ' ')" \
      "top_rss_mb $(ps -eo rss=,comm= --sort=-rss | head -n 3 | awk '{printf "%d:%s ", $1/1024, $2}')" \
      || true
    sleep 30
  done
) >> "$KEPT/monitor.log" 2>&1 &

CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$FIN/Refine10/run_optimiser.star" "$FIN/Refine10/run_data.star" \
         "$FIN/Refine10/run_class001.mrc" "$FOC/Focus0/mask.mrc" "$FOC/Bodies/body2.mrc" \
         "$BLU/Blush/run_it010_data.star" "$HERE/remap_poses.py" "$HERE/remap_box.py"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
# the 256 stacks are deleted before the 338 subtraction, so the peak is polish + 338 stacks
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( ${#CHUNKS[@]} * 22 + 989 * BIGBOX * BIGBOX * 2 / 1000000 + 40 ))
echo "chunks: ${#CHUNKS[@]}; scratch free ${free_gb} GB, need about ${need_gb} GB"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

keep() { mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/; }
if [[ -d "$KEPT/.done" ]]; then
  rsync -a --exclude monitor.log --exclude 'Sub*' "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' '))"
fi
cd "$PROJ"
# node-local stacks do not survive a move to another node
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep .done/sub256 .done/sub338
echo "$NCPU cpus; boxes 256 and $BIGBOX; $NP_SUB subtract ranks"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_refs() {  # Refine10 for the subtraction, map02/03 body and mask, map03 poses, 338 mask
  mkdir -p Refine10 Map03 Mask338
  rsync -a --exclude 'run_it*' --exclude Post "$FIN/Refine10/" Refine10/
  rsync -a "$FOC/Bodies" "$FOC/Focus0" ./
  cp "$BLU/Blush/run_it010_data.star" Map03/run_data.star
  relion_image_handler --i Focus0/mask.mrc --new_box "$BIGBOX" --o Mask338/mask.mrc > Mask338/log.txt 2>&1 \
    || { cat Mask338/log.txt; return 1; }
  python3 -I - Focus0/mask.mrc Mask338/mask.mrc <<'EOF'
import sys, mrcfile, numpy as np
a = np.asarray(mrcfile.read(sys.argv[1]), dtype=np.float64)
b = np.asarray(mrcfile.read(sys.argv[2]), dtype=np.float64)
p = (b.shape[0] - a.shape[0]) // 2
inner = b[p:p + a.shape[0], p:p + a.shape[1], p:p + a.shape[2]]
print('mask 256 sum %.1f, 338 sum %.1f, inner max diff %.2e, box %s' % (
    a.sum(), b.sum(), np.abs(inner - a).max(), b.shape))
sys.exit(0 if abs(a.sum() - b.sum()) < 1e-3 * a.sum() and np.abs(inner - a).max() < 1e-4 else 1)
EOF
  keep Mask338
}

subtract() {  # BOX OUTDIR, identical to final_focus2.sh apart from the box
  local box=$1 out=$2 t0=$SECONDS c star outs=() total=0 n
  rm -rf "$out"; mkdir -p "$out/in"
  for c in "${CHUNKS[@]}"; do
    awk -v c="$c/" 'tolower($0) ~ /^data_particles/ {p=1}
      { if (!p || index($0, "@") == 0 || index($0, c)) print }' \
      Refine10/run_data.star > "$out/in/$c.star"
    n=$(nparticles "$out/in/$c.star"); total=$((total + n))
    $MPI -np "$NP_SUB" relion_particle_subtract_mpi --i Refine10/run_optimiser.star \
      --data "$out/in/$c.star" --mask Bodies/body2.mrc --o "$out/$c/" \
      --recenter_on_mask --new_box "$box" --float16 > "$out/$c.log" 2>&1 \
      || { tail -n 20 "$out/$c.log"; return 1; }
    star=$(ls "$out/$c/particles_subtracted.star" 2>/dev/null \
      || ls "$out/$c/"*.star | head -n 1)
    outs+=("$star")
  done
  local want; want=$(nparticles Refine10/run_data.star)
  (( total == want )) || { echo "chunk split covers $total of $want particles"; return 1; }
  relion_star_handler --combine --i "${outs[*]}" --o "$out/combined.star" > "$out/combine.log"
  ln -sf combined.star "$out/input.star"
  n=$(nparticles "$out/input.star")
  (( n == want )) || { echo "combined star has $n of $want particles"; return 1; }
  echo "subtracted $n particles at box $box in $((SECONDS-t0)) s ($(du -sm "$out" | cut -f1) MB)"
}

recon() {  # STAR OUTDIR MASK NP J
  local t0=$SECONDS h
  mkdir -p "$2"
  for h in 1 2; do
    $MPI -np "$4" relion_reconstruct_mpi --i "$1" --o "$2/run_half${h}_class001_unfil.mrc" \
      --subset "$h" --ctf --j "$5" > "$2/half$h.log" 2>&1 || { tail -n 20 "$2/half$h.log"; return 1; }
  done
  relion_postprocess --i "$2/run_half1_class001_unfil.mrc" --mask "$3" \
    --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o "$2/postprocess" > "$2/post.log"
  echo "RESULT $2: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$2/postprocess.star" | tr -s ' ' | tr '\n' ' ') ($((SECONDS-t0)) s)"
  keep "$2"
}

do_sub256() { subtract 256 Sub256; }
do_map256() { python3 -I "$HERE/remap_poses.py" Map03/run_data.star Sub256/input.star Sub256/poses.star; }
do_rec256() { recon Sub256/poses.star Recon256 Focus0/mask.mrc "$NP_REC" "$J_CPU"; rm -rf Sub256; }
do_sub338() { subtract "$BIGBOX" Sub338; }
do_map338() { python3 -I "$HERE/remap_box.py" Map03/run_data.star Sub338/input.star Sub338/poses.star; }
do_rec338() { recon Sub338/poses.star Recon338 Mask338/mask.mrc "$NP_REC_BIG" "$J_REC_BIG"; }

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage refs 1 0 do_refs
stage sub256 "$NCPU" 0 do_sub256
stage map256 1 0 do_map256
stage rec256 "$NCPU" 0 do_rec256
stage sub338 "$NCPU" 0 do_sub338
stage map338 1 0 do_map338
stage rec338 "$NCPU" 0 do_rec338
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
