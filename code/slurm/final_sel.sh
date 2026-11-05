#!/usr/bin/env bash
#SBATCH --no-requeue
# Quality-selected refinement at bin 1. Refine10 (989,338 polished rods, 4.33 A) is limited by
# alignment, not by the images (Refine8 at bin 2 gave the same 4.33 A), so this keeps only the
# best particles, the rehearsal's arm B (4.04 A against 4.92 A for random at the same N):
# micrographs with a CTF fit at least as good as the median, defocus under 2.5 um, then the
# more confident half of those by Refine10's pose confidence (~250k, half sets kept). They are
# refined locally from Refine10's poses and map (low-passed to INI_HIGH), and post-processed
# under Refine8's mask like every RESULT line.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g5-12xlarge -t 6:00:00 slurm/final_sel.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/sel} SCRATCH=${SCRATCH:-/scratch/sel/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the staged stacks are copies; keep() saves the outputs instead
FIN=${FIN:-$PERSIST_DIR/final}   # Refine10, its bin-1 solvent mask and post mask
KEEP_CTF=${KEEP_CTF:-0.5}; KEEP_CONF=${KEEP_CONF:-0.5}; DFMAX=${DFMAX:-25000}
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$FIN/Refine10/run_data.star" "$FIN/Refine10/run_class001.mrc" \
         "$FIN/Refs/solvent_mask.mrc" "$FIN/Mask/mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( ${#CHUNKS[@]} * 30 ))
echo "chunks: ${#CHUNKS[@]}; scratch free ${free_gb} GB, need about ${need_gb} GB"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

keep() {  # save outputs (small) and the stage markers to $KEPT
  mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/
}
if [[ -d "$KEPT/.done" ]]; then
  rsync -a "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' '))"
fi
cd "$PROJ"
# staged stacks do not survive a move to another node
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep

REFDIAM=${REFDIAM:-220}; INI_HIGH=${INI_HIGH:-7}; SEED=${SEED:-20261026}
echo "$NCPU cpus, $NGPU gpus; keep ctf $KEEP_CTF, conf $KEEP_CONF, defocus <= $DFMAX A"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_select() {
  mkdir -p Refs Mask
  cp "$FIN/Refine10/run_class001.mrc" Refs/ref10.mrc
  cp "$FIN/Refs/solvent_mask.mrc" Refs/
  cp "$FIN/Mask/mask.mrc" Mask/
  python3 -I "$HERE/select_quality.py" "$FIN/Refine10/run_data.star" sel.star \
    "$KEEP_CTF" "$KEEP_CONF" "$DFMAX" | tee select.txt
  keep sel.star select.txt Refs Mask
}

refine() {  # NAME INPUT REF   local searches from the input poses
  local name=$1 in=$2 ref=$3
  mkdir -p "$name"
  local opt; opt=$(ls -t "$name"/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s "$name/run_class001.mrc" ]]; then
    echo "continuing $name from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o "$name/run" \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG"
  else
    $MPI -np "$NP_GPU" relion_refine_mpi --i "$in" --o "$name/run" --auto_refine \
      --split_random_halves --ref "$ref" --trust_ref_size --ini_high "$INI_HIGH" \
      --solvent_mask Refs/solvent_mask.mrc --solvent_correct_fsc \
      --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
      --flatten_solvent --zero_mask --oversampling 1 --healpix_order 4 \
      --auto_local_healpix_order 4 --offset_range 5 --offset_step 1 --sym C1 \
      --low_resol_join_halves 40 --norm --scale --random_seed "$SEED" \
      --j "$THREADS_GPU" --gpu "$GPUARG"
  fi
  keep "$name"
}

post() {  # NAME   Refine8's mask at bin 1, B-factor fitted from 10 A to the final resolution
  local name=$1
  mkdir -p "$name/Post"
  relion_postprocess --i "$name/run_half1_class001_unfil.mrc" --mask Mask/mask.mrc \
    --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o "$name/Post/postprocess" > "$name/Post/log.txt"
  echo "RESULT $name: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$name/Post/postprocess.star" | tr -s ' ' | tr '\n' ' ') particles $(nparticles "$name/run_data.star")"
  keep "$name"
}

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage select 1 0 do_select
stage ref12 "$NCPU" "$NGPU" refine Refine12 sel.star Refs/ref10.mrc
stage post12 1 0 post Refine12
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
