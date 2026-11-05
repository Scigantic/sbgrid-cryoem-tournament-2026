#!/usr/bin/env bash
# Controlled comparison for job 46. Refine6 changed two things at once (RELION's own motion
# correction and a different refinement: local from the start, ini_high 8, the tighter
# post-processing mask) and came out worse than Refine5 (6.64 A / B 275 against 6.36 / 223).
# Here both of job 46's stacks are refined with exactly Refine5's command, reference and
# solvent mask, and post-processed under the same mask as every RESULT line so far:
#   Refine7   Extract6 (RELION's motion correction, re-extracted)  vs Refine5: the micrographs
#   Refine7p  Polish/shiny.star (polished)                         vs Refine7: polishing
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/chunk_ctrl.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/chunk2c} SCRATCH=${SCRATCH:-/scratch/chunk2c/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the inputs are copies (20 GB); only the refinements are kept, at the end
FROM=${FROM:-$PERSIST_DIR/chunk2p}   # job 46: Extract6, Polish, Mask
BASE=${BASE:-$PERSIST_DIR/chunk2}    # jobs 42/43: Refine5's reference and solvent mask
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

for f in "$FROM/Extract6/particles.star" "$FROM/Polish/shiny.star" "$FROM/Mask/mask.mrc" \
         "$BASE/Refine5/ref.mrc" "$BASE/Refine5/solvent_mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
KEPT="$PERSIST_DIR/$(basename "$PROJ")"
cd "$PROJ"
t0=$SECONDS
rsync -a "$FROM/Extract6" "$FROM/Polish" "$FROM/Mask" .
mkdir -p Refine5; cp -n "$BASE/Refine5/ref.mrc" "$BASE/Refine5/solvent_mask.mrc" Refine5/
echo "staged inputs in $((SECONDS-t0)) s"

REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIXR=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.4f", a*b/s}')

refine5like() {  # NAME INPUT   Refine5's command (chunk_scale.sh do_refine5), unchanged
  local name=$1 in=$2
  mkdir -p "$name"
  local opt; opt=$(ls -t "$name"/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s "$name/run_class001.mrc" ]]; then
    echo "continuing $name from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o "$name/run" \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
    return
  fi
  $MPI -np "$NP_GPU" relion_refine_mpi --i "$in" --o "$name/run" \
    --auto_refine --split_random_halves --ref Refine5/ref.mrc --trust_ref_size \
    --ini_high 12 --solvent_mask Refine5/solvent_mask.mrc --solvent_correct_fsc \
    --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
    --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 \
    --auto_local_healpix_order 4 --offset_range 10 --offset_step 2 --sym C1 \
    --low_resol_join_halves 40 --norm --scale --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
}

post() {  # NAME   same mask and B-factor fit as every RESULT line from job 46
  local name=$1
  mkdir -p "$name/Post"
  relion_postprocess --i "$name/run_half1_class001_unfil.mrc" --mask Mask/mask.mrc \
    --angpix "$APIXR" --auto_bfac --autob_lowres 10 --o "$name/Post/postprocess" > "$name/Post/log.txt"
  echo "RESULT $name: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$name/Post/postprocess.star" | tr -s ' ' | tr '\n' ' ') particles $(nparticles "$name/run_data.star")"
}

stage ref7 "$NCPU" "$NGPU" refine5like Refine7 Extract6/particles.star
stage post7 1 0 post Refine7
stage ref7p "$NCPU" "$NGPU" refine5like Refine7p Polish/shiny.star
stage post7p 1 0 post Refine7p
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
mkdir -p "$KEPT"; rsync -a Refine7 Refine7p timing.csv "$KEPT"/
