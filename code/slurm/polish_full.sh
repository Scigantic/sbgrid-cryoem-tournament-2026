#!/usr/bin/env bash
# Bayesian polishing of Refine8 (989,338 rods, 4.33 A at bin 2), one SLURM array task per
# chunk of job 48. Each task takes its chunk's particles out of Refine8 (the merged star
# prefixes every stack with its chunk), stages that chunk's raw movies again, and polishes
# them against Refine8's half maps and mask with the motion metadata RELION's own motion
# correction wrote in job 48 (kept on EFS). The shiny particles come out at bin 1 (338 px,
# 0.8837 A, float16, ~22 GB a chunk) and persist to ~/results/full_tNN/Polish, where the
# final refinement stages them from. Polishing is CPU only; the GPUs stay idle.
# Untrained sigmas 0.2/5000/2, as in the rehearsal's best run and job 46.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch --array=0-9 slurm/polish_full.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
command -v relion_motion_refine_mpi > /dev/null || { echo "relion_motion_refine_mpi not found"; exit 2; }
I=${TASK:-${SLURM_ARRAY_TASK_ID:?run as an array job or set TASK}}
TAG=$(printf 'full_t%02d' "$I")
export PROJ=${PROJ:-/scratch/polish_$TAG} MOVIES=${MOVIES:-/scratch/polish_${TAG}_movies}
export SCRATCH=${SCRATCH:-$PROJ/scratch} PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the inputs are copies; keep() saves Polish to the chunk
export NGPU=${NGPU:-1} # common.sh divides by NGPU; polishing uses none
CHUNK=$PERSIST_DIR/$TAG
REF=${REF:-$PERSIST_DIR/merge}   # Refine8, its Post and the mask of every RESULT line
for f in "$CHUNK/pick.txt" "$CHUNK/gain.mrc" "$CHUNK/MotionCorr/corrected_micrographs.star" \
         "$REF/Refine8/run_data.star" "$REF/Refine8/run_half1_class001_unfil.mrc" \
         "$REF/Refine8/run_half2_class001_unfil.mrc" "$REF/Refine8/Post/postprocess.star" \
         "$REF/Mask/mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
# a rerun starts clean: the staged movies do not survive a move to another node anyway
rm -rf "$PROJ" "$MOVIES"
mkdir -p "$MOVIES" "$PROJ/.done"
source "$HERE/common.sh"

nmov=$(wc -l < "$CHUNK/pick.txt")
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( nmov * 25 / 100 ))   # raw movie ~0.19 GB, shiny particles ~0.015 GB
echo "task $I: $nmov movies; scratch free ${free_gb} GB, need about ${need_gb} GB"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

REFBOX_A=${REFBOX_A:-300}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
# polishing ranks reached 12.8-19.5 GB RSS on the rehearsal; job 46 ran 8 on 186 GB here
NP_POL=${NP_POL:-$(( RAM_GB / 22 < NCPU / 6 ? RAM_GB / 22 : NCPU / 6 ))}
J_POL=${J_POL:-$(( NCPU / NP_POL ))}
echo "$NCPU cpus, $RAM_GB GB; polish $NP_POL x $J_POL; output box $RBOX at $ANGPIX A"

keep() { mkdir -p "$CHUNK"; rsync -a "$@" "$CHUNK"/; }
cd "$PROJ"
ln -sfn "$MOVIES" Movies
cp "$CHUNK/gain.mrc" "$CHUNK/pick.txt" .
# motion metadata only (one star per movie); polishing reads the raw frames, not the sums
rsync -a --include '*/' --include '*.star' --exclude '*' "$CHUNK/MotionCorr" .
mkdir -p Refine8 Mask
cp "$REF"/Refine8/run_half{1,2}_class001_unfil.mrc Refine8/
rsync -a "$REF/Refine8/Post" Refine8/
cp "$REF/Mask/mask.mrc" Mask/

do_subset() {
  # this chunk's rows of the merged star, with the chunk prefix taken off the stack path
  awk -v t="@$TAG/" '!/\.mrcs/ {print; next} index($0, t) {sub(t, "@"); print}' \
    "$REF/Refine8/run_data.star" > particles.star
  local n; n=$(nparticles particles.star)
  echo "$TAG: $n of $(nparticles "$REF/Refine8/run_data.star") Refine8 particles"
  (( n > 0 )) || { echo "no particles for $TAG"; exit 2; }
}

do_stage() {
  local t0=$SECONDS
  xargs -a pick.txt -P 12 -I{} cp -n {} "$MOVIES"/
  local n; n=$(ls "$MOVIES" | wc -l)
  echo "staged $n movies ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
  (( n >= nmov )) || { echo "only $n of $nmov movies staged"; exit 3; }
}

do_polish() {
  mkdir -p Polish
  $MPI -np "$NP_POL" relion_motion_refine_mpi --i particles.star \
    --f Refine8/Post/postprocess.star --corr_mic MotionCorr/corrected_micrographs.star \
    --first_frame 1 --last_frame -1 --o Polish/ --s_vel 0.2 --s_div 5000 --s_acc 2 \
    --combine_frames --bfac_minfreq 20 --bfac_maxfreq -1 --window "$RBOX" --scale "$RBOX" \
    --float16 --j "$J_POL" --only_do_unfinished > Polish/log.txt 2>&1 \
    || { echo "polishing failed:"; tail -n 5 Polish/log.txt | cut -c1-160; exit 3; }
  local n; n=$(nparticles Polish/shiny.star)
  echo "shiny particles: $n ($(du -sm Polish | cut -f1) MB)"
  (( n >= $(nparticles particles.star) * 95 / 100 )) || { echo "too few shiny particles"; exit 3; }
  keep Polish
}

t0=$SECONDS
stage subset 1 0 do_subset
stage copy "$NCPU" 0 do_stage
stage polish "$NCPU" 0 do_polish
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
cp "$PROJ/timing.csv" "$CHUNK/Polish/timing.csv"
cd /; rm -rf "$PROJ" "$MOVIES"
