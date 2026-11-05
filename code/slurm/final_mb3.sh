#!/usr/bin/env bash
#SBATCH --no-requeue
# Multi-body test on Refine12 (247,305 quality-selected polished rods, 4.67 A). Local
# resolution of the consensus is flat over the middle ~100 A of the rod and worsens toward
# both ends and away from the axis, the blur of parts moving relative to each other. Three
# bodies (end bead, middle beads, end bead; make_body_masks.py) are refined from Refine12's
# last iteration, each body post-processed under its own mask. Same particles as Refine12,
# so a middle body better than 4.67 A says flexibility limits us and the full 989k set
# should get the same treatment.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 6:00:00 slurm/final_mb3.sh
# final_mb3: job 85 ran out of GPU memory in the last (full-resolution) iteration on 16 GB
# T4s, three bodies at padding 2. A continue reads the padding from the model stars, so
# PAD (default 1) is written into the newest pair before continuing: about 8x less memory.
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
# Refine12's optimiser names its files relative to /scratch/sel, so this runs there too
export PROJ=${PROJ:-/scratch/sel} SCRATCH=${SCRATCH:-/scratch/sel/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$KEPT/Refine12/run_it012_optimiser.star" "$KEPT/Refine12/run_class001.mrc" \
         "$KEPT/Refs/solvent_mask.mrc" "$KEPT/.done/post12"; do
  [[ -e "$f" ]] || { echo "missing $f"; exit 2; }
done
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( ${#CHUNKS[@]} * 30 ))
echo "chunks: ${#CHUNKS[@]}; scratch free ${free_gb} GB, need about ${need_gb} GB"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

keep() { mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/; }
rsync -a "$KEPT"/ "$PROJ"/
echo "restored ($(ls "$PROJ/.done" | tr '\n' ' '))"
cd "$PROJ"
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep
echo "$NCPU cpus, $NGPU gpus"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_bodies() {
  python3 -I "$HERE/make_body_masks.py" Refine12/run_class001.mrc Refs/solvent_mask.mrc \
    "$ANGPIX" Bodies "${SOFT:-12}"
  keep Bodies
}

do_mb() {
  mkdir -p MultiBody
  # each iteration takes over an hour on T4s: save what exists every 10 min from inside the
  # job, so a timeout or a lost node costs at most one iteration (a resubmit continues)
  ( while sleep 600; do rsync -a MultiBody "$KEPT"/ 2>/dev/null; done ) &
  local saver=$!
  local opt; opt=$(ls -t MultiBody/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" ]]; then
    local it=${opt%_optimiser.star}
    sed -i -E "s/^(_rlnPaddingFactor[[:space:]]+).*/\1${PAD:-1}.000000/" \
      "${it}"_half1_model.star "${it}"_half2_model.star
    grep -H _rlnPaddingFactor "${it}"_half?_model.star
    echo "continuing from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o MultiBody/run \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG"
  else
    # the last numbered iteration: the final run_optimiser.star has the half sets joined,
    # which multi-body refuses
    local start; start=$(ls Refine12/run_it*_optimiser.star | sort | tail -n 1)
    echo "multi-body from $start"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$start" \
      --o MultiBody/run --multibody_masks Bodies/bodies.star --solvent_correct_fsc \
      --oversampling 1 --healpix_order 4 --auto_local_healpix_order 4 \
      --offset_range 3 --offset_step 0.75 --reconstruct_subtracted_bodies \
      --dont_combine_weights_via_disc --pool 30 --pad 2 --j "$THREADS_GPU" --gpu "$GPUARG"
  fi
  kill "$saver" 2>/dev/null || true
  keep MultiBody
}

do_postmb() {
  local n
  for n in 1 2 3; do
    mkdir -p "MultiBody/Post$n"
    relion_postprocess --i "MultiBody/run_half1_body00${n}_unfil.mrc" --mask "Bodies/body$n.mrc" \
      --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o "MultiBody/Post$n/postprocess" \
      > "MultiBody/Post$n/log.txt"
    echo "RESULT body$n: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "MultiBody/Post$n/postprocess.star" | tr -s ' ' | tr '\n' ' ')"
  done
  keep MultiBody
}

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage bodies 1 0 do_bodies
stage mb "$NCPU" "$NGPU" do_mb
stage postmb 1 0 do_postmb
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
