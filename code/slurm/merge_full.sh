#!/usr/bin/env bash
# Merge the ten chunks of job 48 (~/results/full_tNN). Job A stages every chunk's bin-4 rod
# stack on node-local scratch, merges the stars, runs one 2D classification of all rods
# (VDAM, the gradient algorithm meant for a million particles; it runs as one process
# over all GPUs), scores the classes with RELION's class ranker and draws the class sheet.
# Job B applies a selection (SEL=1,2,5,... read off the sheet, or SEL=auto for the
# ranker's) to the merged bin-2 stack and refines it with Refine5's command, reference and
# solvent mask, post-processed under the same mask as every RESULT line so far.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/merge_full.sh                     (job A: through the sheet)
#   export SEL=auto; sbatch slurm/merge_full.sh    (job B: bin-2 refinement)
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/merge} SCRATCH=${SCRATCH:-/scratch/merge/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the staged stacks are copies; keep() saves the outputs instead
BASE=${BASE:-$PERSIST_DIR/chunk2}    # Refine5's reference and solvent mask
MASKD=${MASKD:-$PERSIST_DIR/chunk2p} # the post-processing mask of every RESULT line
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  if [[ -e "$d/.done/extractb1" ]]; then CHUNKS+=("$(basename "$d")")
  else echo "$(basename "$d") is not finished"; fi
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks finished"; exit 2; }
echo "chunks: ${CHUNKS[*]}"

keep() {  # save outputs (small) and the stage markers to $KEPT
  mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/
}
if [[ -d "$KEPT/.done" ]]; then
  rsync -a "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' '))"
fi
cd "$PROJ"
# staged stacks do not survive a move to another node
[[ -d "${CHUNKS[0]}/ExtractRods" ]] || rm -f .done/stagerods .done/mergerods
[[ -d "${CHUNKS[0]}/Extract2" ]] || rm -f .done/stage2 .done/merge2

RODBOX_A=${RODBOX_A:-500}; REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIXR=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.4f", a*b/s}')
RODDIAM=$(awk -v b="$RODBOX_A" 'BEGIN{printf "%d", b*0.9}')
K2D=${K2D:-100}; ITER2D=${ITER2D:-200}; LAST=$(printf 'run_it%03d' "$ITER2D")
ALLGPU=$(seq -s: 0 $((NGPU - 1)))
# fixed seeds: a rerun gives the same answer up to GPU arithmetic; the kept particle
# lists (Class2D data star, Refine8/sel4.star) are saved so a rerun can reuse the selection
SEED=${SEED:-20261022}

stage_dir() {  # SUBDIR: copy that stack of every chunk, 10 in parallel
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/$1' {}/"
  echo "staged $1 of ${#CHUNKS[@]} chunks ($(du -scm */"$1" | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

merge_dir() {  # SUBDIR OUT
  local a=() c
  for c in "${CHUNKS[@]}"; do a+=("$c" "$c/$1/particles.star"); done
  python3 -I "$HERE/merge_stars.py" "$2" "${a[@]}" | tail -n 1
}

do_class2d() {
  mkdir -p Class2D
  local opt; opt=$(ls -t Class2D/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  local src=(--i rods.star --K "$K2D" --tau2_fudge 2 --particle_diameter "$RODDIAM"
             --flatten_solvent --zero_mask --center_classes --oversampling 1 --psi_step 6
             --offset_range 8 --offset_step 2 --norm --scale --ctf --random_seed "$SEED")
  [[ -n "$opt" ]] && { echo "continuing from $opt"; src=(--continue "$opt"); }
  relion_refine --o Class2D/run "${src[@]}" --grad --iter "$ITER2D" --grad_write_iter 10 \
    --class_inactivity_threshold 0.1 --dont_combine_weights_via_disc --pool 30 --pad 2 \
    --j "${J2D:-16}" --gpu "$ALLGPU" --preread_images
  keep Class2D rods.star
}

do_rank() {
  # the ranker was trained on single particles, so its scores are advice for rods; a
  # failure (no torch in the module's python) leaves the sheet as the way to select
  mkdir -p Select
  relion_class_ranker --opt "Class2D/${LAST}_optimiser.star" --o Select/ --auto_select \
    --min_score "${MIN_SCORE:-0.25}" --fn_sel_parts particles.star \
    --fn_sel_classavgs class_averages.star --do_granularity_features \
    ${RELION_PYTHON_EXECUTABLE:+--python "$RELION_PYTHON_EXECUTABLE"} > Select/log.txt 2>&1 \
    && echo "ranker kept $(nparticles Select/particles.star) particles" \
    || { echo "class ranker failed:"; tail -n 5 Select/log.txt; }
  keep Select
}

do_sheet() {
  python3 -I "$HERE/view_classes.py" "Class2D/${LAST}_model.star" \
    "Class2D/${LAST}_classes.mrcs" Class2D/top_classes.jpg "$K2D"
  keep Class2D
}

do_selectb() {
  mkdir -p Refine8
  if [[ "$SEL" == auto ]]; then
    [[ -s Select/particles.star ]] || { echo "no ranker selection; set SEL to class numbers"; exit 2; }
    cp Select/particles.star Refine8/sel4.star
  else
    python3 -I "$HERE/select_classes.py" "Class2D/${LAST}_data.star" Refine8/sel4.star "$SEL"
  fi
  python3 -I "$HERE/join_coords.py" Refine8/sel4.star bin2.star Refine8/input.star
  cp "$BASE/Refine5/ref.mrc" "$BASE/Refine5/solvent_mask.mrc" Refine8/
  mkdir -p Mask; cp "$MASKD/Mask/mask.mrc" Mask/
  echo "selection $SEL" > Refine8/selection.txt
  keep Refine8 Mask
}

do_refine8() {  # Refine5's command, unchanged (chunk_scale.sh do_refine5)
  local opt; opt=$(ls -t Refine8/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s Refine8/run_class001.mrc ]]; then
    echo "continuing from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o Refine8/run \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG"
  else
    # a million bin-2 images (~125 GB) would be read into every rank's memory with
    # --preread_images; they are read from local scratch instead
    $MPI -np "$NP_GPU" relion_refine_mpi --i Refine8/input.star --o Refine8/run \
      --auto_refine --split_random_halves --ref Refine8/ref.mrc --trust_ref_size \
      --ini_high 12 --solvent_mask Refine8/solvent_mask.mrc --solvent_correct_fsc \
      --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
      --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 \
      --auto_local_healpix_order 4 --offset_range 10 --offset_step 2 --sym C1 \
      --low_resol_join_halves 40 --norm --scale --random_seed "$SEED" \
      --j "$THREADS_GPU" --gpu "$GPUARG"
  fi
  keep Refine8
}

do_post8() {
  mkdir -p Refine8/Post
  relion_postprocess --i Refine8/run_half1_class001_unfil.mrc --mask Mask/mask.mrc \
    --angpix "$APIXR" --auto_bfac --autob_lowres 10 --o Refine8/Post/postprocess > Refine8/Post/log.txt
  echo "RESULT Refine8: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' Refine8/Post/postprocess.star | tr -s ' ' | tr '\n' ' ') particles $(nparticles Refine8/run_data.star)"
  keep Refine8
}

t0=$SECONDS
if [[ -z "${SEL:-}" ]]; then
  stage stagerods "$NCPU" 0 stage_dir ExtractRods
  stage mergerods 1 0 merge_dir ExtractRods rods.star
  stage class2d "$NCPU" "$NGPU" do_class2d
  stage rank "$NCPU" "$NGPU" do_rank
  stage sheet 1 0 do_sheet
else
  stage stage2 "$NCPU" 0 stage_dir Extract2
  stage merge2 1 0 merge_dir Extract2 bin2.star
  stage selectb 1 0 do_selectb
  stage refine8 "$NCPU" "$NGPU" do_refine8
  stage post8 1 0 do_post8
fi
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
