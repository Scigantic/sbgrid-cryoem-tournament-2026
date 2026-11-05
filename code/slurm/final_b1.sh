#!/usr/bin/env bash
# Final refinement at bin 1 (338 px, 0.8837 A) of the polished particles of job 66
# (~/results/full_tNN/Polish, ~22 GB float16 a chunk). Stages every chunk's shiny stack on
# node-local scratch, merges the stars, scales Refine8's map and both masks to bin 1, then
#   ref10  local refinement of the shiny particles from Refine8's map (low-passed to
#          INI_HIGH so the half sets stay independent above it; the shiny star keeps
#          Refine8's half-set split)
#   ctfm, ctfa, ctfd  anisotropic magnification; beam tilt + odd and even aberrations;
#          per-particle defocus. After polishing, which is the order that helped in the
#          rehearsal; job 46 showed CTF refinement hurts at 6.6 A, Refine10 is far better
#   ref11  local refinement after CTF refinement
# Every map is post-processed under Refine8's mask, scaled to bin 1, with a fitted
# B-factor, so the RESULT lines compare with every earlier one.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/final_b1.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/final} SCRATCH=${SCRATCH:-/scratch/final/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the staged stacks are copies; keep() saves the outputs instead
REF=${REF:-$PERSIST_DIR/merge}   # Refine8 and the mask of every RESULT line
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  if [[ -s "$d/Polish/shiny.star" ]]; then CHUNKS+=("$(basename "$d")")
  else echo "$(basename "$d") has no polished particles"; fi
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$REF/Refine8/run_class001.mrc" "$REF/Refine8/solvent_mask.mrc" "$REF/Mask/mask.mrc"; do
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

REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIX2=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.5f", a*b/s}')
INI_HIGH=${INI_HIGH:-7}
SEED=${SEED:-20261025}
echo "bin 2 maps $RSCALE px at $APIX2 A -> $RBOX px at $ANGPIX A; $NCPU cpus, $NGPU gpus"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_merge() {
  local a=() c
  for c in "${CHUNKS[@]}"; do a+=("$c" "$c/Polish/shiny.star"); done
  python3 -I "$HERE/merge_stars.py" shiny.star "${a[@]}" | tail -n 1
  echo "merged $(nparticles shiny.star) polished particles"
}

do_refs() {
  # Refine8's map and masks at bin 1: same 300 A box, twice the samples
  mkdir -p Refs Mask
  local f
  for f in Refine8/run_class001.mrc:Refs/ref8.mrc Refine8/solvent_mask.mrc:Refs/solvent_mask.mrc \
           Mask/mask.mrc:Mask/mask.mrc; do
    relion_image_handler --i "$REF/${f%%:*}" --o "${f#*:}" --angpix "$APIX2" \
      --rescale_angpix "$ANGPIX" --new_box "$RBOX" > /dev/null
  done
  # Fourier rescaling rings slightly past 0..1, which refine and postprocess reject
  python3 -I "$HERE/clip_mask.py" Refs/solvent_mask.mrc Mask/mask.mrc
  keep Refs Mask
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
    # ~220 GB of bin-1 images would not fit in memory; they are read from local scratch
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

ctfref() {  # IN POSTPROCESS OUTDIR args...
  local in=$1 f=$2 o=$3; shift 3
  mkdir -p "$o"
  $MPI -np "$NCPU" relion_ctf_refine_mpi --i "$in" --f "$f" --o "$o/" "$@" --j 1 > "$o/log.txt" 2>&1 \
    || { echo "$o failed:"; tail -n 5 "$o/log.txt" | cut -c1-160; exit 3; }
  echo "$o: $(nparticles "$o/particles_ctf_refine.star") particles"
  keep "$o"
}

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage mergep 1 0 do_merge
stage refs 1 0 do_refs
stage ref10 "$NCPU" "$NGPU" refine Refine10 shiny.star Refs/ref8.mrc
stage post10 1 0 post Refine10
stage ctfm "$NCPU" 0 ctfref Refine10/run_data.star Refine10/Post/postprocess.star CtfM \
  --fit_aniso --kmin_mag 20
stage ctfa "$NCPU" 0 ctfref CtfM/particles_ctf_refine.star Refine10/Post/postprocess.star CtfA \
  --fit_beamtilt --kmin_tilt 20 --odd_aberr_max_n 3 --fit_aberr --kmin_aberr 20
stage ctfd "$NCPU" 0 ctfref CtfA/particles_ctf_refine.star Refine10/Post/postprocess.star CtfD \
  --fit_defocus --kmin_defocus 30 --fit_mode fpmfm
stage ref11 "$NCPU" "$NGPU" refine Refine11 CtfD/particles_ctf_refine.star Refine10/run_class001.mrc
stage post11 1 0 post Refine11
echo "--- results"; grep -h '^RESULT' "$SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID:-x}.out" 2>/dev/null || true
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
