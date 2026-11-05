#!/usr/bin/env bash
# State census for the form (Q2 oligomeric states, Q3 heterogeneity): 3D classification of
# all of Refine8's rods (989,338, bin 2, 1.778 A) without re-aligning them, from Refine8's
# map low-passed to 15 A. Rods of other lengths reach past Refine8's solvent mask, so the
# classes use only a spherical mask almost the size of the box. Expectation is limited to
# 12 A (--strict_highres_exp), which is all a bead count needs and keeps it cheap. Each
# class map is then measured (extent, beads) and its particles written to Classes/class_NN.star
# for the per-state refinements, a separate job once the census is known.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/classify_states.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/states} SCRATCH=${SCRATCH:-/scratch/states/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the staged stacks are copies; keep() saves the outputs instead
REF=${REF:-$PERSIST_DIR/merge}   # Refine8 (poses and map)
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -d "$d/Extract2" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks extracted"; exit 2; }
for f in "$REF/Refine8/run_data.star" "$REF/Refine8/run_class001.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( ${#CHUNKS[@]} * 15 ))
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
[[ -d "${CHUNKS[0]}/Extract2" ]] || rm -f .done/stage2

REFBOX_A=${REFBOX_A:-300}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIX2=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.5f", a*b/s}')
DIAM=${DIAM:-$(awk -v a="$APIX2" -v s="$RSCALE" 'BEGIN{printf "%d", a*s*0.96}')}
K3D=${K3D:-8}; T3D=${T3D:-4}; ITER3D=${ITER3D:-25}; LAST=$(printf 'run_it%03d' "$ITER3D")
SEED=${SEED:-20261026}
echo "bin 2: $RSCALE px at $APIX2 A, mask diameter $DIAM A; K=$K3D T=$T3D iter $ITER3D"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Extract2' {}/"
  echo "staged ${#CHUNKS[@]} bin-2 chunks ($(du -scm */Extract2 | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
  mkdir -p Refine8
  cp "$REF"/Refine8/{run_data.star,run_class001.mrc} Refine8/
  echo "particles: $(nparticles Refine8/run_data.star)"
}

do_class3d() {
  mkdir -p Class3D
  local opt; opt=$(ls -t Class3D/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s "Class3D/${LAST}_model.star" ]]; then
    echo "continuing from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o Class3D/run \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG"
  else
    # ~125 GB of images would be read into every rank with --preread_images; they are
    # read from local scratch instead
    $MPI -np "$NP_GPU" relion_refine_mpi --i Refine8/run_data.star --o Class3D/run \
      --ref Refine8/run_class001.mrc --trust_ref_size --ini_high 15 --K "$K3D" \
      --tau2_fudge "$T3D" --iter "$ITER3D" --skip_align --strict_highres_exp 12 \
      --particle_diameter "$DIAM" --flatten_solvent --zero_mask --ctf --norm --scale \
      --sym C1 --pad 2 --pool 30 --dont_combine_weights_via_disc --random_seed "$SEED" \
      --j "$THREADS_GPU" --gpu "$GPUARG"
  fi
  keep Class3D
}

do_census() {
  local maps=() k
  for k in $(seq 1 "$K3D"); do maps+=("Class3D/${LAST}_class$(printf '%03d' "$k").mrc"); done
  python3 -I "$HERE/measure_classes.py" "Class3D/${LAST}_model.star" "$APIX2" "${maps[@]}" \
    | tee census.txt
  # one star per class (select_3d.py; its kept/random outputs are not used here)
  python3 -I "$HERE/select_3d.py" "Class3D/${LAST}_model.star" "Class3D/${LAST}_data.star" \
    0 /dev/null /dev/null Classes | tail -n 1
  keep census.txt Classes
}

t0=$SECONDS
stage stage2 "$NCPU" 0 do_stage
stage class3d "$NCPU" "$NGPU" do_class3d
stage census 1 0 do_census
echo "--- census"; cat census.txt
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
