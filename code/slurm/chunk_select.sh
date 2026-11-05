#!/usr/bin/env bash
# Does strict 3D selection beat keeping every particle (Toshio's EMPIAR-10581 runs kept
# ~11k of 165k for 2.7 A)? Starts from Refine5 (MotionCor3, 86,467 particles, 6.36 A), whose
# particles already carry orientations: poor micrographs are dropped (CTF fit, defocus), the
# rest classified in 3D without re-aligning, the best classes kept, and those refined with
# Refine5's exact command, reference and solvent mask. A random subset of the same size is
# refined the same way, so a gain from selection is told apart from a gain (or loss) from
# simply using fewer particles. Both are post-processed under the mask of every RESULT line.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/chunk_select.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/sel3d} SCRATCH=${SCRATCH:-/scratch/sel3d/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1   # the inputs are copies; keep() saves the outputs
BASE=${BASE:-$PERSIST_DIR/chunk2}    # Refine5, its Extract2 stack and CTF
MASKD=${MASKD:-$PERSIST_DIR/chunk2p} # the post-processing mask of every RESULT line
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

for f in "$BASE/Refine5/run_data.star" "$BASE/Refine5/run_class001.mrc" "$BASE/Refine5/ref.mrc" \
         "$BASE/Refine5/solvent_mask.mrc" "$BASE/CtfFind/micrographs_ctf.star" "$MASKD/Mask/mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
KEPT="$PERSIST_DIR/$(basename "$PROJ")"
keep() { mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/; }
cd "$PROJ"
t0=$SECONDS
rsync -a "$BASE/Extract2" .
mkdir -p Refine5 Mask CtfFind
cp -n "$BASE"/Refine5/{run_data.star,run_class001.mrc,ref.mrc,solvent_mask.mrc} Refine5/
cp -n "$BASE/CtfFind/micrographs_ctf.star" CtfFind/
cp -n "$MASKD/Mask/mask.mrc" Mask/
echo "staged inputs ($(du -sm Extract2 | cut -f1) MB) in $((SECONDS-t0)) s"

REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIXR=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.4f", a*b/s}')
K3D=${K3D:-6}; T3D=${T3D:-4}; ITER3D=${ITER3D:-25}; LAST=$(printf 'run_it%03d' "$ITER3D")
SEED=${SEED:-20261023}

do_filter() {
  python3 -I "$HERE/filter_mics.py" Refine5/run_data.star filtered.star \
    "${MAXRES:-6}" "${DFMAX:-25000}" CtfFind/micrographs_ctf.star
}

do_class3d() {
  # orientations from Refine5 are kept (--skip_align); the reference is Refine5's map
  # low-passed to 15 A so the classes are not seeded with its high-resolution noise
  mkdir -p Class3D
  $MPI -np "$NP_GPU" relion_refine_mpi --i filtered.star --o Class3D/run \
    --ref Refine5/run_class001.mrc --trust_ref_size --ini_high 15 --K "$K3D" \
    --tau2_fudge "$T3D" --iter "$ITER3D" --skip_align --solvent_mask Refine5/solvent_mask.mrc \
    --particle_diameter "$REFDIAM" --flatten_solvent --zero_mask --ctf --norm --scale \
    --sym C1 --pad 2 --pool 30 --dont_combine_weights_via_disc --random_seed "$SEED" \
    --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
  keep Class3D
}

do_select() {
  python3 -I "$HERE/select_3d.py" "Class3D/${LAST}_model.star" "Class3D/${LAST}_data.star" \
    "${SLACK:-0.5}" selected.star random.star Classes | tee select.txt
  keep selected.star random.star Classes select.txt
}

refine5like() {  # NAME INPUT   Refine5's command (chunk_scale.sh do_refine5), unchanged
  local name=$1 in=$2
  mkdir -p "$name"
  $MPI -np "$NP_GPU" relion_refine_mpi --i "$in" --o "$name/run" \
    --auto_refine --split_random_halves --ref Refine5/ref.mrc --trust_ref_size \
    --ini_high 12 --solvent_mask Refine5/solvent_mask.mrc --solvent_correct_fsc \
    --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
    --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 \
    --auto_local_healpix_order 4 --offset_range 10 --offset_step 2 --sym C1 \
    --low_resol_join_halves 40 --norm --scale --random_seed "$SEED" \
    --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
  keep "$name"
}

post() {  # NAME
  local name=$1
  mkdir -p "$name/Post"
  relion_postprocess --i "$name/run_half1_class001_unfil.mrc" --mask Mask/mask.mrc \
    --angpix "$APIXR" --auto_bfac --autob_lowres 10 --o "$name/Post/postprocess" > "$name/Post/log.txt"
  echo "RESULT $name: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$name/Post/postprocess.star" | tr -s ' ' | tr '\n' ' ') particles $(nparticles "$name/run_data.star")"
  keep "$name"
}

stage filter 1 0 do_filter
stage class3d "$NCPU" "$NGPU" do_class3d
stage select 1 0 do_select
stage ref9s "$NCPU" "$NGPU" refine5like Refine9s selected.star
stage post9s 1 0 post Refine9s
stage ref9r "$NCPU" "$NGPU" refine5like Refine9r random.star
stage post9r 1 0 post Refine9r
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
