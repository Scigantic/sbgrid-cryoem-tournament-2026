#!/usr/bin/env bash
# Scale-up chunk: N more movies (disjoint from the first chunk's 300), motion correction,
# CTF, rod picking inside the allocation (rodpick.py in NCPU parallel parts), whole-rod
# extraction at bin 4 and bin 2, and a 2D classification of every rod at bin 4 for
# cleaning. A second submission with SEL2=<good classes> (read off the class sheet)
# applies the selection to the bin-2 stack and refines it from the first chunk's map.
#   export PRESET=$HOME/pipeline/presets/tournament.env N=1500
#   sbatch ... slurm/chunk_scale.sh              (job A: through the 2D sheet)
#   export SEL2=1,2,5,...; sbatch ... slurm/chunk_scale.sh   (job B: refine)
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7 sbgrid/motioncor3/1.2.4}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export CTFFIND=${CTFFIND:-$(command -v ctffind)}
SRC=${SRC:-/shared/s3/tournament}; N=${N:-1500}; PAT=${MOVIE_PAT:-*_fractions.tiff}
CHUNK=${CHUNK:-chunk2}
export PROJ=${PROJ:-/scratch/$CHUNK} MOVIES=${MOVIES:-/scratch/${CHUNK}_movies}
export SCRATCH=${SCRATCH:-$PROJ/scratch} PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
FIRST=${FIRST:-$PERSIST_DIR/chunk/pick.txt}     # movies already used by the first chunk
REF5=${REF5:-$PERSIST_DIR/chunk/Refine4/run_class001.mrc}
MASK5=${MASK5:-$PERSIST_DIR/chunk/Refine4/solvent_mask.mrc}
mkdir -p "$MOVIES" "$PROJ/.done"
source "$HERE/common.sh"

# fail before spending anything if the node's python lacks the picker's libraries, or if the
# node-local disk cannot hold the raw movies (about 0.19 GB each; job 40 filled it at N=1500
# because raw copies and corrected outputs were both kept). Raw movies are now deleted as
# each one is corrected (MC3_RM_MOVIES) and the non-dose-weighted outputs after CTF.
python3 -I -c 'import numpy, scipy, skimage, PIL' || { echo "node python lacks scipy/skimage/PIL"; exit 2; }
export MC3_RM_MOVIES=${MC3_RM_MOVIES:-1}
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( N * 30 / 100 ))
echo "scratch free ${free_gb} GB, need about ${need_gb} GB for $N movies"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
if [[ -z "${FRESH:-}" && -d "$KEPT/.done" ]]; then
  t0=$SECONDS
  rsync -a --exclude '/movies' --exclude '/Movies' "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' ')) in $((SECONDS-t0)) s"
fi
cd "$PROJ"
# markers from a job that died on another node: its staged movies and corrected micrographs
# are gone with that node, so redo those stages unless their outputs are here
if [[ -s pick.txt ]]; then
  nmov=$(wc -l < pick.txt)
  if [[ $(ls MotionCorr/Movies/*_DW.mrc 2>/dev/null | wc -l) -lt $((nmov * 9 / 10)) ]]; then
    rm -f .done/motioncor3 .done/mc3_star .done/ctf .done/trim
    [[ $(ls "$MOVIES" 2>/dev/null | wc -l) -ge $nmov ]] || rm -f .done/copy
  fi
fi
[[ -n "${GAIN_SRC:-}" ]] || GAIN_SRC=$(find "$SRC" -type f -iname '*gain*' | head -1)
cp -n "$GAIN_SRC" "$PROJ/gain_src.${GAIN_SRC##*.}" 2>/dev/null || true
export GAIN="$PROJ/gain_src.${GAIN_SRC##*.}"

RISE=${RISE:-40}; RODBOX_A=${RODBOX_A:-500}; RODSCALE=${RODSCALE:-80}
REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
BOX=$(awk -v a="$ANGPIX" -v b="$RODBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIXR=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.4f", a*b/s}')
RODDIAM=$(awk -v b="$RODBOX_A" 'BEGIN{printf "%d", b*0.9}')
K2DR=${K2DR:-50}; ITER2DR=${ITER2DR:-15}; LASTR=$(printf 'run_it%03d' "$ITER2DR")

do_select() {
  find "$SRC" -type f -name "$PAT" | sort > all_movies.txt
  if [[ -s "$FIRST" ]]; then grep -v -x -F -f "$FIRST" all_movies.txt > rest.txt; else cp all_movies.txt rest.txt; fi
  local total k; total=$(wc -l < rest.txt); k=$(( total / N )); (( k >= 1 )) || k=1
  # no pipe into head: under pipefail its early exit fails the stage with SIGPIPE (141)
  awk -v k="$k" -v n="$N" 'NR%k==1 && ++c<=n' rest.txt > pick.txt
  echo "available $(wc -l < all_movies.txt) movies, $total not in the first chunk, picked $(wc -l < pick.txt)"
}

do_stage() {
  local t0=$SECONDS
  xargs -a pick.txt -P 12 -I{} cp -n {} "$MOVIES"/
  echo "staged $(ls "$MOVIES" | wc -l) movies ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_ctf() {
  mkdir -p CtfFind
  $MPI -np $((NCPU - 1)) relion_run_ctffind_mpi --i MotionCorr/corrected_micrographs.star \
    --o CtfFind/ --ctffind_exe "$CTFFIND" --Box 512 --ResMin 30 --ResMax 5 \
    --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 --is_ctffind4 --only_do_unfinished
}

do_trim() {
  # CTF used the non-dose-weighted micrographs; extraction uses the _DW ones only
  local f n=0
  for f in MotionCorr/Movies/*.mrc; do
    [[ "$f" == *_DW.mrc ]] && continue
    [[ -s "${f%.mrc}_DW.mrc" ]] && { rm -f "$f"; n=$((n + 1)); }
  done
  echo "removed $n non-DW micrographs; scratch: $(df -h "$PROJ" | awk 'NR==2{print $4" free of "$2}')"
}

do_pickrods() {
  rm -rf RodsPick; mkdir -p RodsPick
  grep 'MotionCorr/Movies/' CtfFind/micrographs_ctf.star > RodsPick/mics.txt
  split -n "l/$NCPU" -d -a 2 RodsPick/mics.txt RodsPick/part_
  local p i=0
  for p in RodsPick/part_*; do
    python3 -I "$HERE/rodpick.py" --star "$p" --root . --out "RodsPick/p$(printf '%02d' "$i")" \
      --angpix "$ANGPIX" --margin "${PICK_MARGIN:-260}" > "RodsPick/log_$i.txt" 2>&1 &
    i=$((i + 1))
  done
  wait
  echo "picker parts: $i"
}

do_census() {
  python3 -I "$HERE/rods_merge.py" RodsPick "$RISE" | tee RodsPick_census.txt
}

do_extractrods() {
  mkdir -p ExtractRods
  $MPI -np "$NCPU" relion_preprocess_mpi --i CtfFind/micrographs_ctf.star \
    --coord_list RodsPick/rods.star --part_star ExtractRods/particles.star \
    --part_dir ExtractRods/ --extract --extract_size "$BOX" --scale "$RODSCALE" --norm \
    --bg_radius $((RODSCALE * 3 / 8)) --white_dust -1 --black_dust "${BLACK_DUST:-5}" --invert_contrast
  echo "rods extracted: $(nparticles ExtractRods/particles.star), box $BOX px scaled to $RODSCALE"
}

do_extract2() {
  mkdir -p Extract2
  $MPI -np "$NCPU" relion_preprocess_mpi --i CtfFind/micrographs_ctf.star \
    --coord_list RodsPick/rods.star --part_star Extract2/particles.star \
    --part_dir Extract2/ --extract --extract_size "$RBOX" --scale "$RSCALE" --norm \
    --bg_radius $((RSCALE * 3 / 8)) --white_dust -1 --black_dust "${BLACK_DUST:-5}" --invert_contrast
  echo "bin-2 stack: $(nparticles Extract2/particles.star) particles, box $RBOX -> $RSCALE px at $APIXR A/px"
}

do_class2drods() {
  mkdir -p Class2DRods
  $MPI -np "$NP_GPU" relion_refine_mpi --i ExtractRods/particles.star --o Class2DRods/run \
    --K "$K2DR" --iter "$ITER2DR" --tau2_fudge 2 --particle_diameter "$RODDIAM" --flatten_solvent \
    --zero_mask --center_classes --oversampling 1 --psi_step 6 --offset_range 8 \
    --offset_step 2 --norm --scale --dont_combine_weights_via_disc --pool 30 --pad 2 \
    --ctf --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
  python3 -I "$HERE/view_classes.py" "Class2DRods/${LASTR}_model.star" \
    "Class2DRods/${LASTR}_classes.mrcs" Class2DRods/top_classes.jpg "$K2DR"
}

# job B
do_selrods2() {
  mkdir -p Refine5
  python3 -I "$HERE/select_classes.py" "Class2DRods/${LASTR}_data.star" Refine5/sel4.star "$SEL2"
  python3 -I "$HERE/join_coords.py" Refine5/sel4.star Extract2/particles.star Refine5/input.star
  cp "$REF5" Refine5/ref.mrc; cp "$MASK5" Refine5/solvent_mask.mrc
}

do_refine5() {
  local opt; opt=$(ls -t Refine5/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s Refine5/run_class001.mrc ]]; then
    echo "continuing refinement from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o Refine5/run \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
    return
  fi
  $MPI -np "$NP_GPU" relion_refine_mpi --i Refine5/input.star --o Refine5/run \
    --auto_refine --split_random_halves --ref Refine5/ref.mrc --trust_ref_size \
    --ini_high "${INIHIGH5:-12}" --solvent_mask Refine5/solvent_mask.mrc --solvent_correct_fsc \
    --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
    --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 \
    --auto_local_healpix_order 4 --offset_range "${OFFSET5:-10}" --offset_step 2 --sym C1 \
    --low_resol_join_halves 40 --norm --scale --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
}

do_post5() {
  mkdir -p Refine5/Post
  local thr; thr=$(python3 -I "$HERE/mask_threshold.py" Refine5/run_class001.mrc)
  relion_mask_create --i Refine5/run_class001.mrc --o Refine5/Post/mask.mrc --lowpass 15 \
    --ini_threshold "$thr" --extend_inimask 3 --width_soft_edge 6 --angpix "$APIXR"
  relion_postprocess --i Refine5/run_half1_class001_unfil.mrc --mask Refine5/Post/mask.mrc \
    --angpix "$APIXR" --adhoc_bfac "${BFAC5:--200}" --o Refine5/Post/postprocess
  echo "RESULT Refine5: $(grep -h '_rlnFinalResolution' Refine5/Post/postprocess.star | tr -s ' ' | tr '\n' ' ') particles $(nparticles Refine5/input.star)"
  python3 -I "$HERE/view3d.py" Refine5/run_model.star Refine5/map.jpg "$RISE"
  python3 -I "$HERE/rodview.py" Refine5/run_class001.mrc Refine5/run_data.star
}

t0=$SECONDS
if [[ -z "${SEL2:-}" ]]; then
  stage select 1 0 do_select
  stage copy "$NCPU" 0 do_stage
  "$HERE/01m_motioncor3.sh"
  stage ctf $((NCPU - 1)) 0 do_ctf
  stage trim 1 0 do_trim
  stage pickrods "$NCPU" 0 do_pickrods
  stage census 1 0 do_census
  stage extractrods "$NCPU" 0 do_extractrods
  stage extract2 "$NCPU" 0 do_extract2
  stage class2drods "$NCPU" "$NGPU" do_class2drods
else
  stage selrods2 1 0 do_selrods2
  stage refine5 "$NCPU" "$NGPU" do_refine5
  stage post5 1 0 do_post5
fi
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
[[ -s RodsPick_census.txt ]] && { echo "--- census"; head -n 12 RodsPick_census.txt; }
