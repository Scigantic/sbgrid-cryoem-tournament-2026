#!/usr/bin/env bash
# Full dataset, one SLURM array task per chunk. Task i of T takes every T-th movie (so each
# chunk spans the whole collection), stages it on node-local scratch, and runs RELION's own
# motion correction (float16 micrographs plus power spectra; polishing needs RELION's motion
# metadata, which MotionCor3 does not write), CTFFIND on those power spectra, the rod picker,
# and extraction at bin 4 (2D cleaning), bin 2 (refinement tests) and bin 1 (float16, final
# refinement). Raw movies are deleted once corrected; the polishing pass stages them again.
# Everything except the raw movies, micrographs included, persists to ~/results/full_tNN, so
# the merge job and the polishing pass can start from any node.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch --array=0-9 slurm/chunk_full.sh          (T defaults to the array size)
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export CTFFIND=${CTFFIND:-$(command -v ctffind)}
SRC=${SRC:-/shared/s3/tournament}; PAT=${MOVIE_PAT:-*_fractions.tiff}
I=${TASK:-${SLURM_ARRAY_TASK_ID:?run as an array job or set TASK}}
T=${NTASK:-${SLURM_ARRAY_TASK_COUNT:?set NTASK}}
TAG=$(printf 'full_t%02d' "$I")
export PROJ=${PROJ:-/scratch/$TAG} MOVIES=${MOVIES:-/scratch/${TAG}_movies}
export SCRATCH=${SCRATCH:-$PROJ/scratch} PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export PERSIST_MC=1          # keep the micrographs and their motion star files (polishing)
mkdir -p "$MOVIES" "$PROJ/.done"
source "$HERE/common.sh"

python3 -I -c 'import numpy, scipy, skimage, PIL' || { echo "node python lacks scipy/skimage/PIL"; exit 2; }

KEPT="$PERSIST_DIR/$(basename "$PROJ")"   # where persist() writes
if [[ -z "${FRESH:-}" && -d "$KEPT/.done" ]]; then
  t0=$SECONDS
  rsync -a --exclude '/Movies' "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' ')) in $((SECONDS-t0)) s"
fi
cd "$PROJ"
ln -sfn "$MOVIES" Movies
# a task that died on another node lost its staged movies; motion correction resumes
# (--only_do_unfinished) from the micrographs restored above, so only the copy is redone
if [[ ! -e .done/mcr ]]; then
  [[ -s pick.txt && $(ls "$MOVIES" | wc -l) -ge $(wc -l < pick.txt) ]] || rm -f .done/copy
fi
[[ -n "${GAIN_SRC:-}" ]] || GAIN_SRC=$(find "$SRC" -type f -iname '*gain*' | head -1)

RISE=${RISE:-40}; RODBOX_A=${RODBOX_A:-500}; RODSCALE=${RODSCALE:-80}
REFBOX_A=${REFBOX_A:-300}
BOX=$(awk -v a="$ANGPIX" -v b="$RODBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
NP_MC=${NP_MC:-$(( NCPU / 4 ))}; J_MC=${J_MC:-4}
MCARGS=(--use_own --first_frame_sum 1 --last_frame_sum -1 --bin_factor 1 --bfactor 150
        --dose_per_frame "$DOSE" --preexposure 0 --patch_x "${MC_PATCH:-5}" --patch_y "${MC_PATCH:-5}"
        --gainref gain.mrc --gain_rot 0 --gain_flip "${GAIN_FLIP:-0}" --dose_weighting
        --float16 --grouping_for_ps 4 --ps_size 512 --only_do_unfinished)
echo "task $I of $T; rod box $BOX -> $RODSCALE, refine box $RBOX -> $RSCALE and $RBOX; mc $NP_MC x $J_MC"

do_select() {
  find "$SRC" -type f -name "$PAT" | sort > all_movies.txt
  awk -v t="$T" -v i="$I" '(NR - 1) % t == i' all_movies.txt > pick.txt
  echo "available $(wc -l < all_movies.txt) movies, task $I takes $(wc -l < pick.txt)"
  # raw movies (~0.19 GB) are deleted after correction; micrographs + spectra ~0.05 GB,
  # particle stacks ~0.03 GB per movie
  local free_gb need_gb
  free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
  need_gb=$(( $(wc -l < pick.txt) * 30 / 100 ))
  echo "scratch free ${free_gb} GB, need about ${need_gb} GB"
  (( free_gb >= need_gb )) || { echo "not enough node-local space"; rm -f pick.txt; exit 2; }
}

do_stage() {
  local t0=$SECONDS
  xargs -a pick.txt -P 12 -I{} cp -n {} "$MOVIES"/
  echo "staged $(ls "$MOVIES" | wc -l) movies ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_import() {
  # RELION multiplies by the gain, and the tournament gain is a multiplier: no inversion
  python3 -I "$HERE/gain_to_mrc.py" "$GAIN_SRC" gain.mrc > /dev/null
  mkdir -p Import MotionCorr/Movies   # relion_import does not create its output directory
  relion_import --do_movies --optics_group_name opticsGroup1 --angpix "$ANGPIX" --kV "$KV" \
    --Cs "$CS" --Q0 0.1 --beamtilt_x 0 --beamtilt_y 0 --i "Movies/*.$MOVIE_EXT" \
    --odir Import/ --ofile movies.star
  echo "imported $(grep -c "\.$MOVIE_EXT" Import/movies.star) movies"
}

do_mcr() {
  local t0=$SECONDS
  $MPI -np "$NP_MC" relion_run_motioncorr_mpi --i Import/movies.star --o MotionCorr/ \
    "${MCARGS[@]}" --j "$J_MC"
  local n; n=$(grep -c 'MotionCorr/Movies/' MotionCorr/corrected_micrographs.star || true)
  echo "corrected $n movies in $((SECONDS-t0)) s ($(awk -v s=$((SECONDS-t0)) -v n="$n" 'BEGIN{printf "%.2f", s/(n>0?n:1)}') s/movie)"
  (( n >= $(wc -l < pick.txt) * 95 / 100 )) || { echo "too few corrected movies"; exit 3; }
  rm -f "$MOVIES"/*
  echo "raw movies removed; scratch: $(df -h "$PROJ" | awk 'NR==2{print $4" free"}')"
}

do_ctf() {
  mkdir -p CtfFind
  $MPI -np $((NCPU - 1)) relion_run_ctffind_mpi --i MotionCorr/corrected_micrographs.star \
    --o CtfFind/ --ctffind_exe "$CTFFIND" --use_given_ps --Box 512 --ResMin 30 --ResMax 5 \
    --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 --is_ctffind4 --only_do_unfinished
  echo "ctf: $(grep -c 'MotionCorr/Movies/' CtfFind/micrographs_ctf.star) micrographs"
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

extract() {  # DIR BOX SCALE extra-args...
  local d=$1 box=$2 sc=$3; shift 3
  local scale=(); (( sc == box )) || scale=(--scale "$sc")
  mkdir -p "$d"
  $MPI -np "$NCPU" relion_preprocess_mpi --i CtfFind/micrographs_ctf.star \
    --coord_list RodsPick/rods.star --part_star "$d/particles.star" \
    --part_dir "$d/" --extract --extract_size "$box" ${scale[@]+"${scale[@]}"} --norm \
    --bg_radius $((sc * 3 / 8)) --white_dust -1 --black_dust "${BLACK_DUST:-5}" --invert_contrast "$@"
  echo "$d: $(nparticles "$d/particles.star") particles, box $box -> $sc px"
}

t0=$SECONDS
stage select 1 0 do_select
stage copy "$NCPU" 0 do_stage
stage import 1 0 do_import
stage mcr "$NCPU" 0 do_mcr
stage ctf $((NCPU - 1)) 0 do_ctf
stage pickrods "$NCPU" 0 do_pickrods
stage census 1 0 do_census
stage extractrods "$NCPU" 0 extract ExtractRods "$BOX" "$RODSCALE"
stage extract2 "$NCPU" 0 extract Extract2 "$RBOX" "$RSCALE"
stage extractb1 "$NCPU" 0 extract ExtractB1 "$RBOX" "$RBOX" --float16
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
[[ -s RodsPick_census.txt ]] && { echo "--- census"; head -n 12 RodsPick_census.txt; }
# every stage is already persisted; free the node scratch, which outlives the job
cd /; rm -rf "$PROJ" "$MOVIES"
