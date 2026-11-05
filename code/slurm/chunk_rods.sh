#!/usr/bin/env bash
# Second job on the same chunk: the rod pass (04r_rods.sh). Restores what the first chunk
# job kept in $HOME/results/chunk (CTF, picks, 2D classes), regenerates the corrected
# micrographs with MotionCor3 (they were too big to keep), then picks rods, counts repeats,
# extracts whole rods and classifies them.
#   export PRESET=$HOME/pipeline/presets/tournament.env PDIAM=150
#   sbatch --partition=ct03g-g5-12xlarge,ct03g-g4dn-12xlarge --gres=gpu:4 --cpus-per-task=48 \
#     --time=04:00:00 --export=ALL slurm/chunk_rods.sh
# PERSIST_MC=1 also keeps MotionCorr/Movies on EFS (about 28 GB) so a third job skips MotionCor3.
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
export PDIAM=${PDIAM:-150}
export DIAM=${DIAM:-200}     # autopick mask, A; must be at least the tube diameter and under the 226 A template box
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7 sbgrid/motioncor3/1.2.4}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export CTFFIND=${CTFFIND:-$(command -v ctffind)}
SRC=${SRC:-/shared/s3/tournament}; PAT=${MOVIE_PAT:-*_fractions.tiff}
# raw movies live beside the project, not inside it, so persist() can never copy them to EFS
export PROJ=${PROJ:-/scratch/chunk} MOVIES=${MOVIES:-/scratch/chunk_movies} SCRATCH=${SCRATCH:-${PROJ:-/scratch/chunk}/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
mkdir -p "$MOVIES" "$PROJ/.done"
KEPT="$PERSIST_DIR/$(basename "$PROJ")"
[[ -s "$KEPT/pick.txt" && -d "$KEPT/CtfFind" ]] || { echo "nothing kept in $KEPT; run slurm/chunk_job.sh first"; exit 1; }
t0=$SECONDS
if command -v rsync > /dev/null 2>&1; then rsync -a --exclude '/movies' --exclude '/Movies' "$KEPT"/ "$PROJ"/
else tar -C "$KEPT" --exclude='./movies' --exclude='./Movies' -cf - . | tar -C "$PROJ" -xf -; fi
echo "restored ($(ls "$PROJ/.done" | tr '\n' ' ')) in $((SECONDS-t0)) s"
cd "$PROJ"
GAIN=$(ls "$PROJ"/gain_src.* | head -1); export GAIN

t0=$SECONDS
if compgen -G "MotionCorr/Movies/*_DW.mrc" > /dev/null && [[ $(ls MotionCorr/Movies/*_DW.mrc | wc -l) -ge $(wc -l < pick.txt) ]]; then
  echo "corrected micrographs were kept; MotionCor3 skipped"
else
  rm -f .done/motioncor3 .done/mc3_star
  xargs -a pick.txt -P 12 -I{} cp {} "$MOVIES"/
  echo "staged $(ls "$MOVIES" | wc -l) movies ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
  "$HERE/01m_motioncor3.sh"
fi

"$HERE/04r_rods.sh"
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
echo "--- census"; cat "$PROJ/RodsPick_census.txt"
