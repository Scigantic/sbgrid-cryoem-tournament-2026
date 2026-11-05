#!/usr/bin/env bash
# One SBCloud allocation for the first half of the pipeline on a chunk of movies: stage N movies
# from the S3 mount to local /scratch (parallel copy), MotionCor3, CTF, LoG picks, extraction, 2D.
# Everything stays on the node's /scratch, which a second job on another node cannot see, and
# there is one cold start instead of one per stage. Results are copied to $HOME/results.
#   sbatch --partition=ct03g-g4dn-12xlarge --gres=gpu:t4:4 --cpus-per-task=48 --time=04:00:00 \
#     --export=ALL,PRESET=$HOME/pipeline/presets/tournament.env,N=300,PDIAM=NNN slurm/chunk_job.sh
# Prints timings and counts only. Pick the 2D classes afterwards (CLASSES=... submit/04).
set -euo pipefail
# SLURM runs a copy of this file from /var/spool/slurmd, so dirname does not find the siblings
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
: "${PDIAM:?set PDIAM, the particle diameter in A}"
# 2D classification mask a little wider than the particle; the rehearsal default (130) is for another protein
export DIAM=${DIAM:-$(awk -v d="$PDIAM" 'BEGIN{printf "%d", d*1.2}')}
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7 sbgrid/motioncor3/1.2.4}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export CTFFIND=${CTFFIND:-$(command -v ctffind)}
SRC=${SRC:-/shared/s3/tournament}; N=${N:-300}; PAT=${MOVIE_PAT:-*_fractions.tiff}
export PROJ=${PROJ:-/scratch/chunk} MOVIES=${PROJ:-/scratch/chunk}/movies SCRATCH=${SCRATCH:-${PROJ:-/scratch/chunk}/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
[[ -n "${GAIN_SRC:-}" ]] || GAIN_SRC=$(find "$SRC" -type f -iname '*gain*' | head -1)
mkdir -p "$MOVIES" "$PROJ/.done"
# What an earlier job finished (CTF, picks, bin-4 particles, done markers) was copied to the
# shared home after each stage; bring it back so a rerun only does what is left. FRESH=1 ignores it.
KEPT="$PERSIST_DIR/$(basename "$PROJ")"
if [[ -z "${FRESH:-}" && -d "$KEPT/.done" ]]; then
  t0=$SECONDS
  if command -v rsync > /dev/null 2>&1; then rsync -a "$KEPT"/ "$PROJ"/
  else tar -C "$KEPT" -cf - . | tar -C "$PROJ" -xf -; fi
  echo "restored finished stages ($(ls "$PROJ/.done" | tr '\n' ' ')) in $((SECONDS-t0)) s"
fi
touch "$PROJ/.done/stage"
cd "$PROJ"

t0=$SECONDS
cp "$GAIN_SRC" "$PROJ/gain_src.${GAIN_SRC##*.}"
export GAIN="$PROJ/gain_src.${GAIN_SRC##*.}"
if [[ -e .done/motioncor3 ]]; then
  echo "motion correction finished in an earlier job; movies not staged"
else
  find "$SRC" -type f -name "$PAT" | sort > all_movies.txt
  total=$(wc -l < all_movies.txt); k=$(( total / N )); (( k >= 1 )) || k=1
  awk -v k="$k" 'NR%k==1' all_movies.txt | head -n "$N" > pick.txt     # spread over the session
  xargs -a pick.txt -P 12 -I{} cp {} "$MOVIES"/
  echo "available $total movies, staged $(ls "$MOVIES" | wc -l) ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
fi

"$HERE/01m_motioncor3.sh"
"$HERE/02_ctf_pick_extract.sh"
"$HERE/03_class2d.sh"
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0)); movies $(ls "$MOVIES" | wc -l)"
echo "picks $(nparticles() { grep -c '\.mrcs' "$1"; }; nparticles Extract4/particles.star)"
