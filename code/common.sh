#!/usr/bin/env bash
# Shared settings and helpers for the EMPIAR-10581 rehearsal pipeline (RELION 5).
# Every stage script sources this. Override any value from the environment, or load a
# preset with PRESET=/data/pipeline/presets/tournament.env.
set -euo pipefail
export PATH=/opt/relion/install/bin:$PATH
if [[ -n "${PRESET:-}" ]]; then source "$PRESET"; fi

: "${PROJ:=/opt/dlami/nvme/full}"          # project directory, keep on local NVMe
: "${MOVIES:=/opt/dlami/nvme/movies}"      # staged movies, keep on local NVMe
: "${SCRATCH:=/opt/dlami/nvme/scratch}"    # RELION particle scratch
: "${CTFFIND:=/opt/ctffind/bin/ctffind}"
: "${TEMPLATES=/data/templates/good6.mrcs}"  # 2D class averages used as picking templates
: "${ANGPIX_REF:=3.52}"                    # pixel size of the templates
: "${ANGPIX:=0.8844}"                      # movie header says 0.8844 A/px
: "${MOVIE_EXT:=mrc}"                       # mrc for EMPIAR-10581, tiff for the tournament
: "${GAIN:=}"                              # gain reference file (mrc); empty = none
: "${GAIN_ROT:=0}"                         # 0..3 = 0, 90, 180, 270 degrees counter-clockwise
: "${GAIN_FLIP:=0}"                        # 0 none, 1 upside down, 2 left-right
: "${MIN_BYTES:=1000000000}"               # smaller movie files are treated as truncated
: "${KV:=300}"                              # acceleration voltage in kV. EMPIAR-10581 is 200, the tournament is 300
: "${CS:=2.7}"
: "${MC_PATCH:=5}"                          # motion-correction patches per axis; 1 = global alignment only
: "${MC_LAST_FRAME:=-1}"                    # last frame summed; -1 = all. See the motion-correction experiment
: "${DOSE:=1.0}"                           # e/A^2/frame, not verified for this dataset
: "${DIAM:=130}"                           # mask diameter in A
: "${SYM:=C3}"
: "${N2D:=100000}"                         # particles sent to 2D classification
: "${K2D:=60}"                             # number of 2D classes
: "${RATE_USD_H:=5.672}"                   # g5.12xlarge on-demand, us-east-1

# Under SLURM the job sees the whole node's GPUs through nvidia-smi unless the cgroup hides
# them, so take the allocation from SLURM when it is set. Both can be overridden.
NCPU=${NCPU:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
if [[ -z "${NGPU:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then NGPU=${SLURM_GPUS_ON_NODE}; else NGPU=$(nvidia-smi -L | wc -l); fi
fi
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
GPUARG=$(seq -s: 0 $((NGPU - 1)))          # "0:1:2:3", one GPU per MPI worker
# --oversubscribe: under SLURM Open MPI counts one slot per task, so -np N fails otherwise
MPI="mpirun --allow-run-as-root --use-hwthread-cpus --oversubscribe"
NP_GPU=$((NGPU + 1))                       # master plus one worker per GPU
THREADS_GPU=$((NCPU / NGPU))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PROJ" "$PROJ/.done" "$SCRATCH"

# Gain reference options for RELION's motion correction. Empty GAIN means the movies are
# already gain corrected (true for EMPIAR-10581). The tournament gain is not normalized,
# so it needs the file plus the right rotation and flip (see 01a_gain_test.sh).
GAINARGS=()
if [[ -n "$GAIN" ]]; then GAINARGS=(--gainref "$GAIN" --gain_rot "$GAIN_ROT" --gain_flip "$GAIN_FLIP"); fi

# Results live on local NVMe for speed, and a stop or a timer wipes it. After every stage
# the project is copied to EBS (PERSIST_DIR), so only the stage in flight can be lost.
# Extract1* (full-size particle stacks, ~17 GB) is skipped: it regenerates in about a minute.
# The excludes are anchored at the project root (leading slash). An unanchored 'Movies' also
# matched MotionCorr/Movies, CtfFind/Movies and AutoPick/Movies, i.e. the corrected
# micrographs and the pick coordinates, which are the outputs worth keeping.
# Under SLURM the project sits on node-local /scratch, which vanishes with the job, so the
# default destination is the shared home (EFS on SBCloud) instead of the rehearsal's /data.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then : "${PERSIST_DIR:=$HOME/results}"; fi
: "${PERSIST_DIR:=/data/results}"
persist() {
  [[ -n "${NO_PERSIST:-}" ]] && return 0
  local dest="$PERSIST_DIR/$(basename "$PROJ")"
  mkdir -p "$dest"
  # Under SLURM also skip the corrected micrographs (about 190 MB per movie from MotionCor3);
  # copying them would cost more node time than regenerating the few that are needed.
  local ex=(--exclude '/Extract1*' --exclude '/Movies' --exclude '/movies')
  # PERSIST_MC=1 keeps them anyway (about 28 GB per 300 movies), so later jobs skip MotionCor3.
  [[ -n "${SLURM_JOB_ID:-}" && -z "${PERSIST_MC:-}" ]] && ex+=(--exclude '/MotionCorr/Movies')
  if command -v rsync > /dev/null 2>&1; then
    rsync -a "${ex[@]}" "$PROJ"/ "$dest"/
  else
    local tx=(); local e
    for e in "${ex[@]}"; do [[ "$e" == --exclude ]] || tx+=(--exclude=".$e"); done
    tar -C "$PROJ" "${tx[@]}" -cf - . | tar -C "$dest" -xf -
  fi
}

# stage NAME CPUS GPUS CMD...   run once, time it, append to timing.csv, skip if done.
# CPUS and GPUS are what the stage uses, kept so the run can be priced under any billing
# model (per instance-hour, per core-hour, per GPU-hour) once the real one is known.
stage() {
  local name=$1 cpus=$2 gpus=$3; shift 3
  if [[ -e "$PROJ/.done/$name" && -z "${FORCE:-}" ]]; then echo "skip $name (done)"; return 0; fi
  echo "== $name  $(date -u +%H:%M:%S)"
  local t0; t0=$(date +%s)
  "$@"
  local sec=$(( $(date +%s) - t0 ))
  awk -v n="$name" -v s="$sec" -v r="$RATE_USD_H" -v c="$cpus" -v g="$gpus" \
    'BEGIN{printf "%s,%d,%.2f,%d,%d\n", n, s, s/3600*r, c, g}' >> "$PROJ/timing.csv"
  touch "$PROJ/.done/$name"
  persist || echo "WARNING: persisting $name to $PERSIST_DIR failed" >&2
  echo "== $name done in ${sec}s"
}

# nparticles STAR  count particle rows in a RELION star file
nparticles() { grep -c "\.mrcs" "$1"; }
