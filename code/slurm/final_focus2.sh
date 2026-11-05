#!/usr/bin/env bash
#SBATCH --no-requeue
# Same as final_focus.sh, except the particle subtraction runs one polished chunk at a time
# (about 99k particles per pass, a fresh relion_particle_subtract each time). Jobs 89 and 90
# lost their node about 8 minutes into a single subtraction over all 989k particles; the
# per-chunk passes bound its memory. A monitor logs memory and disk every 15 s to
# $KEPT/monitor.log so a failure leaves evidence off the node.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 20:00:00 slurm/final_focus2.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/focus} SCRATCH=${SCRATCH:-/scratch/focus/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
FIN=${FIN:-$PERSIST_DIR/final}
NEWBOX=${NEWBOX:-256}; FOCUS_DIAM=${FOCUS_DIAM:-200}; INI_HIGH=${INI_HIGH:-7}
NP_SUB=${NP_SUB:-24}; SEED=${SEED:-20261028}
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
mkdir -p "$KEPT"
(
  echo "--- job ${SLURM_JOB_ID:-?} on $(hostname) $(date -u +%F)"
  while true; do
    echo "$(date -u +%T)" \
      "$(free -m | awk '/^Mem/{print "mem_used_mb",$3,"avail_mb",$7} /^Swap/{print "swap_mb",$3}' | tr '\n' ' ')" \
      "scratch_used $(df -BG --output=used "$PROJ" | tail -n 1 | tr -d ' ')" \
      "root_avail $(df -BG --output=avail / | tail -n 1 | tr -d ' ')" \
      "shm_mb $(df -m --output=used /dev/shm | tail -n 1 | tr -d ' ')" \
      "top_rss_mb $(ps -eo rss=,comm= --sort=-rss | head -n 3 | awk '{printf "%d:%s ", $1/1024, $2}')" \
      || true
    sleep 15
  done
) >> "$KEPT/monitor.log" 2>&1 &

CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$FIN/Refine10/run_optimiser.star" "$FIN/Refine10/run_data.star" \
         "$FIN/Refine10/run_class001.mrc" "$FIN/Refs/solvent_mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( ${#CHUNKS[@]} * 22 + 989 * NEWBOX * NEWBOX * 2 / 1000000 + 40 ))
echo "chunks: ${#CHUNKS[@]}; scratch free ${free_gb} GB, need about ${need_gb} GB"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

keep() { mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/; }
if [[ -d "$KEPT/.done" ]]; then
  rsync -a --exclude monitor.log "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' '))"
fi
cd "$PROJ"
# node-local stacks do not survive a move to another node
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep .done/subtract
echo "$NCPU cpus, $NGPU gpus; box $NEWBOX, diameter $FOCUS_DIAM A; $NP_SUB subtract ranks"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_refs() {  # Refine10's final files (paths in its optimiser are relative to here), body masks
  mkdir -p Refine10 Refs
  rsync -a --exclude 'run_it*' --exclude Post "$FIN/Refine10/" Refine10/
  cp "$FIN/Refs/solvent_mask.mrc" Refs/
  python3 -I "$HERE/make_body_masks.py" Refine10/run_class001.mrc Refs/solvent_mask.mrc \
    "$ANGPIX" Bodies "${SOFT:-12}"
  python3 -I "$HERE/recenter.py" Refine10/run_class001.mrc Bodies/body2.mrc "$ANGPIX" \
    "$NEWBOX" Focus0
  keep Bodies Focus0
}

do_subtract() {  # keeps the density inside body2, recentred on it, in a NEWBOX box; per chunk
  local t0=$SECONDS c star outs=() total=0 n
  rm -rf Subtract; mkdir -p Subtract/in
  for c in "${CHUNKS[@]}"; do
    # optics block and headers as is, particle rows of this chunk only
    awk -v c="$c/" 'tolower($0) ~ /^data_particles/ {p=1}
      { if (!p || index($0, "@") == 0 || index($0, c)) print }' \
      Refine10/run_data.star > "Subtract/in/$c.star"
    n=$(nparticles "Subtract/in/$c.star"); total=$((total + n))
    echo "subtract $c: $n particles, start $(date -u +%T)"
    $MPI -np "$NP_SUB" relion_particle_subtract_mpi --i Refine10/run_optimiser.star \
      --data "Subtract/in/$c.star" --mask Bodies/body2.mrc --o "Subtract/$c/" \
      --recenter_on_mask --new_box "$NEWBOX" --float16 > "Subtract/$c.log" 2>&1 \
      || { tail -n 20 "Subtract/$c.log"; return 1; }
    star=$(ls "Subtract/$c/particles_subtracted.star" 2>/dev/null \
      || ls "Subtract/$c/"*.star | head -n 1)
    outs+=("$star")
    echo "  done $c: $(nparticles "$star") out, $(free -g | awk '/^Mem/{print $3}') GB in use"
  done
  local want; want=$(nparticles Refine10/run_data.star)
  (( total == want )) || { echo "chunk split covers $total of $want particles"; return 1; }
  relion_star_handler --combine --i "${outs[*]}" --o Subtract/combined.star > Subtract/combine.log
  ln -sf combined.star Subtract/input.star
  n=$(nparticles Subtract/input.star)
  (( n == want )) || { echo "combined star has $n of $want particles"; return 1; }
  echo "subtracted $n particles in $((SECONDS-t0)) s ($(du -sm Subtract | cut -f1) MB)"
}

do_refine() {
  mkdir -p Focus
  ( while sleep 600; do rsync -a Focus "$KEPT"/ 2>/dev/null || true; done ) &
  local saver=$!
  local opt; opt=$(ls -t Focus/run_it*_optimiser.star 2>/dev/null | head -n 1 || true)
  if [[ -n "$opt" ]]; then
    echo "continuing from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o Focus/run \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG"
  else
    $MPI -np "$NP_GPU" relion_refine_mpi --i Subtract/input.star --o Focus/run --auto_refine \
      --split_random_halves --ref Focus0/ref.mrc --trust_ref_size --ini_high "$INI_HIGH" \
      --solvent_mask Focus0/mask.mrc --solvent_correct_fsc \
      --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$FOCUS_DIAM" \
      --flatten_solvent --zero_mask --oversampling 1 --healpix_order 4 \
      --auto_local_healpix_order 4 --offset_range 5 --offset_step 1 --sym C1 \
      --low_resol_join_halves 40 --norm --scale --random_seed "$SEED" \
      --j "$THREADS_GPU" --gpu "$GPUARG"
  fi
  kill "$saver" 2>/dev/null || true
  keep Focus
}

do_post() {
  mkdir -p Focus/Post
  relion_postprocess --i Focus/run_half1_class001_unfil.mrc --mask Focus0/mask.mrc \
    --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o Focus/Post/postprocess > Focus/Post/log.txt
  echo "RESULT Focus: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' Focus/Post/postprocess.star | tr -s ' ' | tr '\n' ' ') particles $(nparticles Focus/run_data.star)"
  keep Focus
}

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage refs 1 0 do_refs
stage subtract "$NCPU" 0 do_subtract
stage focus "$NCPU" "$NGPU" do_refine
stage postf 1 0 do_post
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
