#!/usr/bin/env bash
#SBATCH --no-requeue
# CTF refinement test on the focused map (job 91). The subtracted particles were node-local
# and are gone, so they are rebuilt exactly as in final_focus2.sh (same chunks, same ranks),
# then checked name for name against job 91's run_data.star so its poses still apply.
# Then: anisotropic magnification, then beam tilt + odd (3) and even (4) aberrations +
# per-particle defocus, then half-map reconstructions with job 91's poses before (control)
# and after CTF refinement, both post-processed under the same focused mask. No re-alignment;
# that is a separate job if this gains enough.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 8:00:00 slurm/final_ctf.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/ctf} SCRATCH=${SCRATCH:-/scratch/ctf/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
FIN=${FIN:-$PERSIST_DIR/final}
FOC=${FOC:-$PERSIST_DIR/focus}
NEWBOX=${NEWBOX:-256}; NP_SUB=${NP_SUB:-24}; NP_CPU=${NP_CPU:-24}; J_CPU=${J_CPU:-2}
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
mkdir -p "$KEPT"
(
  echo "--- job ${SLURM_JOB_ID:-?} on $(hostname) $(date -u +%F)"
  while true; do
    echo "$(date -u +%T)" \
      "$(free -m | awk '/^Mem/{print "mem_used_mb",$3,"avail_mb",$7}')" \
      "scratch_used $(df -BG --output=used "$PROJ" | tail -n 1 | tr -d ' ')" \
      "top_rss_mb $(ps -eo rss=,comm= --sort=-rss | head -n 3 | awk '{printf "%d:%s ", $1/1024, $2}')" \
      || true
    sleep 30
  done
) >> "$KEPT/monitor.log" 2>&1 &

CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$FIN/Refine10/run_optimiser.star" "$FIN/Refine10/run_data.star" \
         "$FIN/Refine10/run_class001.mrc" "$FIN/Refs/solvent_mask.mrc" \
         "$FOC/Focus/run_data.star" "$FOC/Focus/run_half1_class001_unfil.mrc" \
         "$FOC/Focus/run_half2_class001_unfil.mrc" "$FOC/Focus/Post/postprocess.star" \
         "$FOC/Focus0/mask.mrc" "$FOC/Bodies/body2.mrc"; do
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
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep .done/subtract .done/namecheck
echo "$NCPU cpus; box $NEWBOX; $NP_SUB subtract ranks; ctf/reconstruct $NP_CPU x $J_CPU"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_refs() {  # Refine10 for the subtraction; job 91's bodies, focused mask, poses and post
  mkdir -p Refine10 Refs Focus/Post
  rsync -a --exclude 'run_it*' --exclude Post "$FIN/Refine10/" Refine10/
  cp "$FIN/Refs/solvent_mask.mrc" Refs/
  rsync -a "$FOC/Bodies" "$FOC/Focus0" ./
  cp "$FOC/Focus/run_data.star" "$FOC"/Focus/run_half?_class001_unfil.mrc Focus/
  cp "$FOC/Focus/Post/postprocess.star" Focus/Post/
  keep Bodies Focus0 Focus
}

do_subtract() {  # identical to final_focus2.sh
  local t0=$SECONDS c star outs=() total=0 n
  rm -rf Subtract; mkdir -p Subtract/in
  for c in "${CHUNKS[@]}"; do
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
  done
  local want; want=$(nparticles Refine10/run_data.star)
  (( total == want )) || { echo "chunk split covers $total of $want particles"; return 1; }
  relion_star_handler --combine --i "${outs[*]}" --o Subtract/combined.star > Subtract/combine.log
  ln -sf combined.star Subtract/input.star
  n=$(nparticles Subtract/input.star)
  (( n == want )) || { echo "combined star has $n of $want particles"; return 1; }
  echo "subtracted $n particles in $((SECONDS-t0)) s ($(du -sm Subtract | cut -f1) MB)"
}

names() { awk '{for (i = 1; i <= NF; i++) if (index($i, "@")) {print $i; break}}' "$1" | sort; }

do_namecheck() {  # job 91's poses must point at stacks that exist now, image for image
  names Focus/run_data.star > Subtract/names_focus.txt
  names Subtract/input.star > Subtract/names_new.txt
  local a b d
  a=$(wc -l < Subtract/names_focus.txt); b=$(wc -l < Subtract/names_new.txt)
  d=$(comm -3 Subtract/names_focus.txt Subtract/names_new.txt | wc -l)
  echo "image names: job 91 $a, rebuilt $b, differing $d"
  (( a == b && d == 0 )) || { echo "rebuilt stacks do not match job 91's poses"; return 1; }
  local s; s=$(sed -n 1p Subtract/names_focus.txt | cut -d@ -f2)
  [[ -s "$s" ]] || { echo "first stack missing on disk"; return 1; }
}

do_ctfmag() {
  local t0=$SECONDS
  mkdir -p CtfMag
  $MPI -np "$NP_CPU" relion_ctf_refine_mpi --i Focus/run_data.star --f Focus/Post/postprocess.star \
    --m1 Focus/run_half1_class001_unfil.mrc --m2 Focus/run_half2_class001_unfil.mrc \
    --mask Focus0/mask.mrc --o CtfMag/ --fit_aniso --kmin_mag 30 --j "$J_CPU" > CtfMag/log.txt 2>&1 \
    || { tail -n 20 CtfMag/log.txt; return 1; }
  echo "magnification fit in $((SECONDS-t0)) s: $(nparticles CtfMag/particles_ctf_refine.star) particles"
  awk '/^data_optics/{p=1} /^data_particles/{exit} p' CtfMag/particles_ctf_refine.star | cut -c1-200
  keep CtfMag
}

do_ctfall() {
  local t0=$SECONDS
  mkdir -p CtfAll
  $MPI -np "$NP_CPU" relion_ctf_refine_mpi --i CtfMag/particles_ctf_refine.star \
    --f Focus/Post/postprocess.star --m1 Focus/run_half1_class001_unfil.mrc --m2 Focus/run_half2_class001_unfil.mrc \
    --mask Focus0/mask.mrc --o CtfAll/ \
    --fit_defocus --kmin_defocus 30 --fit_mode fpmff \
    --fit_beamtilt --kmin_tilt 30 --odd_aberr_max_n 3 --fit_aberr \
    --j "$J_CPU" > CtfAll/log.txt 2>&1 || { tail -n 20 CtfAll/log.txt; return 1; }
  echo "ctf fit in $((SECONDS-t0)) s: $(nparticles CtfAll/particles_ctf_refine.star) particles"
  awk '/^data_optics/{p=1} /^data_particles/{exit} p' CtfAll/particles_ctf_refine.star | cut -c1-200
  keep CtfAll
}

recon() {  # STAR OUTDIR: half maps from the given poses and CTFs, then post under the focused mask
  local t0=$SECONDS h
  mkdir -p "$2"
  for h in 1 2; do
    $MPI -np "$NP_CPU" relion_reconstruct_mpi --i "$1" --o "$2/run_half${h}_class001_unfil.mrc" \
      --subset "$h" --ctf --j "$J_CPU" > "$2/half$h.log" 2>&1 || { tail -n 20 "$2/half$h.log"; return 1; }
  done
  relion_postprocess --i "$2/run_half1_class001_unfil.mrc" --mask Focus0/mask.mrc \
    --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o "$2/postprocess" > "$2/post.log"
  echo "RESULT $2: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$2/postprocess.star" | tr -s ' ' | tr '\n' ' ') ($((SECONDS-t0)) s)"
  keep "$2"
}

do_recon0() { recon Focus/run_data.star Recon0; }
do_recon2() { recon CtfAll/particles_ctf_refine.star Recon2; }

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage refs 1 0 do_refs
stage subtract "$NCPU" 0 do_subtract
stage namecheck 1 0 do_namecheck
stage ctfmag "$NCPU" 0 do_ctfmag
stage ctfall "$NCPU" 0 do_ctfall
stage recon0 "$NCPU" 0 do_recon0
stage recon2 "$NCPU" 0 do_recon2
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
