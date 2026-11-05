#!/usr/bin/env bash
# Rod picking parameter sweep on a few micrographs. The first rod pass found 10 rods in
# 300 micrographs because RELION drops helical tracks shorter than --particle_diameter
# (the mask), and 200 A is longer than most rods here. Try small masks and a few
# thresholds on NMIC micrographs, print a census per variant, and draw the picks of each
# variant on the first three micrographs. Needs ~/results/chunk with MotionCorr/Movies kept.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/pick_sweep.sh      (SBATCH_* variables as for chunk_rods.sh, 1 hour is plenty)
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do module load "$m" || echo "WARNING: module $m failed"; done
export PROJ=${PROJ:-/scratch/sweep} SCRATCH=${SCRATCH:-/scratch/sweep/scratch} NO_PERSIST=1
KEPT=${KEPT:-$HOME/results/chunk}
NMIC=${NMIC:-12}
RISE=${RISE:-40}
CLASS_ITER=${CLASS_ITER:-run_it012}; RODS_CLASSES=${RODS_CLASSES:-23,17,5}
source "$HERE/common.sh"
cd "$PROJ"
mkdir -p CtfFind MotionCorr/Movies Class2D
t0=$SECONDS
cp -r "$KEPT/CtfFind/." CtfFind/
cp "$KEPT/Class2D/${CLASS_ITER}_classes.mrcs" Class2D/
awk -v n="$NMIC" '/MotionCorr\/Movies\/.*\.mrc/{if (++c > n) next} {print}' CtfFind/micrographs_ctf.star > CtfFind/mics_sub.star
awk '/MotionCorr\/Movies\/.*\.mrc/{print $1}' CtfFind/mics_sub.star | while read -r f; do cp "$KEPT/$f" "$f"; done
echo "staged $NMIC micrographs and CTF in $((SECONDS-t0)) s"
ANGPIX_REF=$(awk -v a="$ANGPIX" 'BEGIN{printf "%.4f", a*4}')
{ printf '\ndata_\n\nloop_\n_rlnReferenceImage #1\n'
  for c in ${RODS_CLASSES//,/ }; do echo "$c@Class2D/${CLASS_ITER}_classes.mrcs"; done
} > rod_templates.star

# name  mode(log|tpl)  extra autopick arguments
# Neither helical nor plain template picking found the rods (0-4 picks on 12 micrographs at
# any threshold), while LoG blob picking at 100-162 A found 700 per micrograph. So try LoG at
# the bead scale (40-80 A): a pick per bead along each rod, which rods.py joins into rods.
VARIANTS=${VARIANTS:-"log40_80 log --LoG_diam_min 40 --LoG_diam_max 80 --LoG_adjust_threshold 0
log40_80p log --LoG_diam_min 40 --LoG_diam_max 80 --LoG_adjust_threshold 0.5
log50_100 log --LoG_diam_min 50 --LoG_diam_max 100 --LoG_adjust_threshold 0
log30_60 log --LoG_diam_min 30 --LoG_diam_max 60 --LoG_adjust_threshold 0
log60_120 log --LoG_diam_min 60 --LoG_diam_max 120 --LoG_adjust_threshold 0
tpl120_t02 tpl --particle_diameter 120 --threshold 0.02
tpl120_lp30 tpl --particle_diameter 120 --threshold 0.10 --lowpass 30 --ang 10"}
echo "$VARIANTS" | while read -r name mode rest; do
  [[ -n "$name" ]] || continue
  t1=$SECONDS
  mkdir -p "Sweep/$name"
  read -r -a extra <<< "$rest"
  if [[ "$mode" == log ]]; then
    $MPI -np "$NCPU" relion_autopick_mpi --i CtfFind/mics_sub.star --odir "Sweep/$name/" \
      --pickname autopick --LoG --shrink 0 --lowpass 20 --LoG_upper_threshold 5 \
      --particle_diameter 60 "${extra[@]}" > "Sweep/$name/log.txt" 2>&1 < /dev/null \
      || { echo "variant $name FAILED:"; grep -v MotionCorr/Movies "Sweep/$name/log.txt" | tail -5; continue; }
  else
    $MPI -np "$NGPU" relion_autopick_mpi --i CtfFind/mics_sub.star --odir "Sweep/$name/" \
      --pickname autopick --ref rod_templates.star --angpix_ref "$ANGPIX_REF" --ang 5 --lowpass 20 \
      --shrink 0.5 --min_distance "$RISE" --max_stddev_noise -1 --gpu "$GPUARG" \
      "${extra[@]}" > "Sweep/$name/log.txt" 2>&1 < /dev/null \
      || { echo "variant $name FAILED:"; grep -v MotionCorr/Movies "Sweep/$name/log.txt" | tail -5; continue; }
  fi
  echo "=== variant $name ($mode $rest) in $((SECONDS-t1)) s"
  python3 -I "$HERE/rods.py" "Sweep/$name/autopick.star" "$RISE" "Sweep/$name/Rods" "$ANGPIX" | head -14
  i=0
  awk '/MotionCorr\/Movies\/.*\.mrc/{print $1}' CtfFind/mics_sub.star | head -3 | while read -r f; do
    stem=$(basename "$f" .mrc)
    c="Sweep/$name/MotionCorr/Movies/${stem}_autopick.star"
    [[ -s "$c" ]] || { echo "  micrograph $i: no picks"; i=$((i+1)); continue; }
    python3 -I "$HERE/overlay.py" "$f" "$c" "Sweep/$name/overlay_mic$i.jpg" 8 | sed "s/^/  micrograph $i: /"
    i=$((i+1))
  done
done
mkdir -p "$HOME/results/sweep"
for d in Sweep/*/; do mkdir -p "$HOME/results/sweep/$(basename "$d")"; cp "$d"/*.jpg "$d"/log.txt "$HOME/results/sweep/$(basename "$d")/" 2>/dev/null || true; done
echo "overlays copied to ~/results/sweep; total $((SECONDS-t0)) s"
