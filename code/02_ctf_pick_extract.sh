#!/usr/bin/env bash
# CTF estimation (CTFFIND 4.1.14 on the power spectra RELION already wrote),
# 2D-template picking, and extraction at 4x binning (3.52 A/px, 64 px box).
# Rehearsal: 2D-template picks gave ~320 picks/micrograph and ~70% landed in
# particle-like 2D classes. 3D-reference and LoG picks were mostly junk.
source "$(dirname "$0")/common.sh"
cd "$PROJ"

do_ctf() {
  mkdir -p CtfFind
  # RELION's own motion correction writes power spectra; MotionCor3 does not
  local ps=()
  compgen -G "MotionCorr/Movies/*_PS.mrc" > /dev/null && ps=(--use_given_ps)
  $MPI -np $((NCPU - 1)) relion_run_ctffind_mpi --i MotionCorr/corrected_micrographs.star \
    --o CtfFind/ --ctffind_exe "$CTFFIND" --Box 512 --ResMin 30 --ResMax 5 \
    --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 ${ps[@]+"${ps[@]}"} --is_ctffind4 \
    --only_do_unfinished
}

do_pick() {
  mkdir -p AutoPick
  if [[ -z "$TEMPLATES" || ! -s "$TEMPLATES" ]]; then
    # No templates (the tournament start): Laplacian-of-Gaussian picks, the bootstrap for 2D.
    # PDIAM is the particle diameter in A; 0.67x to 1.08x matches the rehearsal's 80-130 for 120.
    : "${PDIAM:?no TEMPLATES, so set PDIAM (particle diameter in A) for LoG picking}"
    local lmin lmax
    lmin=$(awk -v d="$PDIAM" 'BEGIN{printf "%d", d*0.67}'); lmax=$(awk -v d="$PDIAM" 'BEGIN{printf "%d", d*1.08}')
    echo "LoG picking, diameters $lmin-$lmax A"
    $MPI -np "$NCPU" relion_autopick_mpi --i CtfFind/micrographs_ctf.star --odir AutoPick/ \
      --pickname autopick --LoG --LoG_diam_min "$lmin" --LoG_diam_max "$lmax" --shrink 0 \
      --lowpass 20 --LoG_adjust_threshold 0 --LoG_upper_threshold 5 --particle_diameter "$PDIAM"
    return 0
  fi
  # In autopick the leader rank also picks, unlike refinement, so one rank per GPU
  # (NGPU ranks, not NGPU+1). An extra rank has no GPU in the list and segfaults.
  $MPI -np "$NGPU" relion_autopick_mpi --i CtfFind/micrographs_ctf.star --odir AutoPick/ \
    --pickname autopick --ref "$TEMPLATES" --ang 5 --lowpass 20 --angpix_ref "$ANGPIX_REF" \
    --particle_diameter "$DIAM" --shrink 0.5 --threshold 0.25 --min_distance 90 \
    --max_stddev_noise 1.1 --gpu "$GPUARG"
}

do_extract4() {
  mkdir -p Extract4
  $MPI -np "$NCPU" relion_preprocess_mpi --i CtfFind/micrographs_ctf.star \
    --coord_list AutoPick/autopick.star --part_star Extract4/particles.star \
    --part_dir Extract4/ --extract --extract_size 256 --scale 64 --norm --bg_radius 25 \
    --white_dust -1 --black_dust -1 --invert_contrast
  echo "picked particles: $(nparticles Extract4/particles.star)"
}

stage ctf $((NCPU - 1)) 0 do_ctf
stage pick "$NCPU" "$NGPU" do_pick
stage extract4 "$NCPU" 0 do_extract4
