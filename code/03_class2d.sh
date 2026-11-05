#!/usr/bin/env bash
# 2D classification of a random subset of the picks, then render the class averages
# so someone can choose the good classes by eye (04_select_and_init.sh takes the list).
# Rehearsal: 83k particles x 50 classes was ~13 min PER iteration on one A10G and the
# classes collapsed on junk picks, so classify a subset. Average images of good
# classes look like compact particles with internal structure; radial streaks,
# edge halves and noise are junk.
source "$(dirname "$0")/common.sh"
cd "$PROJ"
ITER2D=${ITER2D:-20}      # 2D iterations; on T4 GPUs one iteration of 100k particles took 14 min
LAST=$(printf 'run_it%03d' "$ITER2D")

do_subset() {
  mkdir -p Class2D
  local total; total=$(nparticles Extract4/particles.star)
  if ((N2D >= total)); then
    cp Extract4/particles.star Class2D/input.star
  else
    relion_star_handler --i Extract4/particles.star --o Class2D/split --split \
      --random_order --random_seed 1 --size_split "$N2D" > /dev/null
    cp Class2D/split_split1. Class2D/input.star
  fi
  echo "2D input: $(nparticles Class2D/input.star) of $total particles"
}

do_class2d() {
  $MPI -np "$NP_GPU" relion_refine_mpi --i Class2D/input.star --o Class2D/run \
    --K "$K2D" --iter "$ITER2D" --tau2_fudge 2 --particle_diameter "$DIAM" --flatten_solvent \
    --zero_mask --center_classes --oversampling 1 --psi_step 12 --offset_range 5 \
    --offset_step 2 --norm --scale --dont_combine_weights_via_disc --pool 30 --pad 2 \
    --ctf --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
}

do_view() {
  python3 -I "$HERE/view_classes.py" "Class2D/${LAST}_model.star" "Class2D/${LAST}_classes.mrcs" \
    Class2D/top_classes.jpg 24
}

stage subset 1 0 do_subset
stage class2d "$NCPU" "$NGPU" do_class2d
stage view2d 1 0 do_view
echo "Look at $PROJ/Class2D/top_classes.jpg, then run: CLASSES=13,24,... $HERE/04_select_and_init.sh"
