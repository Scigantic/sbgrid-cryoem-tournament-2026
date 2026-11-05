#!/usr/bin/env bash
# Polishing test on the scale-up chunk (jobs 42/43: 1500 movies, 86,467 rods, 6.36 A at bin 2).
# What do Bayesian polishing and CTF refinement do to the B-factor? That decides how many
# movies the final run needs. MotionCor3 writes no motion model, and polishing needs RELION's,
# so the same movies are corrected again with RELION's own CPU motion correction, the refined
# particles are pointed at those micrographs (remap_mics.py, CTF values carried over) and
# re-extracted at bin 2, then
#   ref6   local refinement from the Refine5 map (absorbs any offset between the two
#          motion corrections' reference frames)
#   polish Bayesian polishing (untrained sigmas, as in the rehearsal), shiny stack at bin 2
#   ref6p  local refinement of the shiny particles
#   ctfm, ctfa, ctfd  anisotropic magnification; beam tilt + odd and even aberrations;
#          per-particle defocus (KEK's order: after polishing, as in the rehearsal's best run)
#   ref6c  local refinement after CTF refinement
# Every map is post-processed under the same mask (Refine5's) with a fitted B-factor, and
# Refine5 itself is re-posted the same way first, so all RESULT lines compare directly.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/chunk_polish.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
SRC=${SRC:-/shared/s3/tournament}
export PROJ=${PROJ:-/scratch/chunk2p} MOVIES=${MOVIES:-/scratch/chunk2_movies}
export SCRATCH=${SCRATCH:-$PROJ/scratch} PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
FROM=${FROM:-$PERSIST_DIR/chunk2}          # jobs 42/43: pick.txt, CtfFind, Refine5
mkdir -p "$MOVIES" "$PROJ/.done"
source "$HERE/common.sh"

python3 -I -c 'import numpy' || { echo "node python lacks numpy"; exit 2; }
for f in "$FROM/pick.txt" "$FROM/CtfFind/micrographs_ctf.star" "$FROM/Refine5/run_data.star" \
         "$FROM/Refine5/run_class001.mrc" "$FROM/Refine5/run_half1_class001_unfil.mrc" \
         "$FROM/Refine5/run_half2_class001_unfil.mrc" "$FROM/Refine5/Post/mask.mrc"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
# raw movies (~0.19 GB each) stay until polishing is done; float16 micrographs add ~0.05 GB
nmov=$(wc -l < "$FROM/pick.txt")
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
need_gb=$(( nmov * 30 / 100 ))
echo "scratch free ${free_gb} GB, need about ${need_gb} GB for $nmov movies"
(( free_gb >= need_gb )) || { echo "not enough node-local space"; exit 2; }

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
if [[ -z "${FRESH:-}" && -d "$KEPT/.done" ]]; then
  t0=$SECONDS
  rsync -a --exclude '/Movies' "$KEPT"/ "$PROJ"/
  echo "restored ($(ls "$PROJ/.done" | tr '\n' ' ')) in $((SECONDS-t0)) s"
fi
cd "$PROJ"
mkdir -p CtfFind Refine5/Post Mask
cp -n "$FROM/pick.txt" .
cp -n "$FROM/CtfFind/micrographs_ctf.star" CtfFind/
for f in run_data.star run_class001.mrc run_half1_class001_unfil.mrc run_half2_class001_unfil.mrc; do
  cp -n "$FROM/Refine5/$f" Refine5/
done
cp -n "$FROM/Refine5/Post/mask.mrc" Mask/mask.mrc
ln -sfn "$MOVIES" Movies
# a job that died on another node lost its movies and micrographs (node-local); polishing
# needs both, so redo those stages unless polishing already finished
if [[ ! -e .done/polish ]]; then
  [[ $(ls "$MOVIES" 2>/dev/null | wc -l) -ge "$nmov" ]] || rm -f .done/copy
  [[ $(ls MotionCorr/Movies/*.mrc 2>/dev/null | wc -l) -ge $((nmov * 9 / 10)) ]] || rm -f .done/mcr .done/gaintest
fi
[[ -n "${GAIN_SRC:-}" ]] || GAIN_SRC=$(find "$SRC" -type f -iname '*gain*' | head -1)

REFBOX_A=${REFBOX_A:-300}; REFDIAM=${REFDIAM:-220}
RBOX=$(awk -v a="$ANGPIX" -v b="$REFBOX_A" 'BEGIN{printf "%d", int(b/a/2)*2}')
RSCALE=$(( RBOX / 2 / 2 * 2 ))
APIXR=$(awk -v a="$ANGPIX" -v b="$RBOX" -v s="$RSCALE" 'BEGIN{printf "%.4f", a*b/s}')
NP_MC=${NP_MC:-$(( NCPU / 4 ))}; J_MC=${J_MC:-4}
# polishing ranks reached 12.8-19.5 GB RSS on the rehearsal's 4k movies; budget 20 GB
NP_POL=${NP_POL:-$(( RAM_GB / 20 < NCPU / 6 ? RAM_GB / 20 : NCPU / 6 ))}; J_POL=${J_POL:-6}
echo "box $RBOX -> $RSCALE px at $APIXR A/px; $NCPU cpus, $NGPU gpus, $RAM_GB GB; mc $NP_MC x $J_MC, polish $NP_POL x $J_POL"

MCARGS=(--use_own --first_frame_sum 1 --last_frame_sum -1 --bin_factor 1 --bfactor 150
        --dose_per_frame "$DOSE" --preexposure 0 --patch_x "${MC_PATCH:-5}" --patch_y "${MC_PATCH:-5}"
        --gainref gain.mrc --gain_rot 0 --dose_weighting --only_do_unfinished)

do_stage() {
  local t0=$SECONDS
  xargs -a pick.txt -P 12 -I{} cp -n {} "$MOVIES"/
  echo "staged $(ls "$MOVIES" | wc -l) movies ($(du -sm "$MOVIES" | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_import() {
  # RELION multiplies by the gain, and the tournament gain is a multiplier: no inversion
  python3 -I "$HERE/gain_to_mrc.py" "$GAIN_SRC" gain.mrc
  mkdir -p Import MotionCorr/Movies   # relion_import does not create its output directory (job 44)
  relion_import --do_movies --optics_group_name opticsGroup1 --angpix "$ANGPIX" --kV "$KV" \
    --Cs "$CS" --Q0 0.1 --beamtilt_x 0 --beamtilt_y 0 --i "Movies/*.$MOVIE_EXT" \
    --odir Import/ --ofile movies.star
  echo "imported $(grep -c "\.$MOVIE_EXT" Import/movies.star) movies"
}

do_gaintest() {
  # the same 4 movies corrected with each gain flip; the right one leaves the least noise
  # near Nyquist. MotionCor3 needed none and RELION takes the same gain options, so flip 0
  # stands unless another flip wins clearly: on synthetic K3-like movies (26 e/px, 5% gain
  # spread, 30 px drift) a wrong flip raised that noise by only 0.3%, so small margins mean
  # nothing, while a structured gain error read the wrong way would show far more than 3%.
  rm -rf GainTest; mkdir -p GainTest
  awk '/\.'"$MOVIE_EXT"'/{n++; if (n > 4) next} {print}' Import/movies.star > GainTest/movies4.star
  local fl
  for fl in 0 1 2; do
    mkdir -p "GainTest/f$fl/Movies"
    $MPI -np 5 relion_run_motioncorr_mpi --i GainTest/movies4.star --o "GainTest/f$fl/" \
      "${MCARGS[@]}" --gain_flip "$fl" --j "$(( NCPU / 5 ))" > "GainTest/f$fl.log" 2>&1
  done
  python3 -I "$HERE/gain_qc.py" GainTest/f0 GainTest/f1 GainTest/f2 | tee GainTest/qc.txt
  awk '/^best/{m = $4; gsub(/[^0-9.]/, "", m); print (m + 0 > 3 ? $2 : 0)}' GainTest/qc.txt > gain_flip.txt
  echo "gain flip used: $(cat gain_flip.txt)"
  [[ "$(cat gain_flip.txt)" == 0 ]] || echo "WARNING: gain flip $(cat gain_flip.txt) beats as-is by more than 3%"
}

do_mcr() {
  local t0=$SECONDS
  $MPI -np "$NP_MC" relion_run_motioncorr_mpi --i Import/movies.star --o MotionCorr/ \
    "${MCARGS[@]}" --gain_flip "$(cat gain_flip.txt)" --j "$J_MC" --float16 --grouping_for_ps 4 --ps_size 512
  local n; n=$(grep -c 'MotionCorr/Movies/' MotionCorr/corrected_micrographs.star || true)
  echo "corrected $n movies in $((SECONDS-t0)) s ($(awk -v s=$((SECONDS-t0)) -v n="$n" 'BEGIN{printf "%.2f", s/(n>0?n:1)}') s/movie); scratch: $(df -h "$PROJ" | awk 'NR==2{print $4" free"}')"
}

do_remap() {
  python3 -I "$HERE/remap_mics.py" MotionCorr/corrected_micrographs.star \
    CtfFind/micrographs_ctf.star Refine5/run_data.star Remap
}

do_ex6() {
  mkdir -p Extract6
  $MPI -np "$NCPU" relion_preprocess_mpi --i Remap/micrographs_ctf.star \
    --reextract_data_star Remap/particles.star --recenter --part_star Extract6/particles.star \
    --part_dir Extract6/ --extract --extract_size "$RBOX" --scale "$RSCALE" --norm \
    --bg_radius $((RSCALE * 3 / 8)) --white_dust -1 --black_dust "${BLACK_DUST:-5}" --invert_contrast
  echo "re-extracted $(nparticles Extract6/particles.star) particles"
}

refine() {  # NAME INPUT REF OFFSET_RANGE   local searches from the input poses
  local name=$1 in=$2 ref=$3 off=$4
  mkdir -p "$name"
  local opt; opt=$(ls -t "$name"/run_it*_optimiser.star 2>/dev/null | head -1 || true)
  if [[ -n "$opt" && ! -s "$name/run_class001.mrc" ]]; then
    echo "continuing $name from $opt"
    $MPI -np "$NP_GPU" relion_refine_mpi --continue "$opt" --o "$name/run" \
      --dont_combine_weights_via_disc --pool 30 --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
    return
  fi
  $MPI -np "$NP_GPU" relion_refine_mpi --i "$in" --o "$name/run" --auto_refine --split_random_halves \
    --ref "$ref" --trust_ref_size --ini_high 8 --solvent_mask Mask/mask.mrc --solvent_correct_fsc \
    --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter "$REFDIAM" \
    --flatten_solvent --zero_mask --oversampling 1 --healpix_order 4 --auto_local_healpix_order 4 \
    --offset_range "$off" --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale \
    --j "$THREADS_GPU" --gpu "$GPUARG" --preread_images
}

post() {  # NAME   same mask for every map, B-factor fitted from 10 A to the final resolution
  local name=$1
  mkdir -p "$name/Post"
  relion_postprocess --i "$name/run_half1_class001_unfil.mrc" --mask Mask/mask.mrc \
    --angpix "$APIXR" --auto_bfac --autob_lowres 10 --o "$name/Post/postprocess" > "$name/Post/log.txt"
  echo "RESULT $name: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$name/Post/postprocess.star" | tr -s ' ' | tr '\n' ' ') particles $(nparticles "$name/run_data.star")"
}

do_polish() {
  mkdir -p Polish
  $MPI -np "$NP_POL" relion_motion_refine_mpi --i Refine6/run_data.star \
    --f Refine6/Post/postprocess.star --corr_mic MotionCorr/corrected_micrographs.star \
    --first_frame 1 --last_frame -1 --o Polish/ --s_vel 0.2 --s_div 5000 --s_acc 2 \
    --combine_frames --bfac_minfreq 20 --bfac_maxfreq -1 --window "$RBOX" --scale "$RSCALE" \
    --j "$J_POL" --only_do_unfinished
  echo "shiny particles: $(nparticles Polish/shiny.star)"
}

ctfref() {  # IN POSTPROCESS OUTDIR args...
  local in=$1 f=$2 o=$3; shift 3
  mkdir -p "$o"
  $MPI -np "$NCPU" relion_ctf_refine_mpi --i "$in" --f "$f" --o "$o/" "$@" --j 1
  echo "$o: $(nparticles "$o/particles_ctf_refine.star") particles"
}

t0=$SECONDS
stage post5b 1 0 post Refine5
stage copy "$NCPU" 0 do_stage
stage import 1 0 do_import
stage gaintest "$NCPU" 0 do_gaintest
stage mcr "$NCPU" 0 do_mcr
stage remap 1 0 do_remap
stage ex6 "$NCPU" 0 do_ex6
stage ref6 "$NCPU" "$NGPU" refine Refine6 Extract6/particles.star Refine5/run_class001.mrc "${OFF6:-12}"
stage post6 1 0 post Refine6
stage polish "$NCPU" 0 do_polish
stage ref6p "$NCPU" "$NGPU" refine Refine6p Polish/shiny.star Refine6/run_class001.mrc 4
stage post6p 1 0 post Refine6p
stage ctfm "$NCPU" 0 ctfref Refine6p/run_data.star Refine6p/Post/postprocess.star CtfM --fit_aniso --kmin_mag 20
stage ctfa "$NCPU" 0 ctfref CtfM/particles_ctf_refine.star Refine6p/Post/postprocess.star CtfA \
  --fit_beamtilt --kmin_tilt 20 --odd_aberr_max_n 3 --fit_aberr --kmin_aberr 20
stage ctfd "$NCPU" 0 ctfref CtfA/particles_ctf_refine.star Refine6p/Post/postprocess.star CtfD \
  --fit_defocus --kmin_defocus 30 --fit_mode fpmfm
stage ref6c "$NCPU" "$NGPU" refine Refine6c CtfD/particles_ctf_refine.star Refine6p/run_class001.mrc 4
stage post6c 1 0 post Refine6c
python3 -I "$HERE/rodview.py" Refine6c/run_class001.mrc Refine6c/run_data.star || true
echo "--- results"; grep -h '^RESULT' "$SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID:-x}.out" 2>/dev/null || true
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
