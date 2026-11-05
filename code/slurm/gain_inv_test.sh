#!/usr/bin/env bash
# Plain or inverted gain for RELION's own motion correction? RELION multiplies every frame by
# the gain file (motioncorr_runner.cpp), and the full run (job 48) gave it the plain gain,
# while MotionCor3 on these movies worked far better with the inverted file (CTF ranking
# 4.49 vs 7.75 A). Those agree only if MotionCor3 divides. Here the same N movies are
# corrected by RELION with each file; the right one leaves less fixed-pattern noise near
# Nyquist (gain_qc.py) and gives better CTF fits. Prints numbers only.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch slurm/gain_inv_test.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7}; do
  module load "$m" || echo "WARNING: module $m failed"
done
CTFFIND=${CTFFIND:-$(command -v ctffind)}
SRC=${SRC:-/shared/s3/tournament}; N=${N:-12}
PICK=${PICK:-$HOME/results/full_t00/pick.txt}
export PROJ=${PROJ:-/scratch/gaininv} SCRATCH=${SCRATCH:-/scratch/gaininv/scratch} NO_PERSIST=1
rm -rf "$PROJ"; mkdir -p "$PROJ/Movies" "$PROJ/.done"
source "$HERE/common.sh"   # MPI
cd "$PROJ"
[[ -n "${GAIN_SRC:-}" ]] || GAIN_SRC=$(find "$SRC" -type f -iname '*gain*' | head -1)
python3 -I "$HERE/gain_to_mrc.py" "$GAIN_SRC" gain.mrc > /dev/null
python3 -I "$HERE/gain_to_mrc.py" "$GAIN_SRC" gain_inv.mrc --invert > /dev/null
awk -v n="$N" 'NR % 120 == 1 && ++c <= n' "$PICK" | xargs -P 12 -I{} cp -n {} Movies/
echo "staged $(ls Movies | wc -l) movies"
mkdir -p Import
relion_import --do_movies --optics_group_name opticsGroup1 --angpix "$ANGPIX" --kV "$KV" \
  --Cs "$CS" --Q0 0.1 --beamtilt_x 0 --beamtilt_y 0 --i "Movies/*.$MOVIE_EXT" \
  --odir Import/ --ofile movies.star > /dev/null
for g in gain gain_inv; do
  mkdir -p "$g/Movies"
  $MPI -np "$N" relion_run_motioncorr_mpi --i Import/movies.star --o "$g/" --use_own \
    --first_frame_sum 1 --last_frame_sum -1 --bin_factor 1 --bfactor 150 --dose_per_frame "$DOSE" \
    --preexposure 0 --patch_x 5 --patch_y 5 --gainref "$g.mrc" --gain_rot 0 --gain_flip 0 \
    --dose_weighting --j 4 > "$g.log" 2>&1
  mkdir -p "$g/Ctf"
  $MPI -np "$N" relion_run_ctffind_mpi --i "$g/corrected_micrographs.star" --o "$g/Ctf/" \
    --ctffind_exe "$CTFFIND" --Box 512 --ResMin 30 --ResMax 3 --dFMin 5000 --dFMax 50000 \
    --FStep 500 --dAst 100 --is_ctffind4 > "$g/ctf.log" 2>&1
  relion_star_printtable "$g/Ctf/micrographs_ctf.star" data_micrographs _rlnCtfMaxResolution \
    _rlnCtfFigureOfMerit | sort -g | awk -v g="$g" '{r[NR]=$1; s+=$2} END {printf "%s: %d micrographs, CTF fit median %.2f A, best %.2f, mean FOM %.3f\n", g, NR, r[int((NR+1)/2)], r[1], s/NR}'
done
python3 -I "$HERE/gain_qc.py" gain/Movies gain_inv/Movies | sed 's#^[^:]*/\(gain[a-z_]*\)/Movies:#\1:#'
cd /; rm -rf "$PROJ"
