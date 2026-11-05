#!/usr/bin/env bash
#SBATCH --no-requeue
# Is the early pipeline losing resolution? One chunk, every other micrograph that has
# Refine10 particles (about 735 movies, about 50k particles), four sets of micrographs:
#   A    RELION motion correction with the plain gain (the full run's micrographs, reused)
#   N    RELION motion correction with NO gain (control: does gain handling matter at all),
#        written as float32 like the others
#   Mi   MotionCor3 with the inverted gain (exactly as the Oct 22 run that looked better)
#   Mp   MotionCor3 with the plain gain
# Each set is re-extracted at bin 2 from Refine10's coordinates and poses, with the same
# per-micrograph CTF, and refined locally with identical settings, one GPU per set.
# Also: gain flatness on the raw counts (multiply/divide x 4 orientations), the orientation
# RELION reads TIFFs in, and CTFFIND on every set plus set A at 0.84 A/px.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 4:00:00 slurm/final_mc.sh
# Job 94 died in post (missing output folder). SKIP="" reruns the gain and CTF checks too.
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2 sbgrid/ctffind4/4.1.14-c7 sbgrid/motioncor3/1.2.4}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export CTFFIND=${CTFFIND:-$(command -v ctffind)}
MC3=${MOTIONCOR3:-$(command -v MotionCor3 || true)}
export PROJ=${PROJ:-/scratch/mc} MOVIES=${MOVIES:-/scratch/mc_movies}
export SCRATCH=${SCRATCH:-/scratch/mc/scratch} PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
CHUNK=${CHUNK:-full_t00}; EVERY=${EVERY:-2}; SEED=${SEED:-20261030}
SRCD=$PERSIST_DIR/$CHUNK; FIN=${FIN:-$PERSIST_DIR/final}
mkdir -p "$MOVIES" "$PROJ/.done"
source "$HERE/common.sh"
TOOLS="$HERE/mc_tools.py"

KEPT="$PERSIST_DIR/$(basename "$PROJ")"
mkdir -p "$KEPT"
(
  echo "--- job ${SLURM_JOB_ID:-?} on $(hostname) $(date -u +%F)"
  while true; do
    echo "$(date -u +%T)" \
      "$(free -m | awk '/^Mem/{print "mem_used_mb",$3,"avail_mb",$7}')" \
      "scratch_used $(df -BG --output=used "$PROJ" | tail -n 1 | tr -d ' ')" || true
    sleep 60
  done
) >> "$KEPT/monitor.log" 2>&1 &

[[ -x "$MC3" ]] || { echo "MotionCor3 not found"; exit 2; }
[[ -n "$CTFFIND" ]] || { echo "ctffind not found"; exit 2; }
python3 -I -c 'import numpy, scipy, tifffile' || { echo "node python lacks numpy/scipy/tifffile"; exit 2; }
for f in "$FIN/Refine10/run_data.star" "$FIN/Refine10/run_class001.mrc" "$FIN/Refs/solvent_mask.mrc" \
         "$SRCD/pick.txt" "$SRCD/gain.mrc" "$SRCD/CtfFind/micrographs_ctf.star" "$TOOLS"; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 2; }
done
free_gb=$(df -BG --output=avail "$(dirname "$PROJ")" | tail -n 1 | tr -dc '0-9')
echo "scratch free ${free_gb} GB, need about 420 GB"
(( free_gb >= 420 )) || { echo "not enough node-local space"; exit 2; }
keep() { mkdir -p "$KEPT"; rsync -a "$@" .done "$KEPT"/; }
cd "$PROJ"
ln -sfn "$MOVIES" Movies
ARMS=(A N Mi Mp)
NP_MC=${NP_MC:-10}; J_MC=${J_MC:-4}
echo "$NCPU cpus, $NGPU gpus; chunk $CHUNK, every $EVERY micrograph(s); mc $NP_MC x $J_MC"

# column number of a STAR label in the particles/micrographs loop of a file
col() { awk -v k="$2" '$1 == k {sub("#", "", $2); print $2; exit}' "$1"; }

do_select() {
  local d=$FIN/Refine10/run_data.star cm
  cm=$(col "$d" _rlnMicrographName)
  awk -v c="$CHUNK/" -v m="$cm" 'index($0, "@") && index($0, c) {print $m}' "$d" \
    | sort -u | awk -v e="$EVERY" 'NR % e == 1' > mics.txt
  sed 's#.*/##; s#\.mrc$##' mics.txt > stems.txt
  # particles of those micrographs, micrograph name pointed at each set's Mic/ folder
  awk -v c="$CHUNK/" -v m="$cm" 'FNR == NR {want[$1] = 1; next}
    tolower($0) ~ /^data_particles/ {p = 1}
    !p || !index($0, "@") {print; next}
    index($0, c) && ($m in want) {s = $m; sub(".*/", "", s); $m = "Mic/" s; print}' \
    mics.txt "$d" > parts.star
  # the same per-micrograph CTF for every set (Refine10 has no per-particle CTF columns)
  local cs=$SRCD/CtfFind/micrographs_ctf.star cn
  cn=$(col "$cs" _rlnMicrographName)
  awk -v m="$cn" 'FNR == NR {want[$1] = 1; next}
    tolower($0) ~ /^data_micrographs/ {p = 1}
    !p || $0 !~ /\.mrc/ {print; next}
    {s = $m; sub(".*/", "", s); sub("\\.mrc$", "", s); if (s in want) {$m = "Mic/" s ".mrc"; print}}' \
    stems.txt "$cs" > mics_ctf.star
  grep -F -f <(sed 's#^#/#; s#$#.tiff#' stems.txt) "$SRCD/pick.txt" > movies.txt || true
  local nm ns nv np
  nm=$(wc -l < stems.txt); nv=$(wc -l < movies.txt)
  ns=$(grep -c 'Mic/' mics_ctf.star || true); np=$(nparticles parts.star)
  echo "micrographs $nm, movies found $nv, ctf rows $ns, particles $np"
  (( nm > 100 && nv == nm && ns == nm && np > 1000 )) || { echo "selection mismatch"; return 1; }
}

do_stage() {
  local t0=$SECONDS s
  xargs -a movies.txt -P 12 -I{} cp -n {} "$MOVIES"/
  mkdir -p A/Mic
  # the full run's micrographs are float16; CTFFIND 4.1 wants float32, so rewrite them
  sed "s#^#$SRCD/MotionCorr/Movies/#; s#\$#.mrc#" stems.txt \
    | xargs -P 24 -I{} sh -c 'python3 -I "$0" f32 "$1" "A/Mic/$(basename "$1")"' "$TOOLS" {}
  cp "$SRCD/gain.mrc" gain.mrc
  python3 -I "$TOOLS" invert gain.mrc gain_inv.mrc
  mkdir -p Refs
  python3 -I "$TOOLS" bin2 "$FIN/Refine10/run_class001.mrc" Refs/ref2.mrc
  python3 -I "$TOOLS" bin2 "$FIN/Refs/solvent_mask.mrc" Refs/mask2.mrc
  echo "staged $(ls "$MOVIES" | wc -l) movies and $(ls A/Mic | wc -l) set-A micrographs in $((SECONDS-t0)) s"
  (( $(ls "$MOVIES" | wc -l) == $(wc -l < stems.txt) && $(ls A/Mic | wc -l) == $(wc -l < stems.txt) ))
}

do_gaincheck() {
  sed -n '1,40p' stems.txt | sed "s#^#$MOVIES/#; s#\$#.tiff#" > gaincheck_movies.txt
  python3 -I "$TOOLS" gaincheck gain.mrc $(cat gaincheck_movies.txt) | tee gaincheck.txt \
    || echo "gaincheck failed (not fatal)"
  keep gaincheck.txt
}

mc3_pass() {  # GAINFILE OUTDIR: MotionCor3 over all staged movies, one process per GPU
  local gain=$1 out=$2 g=0 p pids=()
  mkdir -p "$out"
  ls "$MOVIES"/*.tiff > "$out/list.txt"
  split -n "l/$NGPU" -d "$out/list.txt" "$out/part_"
  for p in "$out"/part_*; do
    (
      while read -r f; do
        s=$(basename "$f" .tiff)
        [[ -s "$out/${s}_DW.mrc" ]] && continue
        "$MC3" -InTiff "$f" -OutMrc "$PROJ/$out/$s.mrc" -Gain "$PROJ/$gain" -PixSize "$ANGPIX" \
          -FmDose "$DOSE" -kV "$KV" -Gpu "$g" -Patch 5 5 > /dev/null 2>&1 || echo "$s" >> "$out/failed.txt"
        rm -f "$out/$s.mrc" "$out/${s}_DWS.mrc" "$out/${s}_Ctf.mrc"
      done < "$p"
    ) &
    pids+=($!); g=$((g + 1))
  done
  for p in "${pids[@]}"; do wait "$p"; done   # not a bare wait: the monitor never exits
  echo "MotionCor3 $out: $(ls "$out"/*_DW.mrc 2>/dev/null | wc -l) micrographs, $(cat "$out/failed.txt" 2>/dev/null | wc -l) failed"
}

do_mc() {
  local t0=$SECONDS n
  mkdir -p Import N_mc
  relion_import --do_movies --optics_group_name opticsGroup1 --angpix "$ANGPIX" --kV "$KV" \
    --Cs "$CS" --Q0 0.1 --beamtilt_x 0 --beamtilt_y 0 --i "Movies/*.tiff" \
    --odir Import/ --ofile movies.star > Import/log.txt
  $MPI -np "$NP_MC" relion_run_motioncorr_mpi --i Import/movies.star --o N_mc/ \
    --use_own --first_frame_sum 1 --last_frame_sum -1 --bin_factor 1 --bfactor 150 \
    --dose_per_frame "$DOSE" --preexposure 0 --patch_x 5 --patch_y 5 --dose_weighting \
    --only_do_unfinished --j "$J_MC" \
    > N_mc/log.txt 2>&1 &
  local rpid=$!
  mc3_pass gain_inv.mrc Mi_mc
  mc3_pass gain.mrc Mp_mc
  wait "$rpid" || { tail -n 20 N_mc/log.txt; return 1; }
  n=$(ls N_mc/Movies/*.mrc 2>/dev/null | grep -v -c '_PS\.mrc' || true)
  echo "RELION no-gain: $n micrographs; all motion correction in $((SECONDS-t0)) s"
  local s1; s1=$(sed -n 1p stems.txt)
  python3 -I "$TOOLS" orient "$MOVIES/$s1.tiff" "N_mc/Movies/$s1.mrc" | tee orient.txt \
    || echo "orientation check failed (not fatal)"
  keep orient.txt
  rm -f "$MOVIES"/*.tiff
  local a s
  for a in N Mi Mp; do mkdir -p "$a/Mic"; done
  while read -r s; do
    ln -sf "$PROJ/N_mc/Movies/$s.mrc" "N/Mic/$s.mrc"
    ln -sf "$PROJ/Mi_mc/${s}_DW.mrc" "Mi/Mic/$s.mrc"
    ln -sf "$PROJ/Mp_mc/${s}_DW.mrc" "Mp/Mic/$s.mrc"
  done < stems.txt
  for a in "${ARMS[@]}"; do
    n=$(find -L "$a/Mic" -name '*.mrc' -size +1M | wc -l)
    echo "set $a: $n micrographs"
    (( n == $(wc -l < stems.txt) )) || { echo "set $a incomplete"; return 1; }
  done
}

ctfdiag() {  # SET PIXEL OUT
  local a=$1 px=$2 out=$3
  mkdir -p "$a/$out"
  awk -v p="$px" 'tolower($0) ~ /^data_optics/ {o = 1} tolower($0) ~ /^data_micrographs/ {o = 0}
    o && $1 == "opticsGroup1" {$3 = p; $7 = p} {print}' \
    <(printf 'data_optics\n\nloop_\n_rlnOpticsGroupName #1\n_rlnOpticsGroup #2\n_rlnMicrographOriginalPixelSize #3\n_rlnVoltage #4\n_rlnSphericalAberration #5\n_rlnAmplitudeContrast #6\n_rlnMicrographPixelSize #7\nopticsGroup1 1 %s %s %s 0.1 %s\n\ndata_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnOpticsGroup #2\n' \
      "$ANGPIX" "$KV" "$CS" "$ANGPIX"; sed 's#^#Mic/#; s#$#.mrc 1#' stems.txt) > "$a/$out/mics.star"
  ( cd "$a" && $MPI -np $((NCPU - 1)) relion_run_ctffind_mpi --i "$out/mics.star" --o "$out/" \
      --ctffind_exe "$CTFFIND" --Box 512 --ResMin 30 --ResMax 3 --dFMin 5000 --dFMax 50000 \
      --FStep 500 --dAst 100 --is_ctffind4 > "$out/log.txt" 2>&1 ) || { tail -n 20 "$a/$out/log.txt"; return 1; }
  echo "RESULT ctf $a at $px A: $(python3 -I "$TOOLS" median "$a/$out/micrographs_ctf.star" \
    _rlnCtfMaxResolution _rlnCtfFigureOfMerit _rlnDefocusU)"
}

do_ctfdiag() {
  local a
  for a in "${ARMS[@]}"; do ctfdiag "$a" "$ANGPIX" CtfDiag; done
  ctfdiag A 0.84 CtfDiag84
}

do_extract() {
  local a n box2=$((338 / 2)) want=""
  for a in "${ARMS[@]}"; do
    mkdir -p "$a/Extract"
    ( cd "$a" && $MPI -np "$NCPU" relion_preprocess_mpi --i ../mics_ctf.star \
        --reextract_data_star ../parts.star --recenter --part_star Extract/particles.star \
        --part_dir Extract/ --extract --extract_size 338 --scale "$box2" --norm \
        --bg_radius $((box2 * 3 / 8)) --white_dust -1 --black_dust 5 --invert_contrast \
        > Extract/log.txt 2>&1 ) || { tail -n 20 "$a/Extract/log.txt"; return 1; }
    n=$(nparticles "$a/Extract/particles.star")
    echo "set $a: $n particles extracted"
    [[ -z "$want" ]] && want=$n
    (( n == want )) || { echo "set $a has $n particles, set A has $want"; return 1; }
  done
}

do_refine() {
  local a g=0 pids=() j=$(( NCPU / NGPU / 2 ))
  for a in "${ARMS[@]}"; do
    mkdir -p "$a/Refine"
    ( cd "$a" && $MPI -np 3 relion_refine_mpi --i Extract/particles.star --o Refine/run \
        --auto_refine --split_random_halves --ref ../Refs/ref2.mrc --trust_ref_size --ini_high 8 \
        --solvent_mask ../Refs/mask2.mrc --solvent_correct_fsc --dont_combine_weights_via_disc \
        --pool 30 --pad 2 --ctf --particle_diameter 280 --flatten_solvent --zero_mask \
        --oversampling 1 --healpix_order 4 --auto_local_healpix_order 4 --offset_range 3 \
        --offset_step 1 --sym C1 --low_resol_join_halves 40 --norm --scale \
        --random_seed "$SEED" --j "$j" --gpu "$g" > Refine/log.txt 2>&1 ) &
    pids+=($!); g=$(( (g + 1) % NGPU ))
  done
  local i rc=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "refine ${ARMS[$i]} failed"; tail -n 20 "${ARMS[$i]}/Refine/log.txt"; rc=1; }
  done
  return $rc
}

do_post() {  # never fatal: a failed set is reported and the others still run
  local a
  for a in "${ARMS[@]}"; do
    [[ -s "$a/Refine/run_half1_class001_unfil.mrc" ]] || { echo "set $a: no half maps"; continue; }
    mkdir -p "$KEPT/$a" "$a/Post"
    rsync -a "$a/Refine/run_half1_class001_unfil.mrc" "$a/Refine/run_half2_class001_unfil.mrc" \
      "$a/Refine/run_model.star" "$a/Refine/log.txt" "$KEPT/$a/" 2>/dev/null || true
    # --o needs an existing folder: RELION writes <dir>/logfile.pdf and dies if it is missing
    relion_postprocess --i "$a/Refine/run_half1_class001_unfil.mrc" --mask Refs/mask2.mrc \
      --angpix 1.7674 --auto_bfac --autob_lowres 10 --o "$a/Post/postprocess" \
      > "$a/Post/post.log" 2>&1 || { echo "postprocess $a failed"; tail -n 3 "$a/Post/post.log"; }
    echo "RESULT refine $a: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' \
      "$a/Post/postprocess.star" 2>/dev/null | tr -s ' ' | tr '\n' ' ') particles $(nparticles "$a/Extract/particles.star")"
    rsync -a "$a/Post/postprocess.star" "$a/Post/postprocess_fsc.xml" "$a/Post/post.log" \
      "$KEPT/$a/" 2>/dev/null || true
  done
  return 0
}

# SKIP lists stages already measured (job 94 kept the gain, orientation and CTF results)
SKIP=${SKIP-gaincheck ctfdiag}
skip() { [[ " $SKIP " == *" $1 "* ]] && echo "== $1 skipped"; }
t0=$SECONDS
stage select 1 0 do_select
stage stagein "$NCPU" 0 do_stage
skip gaincheck || stage gaincheck "$NCPU" 0 do_gaincheck
stage mc "$NCPU" "$NGPU" do_mc
skip ctfdiag || stage ctfdiag "$NCPU" 0 do_ctfdiag
stage extract "$NCPU" 0 do_extract
stage refine "$NCPU" "$NGPU" do_refine
stage post 1 0 do_post
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
cp "$PROJ/timing.csv" "$KEPT/"
cd /; rm -rf "$PROJ" "$MOVIES"
