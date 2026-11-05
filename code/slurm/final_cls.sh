#!/usr/bin/env bash
#SBATCH --no-requeue
# Stage 1 of the last shot at 3.3 A: 3D classification of the middle three beads without
# alignment (job 91's poses), then a half-map reconstruction + post of every class and of a
# random subset the size of the best class, all with relion_reconstruct so they compare
# like with like (same method on all 989k gave 4.35 A in job 92, Recon0). Stage 2 (refine
# the best class) only if a class clearly beats both 4.35 A and its random control.
# Subtracted stacks are rebuilt exactly as in final_focus2.sh and checked against job 91.
#   export PRESET=$HOME/pipeline/presets/tournament.env
#   sbatch -p ct03g-g4dn-12xlarge -t 7:00:00 slurm/final_cls.sh
set -euo pipefail
HERE=${PIPE:-${SLURM_SUBMIT_DIR:-$HOME/pipeline}}; [[ -s "$HERE/common.sh" ]] || HERE=$HOME/pipeline
: "${PRESET:?set PRESET}"; source "$PRESET"
for m in ${MODULES:-sbgrid/relion/5.0.1_cu12.2}; do
  module load "$m" || echo "WARNING: module $m failed"
done
export PROJ=${PROJ:-/scratch/cls} SCRATCH=${SCRATCH:-/scratch/cls/scratch}
export PERSIST_DIR=${PERSIST_DIR:-$HOME/results}
export NO_PERSIST=1
FIN=${FIN:-$PERSIST_DIR/final}
FOC=${FOC:-$PERSIST_DIR/focus}
NEWBOX=${NEWBOX:-256}; NP_SUB=${NP_SUB:-24}; NP_CPU=${NP_CPU:-24}; J_CPU=${J_CPU:-2}
KCLS=${KCLS:-4}; TCLS=${TCLS:-8}; ITCLS=${ITCLS:-25}; HIGHEXP=${HIGHEXP:-5}
SEED=${SEED:-20261030}
mkdir -p "$PROJ/.done"
source "$HERE/common.sh"
LASTIT=$(printf 'run_it%03d' "$ITCLS")

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
    sleep 60
  done
) >> "$KEPT/monitor.log" 2>&1 &

CHUNKS=()
for d in "$PERSIST_DIR"/full_t??; do
  [[ -s "$d/Polish/shiny.star" ]] && CHUNKS+=("$(basename "$d")")
done
(( ${#CHUNKS[@]} >= ${MIN_CHUNKS:-10} )) || { echo "only ${#CHUNKS[@]} chunks polished"; exit 2; }
for f in "$FIN/Refine10/run_optimiser.star" "$FIN/Refine10/run_data.star" \
         "$FIN/Refine10/run_class001.mrc" "$FIN/Refs/solvent_mask.mrc" \
         "$FOC/Focus/run_data.star" "$FOC/Focus/run_class001.mrc" \
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
[[ -d "${CHUNKS[0]}/Polish" ]] || rm -f .done/stagep .done/subtract .done/namecheck
echo "$NCPU cpus, $NGPU gpus; box $NEWBOX; class3d K=$KCLS T=$TCLS iter $ITCLS highres_exp $HIGHEXP A"

do_stage() {
  local t0=$SECONDS
  printf '%s\n' "${CHUNKS[@]}" | xargs -P 10 -I{} sh -c \
    "mkdir -p {} && rsync -a '$PERSIST_DIR/{}/Polish' {}/"
  echo "staged ${#CHUNKS[@]} polished chunks ($(du -scm */Polish | tail -n 1 | cut -f1) MB) in $((SECONDS-t0)) s"
}

do_refs() {
  mkdir -p Refine10 Refs Focus
  rsync -a --exclude 'run_it*' --exclude Post "$FIN/Refine10/" Refine10/
  cp "$FIN/Refs/solvent_mask.mrc" Refs/
  rsync -a "$FOC/Bodies" "$FOC/Focus0" ./
  cp "$FOC/Focus/run_data.star" "$FOC/Focus/run_class001.mrc" Focus/
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

do_namecheck() {
  names Focus/run_data.star > Subtract/names_focus.txt
  names Subtract/input.star > Subtract/names_new.txt
  local a b d s
  a=$(wc -l < Subtract/names_focus.txt); b=$(wc -l < Subtract/names_new.txt)
  d=$(comm -3 Subtract/names_focus.txt Subtract/names_new.txt | wc -l)
  echo "image names: job 91 $a, rebuilt $b, differing $d"
  (( a == b && d == 0 )) || { echo "rebuilt stacks do not match job 91's poses"; return 1; }
  s=$(sed -n 1p Subtract/names_focus.txt | cut -d@ -f2)
  [[ -s "$s" ]] || { echo "first stack missing on disk"; return 1; }
}

do_class3d() {
  local t0=$SECONDS
  mkdir -p Class3D
  $MPI -np "$NP_GPU" relion_refine_mpi --i Focus/run_data.star --o Class3D/run \
    --ref Focus/run_class001.mrc --trust_ref_size --ini_high 8 --K "$KCLS" \
    --tau2_fudge "$TCLS" --iter "$ITCLS" --skip_align --solvent_mask Focus0/mask.mrc \
    --particle_diameter 200 --flatten_solvent --zero_mask --pad 2 --ctf --norm --scale \
    --strict_highres_exp "$HIGHEXP" --dont_combine_weights_via_disc --pool 30 \
    --random_seed "$SEED" --j "$THREADS_GPU" --gpu "$GPUARG"
  echo "class3d in $((SECONDS-t0)) s"
  awk '/^data_model_classes/{p=1} /^data_model_class_1/{exit} p' "Class3D/${LASTIT}_model.star" \
    | grep -v '^#' | cut -c1-160
  mkdir -p Class3D/keep
  cp Class3D/"$LASTIT"_model.star Class3D/"$LASTIT"_data.star Class3D/"$LASTIT"_class0*.mrc Class3D/keep/
  keep Class3D/keep
}

recon() {  # STAR OUTDIR LABEL
  local t0=$SECONDS h n
  mkdir -p "$2"
  n=$(nparticles "$1")
  for h in 1 2; do
    $MPI -np "$NP_CPU" relion_reconstruct_mpi --i "$1" --o "$2/run_half${h}_class001_unfil.mrc" \
      --subset "$h" --ctf --j "$J_CPU" > "$2/half$h.log" 2>&1 || { tail -n 20 "$2/half$h.log"; return 1; }
  done
  relion_postprocess --i "$2/run_half1_class001_unfil.mrc" --mask Focus0/mask.mrc \
    --angpix "$ANGPIX" --auto_bfac --autob_lowres 10 --o "$2/postprocess" > "$2/post.log"
  local res; res=$(awk '$1 == "_rlnFinalResolution" {print $2}' "$2/postprocess.star")
  echo "RESULT $3: $(grep -h '_rlnFinalResolution\|_rlnBfactorUsedForSharpening' "$2/postprocess.star" | tr -s ' ' | tr '\n' ' ') particles $n ($((SECONDS-t0)) s)"
  echo "$res $n $2" >> Cls/results.txt
  mkdir -p "$KEPT/$2"
  rsync -a "$2/postprocess.star" "$2/post.log" "$KEPT/$2/"
}

do_classrecon() {
  local k n
  mkdir -p Cls; rm -f Cls/results.txt
  for k in $(seq 1 "$KCLS"); do
    relion_star_handler --i "Class3D/${LASTIT}_data.star" --o "Cls/class$k.star" \
      --select rlnClassNumber --minval "$k" --maxval "$k" > /dev/null
    n=$(nparticles "Cls/class$k.star")
    if (( n < 20000 )); then echo "class $k: $n particles, skipped"; continue; fi
    recon "Cls/class$k.star" "Cls/C$k" "class $k"
  done
  [[ -s Cls/results.txt ]] || { echo "no class large enough to reconstruct"; return 1; }
  # random control the size of the best-resolving class
  local best best_n best_dir total
  best=$(sort -g Cls/results.txt | sed -n 1p)
  best_n=$(echo "$best" | awk '{print $2}'); best_dir=$(echo "$best" | awk '{print $3}')
  total=$(nparticles Focus/run_data.star)
  awk -v p="$best_n" -v t="$total" 'BEGIN{srand(7)} { if (index($0, "@") == 0 || rand() < p / t) print }' \
    Focus/run_data.star > Cls/random.star
  recon Cls/random.star Cls/R "random control (size of best class, $best_dir)"
  keep Cls/results.txt
}

t0=$SECONDS
stage stagep "$NCPU" 0 do_stage
stage refs 1 0 do_refs
stage subtract "$NCPU" 0 do_subtract
stage namecheck 1 0 do_namecheck
stage class3d "$NCPU" "$NGPU" do_class3d
stage classrecon "$NCPU" 0 do_classrecon
echo "--- timing (stage, seconds, usd at RATE_USD_H, cpus, gpus)"; cat "$PROJ/timing.csv"
echo "total job seconds $((SECONDS-t0))"
keep timing.csv
cd /; rm -rf "$PROJ"
