#!/usr/bin/env bash
# GPU motion correction with MotionCor3, one process per GPU, one call per movie (folder input
# fails on SBCloud's build). Writes MotionCorr/Movies/<stem>.mrc and <stem>_DW.mrc and the same
# MotionCorr/corrected_micrographs.star RELION's own stage writes, so 02_ctf_pick_extract.sh
# runs unchanged. Needs GAIN (TIFF or MRC multiplier) for the tournament; it is passed to
# MotionCor3 inverted, because MotionCor3 divides by the file. Rehearsal: ~3.6 s/movie on 4 T4.
# Resumable: movies whose output already exists are skipped.
# MC3_RM_MOVIES=1 deletes each raw movie once its two outputs exist: on a node-local scratch
# the raw copies plus the outputs of 1500 movies (about 190 MB each side) filled the disk.
source "$(dirname "$0")/common.sh"
cd "$PROJ"
: "${MC_PATCH:=5}"
MC3=${MOTIONCOR3:-MotionCor3}

do_mc3() {
  mkdir -p MotionCorr/Movies
  local gain=()
  if [[ -n "$GAIN" ]]; then
    python3 -I "$HERE/gain_to_mrc.py" "$GAIN" gain_inv.mrc --invert
    gain=(-Gain "$PROJ/gain_inv.mrc")
  fi
  ls "$MOVIES"/*."$MOVIE_EXT" > mc3_movies.txt
  rm -f mc3_part_*
  split -n "l/$NGPU" -d mc3_movies.txt mc3_part_
  local g=0 p
  for p in mc3_part_*; do
    (
      while read -r f; do
        s=$(basename "$f"); s=${s%."$MOVIE_EXT"}
        if [[ ! -s "MotionCorr/Movies/$s.mrc" ]]; then
          "$MC3" -In"$(tr a-z A-Z <<< "${MOVIE_EXT:0:1}")${MOVIE_EXT:1}" "$f" -OutMrc "$PROJ/MotionCorr/Movies/$s.mrc" \
            ${gain[@]+"${gain[@]}"} -PixSize "$ANGPIX" -FmDose "$DOSE" -kV "$KV" \
            -Gpu "$g" -Patch "$MC_PATCH" "$MC_PATCH" > /dev/null 2>&1 || { echo "failed: $s" >> mc3_failed.txt || true; }
        fi
        if [[ -n "${MC3_RM_MOVIES:-}" && -s "MotionCorr/Movies/$s.mrc" && -s "MotionCorr/Movies/${s}_DW.mrc" ]]; then
          rm -f "$f"
        fi
      done < "$p"
    ) &
    g=$((g + 1))
  done
  wait
  echo "corrected $(ls MotionCorr/Movies/*_DW.mrc 2>/dev/null | wc -l) movies; scratch: $(df -h "$PROJ" | awk 'NR==2{print $4" free of "$2}')"
}

do_star() {
  python3 -I - "$ANGPIX" "$KV" "$CS" <<'PY'
import glob, os, sys
px, kv, cs = sys.argv[1:4]
rows = sorted(p for p in glob.glob("MotionCorr/Movies/*.mrc") if not p.endswith("_DW.mrc"))
rows = [r for r in rows if os.path.exists(r[:-4] + "_DW.mrc")]
o = ["data_optics", "", "loop_", "_rlnOpticsGroupName #1", "_rlnOpticsGroup #2",
     "_rlnMicrographOriginalPixelSize #3", "_rlnVoltage #4",
     "_rlnSphericalAberration #5", "_rlnAmplitudeContrast #6",
     "_rlnMicrographPixelSize #7", "opticsGroup1 1 %s %s %s 0.1 %s" % (px, kv, cs, px),
     "", "data_micrographs", "", "loop_", "_rlnMicrographName #1",
     "_rlnMicrographNameNoDW #2", "_rlnOpticsGroup #3"]
o += ["%s %s 1" % (r[:-4] + "_DW.mrc", r) for r in rows]
open("MotionCorr/corrected_micrographs.star", "w").write("\n".join(o) + "\n")
print("micrographs in star: %d" % len(rows))
PY
  [[ ! -s mc3_failed.txt ]] || echo "WARNING: $(wc -l < mc3_failed.txt) movies failed (mc3_failed.txt)"
}

stage motioncor3 "$NCPU" "$NGPU" do_mc3
stage mc3_star 1 0 do_star
