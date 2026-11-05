#!/usr/bin/env bash
# Add the focused map (job 91: middle three beads, end beads subtracted, recentred, box 256,
# 989,338 polished particles) to the submission folder next to map01. Copies only, on the
# login node. Re-running replaces the map02 files and rewrites SHA256SUMS for the folder.
#   bash ~/pipeline/package_focus.sh [OUT]
set -euo pipefail
SRC=${SRC:-$HOME/results/focus}
OUT=${1:-$HOME/submission/ct03}
ID=${ID:-ct03_map02_focused}
HERE=$(cd "$(dirname "$0")" && pwd)
TMP="$OUT/.tmp_$ID"

src=(
  "Focus/run_half1_class001_unfil.mrc:${ID}_half1_unfiltered.mrc"
  "Focus/run_half2_class001_unfil.mrc:${ID}_half2_unfiltered.mrc"
  "Focus/run_class001.mrc:${ID}_unsharpened.mrc"
  "Focus/Post/postprocess.mrc:${ID}_sharpened.mrc"
  "Focus/Post/postprocess_masked.mrc:${ID}_sharpened_masked.mrc"
  "Focus0/mask.mrc:${ID}_fsc_mask.mrc"
  "Focus/Post/postprocess.star:${ID}_postprocess.star"
  "Focus/Post/postprocess_fsc.xml:${ID}_fsc.xml"
  "Focus/Post/postprocess_fsc.dat:${ID}_fsc.dat"
  "Focus/Post/log.txt:${ID}_postprocess_log.txt"
)
[[ -d "$OUT" ]] || { echo "missing $OUT (run package.sh first)"; exit 2; }
for s in "${src[@]}"; do [[ -s "$SRC/${s%%:*}" ]] || { echo "missing ${s%%:*}"; exit 2; }; done

rm -rf "$TMP"; mkdir -p "$TMP"
for s in "${src[@]}"; do cp "$SRC/${s%%:*}" "$TMP/${s##*:}"; done
python3 -I "$HERE/package_check.py" "$TMP" "$ID" | tee "$TMP/${ID}_summary.txt"
rm -f "$OUT/${ID}_"*
mv "$TMP"/* "$OUT"/
rmdir "$TMP"
( cd "$OUT" && rm -f SHA256SUMS && sha256sum -- * > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS )
echo "package: $(ls "$OUT" | wc -l) files, $(du -sm "$OUT" | cut -f1) MB"
