#!/usr/bin/env bash
# Add the Blush map (job 96: map02's particles, mask and settings plus --blush; the final
# iteration hit the time limit, so this is iteration 10's half maps) to the submission
# folder next to map01 and map02. Login node, copies plus one half-map average.
# Re-running replaces the map03 files and rewrites SHA256SUMS for the folder.
#   bash ~/pipeline/package_blush.sh [OUT]
set -euo pipefail
SRC=${SRC:-$HOME/results/blush}
FOC=${FOC:-$HOME/results/focus}
OUT=${1:-$HOME/submission/ct03}
ID=${ID:-ct03_map03_focused_blush}
HERE=$(cd "$(dirname "$0")" && pwd)
TMP="$OUT/.tmp_$ID"
H1="$SRC/Blush/run_it010_half1_class001_unfil.mrc"
H2="$SRC/Blush/run_it010_half2_class001_unfil.mrc"
P="$SRC/Post_it010"

src=(
  "$H1:${ID}_half1_unfiltered.mrc"
  "$H2:${ID}_half2_unfiltered.mrc"
  "$P/pp.mrc:${ID}_sharpened.mrc"
  "$P/pp_masked.mrc:${ID}_sharpened_masked.mrc"
  "$FOC/Focus0/mask.mrc:${ID}_fsc_mask.mrc"
  "$P/pp.star:${ID}_postprocess.star"
  "$P/pp_fsc.xml:${ID}_fsc.xml"
  "$P/pp_fsc.dat:${ID}_fsc.dat"
  "$P/B10.log:${ID}_postprocess_log.txt"
)
[[ -d "$OUT" ]] || { echo "missing $OUT (run package.sh first)"; exit 2; }
for s in "${src[@]}"; do [[ -s "${s%%:*}" ]] || { echo "missing ${s%%:*}"; exit 2; }; done

rm -rf "$TMP"; mkdir -p "$TMP"
for s in "${src[@]}"; do cp "${s%%:*}" "$TMP/${s##*:}"; done
# Unsharpened full map = mean of the two unfiltered half maps.
relion_image_handler --i "$H1" --add "$H2" --o "$TMP/sum.mrc" > /dev/null
relion_image_handler --i "$TMP/sum.mrc" --multiply_constant 0.5 \
  --o "$TMP/${ID}_unsharpened.mrc" > /dev/null
rm -f "$TMP/sum.mrc"
python3 -I "$HERE/package_check.py" "$TMP" "$ID" | tee "$TMP/${ID}_summary.txt"
rm -f "$OUT/${ID}_"*
mv "$TMP"/* "$OUT"/
rmdir "$TMP"
( cd "$OUT" && rm -f SHA256SUMS && sha256sum -- * > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS )
echo "package: $(ls "$OUT" | wc -l) files, $(du -sm "$OUT" | cut -f1) MB"
