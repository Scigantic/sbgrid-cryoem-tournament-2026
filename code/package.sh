#!/usr/bin/env bash
# Build the submission package from the reported consensus map (Refine10, 989,338 polished
# particles, bin 1). Copies only, on the login node: consistent names, a checksum manifest,
# header checks and an FSC plot. Re-running rebuilds the folder from scratch.
#   bash ~/pipeline/package.sh [OUT]
set -euo pipefail
FIN=${FIN:-$HOME/results/final}
OUT=${1:-$HOME/submission/ct03}
ID=${ID:-ct03_map01_consensus}
HERE=$(cd "$(dirname "$0")" && pwd)

src=(
  "Refine10/run_half1_class001_unfil.mrc:${ID}_half1_unfiltered.mrc"
  "Refine10/run_half2_class001_unfil.mrc:${ID}_half2_unfiltered.mrc"
  "Refine10/run_class001.mrc:${ID}_unsharpened.mrc"
  "Refine10/Post/postprocess.mrc:${ID}_sharpened.mrc"
  "Refine10/Post/postprocess_masked.mrc:${ID}_sharpened_masked.mrc"
  "Mask/mask.mrc:${ID}_fsc_mask.mrc"
  "LocRes10/relion_locres.mrc:${ID}_local_resolution.mrc"
  "LocRes10/relion_locres_filtered.mrc:${ID}_local_resolution_filtered.mrc"
  "Refine10/Post/postprocess.star:${ID}_postprocess.star"
  "Refine10/Post/postprocess_fsc.xml:${ID}_fsc.xml"
  "Refine10/Post/postprocess_fsc.dat:${ID}_fsc.dat"
  "Refine10/Post/log.txt:${ID}_postprocess_log.txt"
  "LocRes10/log.txt:${ID}_local_resolution_log.txt"
)
for s in "${src[@]}"; do [[ -s "$FIN/${s%%:*}" ]] || { echo "missing ${s%%:*}"; exit 2; }; done

rm -rf "$OUT"; mkdir -p "$OUT"
for s in "${src[@]}"; do cp "$FIN/${s%%:*}" "$OUT/${s##*:}"; done
python3 -I "$HERE/package_check.py" "$OUT" "$ID" | tee "$OUT/${ID}_summary.txt"
( cd "$OUT" && sha256sum -- * > SHA256SUMS )
echo "package: $(ls "$OUT" | wc -l) files, $(du -sm "$OUT" | cut -f1) MB"
