#!/usr/bin/env bash
# Download the six MOT Challenge test sequences and stage them under data/mot/.
#   MOT15 (2DMOT2015): TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1
#   MOT16:             MOT16-09, MOT16-11
#
# Each staged sequence keeps the standard MOTChallenge layout:
#   data/mot/<SEQ>/{img1/, det/det.txt, gt/gt.txt, seqinfo.ini}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
DEST="$DATA/mot"
TMP="$DATA/_zips"
mkdir -p "$DEST" "$TMP"

MOT15_URL="https://motchallenge.net/data/2DMOT2015.zip"
MOT16_URL="https://motchallenge.net/data/MOT16.zip"

MOT15_SEQS=(TUD-Campus TUD-Stadtmitte KITTI-17 PETS09-S2L1)
MOT16_SEQS=(MOT16-09 MOT16-11)

fetch () {  # url out
    if [ -f "$2" ]; then echo "[skip] $2 exists"; return; fi
    echo "[get ] $1"
    curl -L --fail --retry 3 -o "$2" "$1"
}

echo "== downloading archives (this is a few GB) =="
fetch "$MOT15_URL" "$TMP/2DMOT2015.zip"
fetch "$MOT16_URL" "$TMP/MOT16.zip"

echo "== extracting needed sequences =="
# 2DMOT2015 train sequences live under 2DMOT2015/train/<SEQ>/
for seq in "${MOT15_SEQS[@]}"; do
    unzip -o -q "$TMP/2DMOT2015.zip" "2DMOT2015/train/$seq/*" -d "$TMP/mot15" || true
    src="$TMP/mot15/2DMOT2015/train/$seq"
    [ -d "$src" ] || { echo "!! $seq not found in 2DMOT2015.zip"; exit 1; }
    rm -rf "$DEST/$seq"; mkdir -p "$DEST/$seq"; cp -r "$src/." "$DEST/$seq/"
    echo "[ok  ] $seq"
done

# MOT16 train sequences live under MOT16/train/<SEQ>/
for seq in "${MOT16_SEQS[@]}"; do
    unzip -o -q "$TMP/MOT16.zip" "MOT16/train/$seq/*" -d "$TMP/mot16" || true
    src="$TMP/mot16/MOT16/train/$seq"
    [ -d "$src" ] || { echo "!! $seq not found in MOT16.zip"; exit 1; }
    rm -rf "$DEST/$seq"; mkdir -p "$DEST/$seq"; cp -r "$src/." "$DEST/$seq/"
    echo "[ok  ] $seq"
done

echo "== done. sequences staged under $DEST =="
ls -1 "$DEST"
