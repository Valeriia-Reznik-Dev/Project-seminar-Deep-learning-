#!/usr/bin/env bash
# Download the ORIGINAL DeepSORT appearance descriptor (mars-small128.pb).
# Source: the official Google Drive folder linked from the upstream README
#   https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp
# (folder -> resources/networks/mars-small128.pb)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$ROOT/weights"
mkdir -p "$WEIGHTS"
OUT="$WEIGHTS/mars-small128.pb"

if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; exit 0; fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found -> pip install gdown" >&2
    pip install gdown
fi

FOLDER_ID="18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp"
TMP="$ROOT/data/_resources"
mkdir -p "$TMP"

echo "== downloading official DeepSORT resources folder via gdown =="
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$TMP" || {
    echo "!! gdown folder download failed."
    echo "   Manually download mars-small128.pb from:"
    echo "   https://drive.google.com/open?id=$FOLDER_ID"
    echo "   and place it at: $OUT"
    exit 1
}

PB="$(find "$TMP" -name 'mars-small128.pb' | head -1 || true)"
[ -n "$PB" ] || { echo "!! mars-small128.pb not found after download"; exit 1; }
cp "$PB" "$OUT"
echo "== done: $OUT =="
