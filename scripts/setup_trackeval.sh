#!/usr/bin/env bash
# Clone the official TrackEval (pinned) and make it importable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/third_party"
mkdir -p "$VENDOR"
DEST="$VENDOR/TrackEval"

if [ -d "$DEST/.git" ]; then
    echo "[skip] TrackEval already cloned at $DEST"
else
    git clone https://github.com/JonathonLuiten/TrackEval.git "$DEST"
fi
# pin to a known-good commit for reproducibility
git -C "$DEST" checkout -q 12c8791b303e0a0b50f753af204249bcb9a937fc || true
pip install -e "$DEST"
python -c "import trackeval; print('TrackEval import OK')"
echo "== TrackEval ready at $DEST =="
