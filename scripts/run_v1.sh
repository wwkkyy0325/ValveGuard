#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAR="data/raw/context_windows_256.tar"
CACHE="data/processed/cw256"
EXPECTED=9385585152
PY=".venv/Scripts/python.exe"

mkdir -p reports data/processed

echo "[$(date)] waiting for complete tar"
for _ in $(seq 1 240); do
  if [ -f "$TAR" ] && [ "$(stat -c%s "$TAR")" -ge "$EXPECTED" ]; then
    break
  fi
  sleep 60
done
SIZE=$(stat -c%s "$TAR")
if [ "$SIZE" -lt "$EXPECTED" ]; then
  echo "tar incomplete after wait: $SIZE / $EXPECTED" >&2
  exit 1
fi
echo "[$(date)] tar complete: $SIZE bytes"

echo "[$(date)] building cache"
"$PY" -m valveguard.scripts.build_cache --tar "$TAR" --out "$CACHE"

echo "[$(date)] auditing"
"$PY" -m valveguard.scripts.audit --cache "$CACHE" --out reports/audit.json

echo "[$(date)] running GBDT baseline"
"$PY" -m valveguard.scripts.run_baseline --cache "$CACHE" --out reports/baseline

echo "[$(date)] done"
