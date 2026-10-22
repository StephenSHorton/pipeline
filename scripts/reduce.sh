#!/usr/bin/env bash
# Build lookdev / game / proxy LODs from an existing hero mesh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=_common.sh
source "$ROOT/scripts/_common.sh"
PYTHON="$(factory_python)"
STEM="${1:-radio}"
OUT="$ROOT/out/$STEM"
export STEM OUT
MESH="$OUT/mesh_hero.glb"
[[ -f "$MESH" ]] || MESH="$OUT/mesh.glb"
REDUCE="$PYTHON $ROOT/scripts/reduce_mesh.py"

# DEGRADED <stage> <code>: <message>, also recorded in $DEGRADED_JSON when generate.sh sets it.
degrade() {
  echo "DEGRADED $1 $2: $3"
  [[ -n "${DEGRADED_JSON:-}" ]] || return 0
  "$PYTHON" - "$DEGRADED_JSON" "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
p, row = Path(sys.argv[1]), {"stage": sys.argv[2], "code": sys.argv[3], "message": sys.argv[4]}
try:
    data = json.loads(p.read_text())
except (OSError, ValueError):
    data = {}
rows = data.get("degraded") or []
if row not in rows:
    rows.append(row)
data["degraded"] = rows
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

[[ -f "$MESH" ]] || { echo "reduce: missing hero mesh in $OUT" >&2; exit 1; }
echo "reduce: source $MESH"

echo "=== reduce lookdev (80k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_lookdev.glb" --preset lookdev --report "$OUT/lod_lookdev.json"
cp "$OUT/mesh_lookdev.glb" "$OUT/mesh_lod.glb"

echo "=== reduce game (12k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_game.glb" --preset game --report "$OUT/lod_game.json"

echo "=== reduce proxy (2k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_proxy.glb" --preset proxy --report "$OUT/lod_proxy.json"

# The textured game LOD comes from the photo bake when there is one (what the asset ships), else the paint.
# reduce_mesh.py re-bakes the texture onto the decimated mesh; a game LOD without UVs is degraded, not hidden.
TEXSRC="$OUT/textured_photo.glb"
[[ -f "$TEXSRC" ]] || TEXSRC="$OUT/textured.glb"
rm -f "$OUT/lod_textured_game.json"
if [[ -f "$TEXSRC" ]]; then
  echo "=== reduce $(basename "$TEXSRC") -> game ==="
  if ! $REDUCE "$TEXSRC" -o "$OUT/textured_game.glb" --preset game --report "$OUT/lod_textured_game.json"; then
    degrade lods game_failed "textured game LOD failed to reduce; the game LOD is untextured"
  elif ! grep -q '"has_uvs": true' "$OUT/lod_textured_game.json"; then
    degrade lods game_uvs "textured game LOD lost its UVs; the game LOD is untextured"
  fi
fi

# lods.py builds the list (the workbench's psx tool upserts its entry with the same code).
"$PYTHON" "$ROOT/scripts/lods.py" "$OUT"
echo "REDUCE_RESULT $OUT/lods.json"
