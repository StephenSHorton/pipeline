#!/usr/bin/env bash
# Build the lookdev mesh (80k: the mesh paint and the unwrap work on) from an existing hero mesh, then list the
# finished models in lods.json. No game / proxy LODs any more: they were extra copies at other triangle counts that
# nothing in a make went on to use; the PS1 low-poly is cut from the finished model by psx_lowpoly.py instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=_common.sh
source "$ROOT/scripts/_common.sh"
PYTHON="$(factory_python)"
STEM="${1:-radio}"
OUT="$ROOT/out/$STEM"
MESH="$OUT/mesh_hero.glb"
[[ -f "$MESH" ]] || MESH="$OUT/mesh.glb"
REDUCE="$PYTHON $ROOT/scripts/reduce_mesh.py"

[[ -f "$MESH" ]] || { echo "reduce: missing hero mesh in $OUT" >&2; exit 1; }
echo "reduce: source $MESH"

echo "=== reduce lookdev (80k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_lookdev.glb" --preset lookdev --report "$OUT/lod_lookdev.json"
# pipeline.sh (and generate.sh's fallback) read the lookdev mesh by this older name.
cp "$OUT/mesh_lookdev.glb" "$OUT/mesh_lod.glb"

# lods.py builds the list (generate.sh's lods stage and the workbench's psx tool use the same code).
"$PYTHON" "$ROOT/scripts/lods.py" "$OUT"
echo "REDUCE_RESULT $OUT/lods.json"
