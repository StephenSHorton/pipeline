#!/usr/bin/env bash
# Quad remesh via AutoRemesher (MIT, native macOS). Run before paint so UVs follow quads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AR="$ROOT/bin/autoremesher"
PYTHON="$ROOT/.venv/bin/python"
IN="${1:?usage: quad_remesh.sh <in.glb> <out.glb> [target-quads]}"
OUT="${2:?}"
QUADS="${3:-12000}"
[[ -x "$AR" ]] || { echo "quad_remesh: missing $AR" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"$PYTHON" - <<PY
import trimesh, sys
m = trimesh.load("$IN", force="mesh")
if not isinstance(m, trimesh.Trimesh):
    sys.exit("quad_remesh: not a triangle mesh")
# AutoRemesher wants a clean manifold-ish OBJ
m.merge_vertices()
m.export("$TMP/in.obj")
print(f"quad_remesh: export {len(m.vertices)}v {len(m.faces)}f -> $TMP/in.obj")
PY

"$AR" --input "$TMP/in.obj" --output "$TMP/out.obj" --report "$TMP/report.txt" \
  --target-quads "$QUADS" --adaptivity 1.0 --sharp-edge 90
cat "$TMP/report.txt" || true

"$PYTHON" - <<PY
import trimesh
from pathlib import Path
m = trimesh.load("$TMP/out.obj", force="mesh")
Path("$OUT").parent.mkdir(parents=True, exist_ok=True)
m.export("$OUT")
print(f"quad_remesh: wrote $OUT  {len(m.vertices)}v {len(m.faces)}f")
PY
