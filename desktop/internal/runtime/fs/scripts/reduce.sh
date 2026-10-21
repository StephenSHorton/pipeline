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

[[ -f "$MESH" ]] || { echo "reduce: missing hero mesh in $OUT" >&2; exit 1; }
echo "reduce: source $MESH"

echo "=== reduce lookdev (80k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_lookdev.glb" --preset lookdev --report "$OUT/lod_lookdev.json"
cp "$OUT/mesh_lookdev.glb" "$OUT/mesh_lod.glb"

echo "=== reduce game (12k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_game.glb" --preset game --report "$OUT/lod_game.json"

echo "=== reduce proxy (2k) ==="
$REDUCE "$MESH" -o "$OUT/mesh_proxy.glb" --preset proxy --report "$OUT/lod_proxy.json"

if [[ -f "$OUT/textured.glb" ]]; then
  echo "=== reduce textured -> game ==="
  $REDUCE "$OUT/textured.glb" -o "$OUT/textured_game.glb" --preset game --report "$OUT/lod_textured_game.json" \
    || echo "reduce: textured game failed (UVs often drop); paint the game mesh instead"
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path
import os
out = Path(os.environ["OUT"])
stem = os.environ["STEM"]
lods = []

def add(lid, files, label, metas):
    for f in files:
        p = out / f
        if not p.is_file():
            continue
        item = {"id": lid, "file": f, "label": label}
        for m in metas:
            mp = out / m
            if mp.is_file():
                d = json.loads(mp.read_text())
                item["faces"] = d.get("faces", d.get("triangles"))
                item["vertices"] = d.get("vertices")
                item["has_uvs"] = d.get("has_uvs", True if "atlas" in d else None)
                break
        lods.append(item)
        return

add("pbr", ["pbr.glb"], "PBR", ["inspect_pbr.json"])
add("mv", ["mesh_mv.glb"], "2mv", ["shape_mv.json"])
add("photo", ["textured_photo.glb"], "Photo bake", [])
# PS1 low-poly: psx_lowpoly.py (generate --quality psx).
add("psx", ["psx.glb"], "PSX", ["psx/psx.json"])
add("keep", ["anigen_keep.glb"], "Keep-weights", ["anigen_keep.json"])
add("anigen", ["anigen.glb"], "AniGen", [])
add("m2m", ["m2m.glb"], "Mesh2Motion", ["m2m.json"])
add("hero", ["mesh.glb"], "Hero", ["inspect.json"])
add("lookdev", ["textured.glb", "mesh_lookdev.glb", "mesh_lod.glb"], "Lookdev",
    ["inspect_textured.json", "lod_lookdev.json"])
add("quad", ["mesh_quad.glb"], "Quads", [])
# Quadric decimation drops UVs, so don't advertise a "textured" game LOD unless UVs survived.
game_files = ["mesh_game.glb"]
tg = out / "lod_textured_game.json"
if tg.is_file() and json.loads(tg.read_text()).get("has_uvs"):
    game_files = ["textured_game.glb", "mesh_game.glb"]
add("game", game_files, "Game", ["lod_textured_game.json", "lod_game.json"])
add("proxy", ["mesh_proxy.glb"], "Proxy", ["lod_proxy.json"])
add("born-game", ["mesh_born_game.glb"], "Born game", ["lod_born_game.json"])
if (out / "stack" / "stack.json").is_file():
    lods.append({
        "id": "stack",
        "file": "stack/stack.json",
        "label": "Sprite stack",
        "kind": "stack",
    })
(out / "lods.json").write_text(json.dumps({"stem": stem, "lods": lods}, indent=2) + "\n")
print("reduce: wrote", out / "lods.json")
for x in lods:
    print(" ", x["id"], x["file"], x.get("faces"))
PY
echo "REDUCE_RESULT $OUT/lods.json"
