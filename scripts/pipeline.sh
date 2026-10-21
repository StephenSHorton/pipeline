#!/usr/bin/env bash
# Multi-pass lookdev for a stem that already has a blockout mesh.
#
#   isolate  — Vision lift (already done for radio)
#   blockout — Hunyuan shape (already done)
#   simplify — decimate so UVs are possible
#   surround — extra views live in out/<stem>/views (generated upstream)
#   paint    — Hunyuan RGB texture on the simplified mesh
#   lock     — project the original photo onto camera-facing albedo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin"
PYTHON="$ROOT/.venv/bin/python"
STEM="${1:-radio}"
QUALITY="${2:-rgb}"   # rgb | pbr
OUT="$ROOT/out/$STEM"
MESH="$OUT/mesh.glb"
LOD="$OUT/mesh_lod.glb"
LIFTED="$OUT/lifted.png"
PAINTED="$OUT/textured.glb"
LOCKED="$OUT/textured_locked.glb"
PAINT_ROOT="$ROOT/weights/paint"
PAINT_MODEL="rgb"
[[ "$QUALITY" == "pbr" ]] && PAINT_MODEL="pbr"

[[ -f "$MESH" ]] || { echo "pipeline: missing $MESH — run generate.sh first" >&2; exit 1; }
[[ -f "$LIFTED" ]] || { echo "pipeline: missing $LIFTED" >&2; exit 1; }
[[ -x "$BIN/hy3d" && -f "$BIN/mlx.metallib" ]] || { echo "pipeline: hy3d not bootstrapped" >&2; exit 1; }

if [[ ! -f "$PAINT_ROOT/hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors" ]]; then
  echo "pipeline: paint weights missing"
  exit 1
fi
if [[ "$PAINT_MODEL" == "pbr" && ! -f "$PAINT_ROOT/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors" ]]; then
  echo "pipeline: PBR weights missing (hunyuan3d-paintpbr-v2-1 / dinov2-giant)"
  exit 1
fi
mkdir -p "$PAINT_ROOT/realesrgan"
if [[ -f "$PAINT_ROOT/hunyuan3d-paint-v2-0/realesrgan/rrdbnet_mlx.safetensors" && ! -f "$PAINT_ROOT/realesrgan/rrdbnet_mlx.safetensors" ]]; then
  ln -sfn "$PAINT_ROOT/hunyuan3d-paint-v2-0/realesrgan/rrdbnet_mlx.safetensors" \
    "$PAINT_ROOT/realesrgan/rrdbnet_mlx.safetensors"
fi

echo "=== pass reduce (lookdev / game / proxy) ==="
"$ROOT/scripts/reduce.sh" "$STEM"
LOD="$OUT/mesh_lookdev.glb"
[[ -f "$LOD" ]] || LOD="$OUT/mesh_lod.glb"

if [[ -x "$BIN/autoremesher" ]]; then
  echo "=== pass remesh (AutoRemesher quads) ==="
  "$ROOT/scripts/quad_remesh.sh" "$LOD" "$OUT/mesh_quad.glb" 8000 \
    || echo "pipeline: remesh failed (non-fatal)"
  [[ -f "$OUT/mesh_quad.glb" ]] && LOD="$OUT/mesh_quad.glb"
fi

if [[ "$PAINT_MODEL" == "pbr" ]]; then
  PAINTED="$OUT/pbr.glb"
  echo "=== pass paint (Hunyuan PBR, 15 steps, 4096 atlas) ==="
  "$BIN/hy3d" paint "$LOD" "$LIFTED" -o "$PAINTED" \
    --weights "$PAINT_ROOT" --model pbr --steps 15 --res 512
  [[ -f "$PAINTED" ]] && cp "$PAINTED" "$OUT/textured.glb"
else
  echo "=== pass paint (Hunyuan RGB, 15 steps) ==="
  "$BIN/hy3d" paint "$LOD" "$LIFTED" -o "$PAINTED" \
    --weights "$PAINT_ROOT" --model rgb --steps 15 --res 512 --tex 2048
fi

echo "=== pass lock (legacy) ==="
"$PYTHON" "$ROOT/scripts/lock_albedo.py" "$PAINTED" "$LIFTED" -o "$LOCKED" --strength 0.8 || echo "pipeline: lock failed (non-fatal)"

echo "=== pass retexture (fitted-camera photo bake) ==="
PHOTO="$OUT/textured_photo.glb"
VIEWS=("$LIFTED")
if [[ -d "$OUT/views/lifted" ]]; then
  for v in "$OUT/views/lifted"/*.png; do
    [[ -f "$v" ]] && VIEWS+=("$v")
  done
fi
VIEW_ARGS=()
for v in "${VIEWS[@]}"; do VIEW_ARGS+=(--view "$v"); done
"$PYTHON" "$ROOT/scripts/retexture.py" "$PAINTED" -o "$PHOTO" "${VIEW_ARGS[@]}" || echo "pipeline: retexture failed (non-fatal)"

echo "=== pass reduce (refresh LODs now that paint exists) ==="
"$ROOT/scripts/reduce.sh" "$STEM"

echo "=== inspect ==="
"$PYTHON" "$ROOT/scripts/inspect_glb.py" "$LOCKED" "$OUT/inspect_textured.json"
echo "PIPELINE_RESULT $LOCKED"
cat "$OUT/inspect_textured.json"
