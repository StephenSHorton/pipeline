#!/usr/bin/env bash
# Multi-view Hunyuan3D-DiT-v2-mv shape if a backend is present.
#
# Hunyuan3D-MLX lists MPS for v2-mv but MLX-native 2mv is still "under construction";
# hy3d shape is single-image mini only.
# This wrapper tries, in order:
#   1. vendor/hunyuan3d-mlx  `python main.py shape --shape-preset mv`
#   2. vendor/hunyuan3d-2    PyTorch MPS
# and otherwise exits 2 so generate.sh can fall back to single-image mini.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=_common.sh
source "$ROOT/scripts/_common.sh"
STEM="${1:?usage: shape_mv.sh <stem> [out.glb]}"
OUT_GLB="${2:-$ROOT/out/$STEM/mesh_mv.glb}"
VIEWS="$ROOT/out/$STEM/views/lifted"
[[ -d "$VIEWS" ]] || { echo "shape_mv: missing $VIEWS" >&2; exit 2; }

front="$VIEWS/front.png"
left="$VIEWS/left.png"
back="$VIEWS/back.png"
for f in "$front" "$left" "$back"; do
  [[ -f "$f" ]] || { echo "shape_mv: need front/left/back under $VIEWS (missing $(basename "$f"))" >&2; exit 2; }
done

DEVICE="mps"

MLX="$ROOT/vendor/hunyuan3d-mlx"
if [[ -f "$MLX/main.py" ]]; then
  echo "shape_mv: Hunyuan3D-MLX --shape-preset mv"
  (cd "$MLX" && uv run python main.py shape --shape-preset mv \
      --front "$front" --left "$left" --back "$back" --output "$OUT_GLB") \
    && { echo "SHAPE_MV_RESULT $OUT_GLB"; exit 0; }
fi

HY2="$ROOT/vendor/hunyuan3d-2"
if [[ -f "$HY2/examples/shape_gen_multiview.py" ]]; then
  PY="$(factory_python)"
  [[ -f "$HY2/.venv/bin/python" ]] && PY="$HY2/.venv/bin/python"
  echo "shape_mv: trying Hunyuan3D-2 on $DEVICE"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  "$PY" - <<PY
from pathlib import Path
from PIL import Image
import os, sys
sys.path.insert(0, "$HY2")
try:
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
except Exception as e:
    print("shape_mv: hy3dgen import failed:", e)
    sys.exit(2)
device = "$DEVICE"
print("shape_mv: loading tencent/Hunyuan3D-2mv hunyuan3d-dit-v2-mv-turbo on", device)
try:
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv-turbo",
        use_safetensors=True,
        device=device,
    )
except Exception as e:
    print("shape_mv: from_pretrained failed:", e)
    sys.exit(2)
images = {
    "front": Image.open("$front").convert("RGBA"),
    "left": Image.open("$left").convert("RGBA"),
    "back": Image.open("$back").convert("RGBA"),
}
try:
    mesh = pipe(image=images)[0]
except TypeError:
    mesh = pipe(image=images["front"], extra_view_images=images)[0]
out = Path("$OUT_GLB")
out.parent.mkdir(parents=True, exist_ok=True)
mesh.export(str(out))
print("SHAPE_MV_RESULT", out)
PY
  exit $?
fi

echo "shape_mv: no Mac-runnable 2mv backend (need vendor/hunyuan3d-mlx or vendor/hunyuan3d-2)." >&2
echo "shape_mv: hy3d shape remains single-image mini; extra views still bake in retexture." >&2
exit 2
