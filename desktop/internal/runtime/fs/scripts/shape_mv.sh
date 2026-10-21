#!/usr/bin/env bash
# Multi-view shape: Hunyuan3D-2mv (turbo by default) through vendor/hunyuan3d-2 on MPS.
#
# usage: shape_mv.sh <stem> [out.glb]
# Views come from out/<stem>/views/lifted/{front,left,back,right}.png (Hunyuan names: 'left' = the subject's
# left side, nose toward image left; 'right' = nose toward image right). At least three are needed.
# Env: SHAPE_OCTREE (256), SHAPE_SEED (0), SHAPE_MV_MODEL turbo|full (turbo), SHAPE_MV_STEPS (turbo 5, full 30).
# Weights load from $ROOT/weights/tencent/... (HY3DGEN_MODELS) with the hub offline: no downloads mid-job
# (scripts/bootstrap.sh fetches them).
# On failure prints one line 'SHAPE_MV_FAIL <reason>' and exits non-zero; generate.sh then takes the loud
# single-image path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STEM="${1:?usage: shape_mv.sh <stem> [out.glb]}"
OUT_GLB="${2:-$ROOT/out/$STEM/mesh_mv.glb}"
VIEWS="$ROOT/out/$STEM/views/lifted"

fail() { echo "SHAPE_MV_FAIL $*"; exit 2; }

ARGS=()
for v in front left back right; do
  [[ -f "$VIEWS/$v.png" ]] && ARGS+=("$v=$VIEWS/$v.png")
done
((${#ARGS[@]} >= 3)) || fail "missing_views: need 3+ of front/left/back/right under $VIEWS (have ${#ARGS[@]})"

HY2="$ROOT/vendor/hunyuan3d-2"
PY="$HY2/.venv/bin/python"
[[ -f "$HY2/hy3dgen/shapegen/pipelines.py" && -x "$PY" ]] || fail "no_backend: vendor/hunyuan3d-2 with its .venv is not installed"

MODEL="${SHAPE_MV_MODEL:-turbo}"
case "$MODEL" in
  turbo) SUB="hunyuan3d-dit-v2-mv-turbo"; STEPS="${SHAPE_MV_STEPS:-5}" ;;
  full) SUB="hunyuan3d-dit-v2-mv"; STEPS="${SHAPE_MV_STEPS:-30}" ;;
  *) fail "bad_model: SHAPE_MV_MODEL=$MODEL (turbo|full)" ;;
esac
OCTREE="${SHAPE_OCTREE:-256}"
SEED="${SHAPE_SEED:-0}"
[[ -d "$ROOT/weights/tencent/Hunyuan3D-2mv/$SUB" ]] || fail "no_weights: weights/tencent/Hunyuan3D-2mv/$SUB (run scripts/bootstrap.sh)"

export HY3DGEN_MODELS="$ROOT/weights"
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "shape_mv: Hunyuan3D-2mv $SUB fp16 mps steps=$STEPS octree=$OCTREE seed=$SEED views=${#ARGS[@]} flashvdm=mc"
mkdir -p "$(dirname "$OUT_GLB")"
set +e
"$PY" - "$HY2" "$OUT_GLB" "$SUB" "$STEPS" "$OCTREE" "$SEED" "${ARGS[@]}" <<'PY'
import sys, time

hy2, out, sub, steps, octree, seed = sys.argv[1:7]
views = dict(a.split("=", 1) for a in sys.argv[7:])
sys.path.insert(0, hy2)


def fail(reason):
    print("SHAPE_MV_FAIL", reason, flush=True)
    sys.exit(3)


try:
    import torch
    from PIL import Image
    from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline
except Exception as e:  # noqa: BLE001
    fail(f"import: {e.__class__.__name__}: {e}")
t0 = time.time()
try:
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv", subfolder=sub, variant="fp16", use_safetensors=True,
        device="mps", dtype=torch.float16)
    pipe.enable_flashvdm(mc_algo="mc")
except Exception as e:  # noqa: BLE001
    fail(f"load: {e.__class__.__name__}: {str(e)[:300]}")
t1 = time.time()
try:
    images = {k: Image.open(v).convert("RGBA") for k, v in views.items()}
    mesh = pipe(image=images, num_inference_steps=int(steps), octree_resolution=int(octree), num_chunks=20000,
                generator=torch.manual_seed(int(seed)), output_type="trimesh")[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    if mesh is None or not len(mesh.faces):
        fail("empty: the model returned no surface")
    mesh.export(out)
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    fail(f"run: {e.__class__.__name__}: {str(e)[:300]}")
t2 = time.time()
print(f"shape_mv: load {t1 - t0:.1f}s gen {t2 - t1:.1f}s faces {len(mesh.faces)} -> {out}", flush=True)
print("SHAPE_MV_RESULT", out, flush=True)
PY
rc=$?
set -e
# Exit 3: the python side printed its own SHAPE_MV_FAIL. Anything else non-zero is a crash (signal, OOM).
[[ $rc -eq 3 ]] && exit 2
[[ $rc -eq 0 ]] || fail "crash: exit $rc"
[[ -s "$OUT_GLB" ]] || fail "no_output: $OUT_GLB was not written"
