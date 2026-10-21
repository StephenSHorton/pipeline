#!/usr/bin/env bash
# Vertical slice: image -> lifted PNG -> hy3d GLB -> inspect.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=_common.sh
source "$ROOT/scripts/_common.sh"
BIN="$ROOT/bin"
QUALITY="draft"
IMAGE=""
SEED="0"
STEM=""
POLY="medium"
OCTREE=""
FACES=""
STACK_RES=160
WANT_STACK=1
PROMPT=""
REFS=()
# --quality psx: PS1 low-poly with the photo baked on (psx_lowpoly.py).
FACING="camera"
PSX_HEIGHT=48
PSX_FACES=1200
PSX_ATLAS=256
PSX_UP="z"
PSX_NAME=""

usage() {
  echo "usage: generate.sh --image <path> [--ref <path> ...] [--prompt text] [--quality draft|textured|pbr|stack|psx] [--poly high|medium|low|hero|lookdev|game|proxy] [--octree N] [--faces N] [--seed N] [--stem name] [--stack-res N] [--no-stack]" >&2
  echo "       psx: [--facing camera|right|left|away] [--psx-height N] [--psx-faces N] [--psx-atlas N] [--psx-up z|y] [--psx-name name]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --ref) REFS+=("$2"); shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --poly) POLY="$2"; shift 2 ;;
    --octree) OCTREE="$2"; shift 2 ;;
    --faces) FACES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --stem) STEM="$2"; shift 2 ;;
    --stack-res) STACK_RES="$2"; shift 2 ;;
    --no-stack) WANT_STACK=0; shift ;;
    --facing) FACING="$2"; shift 2 ;;
    --psx-height) PSX_HEIGHT="$2"; shift 2 ;;
    --psx-faces) PSX_FACES="$2"; shift 2 ;;
    --psx-atlas) PSX_ATLAS="$2"; shift 2 ;;
    --psx-up) PSX_UP="$2"; shift 2 ;;
    --psx-name) PSX_NAME="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

POLY_IN="$POLY"
case "$POLY" in
  high) POLY="hero" ;;
  medium) POLY="lookdev" ;;
  low) POLY="proxy" ;;
esac

case "$POLY" in
  hero) OCTREE="${OCTREE:-256}" ;;
  lookdev) OCTREE="${OCTREE:-128}"; FACES="${FACES:-80000}" ;;
  game) OCTREE="${OCTREE:-64}"; FACES="${FACES:-12000}" ;;
  proxy) OCTREE="${OCTREE:-64}"; FACES="${FACES:-2000}" ;;
  *) echo "generate: unknown --poly $POLY_IN (high|medium|low or hero|lookdev|game|proxy)" >&2; exit 2 ;;
esac

if [[ "$QUALITY" == "stack" ]]; then
  # Stack is the shippable product. Paint RGB when we can, then slice.
  QUALITY="textured"
  WANT_STACK=1
fi

WANT_PSX=0
if [[ "$QUALITY" == "psx" ]]; then
  # PS1 look: RGB paint, the photo baked over it, then cut to a PS1 budget. The stack is a different product.
  QUALITY="textured"
  WANT_PSX=1
  WANT_STACK=0
fi
case "$FACING" in
  camera|right|left|away) ;;
  *) echo "generate: unknown --facing $FACING (camera|right|left|away)" >&2; exit 2 ;;
esac

PYTHON="$(factory_python)"

[[ -n "$IMAGE" && -f "$IMAGE" ]] || { echo "generate: --image is required and must exist" >&2; exit 2; }
IMAGE="$(cd "$(dirname "$IMAGE")" && pwd)/$(basename "$IMAGE")"
if [[ -z "$STEM" ]]; then
  STEM="$(basename "$IMAGE")"
  STEM="${STEM%.*}"
fi

OUT="$ROOT/out/$STEM"
mkdir -p "$OUT"
EXT="${IMAGE##*.}"
# A rebuild from this asset's own input.* must not copy the file onto itself.
[[ "$IMAGE" -ef "$OUT/input.$EXT" ]] || cp "$IMAGE" "$OUT/input.$EXT"
if [[ -n "$PROMPT" ]]; then
  printf '%s\n' "$PROMPT" > "$OUT/prompt.txt"
fi

LOG="$OUT/generate.log"
# The factory writes this script's output to the log itself (one copy per line); run by hand, tee it.
if [[ -z "${FACTORY_ROOT:-}" ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi

# Progress for the app: one STAGES line with the plan, then "STAGE <key> <label>" as each one starts.
PLAN="isolate shape reduce"
if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" ]]; then
  PLAN="$PLAN paint"
  { ((${#REFS[@]})) || [[ -d "$OUT/views/lifted" || "$WANT_PSX" -eq 1 ]]; } && PLAN="$PLAN retexture"
  [[ "$WANT_PSX" -eq 1 ]] && PLAN="$PLAN psx"
  PLAN="$PLAN lods"
fi
PLAN="$PLAN inspect"
[[ "$WANT_STACK" -eq 1 ]] && PLAN="$PLAN stack"
[[ -x "$BIN/preview-glb" ]] && PLAN="$PLAN preview"
echo "STAGES $PLAN"
stage() { echo "STAGE $1 $2"; }

echo "generate: root=$ROOT"
echo "generate: image=$IMAGE"
echo "generate: python=$PYTHON"
echo "generate: quality=$QUALITY psx=$WANT_PSX poly=$POLY_IN ($POLY) octree=$OCTREE faces=${FACES:-keep} seed=$SEED stack=$WANT_STACK stack-res=$STACK_RES out=$OUT"
if [[ -n "$PROMPT" ]]; then
  echo "generate: prompt=$PROMPT"
fi
if ((${#REFS[@]})); then
  echo "generate: refs=${#REFS[@]}"
fi

if [[ ! -x "$BIN/hy3d" || ! -x "$BIN/lift-subject" || ! -f "$BIN/mlx.metallib" ]]; then
  echo "generate: factory binaries missing, running bootstrap ($QUALITY)"
  "$ROOT/scripts/bootstrap.sh" "$QUALITY"
fi

LIFTED="$OUT/lifted.png"
stage isolate "Cutting out the subject"
echo "generate: lifting subject via Vision"
"$BIN/lift-subject" "$IMAGE" "$LIFTED"
isolate_one() { "$BIN/lift-subject" "$1" "$2"; }

if ((${#REFS[@]})); then
  echo "generate: isolating ${#REFS[@]} extra reference image(s) into views/"
  mkdir -p "$OUT/views/lifted"
  SLOTS=(front left back threequarter top)
  VIEW_ROWS=()
  idx=0
  for REF in "${REFS[@]}"; do
    [[ -f "$REF" ]] || { echo "generate: missing --ref $REF" >&2; exit 2; }
    REF="$(cd "$(dirname "$REF")" && pwd)/$(basename "$REF")"
    REXT="${REF##*.}"
    if (( idx < ${#SLOTS[@]} )); then
      SLOT="${SLOTS[$idx]}"
    else
      SLOT="ref-$((idx + 1))"
    fi
    cp "$REF" "$OUT/views/$SLOT.$REXT"
    isolate_one "$REF" "$OUT/views/lifted/$SLOT.png"
    VIEW_ROWS+=("$SLOT|$REXT")
    idx=$((idx + 1))
  done
  {
    printf '{\n  "views": [\n'
    printf '    {"id": "input", "file": "input.%s", "role": "shape", "label": "Input image"}' "$EXT"
    for row in "${VIEW_ROWS[@]}"; do
      SLOT="${row%%|*}"
      REXT="${row##*|}"
      printf ',\n    {"id": "%s", "file": "views/%s.%s", "role": "ref", "label": "%s"}' "$SLOT" "$SLOT" "$REXT" "$SLOT"
    done
    printf '\n  ]\n}\n'
  } > "$OUT/views.json"
  echo "generate: wrote $OUT/views.json"
fi

stage shape "Building the shape"
HERO="$OUT/mesh_hero.glb"
MESH="$OUT/mesh.glb"
USED_MV=0
export SHAPE_OCTREE="$OCTREE"
export SHAPE_SEED="$SEED"

if [[ -d "$OUT/views/lifted" && -f "$OUT/views/lifted/front.png" && -f "$OUT/views/lifted/left.png" && -f "$OUT/views/lifted/back.png" ]]; then
  if "$ROOT/scripts/shape_mv.sh" "$STEM" "$HERO"; then
    USED_MV=1
    echo "generate: multi-view shape wrote $HERO"
  else
    echo "generate: 2mv unavailable, single-image"
  fi
fi
if [[ "$USED_MV" -eq 0 ]]; then
  SHAPE_WEIGHTS="$ROOT/weights/shape-small"
  [[ -d "$SHAPE_WEIGHTS" ]] || { echo "generate: missing $SHAPE_WEIGHTS" >&2; exit 1; }
  echo "generate: hy3d shape (octree $OCTREE)"
  "$BIN/hy3d" shape "$LIFTED" -o "$HERO" --weights "$SHAPE_WEIGHTS" --seed "$SEED" --steps 30 --octree "$OCTREE"
fi
ls -lh "$HERO" || ls -l "$HERO"
if [[ -n "$FACES" ]]; then
  echo "generate: reduce to $FACES faces (--poly $POLY)"
  "$PYTHON" "$ROOT/scripts/reduce_mesh.py" "$HERO" -o "$MESH" --faces "$FACES" --preset "$POLY"
else
  cp "$HERO" "$MESH"
fi

stage reduce "Reducing the mesh"
echo "generate: reduce LODs"
"$ROOT/scripts/reduce.sh" "$STEM"

FINAL="$MESH"
if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" ]]; then
  PAINT_MODEL="rgb"
  PAINT_WEIGHTS="$ROOT/weights/paint"
  if [[ "$QUALITY" == "pbr" ]]; then
    PAINT_MODEL="pbr"
    PAINT_WEIGHTS="$ROOT/weights/paint"
  fi
  if [[ ! -d "$PAINT_WEIGHTS" ]]; then
    echo "generate: paint weights missing, downloading"
    "$ROOT/scripts/bootstrap.sh" "$QUALITY"
  fi
  LOD="$OUT/mesh_lookdev.glb"
  [[ -f "$LOD" ]] || LOD="$OUT/mesh_lod.glb"
  if [[ -x "$BIN/autoremesher" ]]; then
    echo "generate: quad remesh before paint"
    "$ROOT/scripts/quad_remesh.sh" "$LOD" "$OUT/mesh_quad.glb" 8000 \
      || echo "generate: remesh failed (non-fatal)"
    [[ -f "$OUT/mesh_quad.glb" ]] && LOD="$OUT/mesh_quad.glb"
  fi
  TEXTURED="$OUT/textured.glb"
  [[ "$PAINT_MODEL" == "pbr" ]] && TEXTURED="$OUT/pbr.glb"
  stage paint "Painting the texture"
  echo "generate: hy3d paint ($PAINT_MODEL) on $(basename "$LOD")"
  "$BIN/hy3d" paint "$LOD" "$LIFTED" -o "$TEXTURED" --weights "$PAINT_WEIGHTS" --model "$PAINT_MODEL"
  FINAL="$TEXTURED"
  if [[ "$PAINT_MODEL" == "pbr" && -f "$TEXTURED" ]]; then
    cp "$TEXTURED" "$OUT/textured.glb"
  fi
  ls -lh "$FINAL"
  if [[ -d "$OUT/views/lifted" || "$WANT_PSX" -eq 1 ]]; then
    stage retexture "Baking the photo on"
    echo "generate: retexture from the photo and any turnaround views"
    VIEW_ARGS=(--view "$LIFTED")
    if [[ "$WANT_PSX" -eq 1 ]]; then
      # Drop the cutout's edge ring; a side-on photo also goes, flipped, on the far side.
      VIEW_ARGS+=(--erode 0.008)
      [[ "$FACING" == "right" || "$FACING" == "left" ]] && VIEW_ARGS+=(--mirror)
    fi
    for v in "$OUT/views/lifted"/*.png; do
      [[ -f "$v" ]] || continue
      [[ "$(basename "$v")" == "$(basename "$LIFTED")" ]] && continue
      VIEW_ARGS+=(--view "$v")
    done
    "$PYTHON" "$ROOT/scripts/retexture.py" "$TEXTURED" -o "$OUT/textured_photo.glb" "${VIEW_ARGS[@]}" \
      || echo "generate: retexture failed (non-fatal)"
    if [[ -f "$OUT/textured_photo.glb" ]]; then
      FINAL="$OUT/textured_photo.glb"
    fi
  fi
  if [[ "$WANT_PSX" -eq 1 ]]; then
    stage psx "Cutting down to PS1 low-poly"
    echo "generate: PSX low-poly ($PSX_FACES tris, ${PSX_ATLAS}px, facing $FACING, height $PSX_HEIGHT, up $PSX_UP)"
    rm -rf "$OUT/psx"
    "$PYTHON" "$ROOT/scripts/psx_lowpoly.py" "$FINAL" --out "$OUT/psx" --name "${PSX_NAME:-$STEM}" --height "$PSX_HEIGHT" \
      --facing "$FACING" --faces "$PSX_FACES" --atlas "$PSX_ATLAS" --up "$PSX_UP" --glb "$OUT/psx.glb"
    FINAL="$OUT/psx.glb"
  fi
  stage lods "Writing detail levels"
  echo "generate: refresh LODs with textured game"
  "$ROOT/scripts/reduce.sh" "$STEM"
fi

INSPECT="$OUT/inspect.json"
stage inspect "Checking the model"
echo "generate: inspecting $FINAL"
"$PYTHON" "$ROOT/scripts/inspect_glb.py" "$FINAL" "$INSPECT"

if [[ "$WANT_STACK" -eq 1 ]]; then
  stage stack "Slicing the sprite stack"
  echo "generate: sprite stack res=$STACK_RES from $(basename "$FINAL")"
  "$PYTHON" "$ROOT/scripts/sprite_stack.py" --stem "$STEM" --glb "$FINAL" --res "$STACK_RES" \
    || echo "generate: sprite stack failed (non-fatal)"
fi

if [[ -x "$BIN/preview-glb" ]]; then
  stage preview "Rendering the thumbnail"
  echo "generate: SceneKit preview"
  "$PYTHON" -c "import trimesh; trimesh.load('$FINAL', force='mesh').export('$OUT/preview.obj')"
  "$BIN/preview-glb" "$OUT/preview.obj" "$OUT/preview.png" || echo "generate: preview failed (non-fatal)"
fi

# Last line is a machine-readable pointer for workflow agents.
echo "GENERATE_RESULT $FINAL"
if [[ -f "$OUT/psx/psx.json" ]]; then
  echo "PSX_RESULT $OUT/psx/psx.json"
fi
if [[ -f "$OUT/stack/stack.json" ]]; then
  echo "STACK_RESULT $OUT/stack/stack.json"
fi
cat "$INSPECT"
