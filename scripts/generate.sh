#!/usr/bin/env bash
# Vertical slice: picture(s) -> lifted PNG(s) -> shape GLB -> paint + photo bake -> inspect.json
#
# No 3D without a picture: a missing or non-image --image, --ref or turnaround view stops here (exit 2).
# Turnaround views (--views-json, written by the 2D draw) drive the multi-view shape (Hunyuan3D-2mv) and the
# photo bake. When the views were drawn from a photo (views.json sheet.reference), that photo bakes over them.
# --mesh <glb> takes the shape from a mesh made elsewhere (the workbench's clay) instead of hy3d; --paint off
# unwraps the mesh instead of painting it, so a supplied mesh with --paint off loads no model at all.
# --fill photo gives every texel the photo: a front-on (or back-on) photo is also mirrored onto the opposite end
# (--mirror-ends off keeps it to the one end), and scripts/workbench/fill.py paints a swatch of the photo's own
# subject onto whatever the bake still did not reach. Off by default: a make without it is as before.
# --delight on takes the photo's own lighting out before the bake (scripts/delight.py, retexture --delight and the
# fill's swatch): a side in shadow in the picture does not bake in as a dark patch. Off by default, as before.
# Anything that falls back prints 'DEGRADED <stage> <code>: <message>' and lands in out/<stem>/degraded.json and
# inspect.json's "degraded".
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
VIEWS_JSON=""
PRE_DEGRADED=()  # --degraded "<stage> <code>: <message>": a fallback the caller (the factory) already chose
REMESH=""
# --quality psx: PS1 low-poly with the photo baked on (psx_lowpoly.py).
FACING="camera"
FACING_SET=0
PSX_HEIGHT=48
PSX_FACES=1200
PSX_ATLAS=256
PSX_UP="y"
PSX_NAME=""
MESH_IN=""
PAINT="on"
FILL="none"
MIRROR_ENDS=""
DELIGHT="off"
NO_PICTURE="3D needs a picture. Draw or add one first."

usage() {
  echo "usage: generate.sh --image <path> [--views-json <views.json>] [--degraded '<stage> <code>: <message>' ...] [--ref <path> ...] [--prompt text] [--quality draft|textured|pbr|stack|psx] [--poly high|medium|low|hero|lookdev|game|proxy] [--octree N] [--faces N] [--seed N] [--stem name] [--stack-res N] [--no-stack] [--remesh N] [--mesh <glb>] [--paint on|off] [--fill photo|none] [--mirror-ends on|off] [--delight on|off]" >&2
  echo "       psx: [--facing camera|right|left|away] [--psx-height N] [--psx-faces N] [--psx-atlas N] [--psx-up z|y] [--psx-name name]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --ref) REFS+=("$2"); shift 2 ;;
    --views-json) VIEWS_JSON="$2"; shift 2 ;;
    --degraded) PRE_DEGRADED+=("$2"); shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --poly) POLY="$2"; shift 2 ;;
    --octree) OCTREE="$2"; shift 2 ;;
    --faces) FACES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --stem) STEM="$2"; shift 2 ;;
    --stack-res) STACK_RES="$2"; shift 2 ;;
    --no-stack) WANT_STACK=0; shift ;;
    --remesh) REMESH="$2"; shift 2 ;;
    --facing) FACING="$2"; FACING_SET=1; shift 2 ;;
    --psx-height) PSX_HEIGHT="$2"; shift 2 ;;
    --psx-faces) PSX_FACES="$2"; shift 2 ;;
    --psx-atlas) PSX_ATLAS="$2"; shift 2 ;;
    --psx-up) PSX_UP="$2"; shift 2 ;;
    --psx-name) PSX_NAME="$2"; shift 2 ;;
    --mesh) MESH_IN="$2"; shift 2 ;;
    --paint) PAINT="$2"; shift 2 ;;
    --fill) FILL="$2"; shift 2 ;;
    --mirror-ends) MIRROR_ENDS="$2"; shift 2 ;;
    --delight) DELIGHT="$2"; shift 2 ;;
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

# Octree per detail level (reduce_mesh.PRESETS keeps the same table).
case "$POLY" in
  hero) OCTREE="${OCTREE:-384}" ;;
  lookdev) OCTREE="${OCTREE:-256}"; FACES="${FACES:-80000}" ;;
  game) OCTREE="${OCTREE:-128}"; FACES="${FACES:-12000}" ;;
  proxy) OCTREE="${OCTREE:-128}"; FACES="${FACES:-2000}" ;;
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
case "$QUALITY" in
  draft|textured|pbr) ;;
  *) echo "generate: unknown --quality $QUALITY (draft|textured|pbr|stack|psx)" >&2; exit 2 ;;
esac
case "$FACING" in
  camera|right|left|away) ;;
  *) echo "generate: unknown --facing $FACING (camera|right|left|away)" >&2; exit 2 ;;
esac
if [[ -n "$REMESH" && ! "$REMESH" =~ ^[0-9]+$ ]]; then
  echo "generate: --remesh takes a quad count" >&2; exit 2
fi
case "$PAINT" in
  on|off) ;;
  *) echo "generate: unknown --paint $PAINT (on|off)" >&2; exit 2 ;;
esac
if [[ "$PAINT" == "off" && "$QUALITY" == "pbr" ]]; then
  echo "generate: --quality pbr needs the PBR paint; --paint off has none" >&2; exit 2
fi
case "$FILL" in
  photo|none) ;;
  *) echo "generate: unknown --fill $FILL (photo|none)" >&2; exit 2 ;;
esac
case "$MIRROR_ENDS" in
  "") [[ "$FILL" == "photo" ]] && MIRROR_ENDS="on" || MIRROR_ENDS="off" ;;
  on|off) ;;
  *) echo "generate: unknown --mirror-ends $MIRROR_ENDS (on|off)" >&2; exit 2 ;;
esac
case "$DELIGHT" in
  on|off) ;;
  *) echo "generate: unknown --delight $DELIGHT (on|off)" >&2; exit 2 ;;
esac
FILL_CLI="$ROOT/scripts/workbench/fill.py"
if [[ "$FILL" == "photo" && ! -f "$FILL_CLI" ]]; then
  echo "generate: --fill photo runs scripts/workbench/fill.py, which is missing" >&2; exit 1
fi
UNWRAP_CLI="$ROOT/scripts/workbench/unwrap_cli.py"
if [[ "$PAINT" == "off" && "$QUALITY" == "textured" && ! -f "$UNWRAP_CLI" ]]; then
  echo "generate: --paint off unwraps with scripts/workbench/unwrap_cli.py, which is missing" >&2; exit 1
fi

PYTHON="$(factory_python)"

# --- No picture, no 3D -------------------------------------------------------------------------------------
# Each file must exist, carry a picture suffix (the factory's UPLOAD_EXTS) and decode.
not_pictures() {
  "$PYTHON" - "$@" <<'PY'
import sys
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    Image = None
bad = []
for arg in sys.argv[1:]:
    p = Path(arg)
    ok = p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    if ok and Image is not None:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            ok = False
    if not ok:
        bad.append(arg)
for b in bad:
    print(b)
PY
}
refuse_picture() {
  echo "$NO_PICTURE" >&2
  echo "generate: not a picture: $1" >&2
  exit 2
}
[[ -n "$IMAGE" ]] || refuse_picture "(no --image)"

# Turnaround views: id<TAB>path rows (Hunyuan names front|left|back|right), plus canonical/facing/consistent.
VIEW_IDS=()
VIEW_FILES=()
VIEWS_CANON=""
VIEWS_FACING=""
VIEWS_CONSISTENT=1
VIEWS_NOTE=""
PHOTO=""
PHOTO_NOTE=""
if [[ -n "$VIEWS_JSON" ]]; then
  [[ -f "$VIEWS_JSON" ]] || refuse_picture "$VIEWS_JSON (--views-json)"
  VROWS="$("$PYTHON" - "$VIEWS_JSON" "$ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
try:
    data = json.loads(src.read_text())
except (OSError, ValueError) as e:
    print(f"error\tviews.json unreadable: {e}")
    sys.exit(0)
naming = data.get("naming", "hunyuan")
if naming != "hunyuan":
    print(f"error\tviews.json naming {naming!r} is not supported (hunyuan)")
    sys.exit(0)
seen = set()
for v in data.get("views") or []:
    vid, f = v.get("id"), v.get("file")
    if vid not in {"front", "left", "back", "right"} or not f or vid in seen:
        continue
    seen.add(vid)
    p = Path(f)
    if not p.is_absolute():
        # 'file' is relative to the asset folder (out/<stem>/flat/views/x.png from out/<stem>/flat/views.json)
        # or to the views.json folder.
        cands = [src.parent.parent / p, src.parent / p]
        p = next((c for c in cands if c.exists()), cands[0])
    print(f"view\t{vid}\t{p}")
print(f"canonical\t{data.get('canonical') or ''}")
print(f"facing\t{data.get('facing') or ''}")
print(f"consistent\t{0 if data.get('consistent') is False else 1}")
flaws = data.get("flaws") or []
if flaws:
    print("flaws\t" + "; ".join(str(x) for x in flaws)[:300])
# A photo draw records the photo it drew from (root-relative); a photo replaced since is not that subject.
ref = (data.get("sheet") or {}).get("reference") or {}
if isinstance(ref, dict) and ref.get("file"):
    p = (Path(sys.argv[2]) / str(ref["file"])).resolve()
    want = str(ref.get("sha256") or "").lower()
    if not p.is_file():
        print(f"photo_note\tthe views' photo {ref['file']} is gone; the drawn views bake alone")
    elif want and hashlib.sha256(p.read_bytes()).hexdigest() != want:
        print(f"photo_note\tthe views' photo {ref['file']} changed since the draw; the drawn views bake alone")
    else:
        print(f"photo\t{p}")
PY
)"
  while IFS=$'\t' read -r kind a b; do
    case "$kind" in
      error) echo "generate: $a" >&2; exit 2 ;;
      view) VIEW_IDS+=("$a"); VIEW_FILES+=("$b") ;;
      canonical) VIEWS_CANON="$a" ;;
      facing) VIEWS_FACING="$a" ;;
      consistent) VIEWS_CONSISTENT="$a" ;;
      flaws) VIEWS_NOTE="$a" ;;
      photo) PHOTO="$a" ;;
      photo_note) PHOTO_NOTE="$a" ;;
    esac
  done <<< "$VROWS"
  ((${#VIEW_IDS[@]})) || refuse_picture "$VIEWS_JSON (no views listed)"
fi

CHECK=("$IMAGE")
((${#REFS[@]})) && CHECK+=("${REFS[@]}")
((${#VIEW_FILES[@]})) && CHECK+=("${VIEW_FILES[@]}")
BAD="$(not_pictures "${CHECK[@]}")"
[[ -z "$BAD" ]] || refuse_picture "$(printf '%s' "$BAD" | head -1)"
# The supplied mesh must be a binary glTF: the shape stage takes it as it is.
if [[ -n "$MESH_IN" ]]; then
  MESH_EXT="$(printf '%s' "${MESH_IN##*.}" | tr '[:upper:]' '[:lower:]')"  # bash 3.2: no ${x,,}
  if [[ ! -f "$MESH_IN" || "$MESH_EXT" != "glb" || "$(head -c 4 "$MESH_IN")" != "glTF" ]]; then
    echo "generate: --mesh wants a .glb file: $MESH_IN" >&2; exit 2
  fi
fi
# The photo the views were drawn from: one that no longer decodes leaves the drawn views to bake alone.
if [[ -n "$PHOTO" && -n "$(not_pictures "$PHOTO")" ]]; then
  PHOTO_NOTE="the views' photo $PHOTO is not a picture; the drawn views bake alone"
  PHOTO=""
fi
# ------------------------------------------------------------------------------------------------------------

abspath() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
IMAGE="$(abspath "$IMAGE")"
if [[ -z "$STEM" ]]; then
  STEM="$(basename "$IMAGE")"
  STEM="${STEM%.*}"
fi

# The canonical view: which way the subject looks in the main picture (retexture's --first-facing, psx --facing).
facing_of() {
  case "$1" in
    right) echo right ;; left) echo left ;; back) echo away ;; *) echo camera ;;
  esac
}
view_of() {
  case "$1" in
    right) echo right ;; left) echo left ;; away) echo back ;; *) echo front ;;
  esac
}
if ((${#VIEW_IDS[@]})); then
  [[ -n "$VIEWS_CANON" ]] || VIEWS_CANON="$(view_of "${VIEWS_FACING:-$FACING}")"
  if [[ "$FACING_SET" -eq 0 ]]; then
    FACING="$(facing_of "$VIEWS_CANON")"
  fi
fi

OUT="$ROOT/out/$STEM"
mkdir -p "$OUT"
EXT="${IMAGE##*.}"
# A rebuild from this asset's own input.* must not copy the file onto itself.
[[ "$IMAGE" -ef "$OUT/input.$EXT" ]] || cp "$IMAGE" "$OUT/input.$EXT"
HERO="$OUT/mesh_hero.glb"
MESH="$OUT/mesh.glb"
# The supplied mesh becomes the hero here, before the clean-up below could remove it (it may be a product of the
# last run, such as textured_photo.glb) and before isolate replaces views/.
if [[ -n "$MESH_IN" ]]; then
  MESH_IN="$(abspath "$MESH_IN")"
  [[ "$MESH_IN" -ef "$HERO" ]] || cp "$MESH_IN" "$HERO"
fi
if [[ -n "$PROMPT" ]]; then
  printf '%s\n' "$PROMPT" > "$OUT/prompt.txt"
fi

LOG="$OUT/generate.log"
# The factory writes this script's output to the log itself (one copy per line); run by hand, tee it.
if [[ -z "${FACTORY_ROOT:-}" ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi

# --- Loud degradation ---------------------------------------------------------------------------------------
export DEGRADED_JSON="$OUT/degraded.json"
printf '{\n  "degraded": []\n}\n' > "$DEGRADED_JSON"
degrade() {
  # degrade <stage> <code> <message>
  echo "DEGRADED $1 $2: $3"
  record_degraded "$@"
}
record_degraded() {
  # The degraded.json row only (the caller already printed the DEGRADED line).
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
merge_degraded() {
  # inspect_glb.py writes inspect.json; the degraded list is merged in here so that script stays as it is.
  [[ -f "$1" ]] || return 0
  "$PYTHON" - "$DEGRADED_JSON" "$1" <<'PY'
import json, sys
from pathlib import Path
deg, ins = Path(sys.argv[1]), Path(sys.argv[2])
try:
    rows = json.loads(deg.read_text()).get("degraded") or []
except (OSError, ValueError):
    rows = []
data = json.loads(ins.read_text())
data["degraded"] = rows
ins.write_text(json.dumps(data, indent=2) + "\n")
PY
}
# The factory's own fallbacks (views stale, views inconsistent, ...) are already in the log; they belong in
# degraded.json and inspect.json too, so the model card says why as well as the job card.
PRE_SINGLE=0  # the factory already said why there is one picture (views stale / inconsistent / none)
for ROW in ${PRE_DEGRADED[@]+"${PRE_DEGRADED[@]}"}; do
  if [[ "$ROW" =~ ^([^[:space:]]+)[[:space:]]+([^[:space:]:]+):[[:space:]]?(.*)$ ]]; then
    record_degraded "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    [[ "${BASH_REMATCH[1]}" == views || "${BASH_REMATCH[2]}" == single_view ]] && PRE_SINGLE=1
  else
    echo "generate: --degraded wants '<stage> <code>: <message>', got '$ROW'" >&2; exit 2
  fi
done
# ------------------------------------------------------------------------------------------------------------

# A rebuild starts clean: products of the last run must not sit next to a new shape.
rm -f "$OUT/shape.json" "$OUT/textured.glb" "$OUT/pbr.glb" "$OUT/textured_photo.glb" "$OUT/textured_photo.albedo.png" \
  "$OUT/textured_photo.fit.json" "$OUT/textured_photo.cover.png" "$OUT/textured_photo.fill.json" "$OUT/textured_game.glb" "$OUT/lod_textured_game.json" "$OUT/mesh_quad.glb" \
  "$OUT/psx.glb" "$OUT/mesh_mv.glb"
# Game / proxy LODs are no longer made; a remake of an older asset drops them so no stale copy outlives its shape.
rm -f "$OUT/mesh_game.glb" "$OUT/lod_game.json" "$OUT/mesh_proxy.glb" "$OUT/lod_proxy.json"

USE_MV=0
if ((${#VIEW_IDS[@]})); then
  if [[ "$VIEWS_CONSISTENT" -eq 1 ]]; then
    USE_MV=1
  fi
fi

# Progress for the app: one STAGES line with the plan, then "STAGE <key> <label>" as each one starts.
PLAN="isolate shape reduce"
if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" ]]; then
  PLAN="$PLAN $([[ "$PAINT" == "off" ]] && echo unwrap || echo paint) retexture"
  [[ "$FILL" == "photo" ]] && PLAN="$PLAN fill"
  [[ "$WANT_PSX" -eq 1 ]] && PLAN="$PLAN psx"
  PLAN="$PLAN lods"
fi
PLAN="$PLAN inspect"
[[ "$WANT_STACK" -eq 1 ]] && PLAN="$PLAN stack"
[[ -x "$BIN/preview-glb" ]] && PLAN="$PLAN preview"
echo "STAGES $PLAN"
STAGE_KEY=""
STAGE_T0=0
stage_done() {
  if [[ -n "$STAGE_KEY" ]]; then
    echo "generate: stage $STAGE_KEY took $(( $(date +%s) - STAGE_T0 ))s"
  fi
}
stage() { stage_done; STAGE_KEY="$1"; STAGE_T0="$(date +%s)"; echo "STAGE $1 $2"; }

echo "generate: root=$ROOT"
echo "generate: image=$IMAGE"
echo "generate: python=$PYTHON"
echo "generate: quality=$QUALITY psx=$WANT_PSX poly=$POLY_IN ($POLY) octree=$OCTREE faces=${FACES:-keep} seed=$SEED stack=$WANT_STACK stack-res=$STACK_RES facing=$FACING out=$OUT"
if [[ -n "$MESH_IN" ]]; then
  echo "generate: mesh=$MESH_IN (no shape model)"
fi
if [[ "$PAINT" == "off" ]]; then
  echo "generate: paint=off (unwrap, then the photo bake)"
fi
if [[ -n "$PHOTO" ]]; then
  echo "generate: photo=$PHOTO (the views were drawn from it)"
fi
if [[ -n "$PHOTO_NOTE" ]]; then
  # Only a multi-view bake would have used the photo; there, baking without it is a fallback like any other.
  if [[ "$USE_MV" -eq 1 && "$QUALITY" != "draft" ]]; then
    degrade retexture photo_skipped "$PHOTO_NOTE"
  else
    echo "generate: $PHOTO_NOTE"
  fi
fi
if [[ -n "$PROMPT" ]]; then
  echo "generate: prompt=$PROMPT"
fi
if ((${#VIEW_IDS[@]})); then
  echo "generate: views=${VIEW_IDS[*]} canonical=$VIEWS_CANON consistent=$VIEWS_CONSISTENT ($VIEWS_JSON)"
fi
if ((${#REFS[@]})); then
  echo "generate: refs=${#REFS[@]}"
fi

# hy3d only when it shapes or paints: a supplied mesh that is not painted needs just the cut-out.
NEED_HY3D=1
if [[ -n "$MESH_IN" && ( "$QUALITY" == "draft" || "$PAINT" == "off" ) ]]; then
  NEED_HY3D=0
fi
if [[ ! -x "$BIN/lift-subject" || ( "$NEED_HY3D" -eq 1 && ( ! -x "$BIN/hy3d" || ! -f "$BIN/mlx.metallib" ) ) ]]; then
  echo "generate: factory binaries missing: run scripts/bootstrap.sh $QUALITY" >&2
  exit 1
fi

LIFTED="$OUT/lifted.png"
stage isolate "Cutting out the subject"
echo "generate: lifting subject via Vision"
"$BIN/lift-subject" "$IMAGE" "$LIFTED"
# A drawn picture's see-through parts (glass, vents) show the magenta key; Vision keeps them as subject. Take the
# key's hue out of them before shape, paint and retexture see it (scripts/key_spill.py; photos have no key.json).
IMAGE_KEY="$(dirname "$IMAGE")/key.json"
if [[ -f "$IMAGE_KEY" ]]; then
  KEY_RGB="$("$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])).get("rgb"); print(",".join(str(int(c)) for c in r) if isinstance(r, list) and len(r) == 3 else "")' "$IMAGE_KEY" 2>/dev/null || true)"
  if [[ -n "$KEY_RGB" ]]; then
    echo "generate: key spill on lifted.png $("$PYTHON" "$ROOT/scripts/key_spill.py" "$LIFTED" "$LIFTED" --key "$KEY_RGB")"
  fi
fi

# Views and refs are built in a fresh folder, then swapped in, so a rebuild may read its own old views.
NEWV="$OUT/views.new"
rm -rf "$NEWV"
mkdir -p "$NEWV/lifted"
ROWS_JSON=()
i=0
while (( i < ${#VIEW_IDS[@]} )); do
  vid="${VIEW_IDS[$i]}"
  src="${VIEW_FILES[$i]}"
  vext="${src##*.}"
  cp "$src" "$NEWV/$vid.$vext"
  ROWS_JSON+=("$vid|views/$vid.$vext|shape")
  i=$((i + 1))
done
# The views' photo, cut out for the bake. Multi-view only: with one picture, lifted.png is what bakes.
if [[ -n "$PHOTO" && "$USE_MV" -eq 1 && "$QUALITY" != "draft" ]]; then
  pext="${PHOTO##*.}"
  cp "$PHOTO" "$NEWV/photo.$pext"
  "$BIN/lift-subject" "$PHOTO" "$NEWV/lifted/photo.png"
  ROWS_JSON+=("photo|views/photo.$pext|reference")
fi
idx=0
if ((${#REFS[@]})); then
  for REF in "${REFS[@]}"; do
    idx=$((idx + 1))
    rext="${REF##*.}"
    cp "$REF" "$NEWV/ref-$idx.$rext"
    "$BIN/lift-subject" "$REF" "$NEWV/lifted/ref-$idx.png"
    ROWS_JSON+=("ref-$idx|views/ref-$idx.$rext|ref")
  done
fi
if ((${#VIEW_IDS[@]})); then
  echo "generate: isolating ${#VIEW_IDS[@]} turnaround view(s) into views/lifted/ (key despill)"
  KEY_JSON="$(dirname "$VIEWS_JSON")/key.json"
  "$PYTHON" - "$NEWV" "$BIN/lift-subject" "$KEY_JSON" "$ROOT/scripts" "${VIEW_IDS[@]}" <<'PY'
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])
from key_spill import despill as key_spill

import numpy as np
from PIL import Image
from scipy import ndimage

views, lift, key_json, ids = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[5:]
key = (255, 0, 255)
try:
    rgb = json.loads(key_json.read_text()).get("rgb")
    if isinstance(rgb, list) and len(rgb) == 3:
        key = tuple(int(c) for c in rgb)
except (OSError, ValueError):
    pass
weak = int(np.argmin(key))
strong = [c for c in range(3) if key[c] - key[weak] > 96]
for vid in ids:
    src = next(p for p in views.iterdir() if p.is_file() and p.stem == vid)
    out = views / "lifted" / f"{vid}.png"
    im = Image.open(src)
    a = np.asarray(im.convert("RGBA"))
    cut = im.mode in ("RGBA", "LA") and (a[..., 3] < 128).mean() > 0.02
    if not cut:
        # No cut-out alpha: Vision lifts the subject as it does the main picture.
        subprocess.run([lift, str(src), str(out)], check=True)
        a = np.asarray(Image.open(out).convert("RGBA"))
    a = a.copy()
    # Despill: on the rim (within 3 px of the ground) the key's strong channels come down to its weak one.
    solid = a[..., 3] >= 128
    rim = (a[..., 3] > 0) & ndimage.binary_dilation(~solid, iterations=3)
    if strong:
        px = a[..., :3].astype(np.int32)
        excess = np.clip(px[..., strong].min(-1) - px[..., weak], 0, None)
        for c in strong:
            px[..., c] = np.where(rim, px[..., c] - excess, px[..., c])
        a[..., :3] = np.clip(px, 0, 255).astype(np.uint8)
    # Then the see-through parts (glass, vents) that show the key: its hue comes out of them too.
    a, spill = key_spill(a, key)
    Image.fromarray(a, "RGBA").save(out)
    note = f", key spill {spill['pixels']} px" if not spill["skipped"] else f", key spill skipped: {spill['skipped']}"
    print(f"generate: views/lifted/{vid}.png ({'cut-out' if cut else 'lifted'}, {int(rim.sum())} rim px despilled{note})")
PY
fi
rm -rf "$OUT/views"
if ((${#ROWS_JSON[@]})); then
  mv "$NEWV" "$OUT/views"
  {
    printf '{\n  "views": [\n'
    printf '    {"id": "input", "file": "input.%s", "role": "shape", "label": "Input image"}' "$EXT"
    for row in "${ROWS_JSON[@]}"; do
      vid="${row%%|*}"; rest="${row#*|}"; file="${rest%%|*}"; role="${rest##*|}"
      printf ',\n    {"id": "%s", "file": "%s", "role": "%s", "label": "%s"}' "$vid" "$file" "$role" "$vid"
    done
    printf '\n  ]\n}\n'
  } > "$OUT/views.json"
  echo "generate: wrote $OUT/views.json"
else
  rm -rf "$NEWV" "$OUT/views.json"
fi

if [[ -n "$MESH_IN" ]]; then
  stage shape "Taking the supplied mesh"
else
  stage shape "Building the shape"
fi
export SHAPE_OCTREE="$OCTREE"
export SHAPE_SEED="$SEED"
SHAPE_T0="$(date +%s)"
SHAPE_MODEL=""
SHAPE_STEPS=""
ORIENT_VIEWS=()

if [[ -n "$MESH_IN" ]]; then
  # Copied to the hero above. A supplied mesh is a choice, not a fallback: no single_view row.
  SHAPE_MODEL="supplied"
  SHAPE_STEPS=0
  if [[ "$USE_MV" -eq 1 ]]; then
    for v in front left back right; do
      [[ -f "$OUT/views/lifted/$v.png" ]] && ORIENT_VIEWS+=(--view "$v=$OUT/views/lifted/$v.png")
    done
  else
    ORIENT_VIEWS=(--view "$(view_of "$FACING")=$LIFTED")
  fi
  echo "generate: shape from $MESH_IN"
elif [[ "$USE_MV" -eq 1 ]]; then
  set +e
  "$ROOT/scripts/shape_mv.sh" "$STEM" "$HERO" 2>&1 | tee "$OUT/shape_mv.log"
  MV_RC=${PIPESTATUS[0]}
  set -e
  MV_LOG="$(cat "$OUT/shape_mv.log")"
  if [[ $MV_RC -eq 0 && -s "$HERO" ]]; then
    SHAPE_MODEL="hunyuan3d-2mv-${SHAPE_MV_MODEL:-turbo}"
    SHAPE_STEPS="${SHAPE_MV_STEPS:-$([[ "${SHAPE_MV_MODEL:-turbo}" == "full" ]] && echo 30 || echo 5)}"
    echo "generate: multi-view shape wrote $HERO"
    for v in front left back right; do
      [[ -f "$OUT/views/lifted/$v.png" ]] && ORIENT_VIEWS+=(--view "$v=$OUT/views/lifted/$v.png")
    done
  else
    REASON="$(printf '%s\n' "$MV_LOG" | sed -n 's/^SHAPE_MV_FAIL //p' | tail -1)"
    degrade shape mv_failed "multi-view shape failed (${REASON:-exit $MV_RC}); single-image shape instead"
    USE_MV=0
  fi
elif ((${#VIEW_IDS[@]})); then
  degrade shape single_view "turnaround views failed the consistency check${VIEWS_NOTE:+ ($VIEWS_NOTE)}; one picture"
elif [[ "$PRE_SINGLE" -eq 0 ]]; then
  degrade shape single_view "one picture"
fi

if [[ -z "$MESH_IN" && "$USE_MV" -eq 0 ]]; then
  # Single-image shape: Hunyuan3D-2 v2-0 (30 steps), then shape-large (8 steps) if hy3d cannot load it.
  # Steps go with the weights set.
  V20="$ROOT/weights/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0"
  LARGE="$ROOT/weights/shape-large"
  SHAPED=0
  if [[ -f "$V20/model.fp16.safetensors" ]]; then
    echo "generate: hy3d shape v2-0 (30 steps, octree $OCTREE)"
    if "$BIN/hy3d" shape "$LIFTED" -o "$HERO" --weights "$V20" --seed "$SEED" --steps 30 --octree "$OCTREE"; then
      SHAPED=1; SHAPE_MODEL="hunyuan3d-dit-v2-0"; SHAPE_STEPS=30
    fi
  fi
  if [[ "$SHAPED" -eq 0 ]]; then
    [[ -d "$LARGE" ]] || { echo "generate: no single-image shape weights: run scripts/bootstrap.sh $QUALITY" >&2; exit 1; }
    degrade shape v2_0_unavailable "hy3d could not run Hunyuan3D-2 v2-0; shape-large (8 steps) instead"
    "$BIN/hy3d" shape "$LIFTED" -o "$HERO" --weights "$LARGE" --seed "$SEED" --steps 8 --octree "$OCTREE"
    SHAPE_MODEL="shape-large"; SHAPE_STEPS=8
  fi
  ORIENT_VIEWS=(--view "$(view_of "$FACING")=$LIFTED")
fi
ls -lh "$HERO" || ls -l "$HERO"

# Turn a multi-view shape so the canonical view sits at az 0 (where paint, retexture and psx expect the picture);
# a single-image shape is already there and only gets its silhouette IoU recorded. So does a supplied mesh: it
# was made (and measured) in the frame it is in, and turning it would undo that.
ORIENT_CANON="$VIEWS_CANON"
ORIENT_FLAGS=()
SHAPE_MODE="$([[ "$USE_MV" -eq 1 ]] && echo multiview || echo single)"
if [[ "$USE_MV" -eq 0 ]]; then
  ORIENT_CANON="$(view_of "$FACING")"
  ORIENT_FLAGS=(--no-turn --min-iou 0)
fi
if [[ -n "$MESH_IN" ]]; then
  ORIENT_FLAGS=(--no-turn --min-iou 0 --meta "source=$MESH_IN")
  SHAPE_MODE="supplied"
fi
set +e
"$PYTHON" "$ROOT/scripts/mv_orient.py" "$HERO" --canonical "$ORIENT_CANON" "${ORIENT_VIEWS[@]}" \
  ${ORIENT_FLAGS[@]+"${ORIENT_FLAGS[@]}"} --json "$OUT/shape.json" \
  --meta "model=$SHAPE_MODEL" --meta "steps=$SHAPE_STEPS" --meta "octree=$OCTREE" --meta "seed=$SEED" \
  --meta "mode=$SHAPE_MODE" \
  --meta "seconds=$(( $(date +%s) - SHAPE_T0 ))"
ORC=$?
set -e
if [[ $ORC -eq 3 ]]; then
  IOU="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["orient"]["iou"])' "$OUT/shape.json")"
  degrade shape orient_low_iou "the shape's silhouettes match the views poorly (IoU $IOU < 0.5)"
elif [[ $ORC -ne 0 ]]; then
  degrade shape orient_failed "mv_orient.py exited $ORC; the shape was left as generated"
fi
# A turned multi-view shape has its canonical view looking from az 0, so its front faces az 0 (a front-canonical
# prop) or 90 (a side-on creature, canonical right): the photo bake's first camera is searched around that.
FRONT_AZ=""
if [[ $ORC -eq 0 && -f "$OUT/shape.json" ]]; then
  FRONT_AZ="$("$PYTHON" -c 'import json,sys
o = json.load(open(sys.argv[1])).get("orient") or {}
az = {"front": 0, "right": 90, "back": 180, "left": 270}.get(o.get("canonical"))
print(az if o.get("turned") and az is not None else "")' "$OUT/shape.json" 2>/dev/null || true)"
fi

# The drawn turnaround can squash a long animal (each view drawn in a square quarter of one sheet): a side-on
# photo's height/length puts the length back. Only a multi-view shape with the views' photo, turned (orient ran,
# ORC 0 or 3), and not a supplied mesh (made and measured in its own frame). A front-on or 3/4 photo, or a length
# already within 5%, leaves the hero alone; the reason lands in shape.json "proportion".
if [[ -z "$MESH_IN" && "$USE_MV" -eq 1 && -f "$OUT/views/lifted/photo.png" && ( $ORC -eq 0 || $ORC -eq 3 ) ]]; then
  if ! "$PYTHON" "$ROOT/scripts/proportion_check.py" "$HERO" --photo "$OUT/views/lifted/photo.png" \
       --canonical "$ORIENT_CANON" --json "$OUT/shape.json"; then
    degrade shape proportion_failed "the photo's proportion check failed; the shape keeps the drawn length"
  fi
fi

stage reduce "Reducing the mesh"
# reduce_mesh.py drops the pieces under 2% of the area (shape models leave floating crumbs) before it decimates.
if [[ -n "$FACES" ]]; then
  echo "generate: reduce to $FACES faces (--poly $POLY)"
  "$PYTHON" "$ROOT/scripts/reduce_mesh.py" "$HERO" -o "$MESH" --faces "$FACES" --preset "$POLY" --min-piece 0.02
else
  "$PYTHON" "$ROOT/scripts/reduce_mesh.py" "$HERO" -o "$MESH" --min-piece 0.02
fi
echo "generate: lookdev mesh"
"$ROOT/scripts/reduce.sh" "$STEM"

FINAL="$MESH"
if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" ]]; then
  PAINT_MODEL="rgb"
  PAINT_WEIGHTS="$ROOT/weights/paint"
  PAINT_UNET="$PAINT_WEIGHTS/hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors"
  if [[ "$QUALITY" == "pbr" ]]; then
    PAINT_MODEL="pbr"
    PAINT_UNET="$PAINT_WEIGHTS/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors"
  fi
  if [[ "$PAINT" == "on" && ! -f "$PAINT_UNET" ]]; then
    echo "generate: paint weights missing ($PAINT_UNET): run scripts/bootstrap.sh $QUALITY" >&2
    exit 1
  fi
  LOD="$OUT/mesh_lookdev.glb"
  [[ -f "$LOD" ]] || LOD="$OUT/mesh_lod.glb"
  if [[ -n "$REMESH" ]]; then
    if [[ -x "$BIN/autoremesher" ]]; then
      echo "generate: quad remesh ($REMESH quads) before paint"
      if "$ROOT/scripts/quad_remesh.sh" "$LOD" "$OUT/mesh_quad.glb" "$REMESH"; then
        LOD="$OUT/mesh_quad.glb"
      else
        degrade remesh failed "quad remesh failed; painted the triangle mesh"
      fi
    else
      degrade remesh unavailable "bin/autoremesher is missing (run scripts/bootstrap.sh); painted the triangle mesh"
    fi
  else
    echo "generate: remesh: off (opt-in)"
  fi
  TEXTURED="$OUT/textured.glb"
  [[ "$PAINT_MODEL" == "pbr" ]] && TEXTURED="$OUT/pbr.glb"
  if [[ "$PAINT" == "off" ]]; then
    # No paint model: UVs and a plain atlas, which the photo bake below fills.
    stage unwrap "Unwrapping the mesh"
    echo "generate: unwrap $(basename "$LOD") (no paint)"
    "$PYTHON" "$UNWRAP_CLI" "$LOD" -o "$TEXTURED" --texture 1024
  else
    stage paint "Painting the texture"
    echo "generate: hy3d paint ($PAINT_MODEL) on $(basename "$LOD")"
    "$BIN/hy3d" paint "$LOD" "$LIFTED" -o "$TEXTURED" --weights "$PAINT_WEIGHTS" --model "$PAINT_MODEL"
  fi
  FINAL="$TEXTURED"
  if [[ "$PAINT_MODEL" == "pbr" && -f "$TEXTURED" ]]; then
    cp "$TEXTURED" "$OUT/textured.glb"
  fi
  ls -lh "$FINAL"

  stage retexture "Baking the photo on"
  # Every named view, canonical first, then the refs. --first-facing says where the first one looks;
  # --mirror only when a side-on picture has no real opposite view.
  # The views' photo goes before them all, with --first-wins: retexture averages the views it bakes, so only that
  # flag lets the real photo win where it reaches while the drawn views fill what it never saw. Its angle is
  # unknown (the sheet is drawn in the kind's canonical turnaround, whatever the photo shows), so no --first-facing:
  # retexture searches it within 50 degrees of the drawn view it looks most like, anchored on a drawn side view.
  VIEW_ARGS=()
  NAMED=()
  FIRST_FACING="$FACING"
  PHOTO_VIEW=""
  if [[ "$USE_MV" -eq 1 ]]; then
    FIRST_FACING="$(facing_of "$VIEWS_CANON")"
    if [[ -f "$OUT/views/lifted/photo.png" ]]; then
      PHOTO_VIEW="$OUT/views/lifted/photo.png"
      VIEW_ARGS+=(--view "$PHOTO_VIEW" --first-wins)
    fi
    VIEW_ARGS+=(--view "$OUT/views/lifted/$VIEWS_CANON.png")
    NAMED+=("$VIEWS_CANON")
    for v in front right back left; do
      [[ "$v" == "$VIEWS_CANON" || ! -f "$OUT/views/lifted/$v.png" ]] && continue
      VIEW_ARGS+=(--view "$OUT/views/lifted/$v.png")
      NAMED+=("$v")
    done
  else
    VIEW_ARGS+=(--view "$LIFTED")
    NAMED+=("$(view_of "$FACING")")
  fi
  for v in "$OUT/views/lifted"/ref-*.png; do
    [[ -f "$v" ]] && VIEW_ARGS+=(--view "$v")
  done
  [[ -n "$PHOTO_VIEW" ]] || VIEW_ARGS+=(--first-facing "$FIRST_FACING")
  [[ "$USE_MV" -eq 1 && -n "$FRONT_AZ" ]] && VIEW_ARGS+=(--front-az "$FRONT_AZ")
  # Drawn views never agree exactly, so averaged they showed as patches (the pig): one view per triangle, colours
  # evened out, seams levelled, the paint under what no view sees (scripts/seam_bake.py).
  [[ "$USE_MV" -eq 1 ]] && VIEW_ARGS+=(--select)
  OPPOSITE=""
  ENDS=0
  case "$FIRST_FACING" in right) OPPOSITE=left ;; left) OPPOSITE=right ;; esac
  # --fill photo: a front-on or back-on picture with no real view of the other end mirrors onto it too.
  # --mirror-ends tells retexture to reflect such views across their line of sight; only then, so a plain
  # --mirror bake (a side-on make without --fill) reflects every view through z exactly as it always did.
  if [[ "$MIRROR_ENDS" == "on" ]]; then
    case "$FIRST_FACING" in camera) OPPOSITE=back; ENDS=1 ;; away) OPPOSITE=front; ENDS=1 ;; esac
  fi
  if [[ -n "$OPPOSITE" && " ${NAMED[*]} " != *" $OPPOSITE "* ]]; then
    VIEW_ARGS+=(--mirror)
    [[ "$ENDS" -eq 1 ]] && VIEW_ARGS+=(--mirror-ends)
  fi
  # Drop the cutout's edge ring at PS1 sizes.
  [[ "$WANT_PSX" -eq 1 ]] && VIEW_ARGS+=(--erode 0.008)
  # Every view de-lit before it bakes (last: the fill below reads the first picture at VIEW_ARGS[1]).
  DELIGHT_ARGS=()
  [[ "$DELIGHT" == "on" ]] && DELIGHT_ARGS=(--delight 1)
  VIEW_ARGS+=(${DELIGHT_ARGS[@]+"${DELIGHT_ARGS[@]}"})
  echo "generate: retexture with ${PHOTO_VIEW:+the photo (wins), then }${NAMED[*]}${OPPOSITE:+ (mirror: $([[ " ${VIEW_ARGS[*]} " == *" --mirror "* ]] && echo on || echo off))}"
  if "$PYTHON" "$ROOT/scripts/retexture.py" "$TEXTURED" -o "$OUT/textured_photo.glb" "${VIEW_ARGS[@]}"; then
    [[ -f "$OUT/textured_photo.glb" ]] && FINAL="$OUT/textured_photo.glb"
  else
    degrade retexture failed "the photo bake failed; the painted texture is used as is"
  fi
  if [[ "$FILL" == "photo" ]]; then
    stage fill "Filling what the photo missed"
    # The picture the bake put first (VIEW_ARGS[1]): the views' photo, else the main cut-out or the canonical view.
    FILL_PHOTO="${VIEW_ARGS[1]}"
    if [[ "$FINAL" != "$OUT/textured_photo.glb" ]]; then
      degrade fill skipped "no photo bake to fill; the texture is used as is"
    elif "$PYTHON" "$FILL_CLI" "$OUT/textured_photo.glb" -o "$OUT/textured_photo.glb" --photo "$FILL_PHOTO" \
         --cover "$OUT/textured_photo.cover.png" ${DELIGHT_ARGS[@]+"${DELIGHT_ARGS[@]}"}; then
      echo "generate: filled the texels the photo missed from $(basename "$FILL_PHOTO")"
    else
      degrade fill failed "filling what the photo missed failed; those texels keep the base texture"
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
  # The photo bake and the PS1 model exist only now, so list them; reduce.sh above listed what was there before
  # paint. Just the list: the meshes are already cut (no more game / proxy copies to re-make).
  stage lods "Listing the finished models"
  echo "generate: list the finished models"
  "$PYTHON" "$ROOT/scripts/lods.py" "$OUT"
fi

INSPECT="$OUT/inspect.json"
stage inspect "Checking the model"
echo "generate: inspecting $FINAL"
"$PYTHON" "$ROOT/scripts/inspect_glb.py" "$FINAL" "$INSPECT"
merge_degraded "$INSPECT"

if [[ "$WANT_STACK" -eq 1 ]]; then
  stage stack "Slicing the sprite stack"
  echo "generate: sprite stack res=$STACK_RES from $(basename "$FINAL")"
  "$PYTHON" "$ROOT/scripts/sprite_stack.py" --stem "$STEM" --glb "$FINAL" --res "$STACK_RES" \
    || degrade stack failed "the sprite stack failed"
fi

if [[ -x "$BIN/preview-glb" ]]; then
  stage preview "Rendering the thumbnail"
  echo "generate: SceneKit preview"
  if ! { "$PYTHON" -c "import sys, trimesh; trimesh.load(sys.argv[1], force='mesh').export(sys.argv[2])" \
           "$FINAL" "$OUT/preview.obj" && "$BIN/preview-glb" "$OUT/preview.obj" "$OUT/preview.png"; }; then
    degrade preview failed "the thumbnail failed to render"
  fi
fi
stage_done
merge_degraded "$INSPECT"

# Last line is a machine-readable pointer for workflow agents.
echo "GENERATE_RESULT $FINAL"
if [[ -f "$OUT/psx/psx.json" ]]; then
  echo "PSX_RESULT $OUT/psx/psx.json"
fi
if [[ -f "$OUT/stack/stack.json" ]]; then
  echo "STACK_RESULT $OUT/stack/stack.json"
fi
cat "$INSPECT"
