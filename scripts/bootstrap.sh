#!/usr/bin/env bash
# Idempotent factory bootstrap: hy3d CLI, Vision subject-lift, autoremesher link, shape (+ optional paint) weights.
# generate.sh never runs this mid-job; it fails with "run scripts/bootstrap.sh <quality>" instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin"
VENDOR="$ROOT/vendor/hunyuan3d-swift"
WEIGHTS="$ROOT/weights"
VENV="$ROOT/.venv/bin"
QUALITY="${1:-draft}"   # draft | textured | pbr | psx | stack

mkdir -p "$BIN" "$WEIGHTS" "$ROOT/in" "$ROOT/out"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }; }
need swift
need python3

if [[ ! -x "$VENV/hf" && ! -x "$VENV/huggingface-cli" ]]; then
  echo "bootstrap: installing huggingface_hub into $ROOT/.venv"
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    uv venv --python 3.14 "$ROOT/.venv"
  fi
  uv pip install --python "$ROOT/.venv/bin/python" huggingface_hub pygltflib pillow
fi
HF="$VENV/hf"

if [[ ! -d "$VENDOR/.git" ]]; then
  echo "bootstrap: cloning Hunyuan3D-Swift"
  mkdir -p "$ROOT/vendor"
  git clone --depth 1 https://github.com/ZimengXiong/Hunyuan3D-Swift.git "$VENDOR"
fi

find_hy3d() {
  local c
  for c in \
    "$VENDOR/.build/release/hy3d" \
    "$VENDOR/.build/arm64-apple-macosx/release/hy3d"
  do
    if [[ -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}
if [[ ! -x "$BIN/hy3d" ]]; then
  HY3D_BUILD="$(find_hy3d || true)"
  if [[ -z "$HY3D_BUILD" ]]; then
    echo "bootstrap: building hy3d (release)"
    (cd "$VENDOR" && swift build -c release --product hy3d)
    HY3D_BUILD="$(find_hy3d)"
  fi
  cp "$HY3D_BUILD" "$BIN/hy3d"
fi
echo "bootstrap: hy3d -> $BIN/hy3d"

# mlx-swift loads mlx.metallib from the executable's directory. Homebrew MLX
# ships a compatible one; copy it next to hy3d so the CLI works outside `swift run`.
if [[ ! -f "$BIN/mlx.metallib" ]]; then
  for lib in \
    /opt/homebrew/opt/mlx/lib/mlx.metallib \
    "$(brew --prefix mlx 2>/dev/null)/lib/mlx.metallib"
  do
    if [[ -f "$lib" ]]; then
      cp "$lib" "$BIN/mlx.metallib"
      echo "bootstrap: mlx.metallib <- $lib"
      break
    fi
  done
fi
if [[ ! -f "$BIN/mlx.metallib" ]]; then
  echo "bootstrap: WARNING no mlx.metallib next to hy3d — Metal load will fail" >&2
fi

if [[ ! -x "$BIN/lift-subject" ]]; then
  echo "bootstrap: building lift-subject (Vision + Core Image)"
  swiftc -O -o "$BIN/lift-subject" "$ROOT/scripts/lift-subject.swift" \
    -framework Vision -framework AppKit -framework CoreImage \
    -framework CoreGraphics -framework ImageIO -framework UniformTypeIdentifiers
fi
echo "bootstrap: lift-subject -> $BIN/lift-subject"

if [[ ! -x "$BIN/preview-glb" ]]; then
  echo "bootstrap: building preview-glb (SceneKit + ModelIO)"
  swiftc -O -o "$BIN/preview-glb" "$ROOT/scripts/preview-glb.swift" \
    -framework SceneKit -framework ModelIO -framework Metal -framework AppKit \
    -framework ImageIO -framework UniformTypeIdentifiers
fi
echo "bootstrap: preview-glb -> $BIN/preview-glb"

# AutoRemesher (MIT) ships in-repo; remesh is opt-in (generate.sh --remesh N) but the link is always made.
AR_APP="$ROOT/vendor/autoremesher.app/Contents/MacOS/autoremesher"
if [[ -x "$AR_APP" ]]; then
  ln -sfn "../vendor/autoremesher.app/Contents/MacOS/autoremesher" "$BIN/autoremesher"
  echo "bootstrap: autoremesher -> $AR_APP"
else
  echo "bootstrap: no $AR_APP (remesh unavailable)"
fi

download() {
  local repo="$1" dest="$2"
  if ls "$dest"/*.safetensors >/dev/null 2>&1 || ls "$dest"/*/*.safetensors >/dev/null 2>&1 || ls "$dest"/*/*/*.safetensors >/dev/null 2>&1; then
    echo "bootstrap: weights already present at $dest"
    return 0
  fi
  echo "bootstrap: downloading $repo -> $dest"
  "$HF" download "$repo" --local-dir "$dest"
}

# One subfolder of a tencent repo in the layout hy3dgen reads with HY3DGEN_MODELS=$WEIGHTS
# ($WEIGHTS/<repo>/<subfolder>). A copy already in the Hugging Face cache is cloned (APFS) instead of fetched.
hy_weights() {
  local repo="$1" sub="$2" dest="$WEIGHTS/$1/$2" snap
  if ls "$dest"/*.safetensors >/dev/null 2>&1; then
    echo "bootstrap: weights already present at $dest"
    return 0
  fi
  mkdir -p "$WEIGHTS/$repo"
  for snap in "$HOME/.cache/huggingface/hub/models--${repo/\//--}/snapshots"/*/"$sub"; do
    if ls "$snap"/*.safetensors >/dev/null 2>&1; then
      echo "bootstrap: $repo/$sub <- $snap"
      rm -rf "$dest"
      cp -cRL "$snap" "$dest" 2>/dev/null || cp -RL "$snap" "$dest"
      return 0
    fi
  done
  echo "bootstrap: downloading $repo/$sub -> $dest"
  "$HF" download "$repo" --include "$sub/*" --local-dir "$WEIGHTS/$repo"
}

download "zimengxiong/hunyuan3d-mlx-shape-small" "$WEIGHTS/shape-small"
# Shape: Hunyuan3D-2mv turbo (multi-view default), Hunyuan3D-2 v2-0 (single-image fallback), and the turbo VAE
# that enable_flashvdm swaps in. The full 2mv model is the SHAPE_MV_MODEL=full opt-in.
hy_weights "tencent/Hunyuan3D-2mv" "hunyuan3d-dit-v2-mv-turbo"
hy_weights "tencent/Hunyuan3D-2" "hunyuan3d-dit-v2-0"
hy_weights "tencent/Hunyuan3D-2" "hunyuan3d-vae-v2-0-turbo"
if [[ "${SHAPE_MV_MODEL:-}" == "full" ]]; then
  hy_weights "tencent/Hunyuan3D-2mv" "hunyuan3d-dit-v2-mv"
fi

PAINT_RGB_UNET="$WEIGHTS/paint/hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors"
PAINT_PBR_UNET="$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors"
PAINT_DINO="$WEIGHTS/paint/dinov2-giant/model.safetensors"
if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" || "$QUALITY" == "psx" || "$QUALITY" == "stack" ]]; then
  if [[ -f "$PAINT_RGB_UNET" ]]; then
    echo "bootstrap: paint weights already present at $WEIGHTS/paint/hunyuan3d-paint-v2-0"
  else
    download "zimengxiong/hunyuan3d-mlx-paint-small" "$WEIGHTS/paint-small"
    mkdir -p "$WEIGHTS/paint"
    if [[ -f "$WEIGHTS/paint-small/unet/diffusion_pytorch_model.safetensors" ]]; then
      mkdir -p "$WEIGHTS/paint/hunyuan3d-paint-v2-0"
      cp -cR "$WEIGHTS/paint-small/unet" "$WEIGHTS/paint/hunyuan3d-paint-v2-0/unet"
      [[ -d "$WEIGHTS/paint-small/vae" ]] && cp -cR "$WEIGHTS/paint-small/vae" "$WEIGHTS/paint/hunyuan3d-paint-v2-0/vae"
      mkdir -p "$WEIGHTS/paint/realesrgan"
      if [[ -f "$WEIGHTS/paint-small/realesrgan/rrdbnet_mlx.safetensors" ]]; then
        ln -sfn "$WEIGHTS/paint-small/realesrgan/rrdbnet_mlx.safetensors" \
          "$WEIGHTS/paint/realesrgan/rrdbnet_mlx.safetensors"
      fi
    fi
  fi
fi
if [[ "$QUALITY" == "pbr" ]]; then
  # The check is on what hy3d reads (weights/paint/...), not on the paint-large download folder, which a
  # finished copy may have left empty.
  if [[ -f "$PAINT_PBR_UNET" && -f "$PAINT_DINO" ]]; then
    echo "bootstrap: pbr paint weights already present at $WEIGHTS/paint/hunyuan3d-paintpbr-v2-1"
  else
    download "zimengxiong/hunyuan3d-mlx-paint-large" "$WEIGHTS/paint-large"
    mkdir -p "$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1"
    if [[ -f "$WEIGHTS/paint-large/unet/diffusion_pytorch_model.safetensors" && ! -f "$PAINT_PBR_UNET" ]]; then
      cp -cR "$WEIGHTS/paint-large/unet" "$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1/unet"
    fi
    if [[ -f "$WEIGHTS/paint-large/dinov2/model.safetensors" && ! -f "$PAINT_DINO" ]]; then
      rm -rf "$WEIGHTS/paint/dinov2-giant"
      cp -cR "$WEIGHTS/paint-large/dinov2" "$WEIGHTS/paint/dinov2-giant"
    elif [[ -f "$WEIGHTS/paint-large/dinov2-giant/model.safetensors" && ! -f "$PAINT_DINO" ]]; then
      rm -rf "$WEIGHTS/paint/dinov2-giant"
      cp -cR "$WEIGHTS/paint-large/dinov2-giant" "$WEIGHTS/paint/dinov2-giant"
    fi
  fi
fi

echo "bootstrap: ready quality=$QUALITY"
echo "bootstrap: hy3d=$BIN/hy3d"
echo "bootstrap: shape_weights=$WEIGHTS/shape-small"
