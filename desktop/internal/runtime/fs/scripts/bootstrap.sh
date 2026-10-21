#!/usr/bin/env bash
# Idempotent factory bootstrap: hy3d CLI, Vision subject-lift, shape (+ optional paint) weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin"
VENDOR="$ROOT/vendor/hunyuan3d-swift"
WEIGHTS="$ROOT/weights"
VENV="$ROOT/.venv/bin"
QUALITY="${1:-draft}"   # draft | textured | pbr

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

download() {
  local repo="$1" dest="$2"
  if ls "$dest"/*.safetensors >/dev/null 2>&1 || ls "$dest"/*/*.safetensors >/dev/null 2>&1 || ls "$dest"/*/*/*.safetensors >/dev/null 2>&1; then
    echo "bootstrap: weights already present at $dest"
    return 0
  fi
  echo "bootstrap: downloading $repo -> $dest"
  "$HF" download "$repo" --local-dir "$dest"
}

download "zimengxiong/hunyuan3d-mlx-shape-small" "$WEIGHTS/shape-small"

if [[ "$QUALITY" == "textured" || "$QUALITY" == "pbr" ]]; then
  download "zimengxiong/hunyuan3d-mlx-paint-small" "$WEIGHTS/paint-small"
  mkdir -p "$WEIGHTS/paint"
  if [[ -f "$WEIGHTS/paint-small/unet/diffusion_pytorch_model.safetensors" && \
        ! -f "$WEIGHTS/paint/hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors" ]]; then
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
if [[ "$QUALITY" == "pbr" ]]; then
  download "zimengxiong/hunyuan3d-mlx-paint-large" "$WEIGHTS/paint-large"
  mkdir -p "$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1"
  if [[ -f "$WEIGHTS/paint-large/unet/diffusion_pytorch_model.safetensors" && \
        ! -f "$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors" ]]; then
    cp -cR "$WEIGHTS/paint-large/unet" "$WEIGHTS/paint/hunyuan3d-paintpbr-v2-1/unet"
  fi
  if [[ -f "$WEIGHTS/paint-large/dinov2/model.safetensors" && \
        ! -f "$WEIGHTS/paint/dinov2-giant/model.safetensors" ]]; then
    rm -rf "$WEIGHTS/paint/dinov2-giant"
    cp -cR "$WEIGHTS/paint-large/dinov2" "$WEIGHTS/paint/dinov2-giant"
  elif [[ -f "$WEIGHTS/paint-large/dinov2-giant/model.safetensors" && \
          ! -f "$WEIGHTS/paint/dinov2-giant/model.safetensors" ]]; then
    rm -rf "$WEIGHTS/paint/dinov2-giant"
    cp -cR "$WEIGHTS/paint-large/dinov2-giant" "$WEIGHTS/paint/dinov2-giant"
  fi
fi

echo "bootstrap: ready quality=$QUALITY"
echo "bootstrap: hy3d=$BIN/hy3d"
echo "bootstrap: shape_weights=$WEIGHTS/shape-small"
