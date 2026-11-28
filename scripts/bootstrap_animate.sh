#!/usr/bin/env bash
# The Animate stage's model (scripts/animate.py): LTX-2.3 with the Union-Control IC-LoRA on the Mac. Idempotent.
#   scripts/bootstrap.sh animate        (or this script directly)
# - vendor/ltx-2-mlx: dgrauet/ltx-2-mlx (MIT) at a pinned commit, never installed: scripts/ltx_run.py runs its
#   source folders under python -I from its own venv (vendor/ltx-2-mlx/.venv), with these libraries only.
# - weights/ltx-2.3-mlx-q8: only the distilled-1.1 transformer, connector, VAEs, audio, vocoder and x2 upscaler
#   (about 30 GB, not the whole 86 GB repo); weights/ic-lora-union: the Union-Control IC-LoRA;
#   weights/gemma-3-12b-it-4bit: the text encoder (the Hugging Face cache's copy is used when it is there). All
#   under the LTX-2 Community License (see NOTICE) except Gemma (Gemma Terms of Use).
# - bin/bodypose: Apple Vision's body pose per frame (scripts/bodypose.swift), for the skeleton check. Built from a
#   checkout (or the copy the app ships beside animate.py); scripts/animate.py builds it itself when it is missing.
# Already have the files elsewhere? Link them in place first (ln -s <dir> weights/ltx-2.3-mlx-q8 ...) and this
# script keeps them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin"
WEIGHTS="$ROOT/weights"
LTX="$ROOT/vendor/ltx-2-mlx"
LTX_COMMIT="747b30ef10e92a011e85b2aa0f6c4ee86ced5218"
HF="$ROOT/.venv/bin/hf"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }; }
need git
need uv
need swiftc
need ffmpeg
mkdir -p "$BIN" "$WEIGHTS" "$ROOT/vendor"

# First, so the skeleton fit works even when a download below stops the script.
if [[ -f "$ROOT/scripts/bodypose.swift" ]] && [[ ! -x "$BIN/bodypose" || "$ROOT/scripts/bodypose.swift" -nt "$BIN/bodypose" ]]; then
  echo "animate: building bodypose (Vision)"
  swiftc -O -o "$BIN/bodypose" "$ROOT/scripts/bodypose.swift" -framework Vision -framework AppKit
fi

if [[ ! -d "$LTX/packages/ltx-pipelines-mlx/src" ]]; then
  echo "animate: cloning dgrauet/ltx-2-mlx at $LTX_COMMIT"
  git clone https://github.com/dgrauet/ltx-2-mlx.git "$LTX"
  git -C "$LTX" checkout --quiet "$LTX_COMMIT"
fi

if [[ ! -x "$LTX/.venv/bin/python" ]]; then
  echo "animate: LTX venv (python 3.12) in $LTX/.venv"
  uv venv --python 3.12 "$LTX/.venv"
  uv pip install --python "$LTX/.venv/bin/python" \
    "mlx==0.32.4" "mlx-arsenal==0.19.0" "mlx-lm==0.32.2" "safetensors==0.8.0" "huggingface_hub==2.4.0" \
    "numpy==2.5.5" "pillow==13.0.0" "tqdm==4.70.1"
fi

if [[ ! -x "$HF" ]]; then
  echo "animate: no $HF (run scripts/bootstrap.sh draft first for the factory venv)" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS/ltx-2.3-mlx-q8/transformer-distilled-1.1.safetensors" ]]; then
  echo "animate: LTX-2.3 q8 weights (distilled 1.1 only)"
  "$HF" download dgrauet/ltx-2.3-mlx-q8 --local-dir "$WEIGHTS/ltx-2.3-mlx-q8" \
    --include "*.json" "LICENSE" "README.md" "transformer-distilled-1.1.safetensors" "connector.safetensors" \
    "vae_encoder.safetensors" "vae_decoder.safetensors" "audio_vae.safetensors" "vocoder.safetensors" \
    "spatial_upscaler_x2_v1_1.safetensors"
fi

LORA="$WEIGHTS/ic-lora-union/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
if [[ ! -f "$LORA" ]]; then
  echo "animate: Union-Control IC-LoRA"
  "$HF" download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control --local-dir "$WEIGHTS/ic-lora-union" \
    --include "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" "README.md"
fi

GEMMA_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--mlx-community--gemma-3-12b-it-4bit/snapshots"
if [[ ! -f "$WEIGHTS/gemma-3-12b-it-4bit/config.json" ]] && ! ls "$GEMMA_CACHE"/*/config.json >/dev/null 2>&1; then
  echo "animate: Gemma 3 12B (4-bit, MLX) text encoder"
  "$HF" download mlx-community/gemma-3-12b-it-4bit --local-dir "$WEIGHTS/gemma-3-12b-it-4bit"
fi

echo "animate: ready (LTX-2.3 in $LTX, weights in $WEIGHTS, bodypose in $BIN)"
