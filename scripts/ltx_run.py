"""Run LTX-2.3 (dgrauet/ltx-2-mlx, MIT) from its source folders, never installed: python -I ltx_run.py <clone> <args>.

scripts/animate.py starts this with the LTX venv's python under -I (no user site, no PYTHON* variables, and
neither this folder nor the working folder on sys.path), so the only code it loads is the venv's libraries and
the two source folders below. <args> are ltx-2-mlx's own command line (ic-lora ...).

One fix on top, for a second --image at a later frame (animate.py's --side): ltx-2-mlx appends such a keyframe's
tokens after the video's, and its ic-lora stage 2 (at the pinned commit, and upstream's main of 2026-11-01) strips
appended tokens only in its control-aware refine, so it fails reshaping them into the video. strip_appended drops
every token past the video's own F*H*W before unpatchify, as its stage 1 already does.
"""
import sys


def strip_appended() -> None:
    try:
        from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
    except ImportError:  # a clone without it (the tests' stand-in) has nothing to fix
        return

    unpatchify = VideoLatentPatchifier.unpatchify

    def video_tokens_only(self, tokens, spatial_dims):
        f, h, w = spatial_dims
        if tokens.shape[1] > f * h * w:  # appended conditioning (a keyframe guide), never part of the video
            tokens = tokens[:, : f * h * w, :]
        return unpatchify(self, tokens, spatial_dims)

    VideoLatentPatchifier.unpatchify = video_tokens_only


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -I ltx_run.py <ltx-2-mlx clone> <ltx-2-mlx args...>", file=sys.stderr)
        return 2
    clone = sys.argv[1].rstrip("/")
    sys.path[:0] = [f"{clone}/packages/ltx-core-mlx/src", f"{clone}/packages/ltx-pipelines-mlx/src"]
    strip_appended()
    from ltx_pipelines_mlx.cli import main as ltx_main

    sys.argv = ["ltx-2-mlx", *sys.argv[2:]]
    return ltx_main() or 0


if __name__ == "__main__":
    sys.exit(main())
