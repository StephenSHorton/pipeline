"""Run LTX-2.3 (dgrauet/ltx-2-mlx, MIT) from its source folders, never installed: python -I ltx_run.py <clone> <args>.

scripts/animate.py starts this with the LTX venv's python under -I (no user site, no PYTHON* variables, and
neither this folder nor the working folder on sys.path), so the only code it loads is the venv's libraries and
the two source folders below. <args> are ltx-2-mlx's own command line (ic-lora ...).
"""
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -I ltx_run.py <ltx-2-mlx clone> <ltx-2-mlx args...>", file=sys.stderr)
        return 2
    clone = sys.argv[1].rstrip("/")
    sys.path[:0] = [f"{clone}/packages/ltx-core-mlx/src", f"{clone}/packages/ltx-pipelines-mlx/src"]
    from ltx_pipelines_mlx.cli import main as ltx_main

    sys.argv = ["ltx-2-mlx", *sys.argv[2:]]
    return ltx_main() or 0


if __name__ == "__main__":
    sys.exit(main())
