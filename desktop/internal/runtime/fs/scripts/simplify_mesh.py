#!/usr/bin/env python3
"""Back-compat wrapper around reduce_mesh.py."""
import sys
from pathlib import Path

# Re-dispatch: simplify_mesh.py in.glb -o out.glb --faces N
argv = sys.argv[1:]
faces = ["--preset", "lookdev"]
if "--faces" in argv:
    faces = []
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [str(Path(__file__).with_name("reduce_mesh.py")), *argv, *faces]
import reduce_mesh as reduce_mesh

raise SystemExit(reduce_mesh.main())
