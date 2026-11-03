#!/usr/bin/env python3
"""The low-poly code make (#360): a hard-surface prop built by a standalone Three.js page, exported to a GLB.

Claude reads the four drawn views (out/<stem>/flat/views/*.png) and writes one HTML page that builds the prop from
Three.js primitives and has an Export button that downloads it as a binary GLB (GLTFExporter). This turns the page
into the asset's model with no image-to-3D model at all, so the output is licence-clean (no Hunyuan territory
limit). On the #351 wheelbarrow it beat GLB-Coder and the TRELLIS.2 path: closest to the drawing, 614 triangles.

    .venv/bin/python scripts/code_model.py export page.html out.glb [--shot page.png]

export opens the page in headless Chrome (Playwright, the installed Chrome, else Playwright's Chromium), presses
Export and saves the download. Three.js comes from the importmap's URLs; when the desktop frontend's node_modules
holds a copy, those requests are answered from it, so a make needs no network and gets one pinned version.

publish() makes the GLB the asset's mesh stage, as generate.sh's PS1 step publishes a cut: psx.glb (and mesh.glb),
psx/psx.json naming the page as its source, inspect.json, the lods.json entry and the thumbnail; the page itself is
kept as workbench/code/model.html, so the version carries the code it was built from. The workbench's code_model
tool calls it and keeps the result as a mesh version.

The page contract (the playbook's rule says the same): metres, Y up, standing on y = 0, the front toward +Z, centred
on x = 0; flat-shaded MeshStandardMaterial colours (no textures needed); a button whose text is Export.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import flow  # noqa: E402
import inspect_glb  # noqa: E402
import lods  # noqa: E402

PAGE = "workbench/code/model.html"
SHOT = "workbench/code/page.png"
# Three.js the page asks for by URL (jsdelivr or unpkg, any version) -> the path inside the three package.
THREE_URL = re.compile(r"^https://(?:cdn\.jsdelivr\.net/npm|unpkg\.com)/three@[^/]+/(.+)$")
EXPORT_BUTTON = re.compile(r"export|glb|download", re.I)


class CodeModelError(RuntimeError):
    pass


def local_three(root: Path = ROOT) -> Path | None:
    for p in (root / "desktop" / "frontend" / "node_modules" / "three",):
        if (p / "build" / "three.module.js").is_file():
            return p
    return None


def three_version(three: Path | None) -> str | None:
    try:
        return json.loads((three / "package.json").read_text())["version"] if three else None
    except (OSError, ValueError, KeyError):
        return None


def export(page: Path, out: Path, shot: Path | None = None, timeout_s: float = 60.0, root: Path = ROOT) -> dict:
    """Open page, press its Export button, save the GLB it downloads to out. Returns what happened, with the page's
    errors (a page can throw and still export). Raises CodeModelError when nothing was downloaded."""
    from playwright.sync_api import Error as PwError, sync_playwright

    page = Path(page).resolve()
    if not page.is_file():
        raise CodeModelError(f"no page at {page}")
    three = local_three(root)
    errors: list[str] = []
    served: set[str] = set()
    with sync_playwright() as pw:
        browser = None
        for kw in ({"channel": "chrome"}, {}):  # the installed Chrome needs no download; else Playwright's own
            try:
                browser = pw.chromium.launch(headless=True, args=["--use-angle=metal",
                                                                   "--enable-unsafe-swiftshader"], **kw)
                break
            except PwError as e:
                errors.append(f"launch {kw or 'chromium'}: {str(e).splitlines()[0]}")
        if browser is None:
            raise CodeModelError("no browser to run the page: install Google Chrome, or "
                                 "`.venv/bin/python -m playwright install chromium`; " + "; ".join(errors))
        errors.clear()
        try:
            ctx = browser.new_context(accept_downloads=True, viewport={"width": 900, "height": 700})
            if three is not None:
                def serve(route, request):
                    m = THREE_URL.match(request.url)
                    f = (three / m.group(1)).resolve() if m else None
                    if f is None or not f.is_file() or three.resolve() not in f.parents:
                        return route.continue_()
                    served.add(m.group(1))
                    route.fulfill(path=str(f), headers={"content-type": "text/javascript",
                                                        "access-control-allow-origin": "*"})
                ctx.route(THREE_URL, serve)
            tab = ctx.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            tab.goto(page.as_uri(), wait_until="load", timeout=timeout_s * 1000)
            button = tab.get_by_role("button", name=EXPORT_BUTTON).first
            try:
                button.wait_for(state="visible", timeout=timeout_s * 1000)
            except PwError:
                raise CodeModelError("the page has no Export button (a <button> whose text says Export); page "
                                     "errors: " + ("; ".join(errors[:5]) or "none")) from None
            tab.wait_for_timeout(500)  # a module script builds the scene after load; let one frame draw
            if shot is not None:
                Path(shot).parent.mkdir(parents=True, exist_ok=True)
                tab.screenshot(path=str(shot))
            try:
                with tab.expect_download(timeout=timeout_s * 1000) as dl:
                    button.click()
            except PwError as e:
                raise CodeModelError(f"pressing Export downloaded nothing ({str(e).splitlines()[0]}); page "
                                     "errors: " + ("; ".join(errors[:5]) or "none")) from None
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            dl.value.save_as(str(out))
        finally:
            browser.close()
    data = out.read_bytes()[:4]
    if data != b"glTF":
        raise CodeModelError("the download is not a binary GLB (export with GLTFExporter's binary: true)")
    return {"glb": str(out), "bytes": out.stat().st_size, "errors": errors,
            "three": {"local": three_version(three), "served": sorted(served)} if three else {"local": None}}


def thumbnail(glb: Path, png: Path, root: Path = ROOT) -> dict:
    """generate.sh's thumbnail: the model flattened to an OBJ, rendered by bin/preview-glb."""
    tool = root / "bin" / "preview-glb"
    if not tool.is_file():
        return {"ok": False, "error": "no bin/preview-glb (scripts/bootstrap.sh builds it)"}
    import trimesh

    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "preview.obj"
        trimesh.load(str(glb), force="mesh").export(str(obj))
        r = subprocess.run([str(tool), str(obj), str(png)], capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[-300:]}
    return {"ok": True, "png": str(png)}


def publish(folder: Path, stem: str, page_html: bytes, glb: Path, shot: Path | None = None,
            root: Path = ROOT) -> dict:
    """Make glb the asset's model, replacing the mesh stage's live files (the caller keeps a version of them
    first). Returns the inspect report, the lods entry and the thumbnail result."""
    rep = inspect_glb.inspect(str(glb))
    if not rep["ok"]:
        raise CodeModelError(f"the exported GLB has no triangles: {rep['errors']}")
    import trimesh

    mesh = trimesh.load(str(glb), force="mesh")
    # The live files belong to whatever made the last model (a TRELLIS bake, an older cut): none of it describes
    # this one, and a stale degraded.json or textured.glb would mislead the review and the app's tabs.
    for rel in flow.stage_files(folder, "mesh"):
        (folder / rel).unlink(missing_ok=True)
    (folder / PAGE).parent.mkdir(parents=True, exist_ok=True)
    (folder / PAGE).write_bytes(page_html)
    if shot is not None and Path(shot).is_file():
        shutil.copy2(shot, folder / SHOT)
    for name in ("psx.glb", "mesh.glb"):
        tmp = folder / f".{name}.tmp"
        shutil.copy2(glb, tmp)
        tmp.replace(folder / name)
    (folder / "inspect.json").write_text(json.dumps(rep, indent=2) + "\n")
    (folder / "psx").mkdir(exist_ok=True)
    info = {"name": stem, "made_by": "code", "source": PAGE, "faces": rep["faces"],
            "triangles": int(len(mesh.faces)), "vertices": rep["vertices"], "has_uvs": rep["has_uvs"],
            "bounds": [[round(float(x), 4) for x in row] for row in mesh.bounds],
            "height": round(float(mesh.extents[1]), 4)}
    (folder / "psx" / "psx.json").write_text(json.dumps(info, indent=2) + "\n")
    item = lods.entry(folder, "psx")
    lods.put(folder, stem, item)
    thumb = thumbnail(folder / "psx.glb", folder / "preview.png", root)
    return {"inspect": {k: rep[k] for k in ("faces", "vertices", "meshes", "materials", "textures", "has_uvs")},
            "psx": info, "lod": item, "thumbnail": thumb}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="page -> GLB, headless")
    ex.add_argument("page", type=Path)
    ex.add_argument("out", type=Path)
    ex.add_argument("--shot", type=Path)
    ex.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)
    try:
        res = export(args.page, args.out, args.shot, args.timeout)
    except CodeModelError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
