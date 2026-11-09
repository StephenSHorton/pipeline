#!/usr/bin/env python3
"""Screenshots of an asset in the real inspector (viewer/) from fixed angles, headless Chrome.

    .venv/bin/python scripts/viewer_shots.py <stem> <out_dir> [--lod psx] [--panels]
writes <out_dir>/<stem>_{q34,side,front,top,rear34}.png and a contact sheet <stem>_sheet.png. Serves the repo
root on a free loopback port; three.js comes from desktop/frontend/node_modules (as code_model.py does)."""
import argparse, functools, http.server, socketserver, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import code_model  # noqa: E402
from PIL import Image  # noqa: E402

# name -> (theta, phi): theta 0 = looking from +Z (the car's front), pi/2 from +X
VIEWS = {"q34": (0.7, 1.2), "side": (1.5708, 1.5708), "front": (0.0, 1.5708), "top": (0.0, 0.05), "rear34": (-2.4, 1.2)}

ap = argparse.ArgumentParser()
ap.add_argument("stem"); ap.add_argument("out"); ap.add_argument("--lod", default="psx"); ap.add_argument("--panels", action="store_true")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x): pass
srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(ROOT)))
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
three = code_model.local_three(ROOT)

from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome", headless=True, args=["--use-angle=metal", "--enable-unsafe-swiftshader"])
    ctx = b.new_context(viewport={"width": 1000, "height": 700}, device_scale_factor=1)
    def serve(route, request):
        m = code_model.THREE_URL.match(request.url)
        f = (three / m.group(1)).resolve() if m else None
        if f is None or not f.is_file(): return route.continue_()
        route.fulfill(path=str(f), headers={"content-type": "text/javascript", "access-control-allow-origin": "*"})
    ctx.route(code_model.THREE_URL, serve)
    page = ctx.new_page()
    page.on("pageerror", lambda e: print("PAGEERR", str(e)[:300]))
    page.on("requestfailed", lambda r: print("FAILED", r.url[:100]))
    page.on("console", lambda m: print("CONSOLE", m.text[:150]) if m.type == "error" else None)
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.goto(f"http://127.0.0.1:{port}/viewer/?asset={a.stem}&lod={a.lod}", wait_until="commit")
    page.wait_for_function("window.__viewer !== undefined", timeout=30000)
    page.wait_for_timeout(4000)
    if not a.panels:
        page.evaluate("""() => { for (const b of document.querySelectorAll('button')) if (/hide panels/i.test(b.textContent)) { b.click(); break; } }""")
        page.wait_for_timeout(500)
    page.evaluate("window.__viewer.pause()")
    files = []
    for name, (th, ph) in VIEWS.items():
        page.evaluate(f"window.__viewer.setView({th}, {ph}, 2.6)")
        page.wait_for_timeout(700)
        p = out / f"{a.stem}_{name}.png"
        page.screenshot(path=str(p), clip={"x": 0, "y": 0, "width": 640, "height": 700})
        files.append(p)
    b.close()
ims = [Image.open(f) for f in files]
w, h = ims[0].size
sheet = Image.new("RGB", (w * 3, h * 2))
for i, im in enumerate(ims): sheet.paste(im, ((i % 3) * w, (i // 3) * h))
sheet.save(out / f"{a.stem}_sheet.png")
print("page errors:", errs[:5])
print(out / f"{a.stem}_sheet.png")
