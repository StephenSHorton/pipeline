#!/usr/bin/env python3
"""Installed host must own factory_http and surface real start errors.

The desktop UI is the React app (desktop/frontend/src). The old single-file page (frontend/index.html, served
as /legacy.html) is retired, so the UI halves of these tests read the React sources; the modules' behaviour is
pinned in vitest beside each one. These keep the host / factory / UI pairings that span languages.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "desktop" / "frontend"
SRC = FRONTEND / "src"
SIDECAR = (ROOT / "desktop" / "internal" / "sidecar" / "sidecar.go").read_text(encoding="utf-8")
MAIN = (ROOT / "desktop" / "main.go").read_text(encoding="utf-8")
RUNTIME = (ROOT / "desktop" / "internal" / "runtime" / "runtime.go").read_text(encoding="utf-8")
EMBED_HTTP = ROOT / "desktop" / "internal" / "runtime" / "fs" / "scripts" / "factory_http.py"


def src(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def viewer_page() -> str:
    return (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")


def factory_http() -> str:
    return (ROOT / "scripts" / "factory_http.py").read_text(encoding="utf-8")


API = src("lib/api.ts")
JOBS = src("lib/jobs.ts")
MAKE = src("screens/home/make.ts")


class DesktopShellTests(unittest.TestCase):
    def test_same_origin_factory(self) -> None:
        # The app calls the factory on its own origin (relative paths; the Go host proxies them).
        self.assertIn("res = await fetch(path, opts);", API)
        self.assertNotIn("127.0.0.1:8766", API)
        self.assertNotIn("Start scripts/factory_http", API)

    def test_ui_shows_json_error_not_status_text(self) -> None:
        self.assertIn("JSON.parse(text)", API)
        self.assertIn("data.error", API)
        self.assertIn("Factory failed to start", API)

    def test_proxy_returns_json_when_down(self) -> None:
        self.assertIn("writeDown", SIDECAR)
        self.assertIn("application/json", SIDECAR)
        self.assertNotIn("factory starting", SIDECAR)
        self.assertIn('p == "clean"', SIDECAR)

    def test_host_extracts_factory_and_python(self) -> None:
        self.assertIn("sidecar.New(version)", MAIN)
        self.assertIn("WriteFile(filepath.Join(paths.DataDir(), \"deps.json\")", MAIN)
        self.assertIn("ExtractTo", RUNTIME)
        self.assertIn("factoryScriptsStale", RUNTIME)
        self.assertIn(".embed-hash", RUNTIME)
        self.assertTrue(EMBED_HTTP.is_file(), "run node desktop/frontend/scripts/copy-factory.mjs")
        self.assertEqual(EMBED_HTTP.read_text(encoding="utf-8"), factory_http())

    def test_no_pipeline_root_install_instruction(self) -> None:
        self.assertNotIn("set PIPELINE_ROOT", MAIN)
        self.assertNotIn("set PIPELINE_ROOT", SIDECAR)

    def test_ui_shows_build_version(self) -> None:
        self.assertIn('factoryJson<VersionInfo & { label?: string }>("/version")', API)
        self.assertIn("export function rawVersion", src("screens/settings/update.ts"))
        self.assertIn("windowTitle(version)", MAIN)
        self.assertIn('r.URL.Path == "/version"', MAIN)
        self.assertNotIn('Title:            "Pipeline"', MAIN)

    def test_mesh_loads_via_glb_proxy(self) -> None:
        self.assertIn("export function meshUrls", API)
        self.assertIn("`/glb/${enc(stem)}`", API)
        self.assertIn("preferLod", API)
        self.assertIn("Loading mesh", src("screens/viewer/Hud.tsx"))
        self.assertIn("factory.ServeOut", MAIN)

    def test_generate_does_not_post_file_body(self) -> None:
        self.assertNotIn("body: file", API)
        self.assertIn("export async function uploadImage", API)
        self.assertIn("arrayBufferToBase64", API)
        self.assertIn("application/json", API)
        self.assertIn("SaveUpload", API)
        self.assertIn("HandleUploadHTTP", MAIN)

    def test_home_is_prompt_page(self) -> None:
        # Home asks what you're making first. The standalone viewer/ keeps its own page and layout.
        self.assertIn('aria-label="What are you making"', src("screens/home/MakePanel.tsx"))
        for output in ("ps1", "model", "stack", "sprite"):
            self.assertIn(f'id: "{output}"', MAKE)
        viewer = viewer_page()
        self.assertIn("home-cluster", viewer)
        self.assertIn("min(1280px, 74vw)", viewer)

    def test_poly_high_medium_low(self) -> None:
        for poly in ('id: "high"', 'id: "medium"', 'id: "low"'):
            self.assertIn(poly, MAKE)
        self.assertIn("Maximum fidelity", MAKE)
        self.assertIn("Keep fidelity, faster", MAKE)
        self.assertIn("Low-poly attempt", MAKE)
        # Medium is octree 256 then reduce_mesh.py, which only caps: "up to", never a promise of ~80k.
        self.assertIn("up to 80k faces", MAKE)
        self.assertNotIn("~80k faces", MAKE)
        self.assertIn("~2k faces", MAKE)
        self.assertNotIn("Mesh density", MAKE)
        gen = (ROOT / "scripts" / "generate.sh").read_text(encoding="utf-8")
        self.assertIn("high|medium|low", gen)
        self.assertIn('high) POLY="hero"', gen)
        self.assertIn('medium) POLY="lookdev"', gen)
        self.assertIn('low) POLY="proxy"', gen)
        self.assertRegex(gen, r'lookdev\).*FACES="\$\{FACES:-80000\}"')

    def test_viewer_cubbies(self) -> None:
        viewer = viewer_page()
        self.assertIn("function hideAllCubbies", viewer)
        self.assertIn("cubby-rail-left", viewer)
        self.assertIn('data-cubby="stack-look"', viewer)

    def test_jobs_show_stages_cancel_and_reattach(self) -> None:
        # Progress comes from the factory's stage list, not from guessing at log text.
        self.assertNotIn("function inferProgress", JOBS)
        self.assertIn("progress.stages", JOBS)
        self.assertIn("`/cancel/${enc(jobId)}`", API)
        card = src("screens/home/JobCard.tsx")
        self.assertIn("Run in background", card)
        self.assertIn("Try again", card)
        self.assertIn("reattach", JOBS)
        # The native StartGenerate drops PS1 settings, so the app never calls it.
        for rel in ("lib/api.ts", "lib/jobs.ts", "screens/home/MakePanel.tsx", "screens/home/chain.ts"):
            self.assertNotIn("StartGenerate", src(rel))
        app_html = (FRONTEND / "app" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<script type="module" src="/wails/runtime.js" vite-ignore></script>', app_html)

    def test_delete_goes_to_the_factory(self) -> None:
        self.assertIn('factoryJson<DeleteResponse>(`/assets/${enc(stem)}`, { method: "DELETE" })', API)
        self.assertIn("await deleteAsset(a.id)", src("screens/home/Library.tsx"))

    def test_updater_sends_user_agent(self) -> None:
        upd = (ROOT / "desktop" / "internal" / "update" / "service.go").read_text(encoding="utf-8")
        self.assertIn("User-Agent", upd)
        self.assertIn("Pipeline-Host", upd)
        self.assertNotIn("Timeout: 30 * time.Second", upd)
        self.assertIn("startSilentInstaller", upd)
        self.assertIn("selectUpdateAsset", upd)
        self.assertIn("rel.Draft || rel.Prerelease", upd)
        self.assertIn("private repo", upd)
        self.assertIn("GITHUB_TOKEN", upd)
        self.assertIn("ApplicationStarted", MAIN)
        self.assertIn("go upd.AutoUpdate()", MAIN)
        update = src("screens/settings/update.ts")
        self.assertIn('CHECK_LABEL = "Check for Updates"', update)
        self.assertIn("update:installing", update)
        self.assertIn("OpenDataDir", (ROOT / "desktop" / "internal" / "factory" / "factory.go").read_text(encoding="utf-8"))

    def test_sidecar_and_plist(self) -> None:
        self.assertIn("configureFactoryCmd", SIDECAR)
        plist = (ROOT / "desktop" / "build" / "darwin" / "Info.plist").read_text(encoding="utf-8")
        dev = (ROOT / "desktop" / "build" / "darwin" / "Info.dev.plist").read_text(encoding="utf-8")
        self.assertIn("<string>pipeline</string>", plist)
        self.assertIn("<string>14.0</string>", plist)
        self.assertIn("<string>pipeline</string>", dev)
        self.assertIn("export function healthLabel", JOBS)
        self.assertIn("Apple Silicon / MLX", JOBS)
        self.assertIn("function healthLabel", viewer_page())
        self.assertIn("def factory_features", factory_http())

    def test_generate_quality_follows_output_and_renames_in_inspector(self) -> None:
        # The output decides the factory quality; there is no quality picker.
        self.assertIn('quality: "psx"', MAKE)
        self.assertIn('intent.output === "model" ? "textured" : "stack"', MAKE)
        self.assertIn("intent: intentBody(", MAKE)
        self.assertIn('postJson<RenameResponse>(`/assets/${enc(stem)}`, { name }, "PATCH")', API)
        self.assertIn("renameAsset(stem, value)", src("screens/viewer/Hud.tsx"))
        viewer = viewer_page()
        self.assertNotIn('id="gen-quality"', viewer)
        self.assertNotIn('id="gen-stem"', viewer)
        self.assertIn('quality: "stack"', viewer)
        self.assertIn("function commitRename", viewer)
        http = factory_http()
        self.assertIn("def rename_asset", http)
        self.assertIn("auto_stem", http)
        env = (ROOT / "scripts" / "factory_env.py").read_text(encoding="utf-8")
        self.assertIn("def auto_stem", env)
        self.assertIn("def caption_from_image", env)
        self.assertIn("asset-", env)

    def test_facing_is_asked_not_guessed(self) -> None:
        # The picture's shape never decides the facing: it is detected or picked, and a drawn sheet's 3D build
        # reads the drawing's own facing (flat/key.json), not the old camera default.
        self.assertNotIn("naturalWidth / Math.max", MAKE)
        self.assertIn('postJson<FacingDetectResponse>("/facing/detect", body)', API)
        self.assertIn('(stem, "flat/key.json")', src("screens/home/chain.ts"))
        build = MAKE[MAKE.index("The drawn sheet → 3D"):]
        build = build[:build.index("\n}\n")]
        self.assertIn("key.facing", build)
        self.assertNotIn('facing: "camera"', build)

    def test_rig_stage_surfaces(self) -> None:
        # The rig check's own patterns and copy are pinned in src/screens/asset/rig/inspect.test.ts.
        self.assertIn('factoryJson<{ ok: boolean; plans: RigPlan[] }>("/rig/plans")', API)
        rig = src("screens/asset/rig/rig.ts")
        self.assertIn("made for this model", rig)
        self.assertIn("clips_report.json", rig)
        self.assertIn("FBX checked:", src("screens/asset/export/export.ts"))

    def test_shape_mv_tooltip_reads_the_probed_features(self) -> None:
        # degradedLine and shapeMvNote are pinned case by case in src/lib/jobs.test.ts.
        self.assertIn("export function degradedLine", JOBS)
        self.assertIn("export function shapeMvNote", JOBS)
        http = factory_http()
        for key in ('"shape_mv":', '"shape_mv_venv":', '"shape_mv_weights":'):
            self.assertIn(key, http)

    def test_no_picture_words_match_the_factory(self) -> None:
        env = (ROOT / "scripts" / "factory_env.py").read_text(encoding="utf-8")
        self.assertIn('NO_PICTURE = "3D needs a picture. Draw or add one first."', env)
        self.assertIn('export const NO_PICTURE = "3D needs a picture. Draw or add one first.";', MAKE)

    def test_viewer_fog_disabled(self) -> None:
        viewer = viewer_page()
        self.assertIn("scene.fog = null", viewer)
        self.assertNotIn("THREE.Fog", viewer)

    def test_classic_page_is_gone(self) -> None:
        # Stephen retired the non-Mokei page: no file, no build step, no link back to it.
        self.assertFalse((FRONTEND / "index.html").exists())
        for name in ("rig-inspect.js", "stack-lab.js", "pixel-look.js", "stack"):
            self.assertFalse((FRONTEND / "public" / name).exists(), name)
        self.assertNotIn("legacy", (FRONTEND / "vite.config.ts").read_text(encoding="utf-8"))
        for path in SRC.rglob("*.ts*"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("legacy.html", text, str(path))
            self.assertNotIn("classic view", text.lower(), str(path))


if __name__ == "__main__":
    unittest.main()
