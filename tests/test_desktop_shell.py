#!/usr/bin/env python3
"""Installed host must own factory_http and surface real start errors."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP_HTML = (ROOT / "desktop" / "frontend" / "index.html").read_text(encoding="utf-8")
SIDECAR = (ROOT / "desktop" / "internal" / "sidecar" / "sidecar.go").read_text(encoding="utf-8")
MAIN = (ROOT / "desktop" / "main.go").read_text(encoding="utf-8")
RUNTIME = (ROOT / "desktop" / "internal" / "runtime" / "runtime.go").read_text(encoding="utf-8")
EMBED_HTTP = ROOT / "desktop" / "internal" / "runtime" / "fs" / "scripts" / "factory_http.py"


class DesktopShellTests(unittest.TestCase):
    def test_same_origin_factory(self) -> None:
        self.assertIn('const FACTORY = "";', DESKTOP_HTML)
        self.assertNotIn("Start scripts/factory_http", DESKTOP_HTML)

    def test_ui_shows_json_error_not_status_text(self) -> None:
        self.assertIn("JSON.parse(text)", DESKTOP_HTML)
        self.assertIn("data.error", DESKTOP_HTML)
        self.assertIn("Factory failed to start", DESKTOP_HTML)

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
        shipped = EMBED_HTTP.read_text(encoding="utf-8")
        src = (ROOT / "scripts" / "factory_http.py").read_text(encoding="utf-8")
        self.assertEqual(shipped, src)

    def test_no_pipeline_root_install_instruction(self) -> None:
        self.assertNotIn("set PIPELINE_ROOT", MAIN)
        self.assertNotIn("set PIPELINE_ROOT", SIDECAR)

    def test_ui_shows_build_version(self) -> None:
        self.assertIn('id="app-version"', DESKTOP_HTML)
        self.assertIn("applyVersion", DESKTOP_HTML)
        self.assertIn('fetch("/version")', DESKTOP_HTML)
        self.assertIn("windowTitle(version)", MAIN)
        self.assertIn('r.URL.Path == "/version"', MAIN)
        self.assertNotIn('Title:            "Pipeline"', MAIN)

    def test_mesh_loads_via_glb_proxy(self) -> None:
        self.assertIn("function meshUrls", DESKTOP_HTML)
        self.assertIn("/glb/", DESKTOP_HTML)
        self.assertIn("preferLod", DESKTOP_HTML)
        self.assertIn("Loading mesh", DESKTOP_HTML)
        self.assertIn("openAsset(stem", DESKTOP_HTML)
        self.assertIn("factory.ServeOut", MAIN)
        self.assertIn("loader.parse", DESKTOP_HTML)

    def test_generate_does_not_post_file_body(self) -> None:
        self.assertNotIn("body: file", DESKTOP_HTML)
        self.assertIn("uploadImage", DESKTOP_HTML)
        self.assertIn("arrayBufferToBase64", DESKTOP_HTML)
        self.assertIn("application/json", DESKTOP_HTML)
        self.assertIn("SaveUpload", DESKTOP_HTML)
        self.assertIn("HandleUploadHTTP", MAIN)

    def test_home_is_prompt_page(self) -> None:
        self.assertIn("Describe the thing…", DESKTOP_HTML)
        self.assertIn("home-life", DESKTOP_HTML)
        self.assertIn("life-dots", DESKTOP_HTML)
        self.assertIn("home-logo", DESKTOP_HTML)
        self.assertIn("home-float", DESKTOP_HTML)
        self.assertIn("home-suggest", DESKTOP_HTML)
        self.assertIn('id="gen-prompt"', DESKTOP_HTML)
        self.assertIn("multiple", DESKTOP_HTML)
        self.assertIn("Attach reference images", DESKTOP_HTML)
        self.assertIn('id="home-assets"', DESKTOP_HTML)
        self.assertIn("body.is-home", DESKTOP_HTML)
        self.assertIn("function showHome", DESKTOP_HTML)
        self.assertIn("pickAsset()", DESKTOP_HTML)
        self.assertNotIn('params.get("asset") || "palm-isle"', DESKTOP_HTML)
        self.assertNotIn("home-wrap", DESKTOP_HTML)
        self.assertIn("home-cluster", DESKTOP_HTML)
        self.assertIn("min(1280px, 74vw)", DESKTOP_HTML)
        self.assertIn("clamp(120px, 16vh, 180px)", DESKTOP_HTML)
        self.assertIn("translateX(58vw)", DESKTOP_HTML)
        self.assertIn("background: #0e0f0c", DESKTOP_HTML)
        self.assertIn("life-line l4", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn("home-cluster", viewer)
        self.assertIn("min(1280px, 74vw)", viewer)

    def test_poly_high_medium_low(self) -> None:
        self.assertIn('value="high"', DESKTOP_HTML)
        self.assertIn('value="medium"', DESKTOP_HTML)
        self.assertIn('value="low"', DESKTOP_HTML)
        self.assertIn("Maximum fidelity", DESKTOP_HTML)
        self.assertIn("Keep fidelity, faster", DESKTOP_HTML)
        self.assertIn("Low-poly attempt", DESKTOP_HTML)
        self.assertIn("~80k faces", DESKTOP_HTML)
        self.assertIn("~2k faces", DESKTOP_HTML)
        self.assertNotIn("Mesh density", DESKTOP_HTML)
        self.assertNotIn("Proxy — ~2k faces", DESKTOP_HTML)
        gen = (ROOT / "scripts" / "generate.sh").read_text(encoding="utf-8")
        self.assertIn("high|medium|low", gen)
        self.assertIn('high) POLY="hero"', gen)
        self.assertIn('medium) POLY="lookdev"', gen)
        self.assertIn('low) POLY="proxy"', gen)

    def test_gen_cubbies_and_required_levers(self) -> None:
        self.assertIn("cubby-toggle", DESKTOP_HTML)
        self.assertIn("cubby-hide", DESKTOP_HTML)
        self.assertIn("cubby-rail-left", DESKTOP_HTML)
        self.assertIn("cubby-rail-right", DESKTOP_HTML)
        self.assertIn("width: 348px", DESKTOP_HTML)
        self.assertIn('data-cubby="generate"', DESKTOP_HTML)
        self.assertIn('data-cubby="mesh"', DESKTOP_HTML)
        self.assertIn('data-cubby="stack-look"', DESKTOP_HTML)
        self.assertIn('data-cubby="stack-sliders"', DESKTOP_HTML)
        self.assertIn('data-cubby="stack-palette"', DESKTOP_HTML)
        self.assertIn("levers-required", DESKTOP_HTML)
        self.assertIn("is-required", DESKTOP_HTML)
        self.assertIn("function hideAllCubbies", DESKTOP_HTML)
        self.assertIn("hideAllCubbies()", DESKTOP_HTML)
        self.assertIn("function addRefFiles", DESKTOP_HTML)
        self.assertIn("function abandonGenerateWatch", DESKTOP_HTML)
        self.assertIn("genWatching = true", DESKTOP_HTML)
        self.assertNotIn("float-tl", DESKTOP_HTML)
        self.assertNotIn("float-br", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn("function hideAllCubbies", viewer)
        self.assertIn("cubby-rail-left", viewer)
        self.assertIn('data-cubby="stack-look"', viewer)

    def test_delete_clears_list_and_view(self) -> None:
        self.assertIn("function deleteAsset", DESKTOP_HTML)
        self.assertIn("removeAssetNodes", DESKTOP_HTML)
        self.assertIn("clearView", DESKTOP_HTML)
        self.assertIn("data-asset-id", DESKTOP_HTML)

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
        self.assertIn("Check for Updates", DESKTOP_HTML)
        self.assertIn('id="settings"', DESKTOP_HTML)
        self.assertIn("home-settings-btn", DESKTOP_HTML)
        self.assertIn("update:installing", DESKTOP_HTML)
        self.assertIn("function showSettings", DESKTOP_HTML)
        self.assertIn("OpenDataDir", (ROOT / "desktop" / "internal" / "factory" / "factory.go").read_text(encoding="utf-8"))

    def test_sidecar_and_plist(self) -> None:
        self.assertIn("configureFactoryCmd", SIDECAR)
        plist = (ROOT / "desktop" / "build" / "darwin" / "Info.plist").read_text(encoding="utf-8")
        dev = (ROOT / "desktop" / "build" / "darwin" / "Info.dev.plist").read_text(encoding="utf-8")
        self.assertIn("<string>pipeline</string>", plist)
        self.assertIn("<string>14.0</string>", plist)
        self.assertIn("<string>pipeline</string>", dev)
        self.assertIn("function healthLabel", DESKTOP_HTML)
        self.assertIn("Apple Silicon / MLX", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn("function healthLabel", viewer)
        http = (ROOT / "scripts" / "factory_http.py").read_text(encoding="utf-8")
        self.assertIn("def factory_features", http)

    def test_generate_is_stack_only_and_renames_in_inspector(self) -> None:
        self.assertNotIn('id="gen-quality"', DESKTOP_HTML)
        self.assertNotIn('id="gen-stem"', DESKTOP_HTML)
        self.assertNotIn('<option value="draft">', DESKTOP_HTML)
        self.assertNotIn('<option value="textured">', DESKTOP_HTML)
        self.assertNotIn('<option value="pbr">', DESKTOP_HTML)
        self.assertIn('quality: "stack"', DESKTOP_HTML)
        self.assertIn("function commitRename", DESKTOP_HTML)
        self.assertIn("function setTitleState", DESKTOP_HTML)
        self.assertIn('method: "PATCH"', DESKTOP_HTML)
        self.assertIn('id="title"', DESKTOP_HTML)
        self.assertNotIn("<h1 id=\"title\">", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('id="gen-quality"', viewer)
        self.assertNotIn('id="gen-stem"', viewer)
        self.assertIn('quality: "stack"', viewer)
        self.assertIn("function commitRename", viewer)
        http = (ROOT / "scripts" / "factory_http.py").read_text(encoding="utf-8")
        self.assertIn("def rename_asset", http)
        self.assertIn("auto_stem", http)
        env = (ROOT / "scripts" / "factory_env.py").read_text(encoding="utf-8")
        self.assertIn("def auto_stem", env)
        self.assertIn("def caption_from_image", env)
        self.assertIn("asset-", env)

    def test_viewer_fog_disabled(self) -> None:
        self.assertIn("scene.fog = null", DESKTOP_HTML)
        self.assertNotIn("THREE.Fog", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn("scene.fog = null", viewer)
        self.assertNotIn("THREE.Fog", viewer)


if __name__ == "__main__":
    unittest.main()
