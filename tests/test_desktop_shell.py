#!/usr/bin/env python3
"""Installed host must own factory_http and surface real start errors."""
from __future__ import annotations

import json
import shutil
import subprocess
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
        # Home asks what you're making first; each output shows its own options and a route line.
        self.assertIn("What are you making?", DESKTOP_HTML)
        for output in ("ps1", "model", "stack", "sprite"):
            self.assertIn(f'name="gen-output" value="{output}"', DESKTOP_HTML)
        self.assertIn('id="gen-route"', DESKTOP_HTML)
        self.assertIn("function updateRoute", DESKTOP_HTML)
        self.assertIn('id="lib-search"', DESKTOP_HTML)
        self.assertIn("home-life", DESKTOP_HTML)
        self.assertIn("life-dots", DESKTOP_HTML)
        self.assertIn("home-logo", DESKTOP_HTML)
        self.assertIn("home-float", DESKTOP_HTML)
        self.assertIn("home-suggest", DESKTOP_HTML)
        self.assertIn('id="gen-prompt"', DESKTOP_HTML)
        self.assertIn("multiple", DESKTOP_HTML)
        self.assertIn("Add a photo", DESKTOP_HTML)
        self.assertIn('id="home-assets"', DESKTOP_HTML)
        self.assertIn("body.is-home", DESKTOP_HTML)
        self.assertIn("function showHome", DESKTOP_HTML)
        self.assertIn("pickAsset()", DESKTOP_HTML)
        self.assertNotIn('params.get("asset") || "palm-isle"', DESKTOP_HTML)
        self.assertNotIn("home-wrap", DESKTOP_HTML)
        self.assertIn("home-cluster", DESKTOP_HTML)
        self.assertIn("min(1280px, 74vw)", DESKTOP_HTML)
        self.assertIn("aspect-ratio: 4 / 3", DESKTOP_HTML)
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
        # Medium is octree 256 then reduce_mesh.py, which only caps: "up to", never a promise of ~80k.
        self.assertIn("up to 80k faces", DESKTOP_HTML)
        self.assertNotIn("~80k faces", DESKTOP_HTML)
        self.assertIn("Shape at octree 256, then reduced to at most 80,000 faces.", DESKTOP_HTML)
        self.assertIn("~2k faces", DESKTOP_HTML)
        self.assertNotIn("Mesh density", DESKTOP_HTML)
        self.assertNotIn("Proxy — ~2k faces", DESKTOP_HTML)
        gen = (ROOT / "scripts" / "generate.sh").read_text(encoding="utf-8")
        self.assertIn("high|medium|low", gen)
        self.assertIn('high) POLY="hero"', gen)
        self.assertIn('medium) POLY="lookdev"', gen)
        self.assertIn('low) POLY="proxy"', gen)
        self.assertRegex(gen, r'lookdev\).*FACES="\$\{FACES:-80000\}"')

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
        self.assertIn("make-options", DESKTOP_HTML)
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

    def test_jobs_show_stages_cancel_and_reattach(self) -> None:
        # Progress comes from the factory's stage list, not from guessing at log text.
        self.assertNotIn("function inferProgress", DESKTOP_HTML)
        self.assertIn('id="job-stages"', DESKTOP_HTML)
        self.assertIn("progress.stages", DESKTOP_HTML)
        self.assertIn("`/cancel/${job.id}`", DESKTOP_HTML)
        self.assertIn('label: "Run in background"', DESKTOP_HTML)
        self.assertIn('label: "Try again"', DESKTOP_HTML)
        self.assertIn("async function reattachJob", DESKTOP_HTML)
        self.assertIn('id="job-chip"', DESKTOP_HTML)
        # The native StartGenerate drops PS1 settings, so the page never calls it.
        self.assertNotIn("WAILS_GENERATE", DESKTOP_HTML)
        self.assertIn('<script type="module" src="/wails/runtime.js"></script>', DESKTOP_HTML)

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

    def test_generate_quality_follows_output_and_renames_in_inspector(self) -> None:
        self.assertNotIn('id="gen-quality"', DESKTOP_HTML)
        self.assertNotIn('id="gen-stem"', DESKTOP_HTML)
        self.assertNotIn('<option value="draft">', DESKTOP_HTML)
        self.assertNotIn('<option value="textured">', DESKTOP_HTML)
        self.assertNotIn('<option value="pbr">', DESKTOP_HTML)
        # The output decides the factory quality; there is no quality picker.
        self.assertIn('quality: "psx"', DESKTOP_HTML)
        self.assertIn('want.output === "model" ? "textured" : "stack"', DESKTOP_HTML)
        self.assertIn("intent: intentBody(intent)", DESKTOP_HTML)
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

    def test_facing_is_asked_not_guessed(self) -> None:
        # The picture's shape never decides the facing; the head's side of the picture is asked (section 8).
        self.assertNotIn("naturalWidth / Math.max", DESKTOP_HTML)
        self.assertNotIn("it's probably side-on", DESKTOP_HTML)
        for label in ("Which way does the head point in the picture?", "Looks at the camera",
                      "Head points to the left of the picture", "Head points to the right of the picture",
                      "Back to the camera", "As you look at the picture, not the animal's own left or right.",
                      "side-on, head to the picture's left", "side-on, head to the picture's right",
                      "Drawn in profile, head right"):
            self.assertIn(label, DESKTOP_HTML)
        self.assertNotIn("Side, facing left", DESKTOP_HTML)
        self.assertIn('"/facing/detect"', DESKTOP_HTML)
        self.assertIn("facingTouched = false", DESKTOP_HTML)
        self.assertIn('setFacing(next, "user")', DESKTOP_HTML)
        # buildFromConcept reads the concept's own facing, not the old camera default.
        build = DESKTOP_HTML[DESKTOP_HTML.index("async function buildFromConcept"):]
        build = build[:build.index("\n    }\n")]
        self.assertIn("flat/key.json", build)
        self.assertNotIn('facing: "camera"', build)

    def test_rig_stage_surfaces(self) -> None:
        self.assertIn('factoryJson("/rig/plans")', DESKTOP_HTML)
        self.assertIn("made for this model", DESKTOP_HTML)
        self.assertIn("rig/template/clips_report.json", DESKTOP_HTML)
        self.assertIn("FBX checked:", DESKTOP_HTML)
        self.assertIn('fold.className = "rig-advanced"', DESKTOP_HTML)
        for flag in ('"template"', '"binder"', '"pred"'):
            self.assertIn(flag, DESKTOP_HTML)
        self.assertIn("/render`", DESKTOP_HTML)
        inspect = (ROOT / "desktop" / "frontend" / "public" / "rig-inspect.js").read_text(encoding="utf-8")
        self.assertIn("const LEG = /leg|foot|toe|thigh|shin|calf|knee|hoof|ball|humerus|femur|tibia/i;", inspect)
        self.assertIn("const ARM = /arm|hand|shoulder|clavicle|elbow|wrist|finger/i;", inspect)
        self.assertIn("const NO_FIT = /leaf|tip|_end$|^ear/i;", inspect)
        self.assertIn("Skeleton built for this model", inspect)
        self.assertIn("contact_sheet.png", inspect)
        self.assertIn('data-tab="sheet"', inspect)
        self.assertIn("ROLE_RGB", inspect)
        for key in ("template", "built", "a2", "foot", "parts", "cshape", "slide", "loop_seam", "fbx"):
            self.assertRegex(inspect, rf"[{{ ,]{key}: \"(skeleton|skin|motion)\"")
        self.assertIn('seam: "skin"', inspect)  # seam stays the corner tears

    def _fn(self, name: str) -> str:
        """One top-level-in-the-module function's source, from 'function name(' to its closing '    }'."""
        start = DESKTOP_HTML.index(f"    function {name}(")
        return DESKTOP_HTML[start:DESKTOP_HTML.index("\n    }\n", start) + 6]

    def _node(self, src: str) -> object:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        out = subprocess.run([node, "-e", src], capture_output=True, text=True, timeout=30, check=True)
        return json.loads(out.stdout)

    def test_degraded_is_one_plain_line_on_job_and_asset_cards(self) -> None:
        # The job card: /status/<id> "degraded" is copied onto the job and shown as one line, in every state.
        self.assertIn('<div class="job-degraded" id="job-degraded" role="status" hidden></div>', DESKTOP_HTML)
        follow = DESKTOP_HTML[DESKTOP_HTML.index("async function followJob"):]
        follow = follow[:follow.index("\n    }\n")]
        self.assertIn('"source", "degraded"]', follow)
        self.assertIn('setDegradedLine(document.getElementById("job-degraded"), degraded);', self._fn("showOverlay"))
        draw = self._fn("drawJob")
        self.assertEqual(draw.count("degraded: job.degraded"), 2)  # running, and error/cancelled
        finish = DESKTOP_HTML[DESKTOP_HTML.index("async function finishJob"):]
        finish = finish[:finish.index("\n    }\n")]
        self.assertIn('state: "ok", degraded: job.degraded', finish)
        self.assertIn("degradedLine(job.degraded)", finish)  # the ready toast says it too
        flat = DESKTOP_HTML[DESKTOP_HTML.index("async function finishFlatJob"):]
        flat = flat[:flat.index("\n    }\n")]
        self.assertIn("degraded: job.degraded", flat)
        # The asset card: inspect.json "degraded" for a model, fetched without holding up the list.
        cards = DESKTOP_HTML[DESKTOP_HTML.index("async function refreshAssets"):]
        cards = cards[:cards.index("\n    }\n")]
        self.assertIn('warn.className = "card-degraded"', cards)
        self.assertIn("/inspect.json?t=", cards)
        self.assertIn("setDegradedLine(warn, doc.degraded)", cards)
        self.assertIn("if (a.has_mesh)", cards)
        # The line itself: the DEGRADED grammar without the prefix, nothing when there is nothing.
        got = self._node(self._fn("degradedLine") + """
            console.log(JSON.stringify([
              degradedLine([{stage: "shape", code: "single_view", message: "one picture"},
                            {stage: "rig", code: "clips_dropped", message: "Flap,Glide"}]),
              degradedLine([]), degradedLine(undefined), degradedLine([{}]),
              degradedLine([{stage: "views", code: "stale", message: ""}]),
            ]));""")
        self.assertEqual(got, [
            "Degraded: shape single_view: one picture · rig clips_dropped: Flap,Glide",
            "", "", "", "Degraded: views stale",
        ])
        # Display only: no views gallery, no views toggle, no redraw or suggestion UI came with it.
        for absent in ("views-gallery", "gen-views", "views: 1", "sheet-gallery", "job-degraded-act", "card-degraded button"):
            self.assertNotIn(absent, DESKTOP_HTML)

    def test_shape_mv_tooltip_reads_the_probed_features(self) -> None:
        self.assertIn("genStatus.title = `Factory: ${healthLabel(h)}${shapeMvNote(h.features)}`;", DESKTOP_HTML)
        fn = self._fn("shapeMvNote")
        got = self._node(fn + """
            console.log(JSON.stringify([
              shapeMvNote(undefined), shapeMvNote({}),
              shapeMvNote({shape_mv: true, shape_mv_venv: true, shape_mv_weights: true}),
              shapeMvNote({shape_mv: false, shape_mv_venv: false, shape_mv_weights: true}),
              shapeMvNote({shape_mv: false, shape_mv_venv: true, shape_mv_weights: false}),
              shapeMvNote({shape_mv: false, shape_mv_venv: false, shape_mv_weights: false}),
            ]));""")
        self.assertEqual(got[0], "")
        self.assertEqual(got[1], "")
        self.assertIn("is installed; it is used when a drawing's four views match it", got[2])
        self.assertIn("its Python env is not installed", got[3])
        self.assertIn("its weights are missing", got[4])
        self.assertIn("its Python env and weights are missing", got[5])
        for off in got[3:]:
            self.assertIn("the shape comes from the first picture; extra views only photo-bake", off)
        http = (ROOT / "scripts" / "factory_http.py").read_text(encoding="utf-8")
        for key in ('"shape_mv":', '"shape_mv_venv":', '"shape_mv_weights":'):
            self.assertIn(key, http)

    def test_prompt_only_still_goes_to_ideas(self) -> None:
        # Routing is unchanged: a prompt with no picture draws concepts on the Ideas grid; 3D always has a picture.
        submit = DESKTOP_HTML[DESKTOP_HTML.index('genForm.addEventListener("submit"'):]
        submit = submit[:submit.index("\n    });\n")]
        ideas = submit.index("if (!hasPicture && typed) {\n        await startIdeas(typed, intent);\n        return;\n      }")
        self.assertLess(submit.index('if (intent.output === "sprite")'), ideas)
        self.assertLess(ideas, submit.index("await runJob({"))
        self.assertIn("image: paths[0],", submit)
        self.assertIn('setGenStatus("3D needs a picture. Draw or add one first.", "error");', submit)
        start = DESKTOP_HTML[DESKTOP_HTML.index("async function startIdeas"):]
        start = start[:start.index("\n    }\n")]
        self.assertIn('{ kind: "ideas", stem, prompt, count: 4', start)
        self.assertIn('{ path: "/2d/draw", kind: "ideas", title: "Drawing ideas" }', start)
        route = self._fn("updateRoute")
        self.assertIn("Draw <b>4 concept pictures</b> from your prompt first.", route)
        self.assertIn('label = "Draw concepts";', route)
        env = (ROOT / "scripts" / "factory_env.py").read_text(encoding="utf-8")
        self.assertIn('NO_PICTURE = "3D needs a picture. Draw or add one first."', env)

    def test_viewer_fog_disabled(self) -> None:
        self.assertIn("scene.fog = null", DESKTOP_HTML)
        self.assertNotIn("THREE.Fog", DESKTOP_HTML)
        viewer = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn("scene.fog = null", viewer)
        self.assertNotIn("THREE.Fog", viewer)


if __name__ == "__main__":
    unittest.main()
