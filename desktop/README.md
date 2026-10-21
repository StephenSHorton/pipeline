# Pipeline (desktop)

Wails v3 host for the AI asset factory on Apple Silicon (MLX): `.app` against the checkout factory first, then a Developer ID + notarized DMG later.

The host ships the viewer and `factory_http.py` and owns port 8766. It also listens on `127.0.0.1:8777` (`PIPELINE_CONTROL_ADDR` to move it). That port is the same handler as the window. `GET /diagnostics` is the snapshot (version, paths, sidecar, factory health, 2D status, log tails, tool list). `GET /control/workflow?stem=` and `GET /control/sheet?stem=&part=` are what the window uses to choose 3D or 2D and to read the audio subject. `POST /control` runs `sheet`, `split`, `drop`, `sound`, or `speak`. The sidecar uses `generate.sh`, `~/Library/Application Support/Pipeline`, and checkout `bin/hy3d` + `.venv` — not Homebrew.

## Develop

```sh
cd desktop
node frontend/scripts/copy-viewer.mjs
node frontend/scripts/copy-factory.mjs
go build -o bin/pipeline
./bin/pipeline
```

From a factory checkout the host uses `.venv` / `python3`. The installed app extracts scripts into `~/Library/Application Support/Pipeline`. Do not point the installed app at the git checkout.

```sh
wails3 task darwin:package PACKAGE_MANAGER=npm VERSION=0.1.0
```

## Release

Push a `v*.*.*` tag (`git tag v0.0.14 && git push origin v0.0.14`). `.github/workflows/pipeline-release.yml` runs the Go tests on an Apple Silicon runner, then `build/darwin/package.sh <version>`, and publishes `pipeline-<version>-darwin-arm64.dmg` plus `SHA256SUMS` as a GitHub Release. Run the same script locally from `desktop/` to get the DMG in `bin/`. The app is ad-hoc signed (no Developer ID), so macOS asks before the first open.

The in-app updater finds that DMG on the latest release. Installing it from the app is not implemented yet: open the DMG and drag Pipeline into Applications.

## Deps

See `deps.json`. Do not commit `weights/` or `vendor/hunyuan3d-2` binaries.
