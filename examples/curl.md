# Curl the local factory (no MCP)

Server: `http://127.0.0.1:8766` (loopback only).

Start it (leave the window open):

```sh
.venv/bin/python scripts/factory_http.py
```

## Health

```sh
curl -sS http://127.0.0.1:8766/health
```

## Inspect existing `radio`

```sh
curl -sS http://127.0.0.1:8766/inspect/radio
```

## Generate draft from `in/radio-concept.jpg`

Cheap proxy LOD:

```sh
curl -sS -X POST http://127.0.0.1:8766/generate -H "Content-Type: application/json" --data-binary "@examples/generate-radio.json"
```

Poll:

```sh
curl -sS http://127.0.0.1:8766/status
```

## 2D lane

Drop is a base64 PNG in JSON. Split does not need a model. `sfx_mode` is `stub` (stand-in wav), `no-runner` (weights installed, `FLAT2D_SFX_CMD` empty), or `ready`. `voice` is true when `mlx-audio` is installed. `POST /2d/speak` with `{"stem","part":{"id","line","speaker"}}` writes `voices/<id>.wav`.

```sh
curl -sS http://127.0.0.1:8766/2d/status
curl -sS http://127.0.0.1:8766/2d/unit
curl -sS -X POST http://127.0.0.1:8766/2d/split -H "Content-Type: application/json" -d '{"stem":"unit"}'
```

## Fetch the GLB

```sh
curl -sS -o radio.glb http://127.0.0.1:8766/glb/radio
```

Omit `stem` to auto-name from the prompt, a cheap caption of the refs (filename or JPEG/PNG text), or `asset-1`. Desktop generate always sends `--quality stack`. CLI still accepts `draft` / `textured` / `pbr`.

Rename an existing folder without regenerating:

```sh
curl -sS -X PATCH http://127.0.0.1:8766/assets/radio -H "Content-Type: application/json" --data-binary '{"name":"Brass Lantern"}'
```

JSON everywhere except `/glb/<stem>`, which is raw glTF binary (`glTF` magic). One generate at a time; a second POST while busy returns 409.

## Sprite stack

`--quality stack` (or any generate) writes `out/<stem>/stack/`. Inspect includes a `stack` object when present:

```sh
curl -sS http://127.0.0.1:8766/stack/radio
```

Inspector: `http://127.0.0.1:8780/viewer/?asset=radio` (mesh left, stack right).
