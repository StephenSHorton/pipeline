#!/usr/bin/env python3
"""The agent workbench as an MCP server: Streamable HTTP, JSON responses, on 127.0.0.1:8768/mcp.

    .venv/bin/python scripts/workbench/mcp_server.py [--port 8768]
    claude mcp add --transport http workbench http://127.0.0.1:8768/mcp

Hand-written rather than the MCP SDK: the repo takes no new dependencies, and the part Claude Code needs is small.
Each POST carries one JSON-RPC message (or a batch) and gets one JSON answer: initialize, ping, tools/list,
tools/call. Notifications and responses get 202 with no body. There is no server-to-client stream, so GET (the SSE
channel) is 405, as the spec allows. Tools are in tools.py.

It listens on loopback only and refuses a request whose Origin is not a local page, so a web page in the browser
cannot drive it (DNS rebinding). Model work is not done here: the factory on :8766 runs it, as a client of that.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory_env import repo_root  # noqa: E402
from workbench.factory_client import FactoryClient  # noqa: E402
from workbench.tools import ArgError, Workbench, error_text  # noqa: E402

DEFAULT_PORT = 8768
PATH = "/mcp"
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "workbench", "title": "Asset pipeline workbench", "version": "0.1.0"}
INSTRUCTIONS = (
    "Make, measure and fix 3D assets in the pipeline (out/<stem>/). measure and look before and after every change. "
    "Clay (clay_set, clay_mesh) blocks out shapes a photo can't give depth to; generate with mesh + paint false "
    "turns a clay mesh into an asset without loading a model. draw_views and generate are factory jobs: poll "
    "job_status. Each write is kept as a mesh take; pick_take goes back."
)

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


def _error(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _result(rid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def call_tool(bench: Workbench, params: dict) -> dict:
    """tools/call's result: the tool's answer as JSON text, plus its PNG when it made one. A refusal or failure is
    isError with the reason, so the model can read it and try again."""
    try:
        out = bench.call(params["name"], params.get("arguments"))
    except ArgError as e:
        return {"content": [{"type": "text", "text": f"bad arguments: {e}"}], "isError": True}
    except Exception as e:  # noqa: BLE001 - every tool failure is reported to the caller
        return {"content": [{"type": "text", "text": error_text(e)}], "isError": True}
    image = out.pop("_image", None)
    content = [{"type": "text", "text": json.dumps(out, indent=1)}]
    if image:
        try:
            data = base64.b64encode(Path(image).read_bytes()).decode("ascii")
            content.append({"type": "image", "data": data, "mimeType": "image/png"})
        except OSError as e:
            content[0]["text"] += f"\n(image not attached: {e})"
    return {"content": content, "isError": False}


def handle_rpc(bench: Workbench, msg: Any) -> dict | None:
    """One JSON-RPC message in, its response out (None for a notification or a client's response)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
    method = msg.get("method")
    if "id" not in msg or method is None:
        return None  # a notification (notifications/initialized, cancelled) or a response to us
    rid = msg["id"]
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        return _error(rid, INVALID_PARAMS, "params must be an object")
    if method == "initialize":
        asked = params.get("protocolVersion")
        return _result(rid, {
            "protocolVersion": asked if asked in VERSIONS else VERSIONS[1],
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": bench.listing()})
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str) or name not in bench.tools:
            return _error(rid, INVALID_PARAMS, f"unknown tool {name!r}")
        if not isinstance(params.get("arguments", {}), dict):
            return _error(rid, INVALID_PARAMS, "arguments must be an object")
        return _result(rid, call_tool(bench, params))
    return _error(rid, METHOD_NOT_FOUND, f"method not found: {method}")


def local_origin(origin: str | None) -> bool:
    """No Origin (curl, Claude Code) or a page on this machine."""
    if not origin:
        return True
    return (urlparse(origin).hostname or "") in LOOPBACK


def make_handler(bench: Workbench):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, payload: Any = None) -> None:
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            if payload is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _refuse(self) -> bool:
            if not local_origin(self.headers.get("Origin")):
                self._send(403, _error(None, INVALID_REQUEST, "origin not allowed"))
                return True
            if urlparse(self.path).path.rstrip("/") != PATH:
                self._send(404, _error(None, INVALID_REQUEST, f"not found: {self.path} (the endpoint is {PATH})"))
                return True
            return False

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            if self._refuse():
                return
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send(400, _error(None, PARSE_ERROR, f"invalid JSON: {e}"))
                return
            if isinstance(msg, list):
                out = [r for r in (handle_rpc(bench, m) for m in msg) if r is not None]
                self._send(200, out) if out else self._send(202)
                return
            out = handle_rpc(bench, msg)
            self._send(202) if out is None else self._send(200, out)

        def _not_allowed(self) -> None:
            if self._refuse():
                return
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_DELETE = do_PUT = do_PATCH = _not_allowed  # noqa: N815

    return Handler


def make_server(host: str, port: int, bench: Workbench) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise SystemExit(f"loopback only (got {host})")
    httpd = ThreadingHTTPServer((host, port), make_handler(bench))
    httpd.daemon_threads = True
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("WORKBENCH_MCP_PORT") or DEFAULT_PORT))
    ap.add_argument("--root", default=None)
    ap.add_argument("--factory", default=None, help="the factory's URL (default http://127.0.0.1:8766)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root()
    bench = Workbench(root, FactoryClient(args.factory))
    httpd = make_server(args.host, args.port, bench)
    print(f"workbench MCP listening http://{args.host}:{args.port}{PATH}  ({len(bench.tools)} tools, "
          f"factory {bench.factory.url}, root {root})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
