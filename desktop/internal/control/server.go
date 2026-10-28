// Package control is the loopback HTTP server for the desktop host.
// One port serves the window's routes, the factory proxy (including /2d),
// the checkout viewer, and GET /diagnostics.
package control

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
	"github.com/StephenSHorton/pipeline/desktop/internal/sidecar"
)

const DefaultAddr = "127.0.0.1:8777"

// Handler answers /diagnostics and /control, then forwards every other request.
// Mount this on the window and on the loopback port so both see the same tools.
func Handler(next http.Handler, sc *sidecar.Sidecar, version string) http.Handler {
	probeClient := &http.Client{Timeout: 2 * time.Second}
	opClient := &http.Client{Timeout: 3 * time.Minute}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !LocalRequest(r) {
			writeJSONCode(w, http.StatusForbidden, map[string]any{"ok": false, "error": "origin not allowed"})
			return
		}
		if r.URL.Path == "/diagnostics" {
			writeJSON(w, Diagnostics(sc, version, probeClient))
			return
		}
		if r.URL.Path == "/control" || strings.HasPrefix(r.URL.Path, "/control/") {
			root := ""
			if sc != nil {
				root = sc.Root()
			}
			serveControl(w, r, root, opClient)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// LocalRequest is false for a web page that isn't the app or this machine, or for a Host that isn't loopback
// (DNS rebinding). No Origin means curl, scripts or agents. The webview sends wails:// or wails.localhost.
func LocalRequest(r *http.Request) bool {
	if o := r.Header.Get("Origin"); o != "" && !localOrigin(o) {
		return false
	}
	if h := r.Host; h != "" {
		host, _, err := net.SplitHostPort(h)
		if err != nil {
			host = strings.Trim(h, "[]")
		}
		if !loopbackHost(host) && host != "wails.localhost" && host != "wails" {
			return false
		}
	}
	return true
}

func localOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if u.Scheme == "wails" {
		return true
	}
	h := u.Hostname()
	return loopbackHost(h) || h == "wails.localhost"
}

func loopbackHost(h string) bool {
	return h == "127.0.0.1" || h == "localhost" || h == "::1"
}

// Start listens on addr (PIPELINE_CONTROL_ADDR, else 127.0.0.1:8777).
// Pass the same handler the window uses. A bare port in the env var is bound on loopback.
func Start(next http.Handler) (string, error) {
	addr := strings.TrimSpace(os.Getenv("PIPELINE_CONTROL_ADDR"))
	if addr == "" {
		addr = DefaultAddr
	} else if !strings.Contains(addr, ":") {
		addr = "127.0.0.1:" + addr
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", err
	}
	srv := &http.Server{Handler: next}
	go func() { _ = srv.Serve(ln) }()
	return ln.Addr().String(), nil
}

// Diagnostics is the agent snapshot: host paths, sidecar, factory health, 2D status, log tails.
func Diagnostics(sc *sidecar.Sidecar, version string, client *http.Client) map[string]any {
	root := ""
	python := ""
	script := ""
	running := false
	var last error
	if sc != nil {
		root = sc.Root()
		python = sc.Python()
		script = sc.Script()
		running = sc.Running()
		last = sc.LastError()
	}
	out := map[string]any{
		"ok":      true,
		"service": "pipeline",
		"version": version,
		"dataDir": paths.DataDir(),
		"root":    root,
		"python":  python,
		"script":  script,
		"sidecar": running,
		"control": DefaultAddr,
		"factory": probe(client, "http://127.0.0.1:8766/health"),
		"flat2d":  probe(client, "http://127.0.0.1:8766/2d/status"),
		"tools":   Tools(),
		"logs": map[string]string{
			"pipeline": tail(paths.DataDir() + "/out/pipeline.log"),
			"sidecar":  tail(root + "/out/factory_sidecar.log"),
		},
	}
	if last != nil {
		out["sidecarError"] = last.Error()
	}
	if !running {
		out["ok"] = false
	}
	return out
}

func probe(client *http.Client, url string) any {
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Second}
	}
	resp, err := client.Get(url)
	if err != nil {
		return map[string]any{"ok": false, "error": err.Error()}
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var payload any
	if json.Unmarshal(body, &payload) == nil {
		return payload
	}
	return map[string]any{"ok": false, "status": resp.StatusCode, "body": string(body)}
}

func tail(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	text := strings.TrimSpace(string(b))
	lines := strings.Split(text, "\n")
	if len(lines) > 40 {
		lines = lines[len(lines)-40:]
	}
	text = strings.TrimSpace(strings.Join(lines, "\n"))
	if len(text) > 4000 {
		text = text[len(text)-4000:]
	}
	return text
}

func writeJSON(w http.ResponseWriter, payload map[string]any) {
	writeJSONCode(w, http.StatusOK, payload)
}

func writeJSONCode(w http.ResponseWriter, code int, payload map[string]any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}
