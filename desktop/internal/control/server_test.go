package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/StephenSHorton/pipeline/desktop/internal/sidecar"
)

func TestDiagnosticsAndForward(t *testing.T) {
	sc := &sidecar.Sidecar{}
	var saw string
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		saw = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	})
	h := Handler(next, sc, "dev")

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, localReq(http.MethodGet, "/diagnostics"))
	if rec.Code != http.StatusOK {
		t.Fatalf("diagnostics %d", rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["version"] != "dev" || payload["sidecar"] != false || payload["ok"] != false {
		t.Fatalf("payload=%v", payload)
	}
	if _, ok := payload["flat2d"].(map[string]any); !ok {
		t.Fatalf("flat2d=%T", payload["flat2d"])
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, localReq(http.MethodGet, "/2d/status"))
	if rec.Code != http.StatusNoContent || saw != "/2d/status" {
		t.Fatalf("forward code=%d path=%s", rec.Code, saw)
	}
	if strings.Contains(rec.Body.String(), "factory is not running") {
		t.Fatal("forwarded request was swallowed")
	}
}

func TestHandlerRefusesOtherWebPages(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	h := Handler(next, nil, "test")
	cases := []struct {
		origin, host string
		want         int
	}{
		{"https://evil.example", "127.0.0.1:8777", http.StatusForbidden},
		{"", "evil.example:8777", http.StatusForbidden},
		{"", "127.0.0.1:8777", http.StatusOK},
		{"wails://localhost", "localhost", http.StatusOK},
		{"http://wails.localhost", "wails.localhost", http.StatusOK},
		{"http://localhost:5173", "127.0.0.1:8777", http.StatusOK},
	}
	for _, c := range cases {
		r := httptest.NewRequest(http.MethodDelete, "/assets/radio", nil)
		r.Host = c.host
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("origin %q host %q: got %d, want %d", c.origin, c.host, w.Code, c.want)
		}
	}
}

// localReq is a request as the loopback port sees it (httptest defaults Host to example.com).
func localReq(method, target string) *http.Request {
	r := httptest.NewRequest(method, target, nil)
	r.Host = "127.0.0.1:8777"
	return r
}
