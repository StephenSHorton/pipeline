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
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/diagnostics", nil))
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
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/2d/status", nil))
	if rec.Code != http.StatusNoContent || saw != "/2d/status" {
		t.Fatalf("forward code=%d path=%s", rec.Code, saw)
	}
	if strings.Contains(rec.Body.String(), "factory is not running") {
		t.Fatal("forwarded request was swallowed")
	}
}
