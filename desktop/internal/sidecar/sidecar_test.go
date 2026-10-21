package sidecar

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandlerReportsStartErrorAsJSON(t *testing.T) {
	s := &Sidecar{
		lastErr: fmt.Errorf("factory python missing: /missing/.venv/bin/python"),
		python:  "/missing/.venv/bin/python",
		script:  "/missing/scripts/factory_http.py",
		root:    "/missing",
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "json") {
		t.Fatalf("content-type %s", ct)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["ok"] != false {
		t.Fatalf("ok=%v", payload["ok"])
	}
	errStr, _ := payload["error"].(string)
	if !strings.Contains(errStr, "factory python missing") {
		t.Fatalf("error=%q", errStr)
	}
	if strings.Contains(rec.Body.String(), "factory starting") {
		t.Fatal("must not use the old blank 503 text")
	}
}

func TestIsFactoryAPI(t *testing.T) {
	for _, p := range []string{"/health", "/assets", "/generate", "/upload", "/clean", "/inspect/walker", "/paint/stack", "/2d/status", "/2d/unit/voices/gun.wav", "/flow/radio", "/flow/radio/pick", "/export/radio", "/cancel/ab12cd34", "/intent/radio", "/rig/radio", "/rig/plans", "/facing/detect"} {
		if !IsFactoryAPI(p) {
			t.Fatalf("%s should be factory API", p)
		}
	}
	if IsFactoryAPI("/stack-lab.js") {
		t.Fatal("static file is not factory API")
	}
}
