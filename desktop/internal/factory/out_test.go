package factory

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestServeOutGLB(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "out", "pipeline-mech")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := []byte("glTF\x00fake")
	if err := os.WriteFile(filepath.Join(dir, "mesh_lookdev.glb"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	h := ServeOut(root)
	req := httptest.NewRequest(http.MethodGet, "/out/pipeline-mech/mesh_lookdev.glb", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("code=%d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "model/gltf-binary" {
		t.Fatalf("content-type %s", ct)
	}
	got, _ := io.ReadAll(rec.Body)
	if string(got) != string(body) {
		t.Fatalf("body mismatch")
	}
}

func TestServeOutRejectsDotDot(t *testing.T) {
	root := t.TempDir()
	h := ServeOut(root)
	req := httptest.NewRequest(http.MethodGet, "/out/../secrets.txt", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 404 {
		t.Fatalf("code=%d", rec.Code)
	}
}
