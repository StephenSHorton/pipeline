package runtime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

func TestExtractToWritesFactoryHTTP(t *testing.T) {
	dest := t.TempDir()
	if err := ExtractTo(dest, "test"); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dest, "scripts", "factory_http.py")
	b, err := os.ReadFile(script)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "ai-asset-pipeline") {
		t.Fatalf("extracted factory_http.py missing service name")
	}
	if _, err := os.Stat(filepath.Join(dest, "scripts", "factory_env.py")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dest, "scripts", "inspect_glb.py")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dest, "deps.json")); err != nil {
		t.Fatal(err)
	}
	stamp, err := os.ReadFile(filepath.Join(dest, "scripts", ".embed-version"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(stamp)) != "test" {
		t.Fatalf("stamp=%q", stamp)
	}
	hash, err := os.ReadFile(filepath.Join(dest, "scripts", ".embed-hash"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(hash)) != embedDigest() {
		t.Fatalf("hash=%q want %q", hash, embedDigest())
	}
}

func TestExtractToSkipsWhenCurrent(t *testing.T) {
	dest := t.TempDir()
	if err := ExtractTo(dest, "1.0.0"); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dest, "scripts", "factory_http.py")
	want, err := os.ReadFile(script)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(script, want, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ExtractTo(dest, "1.0.0"); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(script)
	if string(got) != string(want) {
		t.Fatal("expected current extract to skip")
	}
}

func TestExtractToRefreshesStaleScriptsDespiteStamp(t *testing.T) {
	dest := t.TempDir()
	if err := ExtractTo(dest, "0.0.8"); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dest, "scripts", "factory_http.py")
	if err := os.WriteFile(script, []byte("stale 0.0.7 factory_http"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dest, "scripts", "factory_env.py")); err != nil {
		t.Fatal(err)
	}
	if err := ExtractTo(dest, "0.0.8"); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(script)
	if string(b) == "stale 0.0.7 factory_http" {
		t.Fatal("expected stale factory_http.py to be replaced")
	}
	if _, err := os.Stat(filepath.Join(dest, "scripts", "factory_env.py")); err != nil {
		t.Fatal(err)
	}
}

func TestEnsureRefreshesStaleDataDir(t *testing.T) {
	t.Setenv("PIPELINE_DATA", t.TempDir())
	t.Setenv("PIPELINE_ROOT", "")
	data := paths.DataDir()
	scripts := filepath.Join(data, "scripts")
	if err := os.MkdirAll(scripts, 0o755); err != nil {
		t.Fatal(err)
	}
	fakePy := filepath.Join(data, ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(fakePy), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fakePy, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scripts, "factory_http.py"), []byte("old http"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scripts, ".embed-version"), []byte("0.0.8\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := Ensure("0.0.8")
	if err != nil && res.Script == "" {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(scripts, "factory_env.py")); err != nil {
		t.Fatalf("factory_env.py not extracted: %v", err)
	}
	httpBody, _ := os.ReadFile(filepath.Join(scripts, "factory_http.py"))
	if string(httpBody) == "old http" {
		t.Fatal("DataDir still has the old factory_http.py")
	}
}

func TestSafeJoinRejectsDotDot(t *testing.T) {
	if _, err := safeJoin(t.TempDir(), "../evil"); err == nil {
		t.Fatal("expected error")
	}
}

func TestFindPythonPrefersHomeVenv(t *testing.T) {
	home := t.TempDir()
	t.Setenv("PIPELINE_DATA", t.TempDir())
	exe := filepath.Join(home, ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(exe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(exe, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := FindPython(home)
	if err != nil {
		t.Fatal(err)
	}
	if got != exe {
		t.Fatalf("got %s want %s", got, exe)
	}
}
