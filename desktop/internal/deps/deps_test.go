package deps

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStatusPresentFromRoot(t *testing.T) {
	root := t.TempDir()
	t.Setenv("PIPELINE_ROOT", root)
	t.Setenv("PIPELINE_DATA", t.TempDir())

	m := Manifest{Items: []Item{{
		ID:   "hunyuan-paint",
		Kind: "huggingface",
		Dest: "weights/paint",
	}}}
	s := New(m)
	if s.Status().Items[0].Present {
		t.Fatal("missing dest must not be present")
	}
	if err := os.MkdirAll(filepath.Join(root, "weights", "paint"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !s.Status().Items[0].Present {
		t.Fatal("expected checkout weights to count as present")
	}
}

func TestLoadFileReadsItems(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "deps.json")
	body := `{
	  "version": "0.0.4",
	  "items": [
	    {
	      "id": "hunyuan-paint",
	      "kind": "huggingface",
	      "repo": "tencent/Hunyuan3D-2",
	      "dest": "weights/paint"
	    }
	  ]
	}`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.Items[0].Kind != "huggingface" || m.Items[0].Repo != "tencent/Hunyuan3D-2" {
		t.Fatalf("item=%+v", m.Items[0])
	}
}
