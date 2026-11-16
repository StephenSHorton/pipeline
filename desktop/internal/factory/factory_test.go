package factory

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenDirCmd(t *testing.T) {
	cmd := openDirCmd("/tmp/pipeline-data")
	if cmd.Args[0] != "open" {
		t.Fatalf("args=%v", cmd.Args)
	}
}

func TestExportDirChecksStem(t *testing.T) {
	for _, bad := range []string{"", "..", "../x", "a/b", ".hidden"} {
		if _, err := ExportDir(bad); err == nil {
			t.Fatalf("ExportDir(%q) should fail", bad)
		}
	}
	dir, err := ExportDir("lantern-2")
	if err != nil || filepath.Base(dir) != "export" || filepath.Base(filepath.Dir(dir)) != "lantern-2" {
		t.Fatalf("dir=%q err=%v", dir, err)
	}
}

func TestListAssetsSkipsDotFolders(t *testing.T) {
	root := t.TempDir()
	t.Setenv("PIPELINE_ROOT", root)
	for _, d := range []string{"out/radio", "out/.projects/rust-belt"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got := New().ListAssets()
	if len(got) != 1 || got[0].Stem != "radio" {
		t.Fatalf("assets=%v", got)
	}
}
