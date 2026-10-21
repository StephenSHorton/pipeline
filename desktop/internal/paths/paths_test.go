package paths

import (
	"path/filepath"
	"testing"
)

func TestDataDirHonorsPipelineData(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("PIPELINE_DATA", dir)
	if got := DataDir(); got != dir {
		t.Fatalf("DataDir()=%s want %s", got, dir)
	}
}

func TestPlatformDataDirShape(t *testing.T) {
	t.Setenv("PIPELINE_DATA", "")
	got := DataDir()
	if filepath.Base(got) != AppName {
		t.Fatalf("DataDir base=%s want %s (%s)", filepath.Base(got), AppName, got)
	}
	if filepath.Base(filepath.Dir(got)) != "Application Support" {
		t.Fatalf("DataDir should be under Application Support, got %s", got)
	}
}
