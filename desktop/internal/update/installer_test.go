package update

import "testing"

func TestStartSilentInstallerIsStubbed(t *testing.T) {
	if err := startSilentInstaller("/tmp/pipeline-0.0.10-darwin-arm64.dmg"); err == nil {
		t.Fatal("expected stub error until DMG install exists")
	}
}
