package runtime

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

// FindPython locates a factory interpreter.
// Prefers a checkout .venv (MLX factory), then a standalone CPython under
// DataDir, then python3 on PATH. Does not require Homebrew.
func FindPython(home string) (string, error) {
	data := paths.DataDir()
	cands := []string{
		filepath.Join(home, ".venv", "bin", "python"),
		filepath.Join(home, ".venv", "bin", "python3"),
		filepath.Join(data, ".venv", "bin", "python"),
		filepath.Join(data, ".venv", "bin", "python3"),
		filepath.Join(home, "python", "bin", "python3"),
		filepath.Join(data, "python", "bin", "python3"),
		filepath.Join(paths.ExeDir(), "python", "bin", "python3"),
	}
	for _, c := range cands {
		if fileExists(c) {
			return c, nil
		}
	}
	if p := pathPython(); p != "" {
		return p, nil
	}
	return "", fmt.Errorf("no runnable python3 under %s, %s, or PATH", home, data)
}

func pathPython() string {
	for _, name := range []string{"python3", "python"} {
		p, err := exec.LookPath(name)
		if err == nil && p != "" {
			return p
		}
	}
	return ""
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}
