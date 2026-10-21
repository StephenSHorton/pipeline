package paths

import (
	"os"
	"path/filepath"
)

// AppName is the product folder under Application Support.
const AppName = "Pipeline"

// DataDir is the writable factory tree for an installed host:
// ~/Library/Application Support/Pipeline.
// Override with PIPELINE_DATA (tests and support).
func DataDir() string {
	if v := os.Getenv("PIPELINE_DATA"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		return filepath.Join(home, "Library", "Application Support", AppName)
	}
	base, _ := os.UserConfigDir()
	return filepath.Join(base, AppName)
}

func hasFactoryScripts(dir string) bool {
	if dir == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, "scripts", "factory_http.py"))
	return err == nil
}

// Root is the factory tree (scripts/factory_http.py) or DataDir when installed.
func Root() string {
	if v := os.Getenv("PIPELINE_ROOT"); v != "" {
		return v
	}
	var candidates []string
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, cwd, filepath.Dir(cwd), filepath.Join(cwd, ".."))
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		candidates = append(candidates, d, filepath.Join(d, ".."), filepath.Join(d, "..", ".."))
	}
	for _, c := range candidates {
		if hasFactoryScripts(c) {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	return DataDir()
}

// ExeDir is the folder that contains the host binary
// (Contents/MacOS inside Pipeline.app).
func ExeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	return filepath.Dir(exe)
}

func OutDir() string {
	return filepath.Join(Root(), "out")
}
