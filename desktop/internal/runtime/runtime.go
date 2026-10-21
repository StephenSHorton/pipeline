// Package runtime extracts the factory scripts from the host and
// finds the factory Python interpreter.
package runtime

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

var requiredFactoryFiles = []string{
	"scripts/factory_http.py",
	"scripts/generate.sh",
	"scripts/factory_env.py",
	"deps.json",
}

type Result struct {
	Root   string
	Python string
	Script string
}

// Ensure places factory_http.py (and the rest of scripts/) under DataDir and
// finds a runnable python. A git checkout that
// already has scripts/factory_http.py is used for CLI/dev, but DataDir is
// still refreshed so an exe-only update cannot leave 0.0.7 factory files.
func Ensure(version string) (Result, error) {
	data := paths.DataDir()
	if err := os.MkdirAll(filepath.Join(data, "out"), 0o755); err != nil {
		return Result{Root: data}, err
	}
	if err := os.MkdirAll(filepath.Join(data, "in"), 0o755); err != nil {
		return Result{Root: data}, err
	}
	if err := ExtractTo(data, version); err != nil {
		home := paths.Root()
		if sameDir(home, data) || !scriptExists(home) {
			return Result{Root: data}, fmt.Errorf("extract factory scripts: %w", err)
		}
		log.Printf("pipeline: extract factory scripts into data dir: %v", err)
	}

	home := paths.Root()
	if !sameDir(home, data) && scriptExists(home) {
		py, err := FindPython(home)
		return Result{
			Root:   home,
			Python: py,
			Script: filepath.Join(home, "scripts", "factory_http.py"),
		}, err
	}
	py, err := FindPython(data)
	script := filepath.Join(data, "scripts", "factory_http.py")
	if err != nil {
		return Result{Root: data, Script: script}, err
	}
	return Result{Root: data, Python: py, Script: script}, nil
}

func scriptExists(home string) bool {
	_, err := os.Stat(filepath.Join(home, "scripts", "factory_http.py"))
	return err == nil
}

func sameDir(a, b string) bool {
	aa, errA := filepath.Abs(a)
	bb, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return filepath.Clean(a) == filepath.Clean(b)
	}
	return strings.EqualFold(filepath.Clean(aa), filepath.Clean(bb))
}

func embedDigest() string {
	var names []string
	_ = fs.WalkDir(FS, "fs", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		names = append(names, path)
		return nil
	})
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		data, err := FS.ReadFile(name)
		if err != nil {
			continue
		}
		_, _ = fmt.Fprintf(h, "%s\n%d\n", name, len(data))
		_, _ = h.Write(data)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func factoryScriptsStale(dest string) string {
	for _, rel := range requiredFactoryFiles {
		if _, err := os.Stat(filepath.Join(dest, filepath.FromSlash(rel))); err != nil {
			return "missing " + rel
		}
	}
	return ""
}

func extractCurrent(dest, version, wantHash string) bool {
	if version == "" || version == "dev" {
		return false
	}
	stamp := filepath.Join(dest, "scripts", ".embed-version")
	hashPath := filepath.Join(dest, "scripts", ".embed-hash")
	gotStamp, stampErr := os.ReadFile(stamp)
	gotHash, hashErr := os.ReadFile(hashPath)
	if stampErr != nil || hashErr != nil {
		return false
	}
	if strings.TrimSpace(string(gotStamp)) != version {
		return false
	}
	if strings.TrimSpace(string(gotHash)) != wantHash {
		return false
	}
	return factoryScriptsStale(dest) == ""
}

func ExtractTo(dest, version string) error {
	if dest == "" {
		return fmt.Errorf("empty extract dest")
	}
	wantHash := embedDigest()
	stamp := filepath.Join(dest, "scripts", ".embed-version")
	hashPath := filepath.Join(dest, "scripts", ".embed-hash")
	script := filepath.Join(dest, "scripts", "factory_http.py")
	if extractCurrent(dest, version, wantHash) {
		return nil
	}
	if reason := factoryScriptsStale(dest); reason != "" {
		log.Printf("pipeline: refreshing factory scripts (%s)", reason)
	}

	// Wipe the extracted scripts dir so a 0.0.7 tree cannot mix with this embed.
	// out/, .venv, and vendor/ live beside scripts/ and are left alone.
	if err := os.RemoveAll(filepath.Join(dest, "scripts")); err != nil {
		return fmt.Errorf("clear factory scripts: %w", err)
	}

	err := fs.WalkDir(FS, "fs", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel := strings.TrimPrefix(path, "fs")
		rel = strings.TrimPrefix(rel, "/")
		if rel == "" {
			if d.IsDir() {
				return os.MkdirAll(dest, 0o755)
			}
			return nil
		}
		target, joinErr := safeJoin(dest, rel)
		if joinErr != nil {
			return joinErr
		}
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, readErr := FS.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
	if err != nil {
		return err
	}
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("factory_http.py missing after extract: %w", err)
	}
	if err := os.WriteFile(stamp, []byte(version+"\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(hashPath, []byte(wantHash+"\n"), 0o644); err != nil {
		return err
	}
	log.Printf("pipeline: factory scripts at %s", filepath.Join(dest, "scripts"))
	return nil
}

func safeJoin(root, rel string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(rel))
	if clean == "." {
		return root, nil
	}
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
		return "", fmt.Errorf("refusing path %q", rel)
	}
	p := filepath.Join(root, clean)
	relOut, err := filepath.Rel(root, p)
	if err != nil || strings.HasPrefix(relOut, "..") {
		return "", fmt.Errorf("refusing path %q", rel)
	}
	return p, nil
}
