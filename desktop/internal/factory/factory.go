package factory

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

type Asset struct {
	Stem     string `json:"stem"`
	Textured bool   `json:"textured"`
	Stack    bool   `json:"stack"`
}

type Service struct{}

func New() *Service { return &Service{} }

func (s *Service) Root() string { return paths.Root() }

func (s *Service) DataDir() string { return paths.DataDir() }

func (s *Service) OpenDataDir() error {
	dir := paths.DataDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return OpenDir(dir)
}

var stemRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

// ExportDir is out/<stem>/export, the folder POST /export/<stem> writes.
func ExportDir(stem string) (string, error) {
	if !stemRE.MatchString(stem) {
		return "", fmt.Errorf("invalid stem %q", stem)
	}
	return filepath.Join(paths.OutDir(), stem, "export"), nil
}

// OpenExport shows an asset's export folder in Finder.
func (s *Service) OpenExport(stem string) error {
	dir, err := ExportDir(stem)
	if err != nil {
		return err
	}
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return fmt.Errorf("%s is not exported yet", stem)
	}
	return OpenDir(dir)
}

func OpenDir(dir string) error {
	return openDirCmd(dir).Start()
}

func openDirCmd(dir string) *exec.Cmd {
	return exec.Command("open", dir)
}

func (s *Service) ListAssets() []Asset {
	dir := paths.OutDir()
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []Asset
	for _, e := range ents {
		// out/.projects (and any other dot folder) is factory bookkeeping, not an asset.
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		stem := e.Name()
		base := filepath.Join(dir, stem)
		_, tex := os.Stat(filepath.Join(base, "textured.glb"))
		_, stack := os.Stat(filepath.Join(base, "stack", "stack.json"))
		out = append(out, Asset{Stem: stem, Textured: tex == nil, Stack: stack == nil})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Stem < out[j].Stem })
	return out
}
