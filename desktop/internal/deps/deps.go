// Package deps fetches Hunyuan weights and other bulky runtime files into
// the data dir. They are never committed to git.
package deps

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

type Manifest struct {
	Version string `json:"version"`
	Notes   string `json:"notes"`
	Items   []Item `json:"items"`
}

type Item struct {
	ID      string `json:"id"`
	Kind    string `json:"kind"` // huggingface | url
	Repo    string `json:"repo,omitempty"`
	Sub     string `json:"subfolder,omitempty"`
	URL     string `json:"url,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
	Dest    string `json:"dest"`
	Version string `json:"version,omitempty"`
}

type Status struct {
	DataDir string       `json:"dataDir"`
	Root    string       `json:"root"`
	Items   []ItemStatus `json:"items"`
}

type ItemStatus struct {
	ID      string `json:"id"`
	Dest    string `json:"dest"`
	Present bool   `json:"present"`
}

type Service struct {
	manifest Manifest
}

func New(m Manifest) *Service { return &Service{manifest: m} }

func LoadFile(path string) (Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

func (s *Service) Status() Status {
	st := Status{DataDir: paths.DataDir(), Root: paths.Root()}
	for _, it := range s.manifest.Items {
		dest := resolveDest(it.Dest)
		_, err := os.Stat(dest)
		st.Items = append(st.Items, ItemStatus{ID: it.ID, Dest: dest, Present: err == nil})
	}
	return st
}

func (s *Service) Ensure() error {
	if err := os.MkdirAll(paths.DataDir(), 0o755); err != nil {
		return err
	}
	// First-run: if the factory checkout already has weights, reuse them.
	// Downloads of multi-GB Hunyuan snapshots happen on demand later
	// (huggingface-cli / generate bootstrap). We never vendor them in git.
	st := s.Status()
	for _, it := range st.Items {
		if it.Present {
			continue
		}
		_ = os.MkdirAll(filepath.Dir(it.Dest), 0o755)
	}
	return nil
}

func resolveDest(dest string) string {
	if filepath.IsAbs(dest) {
		return dest
	}
	root := paths.Root()
	p := filepath.Join(root, dest)
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return filepath.Join(paths.DataDir(), dest)
}

func (s *Service) Missing() []string {
	var miss []string
	for _, it := range s.Status().Items {
		if !it.Present {
			miss = append(miss, it.ID)
		}
	}
	return miss
}

func (s *Service) Summary() string {
	miss := s.Missing()
	if len(miss) == 0 {
		return "all runtime deps present"
	}
	return fmt.Sprintf("missing (download on generate): %v", miss)
}
