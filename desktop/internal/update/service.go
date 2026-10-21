// Package update is Pipeline's in-app updater.
// GitHub Releases → darwin DMG asset when present.
package update

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const EventUpdateInstalling = "update:installing"

// Host is the native window. The update package does not import Wails so
// it can compile without WebKit CGO.
type Host interface {
	Quit()
	EmitInstalling(version string)
}

type UpdateInfo struct {
	Version     string `json:"version"`
	Notes       string `json:"notes"`
	AssetURL    string `json:"assetUrl"`
	AssetName   string `json:"assetName"`
	SHA256      string `json:"sha256"`
	PublishedAt string `json:"publishedAt"`
}

type Service struct {
	repo       string
	current    string
	apiBase    string
	goarch     string
	client     *http.Client
	host       Host
	installing atomic.Bool
}

func New(repo, currentVersion string) *Service {
	return &Service{
		repo:    repo,
		current: currentVersion,
		goarch:  runtime.GOARCH,
		// No client Timeout — each call uses a context. A 30s cap was
		// aborting the ~6MB installer download on slower links.
		client: &http.Client{},
	}
}

func (s *Service) arch() string {
	if s.goarch == "" {
		return runtime.GOARCH
	}
	return s.goarch
}

func githubToken() string {
	if t := strings.TrimSpace(os.Getenv("GITHUB_TOKEN")); t != "" {
		return t
	}
	return strings.TrimSpace(os.Getenv("GH_TOKEN"))
}

func (s *Service) applyGitHubHeaders(req *http.Request) {
	ua := "Pipeline-Host"
	if v := strings.TrimPrefix(s.current, "v"); v != "" && v != "dev" {
		ua = "Pipeline-Host/" + v
	}
	req.Header.Set("User-Agent", ua)
	if tok := githubToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
}

func (s *Service) SetHost(h Host) { s.host = h }

func (s *Service) GetVersion() string { return s.current }

func (s *Service) AutoUpdate() {
	info, err := s.CheckForUpdate()
	if err != nil {
		log.Printf("pipeline: auto-update check failed: %v", err)
		return
	}
	if info == nil {
		return
	}
	log.Printf("pipeline: auto-updating to v%s", info.Version)
	if err := s.DownloadAndInstall(*info); err != nil {
		log.Printf("pipeline: auto-update install failed: %v", err)
	}
}

type ghAsset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
}

type ghRelease struct {
	TagName     string    `json:"tag_name"`
	Body        string    `json:"body"`
	Draft       bool      `json:"draft"`
	Prerelease  bool      `json:"prerelease"`
	PublishedAt string    `json:"published_at"`
	Assets      []ghAsset `json:"assets"`
}

func (s *Service) latestRelease(ctx context.Context) (*ghRelease, error) {
	base := s.apiBase
	if base == "" {
		base = "https://api.github.com"
	}
	url := fmt.Sprintf("%s/repos/%s/releases/latest", strings.TrimRight(base, "/"), s.repo)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	s.applyGitHubHeaders(req)
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		if resp.StatusCode == http.StatusNotFound {
			// A private repo 404s the same as "no releases". Toru is public, so
			// /releases/latest works there. Do not treat this as "up to date".
			return nil, fmt.Errorf("github releases: 404 — no latest release visible (private repo or nothing published)")
		}
		return nil, fmt.Errorf("github releases: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	var rel ghRelease
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return nil, err
	}
	return &rel, nil
}

func (s *Service) CheckForUpdate() (*UpdateInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	rel, err := s.latestRelease(ctx)
	if err != nil || rel == nil {
		return nil, err
	}
	if rel.Draft || rel.Prerelease {
		return nil, nil
	}
	if !isNewer(rel.TagName, s.current) {
		return nil, nil
	}
	installer, ok := selectUpdateAsset(rel.Assets, s.arch())
	var sumsURL string
	for _, a := range rel.Assets {
		if strings.ToLower(a.Name) == "sha256sums" {
			sumsURL = a.URL
		}
	}
	if !ok {
		// No darwin DMG for this arch yet — not an error.
		return nil, nil
	}
	var sum string
	if sumsURL != "" {
		sum, _ = s.fetchChecksum(ctx, sumsURL, installer.Name)
	}
	return &UpdateInfo{
		Version:     strings.TrimPrefix(rel.TagName, "v"),
		Notes:       rel.Body,
		AssetURL:    installer.URL,
		AssetName:   installer.Name,
		SHA256:      sum,
		PublishedAt: rel.PublishedAt,
	}, nil
}

func selectUpdateAsset(assets []ghAsset, goarch string) (ghAsset, bool) {
	for _, a := range assets {
		n := strings.ToLower(a.Name)
		if strings.HasSuffix(n, ".dmg") && strings.Contains(n, "darwin") && strings.Contains(n, goarch) {
			return a, true
		}
	}
	return ghAsset{}, false
}

func (s *Service) fetchChecksum(ctx context.Context, url, name string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	s.applyGitHubHeaders(req)
	resp, err := s.client.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	for _, line := range strings.Split(string(b), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) == 2 && filepath.Base(strings.TrimPrefix(fields[1], "*")) == name {
			return strings.ToLower(fields[0]), nil
		}
	}
	return "", fmt.Errorf("no checksum for %s", name)
}

func (s *Service) DownloadAndInstall(info UpdateInfo) error {
	if !s.installing.CompareAndSwap(false, true) {
		return nil
	}
	committed := false
	defer func() {
		if !committed {
			s.installing.Store(false)
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	dir := filepath.Join(os.TempDir(), "pipeline-update")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	dst := filepath.Join(dir, info.AssetName)
	if err := s.download(ctx, info.AssetURL, dst); err != nil {
		return err
	}
	if info.SHA256 != "" {
		got, err := sha256File(dst)
		if err != nil {
			return err
		}
		if !strings.EqualFold(got, info.SHA256) {
			_ = os.Remove(dst)
			return fmt.Errorf("update checksum mismatch: got %s, want %s", got, info.SHA256)
		}
	}
	if err := startSilentInstaller(dst); err != nil {
		return fmt.Errorf("launch installer: %w", err)
	}
	committed = true
	if s.host != nil {
		s.host.EmitInstalling(info.Version)
		s.host.Quit()
	}
	return nil
}

func (s *Service) download(ctx context.Context, url, dst string) (err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	s.applyGitHubHeaders(req)
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()
	_, err = io.Copy(f, resp.Body)
	return err
}

func sha256File(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func isNewer(tag, current string) bool {
	if current == "" || current == "dev" {
		return false
	}
	return cmpSemver(strings.TrimPrefix(tag, "v"), strings.TrimPrefix(current, "v")) > 0
}

func cmpSemver(a, b string) int {
	na, prea := parseVer(a)
	nb, preb := parseVer(b)
	for i := 0; i < 3; i++ {
		if na[i] != nb[i] {
			if na[i] > nb[i] {
				return 1
			}
			return -1
		}
	}
	switch {
	case prea == "" && preb == "":
		return 0
	case prea == "":
		return 1
	case preb == "":
		return -1
	default:
		return strings.Compare(prea, preb)
	}
}

func parseVer(v string) ([3]int, string) {
	v = strings.SplitN(v, "+", 2)[0]
	num, pre, _ := strings.Cut(v, "-")
	var out [3]int
	for i, p := range strings.SplitN(num, ".", 3) {
		if i > 2 {
			break
		}
		n, _ := strconv.Atoi(strings.TrimSpace(p))
		out[i] = n
	}
	return out, pre
}
