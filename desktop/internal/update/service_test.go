package update

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsNewer(t *testing.T) {
	if !isNewer("v0.0.7", "0.0.6") {
		t.Fatal("0.0.7 should be newer than 0.0.6")
	}
	if isNewer("v0.0.6", "0.0.6") {
		t.Fatal("same version is not newer")
	}
	if isNewer("v0.0.7", "dev") {
		t.Fatal("dev builds must not auto-update")
	}
	if isNewer("v0.1.0", "0.0.10") != true {
		t.Fatal("semver still treats 0.1.0 as newer than 0.0.10")
	}
}

func TestSelectUpdateAssetPicksDarwinDMG(t *testing.T) {
	assets := []ghAsset{
		{Name: "pipeline-0.0.9-darwin-arm64.dmg", URL: "http://example/mac.dmg"},
		{Name: "SHA256SUMS", URL: "http://example/sums"},
	}
	mac, ok := selectUpdateAsset(assets, "arm64")
	if !ok || !strings.HasSuffix(mac.Name, ".dmg") {
		t.Fatalf("darwin asset: %+v ok=%v", mac, ok)
	}
	if _, ok := selectUpdateAsset(assets, "amd64"); ok {
		t.Fatal("intel Mac is out of scope")
	}
	if _, ok := selectUpdateAsset(assets[1:], "arm64"); ok {
		t.Fatal("no DMG means no update asset")
	}
}

func TestGitHubRequestsSendUserAgent(t *testing.T) {
	var ua string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ua = r.Header.Get("User-Agent")
		if r.URL.Path == "/repos/StephenSHorton/pipeline/releases/latest" {
			io.WriteString(w, `{"tag_name":"v0.0.7","draft":false,"prerelease":false,"published_at":"2026-09-15T02:09:16Z","assets":[{"name":"pipeline-0.0.7-darwin-arm64.dmg","browser_download_url":"http://example/mac.dmg"}]}`)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.6")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	info, err := s.CheckForUpdate()
	if err != nil {
		t.Fatal(err)
	}
	if info == nil || info.Version != "0.0.7" {
		t.Fatalf("update info: %+v", info)
	}
	if !strings.HasPrefix(ua, "Pipeline-Host/") {
		t.Fatalf("User-Agent %q; GitHub rejects requests without one", ua)
	}
}

func TestCheckForUpdateWithoutDMGIsNotAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"tag_name":"v0.0.9","draft":false,"prerelease":false,"published_at":"2026-09-15T02:09:16Z","assets":[{"name":"SHA256SUMS","browser_download_url":"http://example/sums"}]}`)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.8")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	info, err := s.CheckForUpdate()
	if err != nil {
		t.Fatalf("missing darwin DMG must not be a fatal error: %v", err)
	}
	if info != nil {
		t.Fatalf("no DMG, no update: %+v", info)
	}
}

func TestCheckForUpdateDarwinPicksDMG(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"tag_name":"v0.0.10","draft":false,"prerelease":false,"published_at":"2026-09-15T02:09:16Z","assets":[{"name":"pipeline-0.0.10-darwin-arm64.dmg","browser_download_url":"http://example/mac.dmg"}]}`)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.9")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	info, err := s.CheckForUpdate()
	if err != nil {
		t.Fatal(err)
	}
	if info == nil || !strings.HasSuffix(info.AssetName, ".dmg") {
		t.Fatalf("darwin update: %+v", info)
	}
}

func TestCheckForUpdateSkipsDraftAndPrerelease(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"tag_name":"v0.0.11","draft":false,"prerelease":true,"published_at":"2026-09-15T02:09:16Z","assets":[{"name":"pipeline-0.0.11-darwin-arm64.dmg","browser_download_url":"http://example/mac.dmg"}]}`)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.10")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	info, err := s.CheckForUpdate()
	if err != nil {
		t.Fatal(err)
	}
	if info != nil {
		t.Fatalf("prerelease must not be offered: %+v", info)
	}
}

func TestCheckForUpdateNotFoundIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.10")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	info, err := s.CheckForUpdate()
	if err == nil {
		t.Fatal("private-repo / missing latest must not look like up-to-date")
	}
	if info != nil {
		t.Fatalf("info=%+v", info)
	}
	if !strings.Contains(err.Error(), "404") {
		t.Fatalf("err=%v", err)
	}
}

func TestGitHubTokenHeader(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "test-token")
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		io.WriteString(w, `{"tag_name":"v0.0.10","draft":false,"prerelease":false,"assets":[]}`)
	}))
	t.Cleanup(srv.Close)

	s := New("StephenSHorton/pipeline", "0.0.10")
	s.goarch = "arm64"
	s.apiBase = srv.URL
	s.client = srv.Client()

	_, err := s.CheckForUpdate()
	if err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer test-token" {
		t.Fatalf("Authorization=%q", auth)
	}
}
