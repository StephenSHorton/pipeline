package factory

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
)

const maxUpload = 32 << 20

var uploadNameRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

type UploadResult struct {
	OK    bool   `json:"ok"`
	Path  string `json:"path,omitempty"`
	Bytes int    `json:"bytes,omitempty"`
	Stem  string `json:"stem,omitempty"`
	Error string `json:"error,omitempty"`
}

type uploadJSON struct {
	Name  string `json:"name"`
	Data  string `json:"data"`
	Bytes string `json:"bytes"`
}

// SaveUpload is the Wails binding. `data` is JPEG/PNG bytes (JS sends base64).
func (s *Service) SaveUpload(name string, data []byte) (UploadResult, error) {
	res, err := saveBytes(name, data)
	if err != nil {
		return res, err
	}
	log.Printf("pipeline: upload %s (%d bytes) -> %s", name, res.Bytes, res.Path)
	return res, nil
}

func HandleUploadHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxUpload+1))
	if err != nil {
		writeUploadJSON(w, http.StatusBadRequest, UploadResult{OK: false, Error: err.Error()})
		return
	}
	if len(body) > maxUpload {
		writeUploadJSON(w, http.StatusRequestEntityTooLarge, UploadResult{OK: false, Error: "upload too large"})
		return
	}
	name := r.URL.Query().Get("name")
	data, name, err := DecodeUpload(name, r.Header.Get("Content-Type"), body)
	if err != nil {
		writeUploadJSON(w, http.StatusBadRequest, UploadResult{OK: false, Error: err.Error()})
		return
	}
	res, err := saveBytes(name, data)
	if err != nil {
		writeUploadJSON(w, http.StatusBadRequest, res)
		return
	}
	log.Printf("pipeline: upload %s (%d bytes) -> %s", name, res.Bytes, res.Path)
	writeUploadJSON(w, http.StatusCreated, res)
}

func DecodeUpload(name, contentType string, body []byte) (data []byte, outName string, err error) {
	outName = name
	trim := bytes.TrimSpace(body)
	isJSON := strings.Contains(strings.ToLower(contentType), "json") || bytes.HasPrefix(trim, []byte("{"))
	if isJSON && len(trim) > 0 {
		var payload uploadJSON
		if err := json.Unmarshal(trim, &payload); err != nil {
			return nil, outName, fmt.Errorf("invalid upload json: %w", err)
		}
		if payload.Name != "" {
			outName = payload.Name
		}
		b64 := payload.Data
		if b64 == "" {
			b64 = payload.Bytes
		}
		if b64 == "" {
			return nil, outName, fmt.Errorf("empty upload")
		}
		data, err = base64.StdEncoding.DecodeString(b64)
		if err != nil {
			data, err = base64.RawStdEncoding.DecodeString(b64)
		}
		if err != nil {
			return nil, outName, fmt.Errorf("invalid upload base64: %w", err)
		}
		return data, outName, nil
	}
	if len(body) == 0 {
		return nil, outName, fmt.Errorf("empty upload")
	}
	return body, outName, nil
}

func saveBytes(name string, data []byte) (UploadResult, error) {
	if len(data) == 0 {
		return UploadResult{OK: false, Error: "empty upload"}, fmt.Errorf("empty upload")
	}
	raw := filepath.Base(name)
	if raw == "" || raw == "." {
		raw = "upload.jpg"
	}
	ext := strings.ToLower(filepath.Ext(raw))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".webp":
	default:
		ext = ".jpg"
	}
	stem := uploadNameRe.ReplaceAllString(strings.Trim(strings.TrimSuffix(raw, filepath.Ext(raw)), "-._"), "-")
	if stem == "" {
		stem = "upload"
	}
	if len(stem) > 48 {
		stem = stem[:48]
	}
	root := paths.Root()
	dir := filepath.Join(root, "in")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return UploadResult{OK: false, Error: err.Error()}, err
	}
	dest := filepath.Join(dir, stem+ext)
	if err := os.WriteFile(dest, data, 0o644); err != nil {
		return UploadResult{OK: false, Error: err.Error()}, err
	}
	return UploadResult{
		OK:    true,
		Path:  "in/" + filepath.Base(dest),
		Bytes: len(data),
		Stem:  stem,
	}, nil
}

func writeUploadJSON(w http.ResponseWriter, status int, res UploadResult) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(res)
}
