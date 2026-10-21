package factory

import (
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// ServeOut serves factory out/<stem>/... files with explicit types so the
// Wails webview can load GLBs (http.FileServer 404s if Root() is wrong).
func ServeOut(factoryRoot string) http.Handler {
	out := filepath.Join(factoryRoot, "out")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rel := strings.TrimPrefix(r.URL.Path, "/out/")
		rel = path.Clean("/" + rel)
		rel = strings.TrimPrefix(rel, "/")
		if rel == "" || rel == "." {
			http.NotFound(w, r)
			return
		}
		full, err := safeFile(out, rel)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		st, err := os.Stat(full)
		if err != nil || st.IsDir() {
			log.Printf("pipeline: out 404 %s (%s)", r.URL.Path, full)
			http.NotFound(w, r)
			return
		}
		switch strings.ToLower(filepath.Ext(full)) {
		case ".glb":
			w.Header().Set("Content-Type", "model/gltf-binary")
		case ".png":
			w.Header().Set("Content-Type", "image/png")
		case ".jpg", ".jpeg":
			w.Header().Set("Content-Type", "image/jpeg")
		case ".json":
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
		case ".webp":
			w.Header().Set("Content-Type", "image/webp")
		}
		w.Header().Set("Cache-Control", "no-store")
		http.ServeFile(w, r, full)
	})
}

func safeFile(root, rel string) (string, error) {
	full := filepath.Join(root, filepath.FromSlash(rel))
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	absFull, err := filepath.Abs(full)
	if err != nil {
		return "", err
	}
	sep := string(os.PathSeparator)
	if absFull != absRoot && !strings.HasPrefix(absFull, absRoot+sep) {
		return "", os.ErrNotExist
	}
	return absFull, nil
}
