package main

import (
	"context"
	"embed"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/StephenSHorton/pipeline/desktop/internal/control"
	"github.com/StephenSHorton/pipeline/desktop/internal/deps"
	"github.com/StephenSHorton/pipeline/desktop/internal/factory"
	"github.com/StephenSHorton/pipeline/desktop/internal/paths"
	"github.com/StephenSHorton/pipeline/desktop/internal/sidecar"
	"github.com/StephenSHorton/pipeline/desktop/internal/update"
	"github.com/wailsapp/wails/v3/pkg/application"
	"github.com/wailsapp/wails/v3/pkg/events"
)

//go:embed all:frontend/dist
var assets embed.FS

//go:embed deps.json
var depsJSON []byte

var version = "dev"

const updateRepo = "StephenSHorton/pipeline"

type updateHost struct {
	app *application.App
}

func (h updateHost) Quit() {
	if h.app != nil {
		h.app.Quit()
	}
}

func (h updateHost) EmitInstalling(version string) {
	if h.app != nil {
		h.app.Event.Emit(update.EventUpdateInstalling, version)
	}
}

func main() {
	setupLog()
	log.Printf("pipeline: version %s", version)

	var manifest deps.Manifest
	if err := json.Unmarshal(depsJSON, &manifest); err != nil {
		log.Printf("pipeline: deps.json: %v", err)
	}
	depSvc := deps.New(manifest)
	_ = depSvc.Ensure()
	if err := os.WriteFile(filepath.Join(paths.DataDir(), "deps.json"), depsJSON, 0o644); err != nil {
		log.Printf("pipeline: write deps.json: %v", err)
	}

	fact := factory.New()
	upd := update.New(updateRepo, version)

	sc := sidecar.New(version)
	if err := sc.LastError(); err != nil {
		log.Printf("pipeline: factory runtime: %v", err)
	}
	startCtx, startCancel := context.WithTimeout(context.Background(), 25*time.Second)
	if err := sc.Start(startCtx); err != nil {
		log.Printf("pipeline: factory sidecar: %v", err)
	}
	startCancel()
	defer sc.Stop()

	ui := control.Handler(assetHandler(sc.Root(), sc), sc, version)
	if addr, err := control.Start(ui); err != nil {
		log.Printf("pipeline: control server: %v", err)
	} else {
		log.Printf("pipeline: control http://%s/diagnostics", addr)
	}

	app := application.New(application.Options{
		Name:        "Pipeline",
		Description: "AI asset factory: generate, paint, sprite-stack",
		Services: []application.Service{
			application.NewService(fact),
			application.NewService(depSvc),
			application.NewService(upd),
		},
		Assets: application.AssetOptions{
			Handler: ui,
		},
		Mac: application.MacOptions{
			ApplicationShouldTerminateAfterLastWindowClosed: true,
		},
		// Quit on macOS (menu, last window, SIGTERM via Wails' handler) can exit
		// without app.Run returning, so the deferred Stop never runs.
		OnShutdown: sc.Stop,
	})
	upd.SetHost(updateHost{app})

	// SIGTERM/SIGINT otherwise kill the host and orphan the factory, which keeps
	// port 8766 and the loaded models until the next launch.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-quit
		sc.Stop()
		app.Quit()
	}()

	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Title:            windowTitle(version),
		Width:            1480,
		Height:           920,
		BackgroundColour: application.NewRGB(16, 17, 15),
		URL:              "/?look=voxel",
	})

	app.Event.OnApplicationEvent(events.Common.ApplicationStarted, func(*application.ApplicationEvent) {
		go upd.AutoUpdate()
	})

	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}

func setupLog() {
	dir := filepath.Join(paths.DataDir(), "out")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dir, "pipeline.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	log.SetOutput(io.MultiWriter(f, os.Stderr))
}

func assetHandler(factoryRoot string, sc *sidecar.Sidecar) http.Handler {
	ui := application.BundledAssetFileServer(assets)
	files := factory.ServeOut(factoryRoot)
	api := sc.Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/version" {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			label := formatVersion(version)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":      true,
				"version": label,
				"raw":     version,
			})
			return
		}
		if r.URL.Path == "/paths" {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":      true,
				"dataDir": paths.DataDir(),
				"version": formatVersion(version),
				"raw":     version,
			})
			return
		}
		if r.URL.Path == "/open-data-dir" && (r.Method == http.MethodPost || r.Method == http.MethodGet) {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			if err := factory.New().OpenDataDir(); err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": err.Error()})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "dataDir": paths.DataDir()})
			return
		}
		if r.URL.Path == "/open-export" && r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			if err := factory.New().OpenExport(r.URL.Query().Get("stem")); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": err.Error()})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
			return
		}
		if r.URL.Path == "/upload" && r.Method == http.MethodPost {
			factory.HandleUploadHTTP(w, r)
			return
		}
		if sidecar.IsFactoryAPI(r.URL.Path) {
			api.ServeHTTP(w, r)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/out/") {
			files.ServeHTTP(w, r)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/viewer/") {
			http.StripPrefix("/viewer/", http.FileServer(http.Dir(filepath.Join(factoryRoot, "viewer")))).ServeHTTP(w, r)
			return
		}
		ui.ServeHTTP(w, r)
	})
}
