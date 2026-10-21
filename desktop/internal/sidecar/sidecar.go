package sidecar

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/StephenSHorton/pipeline/desktop/internal/runtime"
)

const Port = 8766

type Sidecar struct {
	cmd     *exec.Cmd
	mu      sync.Mutex
	owned   bool
	python  string
	script  string
	root    string
	lastErr error
}

func New(version string) *Sidecar {
	s := &Sidecar{}
	res, err := runtime.Ensure(version)
	s.root = res.Root
	s.python = res.Python
	s.script = res.Script
	s.lastErr = err
	return s
}

func (s *Sidecar) Root() string { return s.root }

// SetRoot points the host at a factory tree. Tests use it; the running app uses Ensure.
func (s *Sidecar) SetRoot(dir string) { s.root = dir }

func (s *Sidecar) Python() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.python
}

func (s *Sidecar) Script() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.script
}

func (s *Sidecar) LastError() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastErr
}

func (s *Sidecar) Running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.runningLocked()
}

func (s *Sidecar) runningLocked() bool {
	return s.owned && s.cmd != nil && s.cmd.Process != nil
}

func (s *Sidecar) failLocked(err error) error {
	s.lastErr = err
	return err
}

func (s *Sidecar) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastErr != nil && (s.python == "" || s.script == "") {
		return s.lastErr
	}
	s.killStale()
	if err := waitPortFree(Port, 3*time.Second); err != nil {
		s.takeOver()
		_ = waitPortFree(Port, 3*time.Second)
	}
	if _, err := os.Stat(s.python); err != nil {
		return s.failLocked(fmt.Errorf("factory python missing: %s", s.python))
	}
	if _, err := os.Stat(s.script); err != nil {
		return s.failLocked(fmt.Errorf("factory_http.py missing: %s", s.script))
	}
	if err := os.MkdirAll(filepath.Join(s.root, "out"), 0o755); err != nil {
		return s.failLocked(err)
	}
	logPath := filepath.Join(s.root, "out", "factory_sidecar.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return s.failLocked(fmt.Errorf("factory log: %w", err))
	}
	cmd := exec.Command(s.python, s.script, "--host", "127.0.0.1", "--port", strconv.Itoa(Port), "--root", s.root)
	cmd.Dir = s.root
	cmd.Env = append(os.Environ(),
		"FACTORY_ROOT="+s.root,
		"PYTHONPATH="+filepath.Join(s.root, "scripts"),
		"PYTHONUTF8=1",
		"PYTHONUNBUFFERED=1",
	)
	configureFactoryCmd(cmd)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return s.failLocked(fmt.Errorf("start factory_http: %w", err))
	}
	s.cmd = cmd
	s.owned = true
	go func() {
		_ = cmd.Wait()
		_ = logFile.Close()
	}()
	ready, err := waitHealth(ctx, 20*time.Second)
	if err != nil {
		s.stopLocked()
		return s.failLocked(fmt.Errorf("factory health: %w\n%s", err, tailFile(logPath)))
	}
	if !ready {
		s.stopLocked()
		return s.failLocked(fmt.Errorf("factory_http did not become healthy on :%d\n%s", Port, tailFile(logPath)))
	}
	s.lastErr = nil
	return nil
}

func (s *Sidecar) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopLocked()
}

func (s *Sidecar) stopLocked() {
	if s.cmd != nil && s.cmd.Process != nil {
		killPID(s.cmd.Process.Pid)
		s.cmd = nil
	}
	s.owned = false
	s.killStale()
}

func (s *Sidecar) killStale() {
	pidPath := filepath.Join(s.root, "out", "factory_http.pid")
	if b, err := os.ReadFile(pidPath); err == nil {
		pid := strings.TrimSpace(string(b))
		if n, err := strconv.Atoi(pid); err == nil && n > 0 {
			killPID(n)
		}
		_ = os.Remove(pidPath)
	}
	if pid := pidOnPort(Port); pid > 0 {
		killPID(pid)
	}
}

func (s *Sidecar) takeOver() {
	s.killStale()
}

func (s *Sidecar) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.Running() {
			s.writeDown(w, "")
			return
		}
		var body io.Reader = r.Body
		var n int64
		if r.Body != nil {
			buf, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
			if err != nil {
				s.writeDown(w, err.Error())
				return
			}
			n = int64(len(buf))
			body = bytes.NewReader(buf)
		}
		req, err := http.NewRequestWithContext(r.Context(), r.Method, "http://127.0.0.1:"+strconv.Itoa(Port)+r.URL.RequestURI(), body)
		if err != nil {
			s.writeDown(w, err.Error())
			return
		}
		req.Header = r.Header.Clone()
		req.ContentLength = n
		if n > 0 {
			req.Header.Set("Content-Length", strconv.FormatInt(n, 10))
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			s.writeDown(w, err.Error())
			return
		}
		defer func() { _ = resp.Body.Close() }()
		for k, vs := range resp.Header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	})
}

func (s *Sidecar) writeDown(w http.ResponseWriter, extra string) {
	s.mu.Lock()
	errStr := "factory is not running"
	if s.lastErr != nil {
		errStr = s.lastErr.Error()
	}
	if extra != "" && s.lastErr == nil {
		errStr = extra
	}
	payload := map[string]any{
		"ok":      false,
		"service": "ai-asset-pipeline",
		"error":   errStr,
		"python":  s.python,
		"script":  s.script,
		"root":    s.root,
	}
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(payload)
}

func waitPortFree(port int, d time.Duration) error {
	deadline := time.Now().Add(d)
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	for time.Now().Before(deadline) {
		c, err := net.DialTimeout("tcp", addr, 150*time.Millisecond)
		if err != nil {
			return nil
		}
		_ = c.Close()
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("port %d still busy", port)
}

func waitHealth(ctx context.Context, d time.Duration) (bool, error) {
	deadline := time.Now().Add(d)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	url := fmt.Sprintf("http://127.0.0.1:%d/health", Port)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		default:
		}
		resp, err := client.Get(url)
		if err == nil {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			_ = resp.Body.Close()
			if resp.StatusCode == 200 && strings.Contains(string(b), "ai-asset-pipeline") {
				return true, nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	return false, nil
}

func tailFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) > 40 {
		lines = lines[len(lines)-40:]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func IsFactoryAPI(path string) bool {
	p := strings.TrimPrefix(path, "/")
	if p == "health" || p == "assets" || p == "status" || p == "generate" || p == "upload" || p == "clean" || p == "2d" {
		return true
	}
	for _, pre := range []string{"inspect/", "status/", "glb/", "stack/", "assets/", "paint/", "2d/", "flow/", "export/", "cancel/", "intent/", "rig/", "facing/"} {
		if strings.HasPrefix(p, pre) {
			return true
		}
	}
	return false
}
