package control

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/StephenSHorton/pipeline/desktop/internal/sidecar"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestWorkflowAndSheet(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("out/sheet/flat/source.png", "png")
	write("out/sheet/flat/parts.json", `{"parts":[{"id":"part-02","file":"parts/part-02.png","name":""}]}`)
	write("out/sheet/flat/mechanics.json", `{"note":"browser pass","parts":[{"id":"gun","file":"parts/part-02.png","name":"Gun","role":"weapon","projectile":"small energy bolt","line":"Hold fire.","speaker":"af_heart","sound":"sounds/gun.wav","voice":"voices/gun.wav"}]}`)
	write("out/radio/lods.json", `{"lods":[]}`)

	flat, err := workflowView(root, "sheet")
	if err != nil || flat["workflow"] != "2d" || flat["choose"] != false {
		t.Fatalf("flat=%v err=%v", flat, err)
	}
	mesh, err := workflowView(root, "radio")
	if err != nil || mesh["workflow"] != "3d" {
		t.Fatalf("mesh=%v err=%v", mesh, err)
	}
	view, err := sheetView(root, "sheet", "gun")
	if err != nil {
		t.Fatal(err)
	}
	if view["who"] != "Gun" || view["note"] != "browser pass" {
		t.Fatalf("view=%v", view)
	}
	audio := view["audio"].(map[string]any)
	if audio["line"] != "Hold fire." || audio["effect"] != "small energy bolt" {
		t.Fatalf("audio=%v", audio)
	}

	sc := &sidecar.Sidecar{}
	sc.SetRoot(root)
	h := Handler(http.NotFoundHandler(), sc, "dev")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, localReq(http.MethodGet, "/control/sheet?stem=sheet&part=gun"))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["workflow"] != "2d" {
		t.Fatalf("payload=%v", payload)
	}
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, localReq(http.MethodGet, "/diagnostics"))
	if rec.Code != 200 || !json.Valid(rec.Body.Bytes()) {
		t.Fatalf("diagnostics %d", rec.Code)
	}
}

func TestStepsOpForwardsFieldsTheOldStructDropped(t *testing.T) {
	for _, tool := range Tools() {
		if tool["path"] == "/control" && tool["does"] != "op sheet, split, drop, pack, save, sound, speak, or steps" {
			t.Fatalf("tools=%v", tool)
		}
	}
	root := t.TempDir()
	post := func(raw string) map[string]any {
		t.Helper()
		var cmd command
		if err := json.Unmarshal([]byte(raw), &cmd); err != nil {
			t.Fatal(err)
		}
		var got map[string]any
		var path string
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			path = r.URL.Path
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatal(err)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader([]byte(`{"ok":true,"hear":"sounds/asset.wav"}`))),
				Request:    r,
			}, nil
		})}
		out, err := runCommand(root, client, cmd)
		if err != nil {
			t.Fatal(err)
		}
		if path != "/2d/steps" || out["op"] != "steps" || out["factory"] == nil {
			t.Fatalf("path=%s out=%v", path, out)
		}
		return got
	}
	got := post(`{
		"op":"steps",
		"stem":"spider-mech",
		"when":"home",
		"prompt":"top down spider mech",
		"steps":["sound"],
		"pieces":"hull, legs",
		"motion":"Walk cycle",
		"say":"hold",
		"sound":"clank",
		"speaker":"af_heart",
		"part":"asset",
		"video":"",
		"frames":[],
		"fps":10,
		"count":8
	}`)
	for _, key := range []string{"when", "steps", "pieces", "motion", "say", "sound"} {
		if _, ok := got[key]; !ok {
			t.Fatalf("dropped %s in %v", key, got)
		}
	}
	if got["when"] != "home" || got["sound"] != "clank" || got["motion"] != "Walk cycle" || got["pieces"] != "hull, legs" || got["say"] != "hold" {
		t.Fatalf("got=%v", got)
	}
	steps, _ := got["steps"].([]any)
	if len(steps) != 1 || steps[0] != "sound" {
		t.Fatalf("steps=%v", got["steps"])
	}
	if _, ok := got["video"]; ok {
		t.Fatalf("empty video forwarded: %v", got)
	}
	if _, ok := got["frames"]; ok {
		t.Fatalf("empty frames forwarded: %v", got)
	}
	if _, ok := got["op"]; ok {
		t.Fatalf("op forwarded: %v", got)
	}
	withClip := post(`{"op":"steps","stem":"spider-mech","when":"revise","steps":["motion"],"video":"QQ==","frames":["QQ=="]}`)
	if withClip["video"] != "QQ==" {
		t.Fatalf("video=%v", withClip["video"])
	}
	frames, _ := withClip["frames"].([]any)
	if len(frames) != 1 || frames[0] != "QQ==" {
		t.Fatalf("frames=%v", withClip["frames"])
	}
}
