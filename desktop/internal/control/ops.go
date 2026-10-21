package control

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var stemRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

// Tools is the agent catalog. The window uses the same routes.
func Tools() []map[string]string {
	return []map[string]string{
		{"method": "GET", "path": "/control/workflow?stem=", "does": "2d or 3d for an asset, from disk"},
		{"method": "GET", "path": "/control/sheet?stem=&part=", "does": "parts, note, and the audio subject the window would show"},
		{"method": "POST", "path": "/control", "does": "op sheet, split, drop, pack, save, sound, speak, or steps"},
	}
}

type command struct {
	Op      string          `json:"op"`
	Stem    string          `json:"stem"`
	Part    string          `json:"part"`
	Line    string          `json:"line"`
	Speaker string          `json:"speaker"`
	Effect  string          `json:"effect"`
	Data    string          `json:"data"`
	Note    string          `json:"note"`
	Parts   json.RawMessage `json:"parts"`
	Action  string          `json:"action"`
	Prompt  string          `json:"prompt"`
	Frames  []string        `json:"frames"`
	Video   string          `json:"video"`
	Fps     int             `json:"fps"`
	Count   int             `json:"count"`
	When    string          `json:"when"`
	Steps   []string        `json:"steps"`
	Pieces  string          `json:"pieces"`
	Motion  string          `json:"motion"`
	Say     string          `json:"say"`
	Sound   string          `json:"sound"`
}

func serveControl(w http.ResponseWriter, r *http.Request, root string, client *http.Client) {
	switch {
	case r.URL.Path == "/control/tools" && r.Method == http.MethodGet:
		writeJSON(w, map[string]any{"ok": true, "tools": Tools()})
	case r.URL.Path == "/control/workflow" && r.Method == http.MethodGet:
		view, err := workflowView(root, r.URL.Query().Get("stem"))
		writeResult(w, view, err)
	case r.URL.Path == "/control/sheet" && r.Method == http.MethodGet:
		view, err := sheetView(root, r.URL.Query().Get("stem"), r.URL.Query().Get("part"))
		writeResult(w, view, err)
	case r.URL.Path == "/control" && r.Method == http.MethodPost:
		var cmd command
		if err := json.NewDecoder(io.LimitReader(r.Body, 32<<20)).Decode(&cmd); err != nil {
			writeJSONCode(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "invalid json"})
			return
		}
		out, err := runCommand(root, client, cmd)
		writeResult(w, out, err)
	default:
		writeJSONCode(w, http.StatusNotFound, map[string]any{"ok": false, "error": "not a control route"})
	}
}

func runCommand(root string, client *http.Client, cmd command) (map[string]any, error) {
	stem, err := cleanStem(cmd.Stem)
	if err != nil {
		return nil, err
	}
	op := strings.TrimSpace(cmd.Op)
	if op == "" || op == "sheet" {
		return sheetView(root, stem, cmd.Part)
	}
	if client == nil {
		client = &http.Client{Timeout: 3 * time.Minute}
	}
	var factory any
	switch op {
	case "pack":
		body := map[string]any{"stem": stem, "action": cmd.Action, "fps": cmd.Fps, "count": cmd.Count}
		if cmd.Video != "" {
			body["video"] = cmd.Video
		}
		if len(cmd.Frames) > 0 {
			body["frames"] = cmd.Frames
		}
		factory, err = factoryPost(client, "/2d/pack", body)
	case "split":
		body := map[string]any{"stem": stem}
		if strings.TrimSpace(cmd.Prompt) != "" {
			body["prompt"] = cmd.Prompt
		}
		factory, err = factoryPost(client, "/2d/split", body)
	case "drop":
		if strings.TrimSpace(cmd.Data) == "" {
			return nil, errBad("empty image")
		}
		factory, err = factoryPost(client, "/2d/drop", map[string]any{"stem": stem, "data": cmd.Data})
	case "save":
		factory, err = factoryPost(client, "/2d/save", map[string]any{"stem": stem, "note": cmd.Note, "parts": json.RawMessage(cmd.Parts)})
	case "sound", "speak":
		sheet, serr := sheetView(root, stem, cmd.Part)
		if serr != nil {
			return nil, serr
		}
		audio, _ := sheet["audio"].(map[string]any)
		if audio == nil {
			return nil, errBad("no audio subject")
		}
		part := map[string]any{
			"id":         audio["id"],
			"file":       audio["file"],
			"name":       audio["name"],
			"line":       firstText(cmd.Line, str(audio["line"])),
			"speaker":    firstText(cmd.Speaker, str(audio["speaker"])),
			"projectile": firstText(cmd.Effect, str(audio["effect"])),
		}
		path := "/2d/sound"
		if op == "speak" {
			path = "/2d/speak"
		}
		factory, err = factoryPost(client, path, map[string]any{"stem": stem, "part": part})
	case "steps":
		steps := cmd.Steps
		if steps == nil {
			steps = []string{}
		}
		body := map[string]any{
			"stem":    stem,
			"when":    cmd.When,
			"prompt":  cmd.Prompt,
			"steps":   steps,
			"pieces":  cmd.Pieces,
			"motion":  cmd.Motion,
			"say":     cmd.Say,
			"sound":   cmd.Sound,
			"speaker": cmd.Speaker,
			"part":    cmd.Part,
			"fps":     cmd.Fps,
			"count":   cmd.Count,
		}
		if strings.TrimSpace(cmd.Video) != "" {
			body["video"] = cmd.Video
		}
		if len(cmd.Frames) > 0 {
			body["frames"] = cmd.Frames
		}
		factory, err = factoryPost(client, "/2d/steps", body)
	default:
		return nil, errBad("unknown op " + op)
	}
	if err != nil {
		return nil, err
	}
	sheet, serr := sheetView(root, stem, cmd.Part)
	if serr != nil {
		return map[string]any{"ok": true, "op": op, "factory": factory}, nil
	}
	sheet["op"] = op
	sheet["factory"] = factory
	return sheet, nil
}

func workflowView(root, stem string) (map[string]any, error) {
	name, err := cleanStem(stem)
	if err != nil {
		return nil, err
	}
	hasMesh, hasFlat := assetFacts(root, name)
	workflow := ""
	switch {
	case hasFlat && !hasMesh:
		workflow = "2d"
	case hasMesh && !hasFlat:
		workflow = "3d"
	case !hasMesh && !hasFlat:
		workflow = "3d"
	}
	return map[string]any{
		"ok":       true,
		"stem":     name,
		"hasMesh":  hasMesh,
		"hasFlat":  hasFlat,
		"workflow": workflow,
		"choose":   hasMesh && hasFlat,
	}, nil
}

func sheetView(root, stem, part string) (map[string]any, error) {
	base, err := workflowView(root, stem)
	if err != nil {
		return nil, err
	}
	name := base["stem"].(string)
	parts, note := mergedParts(root, name)
	base["note"] = note
	base["source"] = fileExists(filepath.Join(root, "out", name, "flat", "source.png"))
	base["parts"] = parts
	audio, who := audioSubject(base["workflow"].(string), parts, part)
	base["audio"] = audio
	base["who"] = who
	manifest := readJSON(filepath.Join(root, "out", name, "flat", "sheets", "manifest.json"))
	if actions := manifest["actions"]; actions != nil {
		base["actions"] = actions
	} else {
		base["actions"] = map[string]any{}
	}
	return base, nil
}

func audioSubject(workflow string, parts []map[string]any, want string) (map[string]any, string) {
	want = strings.TrimSpace(want)
	if want != "" {
		for _, p := range parts {
			if str(p["id"]) == want || str(p["name"]) == want {
				return labelAudio(p)
			}
		}
		if workflow == "2d" {
			return nil, "Select a part"
		}
	}
	if workflow != "2d" {
		for _, p := range parts {
			if str(p["id"]) == "asset" {
				return labelAudio(p)
			}
		}
		return map[string]any{"id": "asset", "speaker": "af_heart", "line": "", "effect": "", "sound": "", "voice": ""}, "This asset"
	}
	if len(parts) == 0 {
		return nil, "Select a part"
	}
	return labelAudio(parts[0])
}

func labelAudio(p map[string]any) (map[string]any, string) {
	p["effect"] = effectOf(p)
	if str(p["id"]) == "asset" {
		return p, "This asset"
	}
	who := str(p["name"])
	if who == "" {
		who = str(p["id"])
	}
	return p, who
}

func mergedParts(root, stem string) ([]map[string]any, string) {
	flat := readJSON(filepath.Join(root, "out", stem, "flat", "parts.json"))
	mech := readJSON(filepath.Join(root, "out", stem, "flat", "mechanics.json"))
	note := str(mech["note"])
	if note == "" {
		note = str(flat["note"])
	}
	by := map[string]map[string]any{}
	for _, row := range objectList(mech["parts"]) {
		if id := str(row["id"]); id != "" {
			by[id] = row
		}
		if file := str(row["file"]); file != "" {
			by[file] = row
		}
	}
	var out []map[string]any
	for _, p := range objectList(flat["parts"]) {
		extra := by[str(p["file"])]
		if extra == nil {
			extra = by[str(p["id"])]
		}
		out = append(out, mergePart(p, extra))
	}
	seen := map[string]bool{}
	for _, p := range out {
		if id := str(p["id"]); id != "" {
			seen[id] = true
		}
	}
	for _, row := range objectList(mech["parts"]) {
		id := str(row["id"])
		if id != "" && !seen[id] {
			out = append(out, mergePart(row, nil))
		}
	}
	if out == nil {
		out = []map[string]any{}
	}
	return out, note
}

func mergePart(p, extra map[string]any) map[string]any {
	row := map[string]any{}
	for k, v := range p {
		row[k] = v
	}
	if extra == nil {
		row["effect"] = effectOf(row)
		return row
	}
	for _, key := range []string{"name", "role", "motion", "projectile", "line", "speaker", "sound", "voice", "id"} {
		if str(extra[key]) != "" {
			row[key] = extra[key]
		}
	}
	row["effect"] = effectOf(row)
	return row
}

func effectOf(p map[string]any) string {
	if v := str(p["projectile"]); v != "" {
		return v
	}
	return str(p["motion"])
}

func assetFacts(root, stem string) (hasMesh, hasFlat bool) {
	dir := filepath.Join(root, "out", stem)
	hasFlat = fileExists(filepath.Join(dir, "flat", "source.png")) ||
		fileExists(filepath.Join(dir, "flat", "parts.json")) ||
		fileExists(filepath.Join(dir, "flat", "sheets", "manifest.json"))
	if fileExists(filepath.Join(dir, "lods.json")) {
		hasMesh = true
		return hasMesh, hasFlat
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.glb"))
	hasMesh = len(matches) > 0
	return hasMesh, hasFlat
}

func factoryPost(client *http.Client, path string, payload any) (any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:8766"+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	var parsed any
	if json.Unmarshal(raw, &parsed) != nil {
		parsed = map[string]any{"ok": false, "status": resp.StatusCode, "body": string(raw)}
	}
	if resp.StatusCode >= 400 {
		msg := "factory " + resp.Status
		if m, ok := parsed.(map[string]any); ok && str(m["error"]) != "" {
			msg = str(m["error"])
		}
		return parsed, errBad(msg)
	}
	return parsed, nil
}

func cleanStem(stem string) (string, error) {
	stem = strings.TrimSpace(stem)
	if !stemRE.MatchString(stem) {
		return "", errBad("invalid stem")
	}
	return stem, nil
}

func readJSON(path string) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}
	}
	var payload map[string]any
	if json.Unmarshal(b, &payload) != nil || payload == nil {
		return map[string]any{}
	}
	return payload
}

func objectList(v any) []map[string]any {
	items, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []map[string]any
	for _, item := range items {
		row, ok := item.(map[string]any)
		if ok {
			out = append(out, row)
		}
	}
	return out
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func firstText(given, fallback string) string {
	if strings.TrimSpace(given) != "" {
		return strings.TrimSpace(given)
	}
	return fallback
}

type badRequest struct{ msg string }

func (e badRequest) Error() string { return e.msg }

func errBad(msg string) error { return badRequest{msg: msg} }

func writeResult(w http.ResponseWriter, payload map[string]any, err error) {
	if err == nil {
		if payload == nil {
			payload = map[string]any{"ok": true}
		}
		writeJSON(w, payload)
		return
	}
	var bad badRequest
	code := http.StatusInternalServerError
	if errors.As(err, &bad) {
		code = http.StatusBadRequest
	}
	writeJSONCode(w, code, map[string]any{"ok": false, "error": err.Error()})
}
