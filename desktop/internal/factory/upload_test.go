package factory

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDecodeUploadJSONBase64(t *testing.T) {
	raw := []byte("\xff\xd8\xfffakejpeg")
	body, _ := json.Marshal(map[string]string{
		"name": "pipeline-mech.jpg",
		"data": base64.StdEncoding.EncodeToString(raw),
	})
	data, name, err := DecodeUpload("ignored.png", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	if name != "pipeline-mech.jpg" {
		t.Fatalf("name=%q", name)
	}
	if !bytes.Equal(data, raw) {
		t.Fatalf("got %q", data)
	}
}

func TestDecodeUploadEmptyJSON(t *testing.T) {
	_, _, err := DecodeUpload("x.jpg", "application/json", []byte(`{"name":"x.jpg"}`))
	if err == nil || err.Error() != "empty upload" {
		t.Fatalf("err=%v", err)
	}
}

func TestDecodeUploadRawBytes(t *testing.T) {
	raw := []byte("\x89PNG\r\n")
	data, name, err := DecodeUpload("Shot.PNG", "image/png", raw)
	if err != nil {
		t.Fatal(err)
	}
	if name != "Shot.PNG" {
		t.Fatalf("name=%q", name)
	}
	if !bytes.Equal(data, raw) {
		t.Fatalf("got %q", data)
	}
}

func TestLiveUploadAndStartGenerate(t *testing.T) {
	if os.Getenv("PIPELINE_LIVE") == "" {
		t.Skip("set PIPELINE_LIVE=1 to hit a running factory")
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://127.0.0.1:8766/health")
	if err != nil {
		t.Skip("factory sidecar not running")
	}
	_ = resp.Body.Close()
	jpg := filepath.Join(os.Getenv("USERPROFILE"), "Downloads", "pipeline-mech.jpg")
	raw, err := os.ReadFile(jpg)
	if err != nil {
		t.Skip(err)
	}
	svc := New()
	up, err := svc.SaveUpload("pipeline-mech.jpg", raw)
	if err != nil {
		t.Fatal(err)
	}
	if !up.OK || up.Bytes != len(raw) {
		t.Fatalf("%+v", up)
	}
	job, err := svc.StartGenerate(up.Path, "pipeline-mech", "draft", "proxy")
	if err != nil {
		t.Fatal(err)
	}
	id, _ := job["job_id"].(string)
	if id == "" {
		t.Fatalf("no job_id: %+v", job)
	}
	t.Logf("job %s state=%v", id, job["state"])
}

func TestHandleUploadHTTPWritesIn(t *testing.T) {
	root := t.TempDir()
	t.Setenv("PIPELINE_ROOT", root)
	raw := []byte("\xff\xd8\xffmech")
	body, _ := json.Marshal(map[string]string{
		"name": "pipeline-mech.jpg",
		"data": base64.StdEncoding.EncodeToString(raw),
	})
	req := httptest.NewRequest(http.MethodPost, "/upload?name=ignored.jpg", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	HandleUploadHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	var res UploadResult
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.Path != "in/pipeline-mech.jpg" || res.Bytes != len(raw) {
		t.Fatalf("%+v", res)
	}
	got, err := os.ReadFile(filepath.Join(root, "in", "pipeline-mech.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatal("file bytes mismatch")
	}
}
