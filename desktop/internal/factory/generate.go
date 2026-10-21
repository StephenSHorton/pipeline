package factory

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/StephenSHorton/pipeline/desktop/internal/sidecar"
)

// NoPicture is the one message for 3D asked for without a picture (scripts/factory_env.py NO_PICTURE).
const NoPicture = "3D needs a picture. Draw or add one first."

// StartGenerate posts the generate job to the in-process factory sidecar.
// The image path is a factory-relative path from SaveUpload (e.g. in/foo.jpg).
func (s *Service) StartGenerate(image, stem, quality, poly string) (map[string]any, error) {
	if image == "" {
		return nil, errors.New(NoPicture)
	}
	if quality == "" {
		quality = "stack"
	}
	if poly == "" {
		poly = "medium"
	}
	payload, err := json.Marshal(map[string]string{
		"image":   image,
		"stem":    stem,
		"quality": quality,
		"poly":    poly,
	})
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 15 * time.Second}
	url := "http://127.0.0.1:" + strconv.Itoa(sidecar.Port) + "/generate"
	resp, err := client.Post(url, "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("factory generate: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("factory generate: %s", bytes.TrimSpace(body))
	}
	if resp.StatusCode >= 400 {
		errStr, _ := out["error"].(string)
		if errStr == "" {
			errStr = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
		return out, fmt.Errorf("%s", errStr)
	}
	return out, nil
}
