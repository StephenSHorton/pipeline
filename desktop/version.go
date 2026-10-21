package main

import "strings"

// formatVersion turns the ldflag/git-describe string into the label shown
// in the window title and UI (e.g. "v0.0.3"). CI passes 0.0.3; local builds
// may pass `git describe --tags --always --dirty`.
func formatVersion(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		v = "dev"
	}
	if v == "dev" {
		return v
	}
	if strings.HasPrefix(v, "v") || strings.HasPrefix(v, "V") {
		return v
	}
	return "v" + v
}

func windowTitle(v string) string {
	return "Pipeline " + formatVersion(v)
}
