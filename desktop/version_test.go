package main

import "testing"

func TestFormatVersion(t *testing.T) {
	cases := map[string]string{
		"":                        "dev",
		"dev":                     "dev",
		"0.0.3":                   "v0.0.3",
		"v0.0.3":                  "v0.0.3",
		"v0.0.3-1-g1db6fb3":       "v0.0.3-1-g1db6fb3",
		"v0.0.3-1-g1db6fb3-dirty": "v0.0.3-1-g1db6fb3-dirty",
	}
	for in, want := range cases {
		if got := formatVersion(in); got != want {
			t.Errorf("formatVersion(%q)=%q want %q", in, got, want)
		}
	}
}

func TestWindowTitle(t *testing.T) {
	if got := windowTitle("0.0.3"); got != "Pipeline v0.0.3" {
		t.Fatalf("got %q", got)
	}
	if got := windowTitle("v0.0.4"); got != "Pipeline v0.0.4" {
		t.Fatalf("got %q", got)
	}
}
