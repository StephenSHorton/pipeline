// Same cases as tests/test_desktop_shell.py (test_degraded_is_one_plain_line_on_job_and_asset_cards,
// test_shape_mv_tooltip_reads_the_probed_features), which run the legacy page's functions under node.
import { describe, expect, it } from "vitest";
import { degradedLine, healthLabel, overlayPct, overlayPctText, shapeMvNote } from "./jobs";

describe("degradedLine", () => {
  it("is one plain-English line, nothing when there is nothing", () => {
    expect([
      degradedLine([
        { stage: "shape", code: "single_view", message: "one picture" },
        { stage: "rig", code: "clips_dropped", message: "Flap,Glide" },
      ]),
      degradedLine([]),
      degradedLine(undefined),
      degradedLine([{}]),
      degradedLine([{ stage: "views", code: "stale", message: "" }]),
      degradedLine([{ stage: "psx", code: "legs_clean", message: "no legs to replace" }]),
      degradedLine([{ stage: "psx", code: "facing_check" }]),
    ]).toEqual([
      "Built from one view, not four · Clips left out: Flap, Glide",
      "",
      "",
      "",
      "The drawing changed since this was built",
      "No legs to replace",
      "Facing check",
    ]);
  });

  it("shows the message alone when there is no stage or code", () => {
    expect(degradedLine([{ message: "x" }])).toBe("X");
  });
});

describe("shapeMvNote", () => {
  const got = [
    shapeMvNote(undefined),
    shapeMvNote({}),
    shapeMvNote({ shape_mv: true, shape_mv_venv: true, shape_mv_weights: true }),
    shapeMvNote({ shape_mv: false, shape_mv_venv: false, shape_mv_weights: true }),
    shapeMvNote({ shape_mv: false, shape_mv_venv: true, shape_mv_weights: false }),
    shapeMvNote({ shape_mv: false, shape_mv_venv: false, shape_mv_weights: false }),
  ];

  it("says nothing without the probe", () => {
    expect(got[0]).toBe("");
    expect(got[1]).toBe("");
  });

  it("names what is installed or missing", () => {
    expect(got[2]).toContain("is installed; it is used when a drawing's four views match it");
    expect(got[3]).toContain("its Python env is not installed");
    expect(got[4]).toContain("its weights are missing");
    expect(got[5]).toContain("its Python env and weights are missing");
    for (const off of got.slice(3)) expect(off).toContain("the shape comes from the first picture; extra views only photo-bake");
  });
});

describe("factory pill and job bar", () => {
  it("labels health like the legacy pill", () => {
    expect(healthLabel(null)).toBe("idle");
    expect(healthLabel({ ok: true, service: "ai-asset-pipeline", busy: true, job_id: "j" })).toBe("busy");
    expect(healthLabel({ ok: true, service: "ai-asset-pipeline", busy: false, job_id: null, backend: "mlx" })).toBe("Apple Silicon / MLX");
  });

  it("weights the bar by last run's stage times and says the time left", () => {
    const progress = { stages: [{ key: "a", label: "A" }, { key: "b", label: "B" }], stage: "b", stage_index: 1, stage_count: 2, sub: 0.5 };
    const timing = { stage_est: { a: 10, b: 30 } };
    expect(overlayPct(progress, "running", timing)).toBe(63); // (10 + 30*0.5) / 40
    expect(overlayPctText(progress, "running", 63, timing)).toBe("B 2/2 · 63% · about 15 s left");
    expect(overlayPct(progress, "ok", timing)).toBe(100);
    expect(overlayPctText(progress, "cancelled", 63, timing)).toBe("Cancelled");
  });
});
