// The Rig stage's controls, word for word from the legacy renderFlow rig branch (index.html 2914-2999) and its
// helpers. The "pins" block mirrors the page half of tests/test_desktop_shell.py test_rig_stage_surfaces ("FBX
// checked:" is export.ts's fbxLine, pinned in export.test.ts).
import { describe, expect, it } from "vitest";
import type { FlowEntry } from "@/lib/flow";
import type { RigInspectDoc } from "./inspect";
import {
  advancedSummary,
  ADVANCED0,
  choicesFor,
  choicesOf,
  clipButton,
  clipGateText,
  CLIPS_REPORT,
  didNotLoad,
  didNotLoadNamed,
  finishAction,
  hasTears,
  inspectKey,
  inspectPath,
  madeBadgeTitle,
  mergeRigInfo,
  originalUrl,
  planOptions,
  PLANS_PATH,
  psxFacts,
  recheckView,
  renderPath,
  RIG_ADVANCED,
  RIG_COPY,
  rigBody,
  rigButton,
  rigCompare,
  rigFacts,
  rigKeptHead,
  rigMadeFrom,
  rigPath,
  rigReadyText,
  rigStatus,
  shownChoice,
  validInspect,
  type RigMade,
} from "./rig";

const doc = (over: Partial<RigInspectDoc> = {}): RigInspectDoc => ({ bones: [], ...over });
const made = (plan = "ground_bird", clips: RigMade["clips"] = {}): RigMade => ({ stem: "cow", plan, clips, failed: [] });

describe("pins (test_rig_stage_surfaces)", () => {
  it("the routes: /rig/plans, /rig/<stem>, /inspect, /render, clips_report.json", () => {
    expect(PLANS_PATH).toBe("/rig/plans");
    expect(rigPath("big cow")).toBe("/rig/big%20cow");
    expect(inspectPath("cow")).toBe("/rig/cow/inspect");
    expect(renderPath("cow")).toBe("/rig/cow/render");
    expect(CLIPS_REPORT).toBe("rig/template/clips_report.json");
  });

  it("made for this model, on the clips and the badge", () => {
    expect(clipButton("Walk", null, made("ground_bird", { Walk: { gates: { stretch_p99: { value: 2.5, limit: 6, ok: true } } } })).title).toBe(
      "Play Walk · made for this model\n✓ stretch 2.5 (limit 6)",
    );
    expect(RIG_COPY.madeBadge).toBe("Clips made for this model");
    expect(madeBadgeTitle("ground_bird")).toBe("A skeleton was built for this model (ground_bird) and every clip passed its checks on it.");
    expect(madeBadgeTitle("")).toBe("A skeleton was built for this model (its body plan) and every clip passed its checks on it.");
  });

  it("the Advanced fold: template, binder, pred, with their words", () => {
    expect(RIG_ADVANCED.map((a) => a[0])).toEqual(["template", "binder", "pred"]);
    expect(RIG_ADVANCED).toEqual([
      ["template", "Skeleton", [["", "auto"], ["build", "build for the model"], ["fixed", "the body plan's template"]],
        "auto builds a skeleton for the model when its body plan can (a ground bird), else uses the body plan's template."],
      ["binder", "Skin weights", [["", "skin-tokens"], ["pred", "predicted rig's"], ["heat", "Blender heat"]], "How the skin weights are learned."],
      ["pred", "Predicted skeleton", [["", "auto"], ["always", "always"], ["never", "never"]],
        "Use the model's own predicted skeleton as the fitting target: auto (when the match is poor), always, never."],
    ]);
    expect(advancedSummary(ADVANCED0)).toBe("Advanced");
    expect(advancedSummary({ template: "build", binder: "", pred: "never" })).toBe("Advanced · 2");
    expect(RIG_COPY.advancedTitle).toBe("Skeleton, skin weights and predicted skeleton for the next Re-rig.");
  });
});

describe("bar copy", () => {
  it("is word for word", () => {
    expect(RIG_COPY).toEqual({
      noMesh: "Make the model first, then rig it here.",
      clips: "Clips",
      pass: "Rig looks right",
      passTitle: "Every check passed. Click to see how it was rigged.",
      problemsTail: "Click to see it on the model.",
      checking: "Checking…",
      checkVersion: "Check this version",
      checkAgainTitle: "This version was checked before its clips were measured. Check it again and play every clip. Takes a few seconds.",
      checkFirstTitle: "Work out the rig checks for this version and measure its clips. Takes a few seconds.",
      madeBadge: "Clips made for this model",
      planTitle: "The body plan decides which skeleton the model gets, and which clips.",
      advancedTitle: "Skeleton, skin weights and predicted skeleton for the next Re-rig.",
      rigIt: "Rig it",
      reRig: "Re-rig",
    });
  });
});

describe("clips", () => {
  it("tears and stretch titles", () => {
    const d = doc({ motion: { clips: { Walk_Fast: { tears: 3, stretch_p99: 2.345 }, Idle: { tears: 0, stretch_max: 1.04 } } } });
    expect(clipButton("Walk_Fast", d, null)).toEqual({ text: "Walk Fast", title: "Walk Fast · tears at 3 points · 1% of edges stretch 2.3×", tears: true, made: false });
    expect(clipButton("Idle", d, null).title).toBe("Play Idle · no tears · worst stretch 1.0×");
    expect(clipButton("Hop", d, null).title).toBe("Play Hop");
  });

  it("clipGateText: words, fmt, tries only past one, energy only when set", () => {
    expect(
      clipGateText({
        gates: {
          stretch_p99: { value: 2.5, limit: 6, ok: true },
          floor_min_H: { value: -0.0125, limit: [-0.02, 0.02], ok: false },
          wing_rise_H: { value: 12.4 },
          odd_gate_name: { value: 0.1 },
          skipped: { value: null },
          bare: 3,
        },
        attempts: [1, 2],
        energy: 0.5,
      }),
    ).toBe("✓ stretch 2.5 (limit 6)\n✗ floor -0.013 (limit -0.02–0.02)\n✓ wing skin rises 12\n✓ odd gate name 0.1\nenergy 0.5 · 2 tries");
    expect(clipGateText({ gates: {}, attempts: [1] , energy: 10.6 })).toBe("energy 11");
    expect(clipGateText({})).toBe("");
  });
});

describe("status and recheck", () => {
  it("the worst finding leads; pass says Rig looks right", () => {
    const bad = doc({ checks: [{ key: "planner", level: "warn", label: "Plan unsure" }, { key: "seam", level: "fail", label: "Tears at 4 points" }] });
    expect(rigStatus(bad)).toEqual({ level: "fail", key: "seam", label: "Tears at 4 points", title: "Tears at 4 points\nPlan unsure\nClick to see it on the model." });
    expect(rigStatus(doc({ checks: [{ key: "seam", level: "pass", label: "No tears" }] }))).toEqual({
      level: "pass",
      key: "",
      label: "Rig looks right",
      title: "Every check passed. Click to see how it was rigged.",
    });
    expect(rigStatus(doc())).toBeNull();
    expect(hasTears(bad)).toBe(true);
  });

  it("Check this version: no checks, or checks before clips were measured", () => {
    expect(recheckView({ rigged: true, loaded: true, doc: null, checking: false })).toEqual({ kind: "button", title: RIG_COPY.checkFirstTitle });
    expect(recheckView({ rigged: true, loaded: true, doc: doc(), checking: false })).toEqual({ kind: "button", title: RIG_COPY.checkAgainTitle });
    expect(recheckView({ rigged: true, loaded: true, doc: doc({ motion: { clips: {} } }), checking: false }).kind).toBe("none");
    expect(recheckView({ rigged: true, loaded: true, doc: doc({ motion: { clips: {} } }), checking: true }).kind).toBe("checking");
    expect(recheckView({ rigged: false, loaded: true, doc: null, checking: false }).kind).toBe("none");
    expect(recheckView({ rigged: true, loaded: false, doc: null, checking: false }).kind).toBe("none");
  });
});

describe("body plan", () => {
  const plans = [
    { id: "ground_bird", label: "Ground bird", blurb: "", class: "avian", aliases: [], built: true, fixed: false, clips: [] },
    { id: "horse", label: "Horse", blurb: "", class: "quadruped", aliases: [], built: false, fixed: true, clips: [] },
  ];
  it("plans when the factory has them, RIG_NAMES otherwise", () => {
    expect(planOptions(plans)).toEqual([["", "Body plan: auto"], ["plan:ground_bird", "Ground bird · built for the model"], ["plan:horse", "Horse"]]);
    expect(planOptions([])).toEqual([["", "Body plan: auto"], ["humanoid", "Person"], ["quadruped", "Four legs"], ["multileg", "Many legs"], ["avian", "Bird"]]);
    expect(planOptions(null)).toEqual(planOptions([]));
  });

  it("shows the built plan until you pick one; an unknown value shows auto", () => {
    const opts = planOptions(plans);
    expect(shownChoice("", made("ground_bird"), opts)).toBe("plan:ground_bird");
    expect(shownChoice("plan:horse", made("ground_bird"), opts)).toBe("plan:horse");
    expect(shownChoice("", null, opts)).toBe("");
    expect(shownChoice("", made("spider"), opts)).toBe("");
  });

  it("choices are per asset (legacy rigClassChoice / rigAdvanced carried over)", () => {
    const a = { ...choicesFor("cow"), choice: "plan:horse", advanced: { template: "build", binder: "", pred: "" }, advancedOpen: true };
    expect(choicesOf(a, "cow")).toBe(a);
    expect(choicesOf(a, "pig")).toEqual({ stem: "pig", choice: "", advanced: { template: "", binder: "", pred: "" }, advancedOpen: false });
  });
});

describe("Rig it / Re-rig", () => {
  it("the button", () => {
    expect(rigButton(false, false, false)).toEqual({
      label: "Rig it",
      title: "Fit a skeleton to the model, learn its skin weights and bake library clips. About 1-2 minutes.",
      primary: true,
    });
    expect(rigButton(true, false, true)).toEqual({
      label: "Re-rig",
      title: "Fit a skeleton to the model, learn its skin weights and bake library clips. About 1-2 minutes. Head kept as made.",
      primary: false,
    });
    expect(rigButton(true, true, false).primary).toBe(true);
  });

  it("the body: auto, plan, class, Advanced, straighten", () => {
    expect(rigBody("", ADVANCED0)).toEqual({});
    expect(rigBody("plan:horse", { template: "fixed", binder: "", pred: "always" })).toEqual({ plan: "horse", template: "fixed", pred: "always" });
    expect(rigBody("avian", ADVANCED0)).toEqual({ class: "avian" });
    expect(rigBody("", ADVANCED0, { keptHead: true })).toEqual({ straighten: "off" });
    expect(rigBody("", ADVANCED0, { keptHead: true, straighten: "auto" })).toEqual({ straighten: "auto" });
  });

  it("the panel's re-rig sends the plan the select shows (legacy sent auto until you picked)", () => {
    const opts = planOptions([{ id: "ground_bird", label: "Ground bird", blurb: "", class: "avian", aliases: [], built: true, fixed: false, clips: [] }]);
    expect(rigBody(shownChoice("", made("ground_bird"), opts), ADVANCED0, { straighten: "off" })).toEqual({ plan: "ground_bird", straighten: "off" });
  });

  it("a head kept as made", () => {
    expect(rigKeptHead({ measured: { turned: true, head_yaw_deg: 30 } })).toBe(true);
    expect(rigKeptHead({ measured: { turned: true, head_yaw_deg: 30 }, straightened: { head_yaw_deg: 0 } })).toBe(false);
    expect(rigKeptHead({ measured: { turned: true, head_yaw_deg: 30 }, before_straightening: true })).toBe(false);
    expect(rigKeptHead(null)).toBe(false);
  });
});

describe("rig.json and the built template", () => {
  it("made only for a built template; rig.json's rest fills an old inspect.json", () => {
    expect(rigMadeFrom("cow", { template: { built: true, plan: "ground_bird" }, clips_failed: ["Fly"] }, { clips: { Walk: {} } })).toEqual({
      stem: "cow",
      plan: "ground_bird",
      clips: { Walk: {} },
      failed: ["Fly"],
    });
    expect(rigMadeFrom("cow", { template: "rigs/quad.glb" } as never, null)).toBeNull();
    const rest = { measured: { turned: false, head_yaw_deg: 1 } };
    expect(mergeRigInfo(doc({ rig: { plan: "x" } }), { class: "avian", rest }).rig).toEqual({ class: "avian", plan: "x", rest });
    expect(validInspect({ bones: [] })).toBe(true);
    expect(validInspect({ checks: [] })).toBe(false);
  });
});

describe("versions", () => {
  const entry = (over: Partial<FlowEntry> = {}): FlowEntry => ({
    stage: "rig",
    pick: "t2",
    takes: [
      { id: "t1", made: 1, files: [], from: { mesh: "m1" }, summary: { tears: 3 } },
      { id: "t2", made: 2, files: [], from: { mesh: "m2" }, summary: { tears: 0 } },
    ],
    live: true,
    files: [],
    unsaved: false,
    stale: [],
    upstream: [],
    ...over,
  });
  it("the version before the picked one", () => {
    expect(rigCompare(entry(), null)).toEqual({ n: 1, id: "t1", summary: { tears: 3 }, current: { tears: 0 }, sameModel: false });
    expect(rigCompare(entry({ pick: "t1" }), null)).toBeNull();
    expect(rigCompare(entry({ unsaved: true }), null)).toBeNull();
  });

  it("the original model: this mesh, or the mesh version the rig came from", () => {
    const info = { source: "mesh.glb", rest: { measured: { turned: true, head_yaw_deg: 30 } } };
    expect(originalUrl("cow", info, entry().takes[1], "m2")).toBe("/out/cow/mesh.glb");
    expect(originalUrl("cow", info, entry().takes[1], "m1")).toBe("/out/cow/takes/mesh/m2/mesh.glb");
    expect(originalUrl("cow", { source: "mesh.glb" }, null, "m1")).toBe("");
  });
});

describe("finish", () => {
  it("Rig ready with rig.json facts, the summary line and degraded", () => {
    expect(rigFacts({ class: "quadruped", clips: ["a", "b"], fbx: "rig/cow.fbx" })).toBe(" · Four legs, 2 clips, FBX for s&box");
    expect(rigFacts({ class: "ground_thing", clips: [] })).toBe(" · ground_thing, 0 clips");
    expect(rigFacts(null)).toBe("");
    expect(psxFacts({ triangles: 1234, atlas: 128 })).toBe(" · 1,234 triangles, 128 px texture");
    expect(psxFacts(null)).toBe("");
    expect(rigReadyText("big cow", " · Four legs, 2 clips", "Tears at 4 points", "Degraded: rig clips_dropped: Fly")).toBe(
      "Rig ready: big cow · Four legs, 2 clips · Tears at 4 points · Degraded: rig clips_dropped: Fly",
    );
    expect(rigReadyText("cow", "", "", "")).toBe("Rig ready: cow");
  });

  it("Inspect a failing rig, Export otherwise, Open in the background", () => {
    expect(finishAction({ here: true, level: "fail" })).toBe("Inspect");
    expect(finishAction({ here: true, level: "warn" })).toBe("Export →");
    expect(finishAction({ here: false, level: "fail" })).toBe("Open");
    expect(inspectKey([{ level: "warn", key: "planner" }, { level: "fail", key: "fit" }, { level: "fail", key: "seam" }])).toBe("seam");
    expect(inspectKey(undefined)).toBe("");
  });

  it("the model did not load", () => {
    expect(didNotLoad("no glb loaded")).toBe("It finished, but the model did not load: no glb loaded");
    expect(didNotLoadNamed("big cow", "no glb loaded")).toBe("big cow finished, but the model did not load: no glb loaded");
  });
});
