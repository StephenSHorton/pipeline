// The Rig check panel's copy and rules, word for word from public/rig-inspect.js. The "pins" block mirrors
// tests/test_desktop_shell.py test_rig_stage_surfaces (the rig-inspect.js half), so that test can move here.
import { describe, expect, it } from "vitest";
import {
  ARM,
  boneCardRows,
  boneDepth,
  boneFound,
  builtBadge,
  canShow,
  checkBody,
  CHECK_HELP,
  compareView,
  fitRows,
  headBlock,
  headWords,
  helpText,
  INSPECT_COPY,
  KEY_STEP,
  LEG,
  legendParts,
  MINOR,
  MODES,
  MOTION_HELP,
  motionNotes,
  NO_FIT,
  niceNames,
  pair,
  planRows,
  PLANS,
  rigChecks,
  roleHex,
  ROLE_RGB,
  SHEET0,
  SHEET_COPY,
  SHEET_FILE,
  sheetView,
  sideOf,
  stepGroups,
  STEPS,
  stickRoles,
  VIEWS,
  worstCheck,
  worstClip,
  worstFirst,
  type RigCheck,
  type RigInspectDoc,
} from "./inspect";

const doc = (over: Partial<RigInspectDoc> = {}): RigInspectDoc => ({ bones: [{ name: "root", parent: -1 }, { name: "head", parent: 0, label: "head" }], ...over });
const chk = (key: string, level: string, more: Partial<RigCheck> = {}): RigCheck => ({ key, level, label: `${key} ${level}`, text: `${key} text`, ...more });

describe("pins (test_rig_stage_surfaces)", () => {
  it("the LEG, ARM and NO_FIT patterns are verbatim", () => {
    expect(`const LEG = /${LEG.source}/${LEG.flags};`).toBe("const LEG = /leg|foot|toe|thigh|shin|calf|knee|hoof|ball|humerus|femur|tibia/i;");
    expect(`const ARM = /${ARM.source}/${ARM.flags};`).toBe("const ARM = /arm|hand|shoulder|clavicle|elbow|wrist|finger/i;");
    expect(`const NO_FIT = /${NO_FIT.source}/${NO_FIT.flags};`).toBe("const NO_FIT = /leaf|tip|_end$|^ear/i;");
    expect(MINOR.source).toBe("leaf|tip|_end$");
    expect(MINOR.flags).toBe("");
  });

  it("the newer checks map to skeleton, skin or motion, and seam stays skin", () => {
    for (const key of ["template", "built", "a2", "foot", "parts", "cshape", "slide", "loop_seam", "fbx"]) expect(["skeleton", "skin", "motion"]).toContain(KEY_STEP[key]);
    expect(KEY_STEP.seam).toBe("skin");
    expect(KEY_STEP).toEqual({
      planner: "plan", rest: "rest", head: "skeleton", fit: "skeleton", moved: "skeleton", facing: "skeleton", match: "skeleton",
      pose: "skeleton", pred: "skeleton", template: "skeleton", built: "skeleton", seam: "skin", folded: "skin", a2: "skin", foot: "skin",
      parts: "skin", clips: "motion", motion: "motion", far: "motion", stretch: "motion", floor: "motion", unsupported: "motion",
      cshape: "motion", slide: "motion", loop_seam: "motion", fbx: "motion",
    });
  });

  it("ROLE_RGB matches rig_render.py", () => {
    expect(ROLE_RGB).toEqual({ leg: [26, 102, 255], wing: [255, 128, 0], spine: [255, 255, 26], head: [255, 26, 255], tail: [26, 255, 77], other: [240, 234, 220], root: [110, 110, 110] });
    expect(roleHex("leg")).toBe("#1a66ff");
    expect(roleHex("mystery")).toBe("#f0eadc");
  });

  it("Skeleton built for this model, with and without a plan", () => {
    expect(builtBadge(doc())).toBeNull();
    expect(builtBadge(doc({ template: { built: true, plan: "ground_bird" } }))).toEqual({
      text: "Skeleton built for this model · Ground bird",
      title: "rig/template/: a skeleton and clips made for this model from the ground bird body plan.",
    });
    expect(builtBadge(doc({ template: { built: true } }))).toEqual({ text: "Skeleton built for this model", title: "rig/template/: a skeleton and clips made for this model." });
  });

  it("the Sheet tab and contact_sheet.png", () => {
    expect(SHEET_FILE).toBe("rig/preview/contact_sheet.png");
    expect(INSPECT_COPY.sheet).toBe("Sheet");
    expect(INSPECT_COPY.sheetTitle).toBe("Every clip rendered on the CPU from fixed cameras: the contact sheet.");
    expect(INSPECT_COPY.checks).toBe("Checks");
    expect(INSPECT_COPY.checksTitle).toBe("What the rigger decided and how it measured.");
  });
});

describe("constants", () => {
  it("modes, views, steps and plan names", () => {
    expect(MODES.map((m) => m[1])).toEqual(["Texture", "Bones", "One bone", "Dropped", "Split"]);
    expect(MODES[2][2]).toBe("How strongly the picked bone pulls each point: red is fully, blue is not at all.");
    expect(VIEWS.map((v) => v[1])).toEqual(["Side", "Front", "Top", "3/4"]);
    expect(STEPS.map((s) => s[1])).toEqual(["Body plan", "Rest pose", "Skeleton", "Skin", "In motion"]);
    expect(PLANS.biped).toBe("Two legs, two arms");
    expect(PLANS.horse).toBe("Four legs");
    expect(MOTION_HELP).toBe(
      "Every clip was played on this rig and measured. Tears: points whose corners come apart. Stretch: how far the worst 1% of edges grow or shrink; low-poly knees always change some.",
    );
  });
});

describe("rigChecks / worstFirst", () => {
  it("drops fit checks on tip and ear bones, fills step and v1 labels", () => {
    const d = doc({
      checks: [
        { key: "fit", level: "warn", bone: "Ear_L", text: "x" },
        { key: "fit", level: "warn", bone: "toes_tip_L", text: "x" },
        { key: "fit", level: "warn", bone: "Thigh_L", text: "Thigh_L is off", label: "Thigh off" },
        { key: "seam", level: "fail", text: "12 points tear" },
        { key: "head", level: "warn", text: "the head is 23 ° off" },
        { key: "mystery", level: "pass", text: "fine" },
      ],
    });
    const out = rigChecks(d);
    expect(out.map((c) => c.key)).toEqual(["fit", "seam", "head", "mystery"]);
    expect(out[1].label).toBe("Tears at 12 points");
    expect(out[1].step).toBe("skin");
    expect(out[2].label).toBe("Head 23° off its bone");
    expect(out[3]).toMatchObject({ step: "skeleton", label: "fine" });
  });

  it("fail before warn, tears first at a level, then step order", () => {
    const d = doc({ checks: [chk("planner", "warn"), chk("fit", "fail", { bone: "Thigh" }), chk("motion", "fail"), chk("rest", "warn")] });
    expect(rigChecks(d).sort(worstFirst).map((c) => c.key)).toEqual(["motion", "fit", "planner", "rest"]);
    expect(worstCheck(d)?.key).toBe("motion");
    expect(worstCheck(doc())).toBeNull();
  });

  it("groups by step, empty steps left out", () => {
    const g = stepGroups(doc({ checks: [chk("seam", "pass"), chk("planner", "pass")] }));
    expect(g.map((x) => x.title)).toEqual(["Body plan", "Skin"]);
  });
});

describe("check rows", () => {
  it("help by level and by function", () => {
    expect(helpText(CHECK_HELP.seam.fix, chk("seam", "fail"), null)).toBe("Re-rig. Rigs made now give every copy of a point the same bones.");
    expect(helpText(CHECK_HELP.seam.fix, chk("seam", "pass"), null)).toBe("");
    expect(helpText(CHECK_HELP.planner.why, chk("planner", "pass", { label: "Four legs, set by hand" }), null)).toBe("");
    expect(helpText(CHECK_HELP.planner.why, chk("planner", "pass"), null)).toBe("The skeleton is picked from the model's picture.");
    expect(helpText(CHECK_HELP.facing.did, chk("facing", "pass", { yaw: 180 }), null)).toBe("Turned the skeleton 180° to face the way the model does.");
    const turned = doc({ rig: { rest: { measured: { turned: true, head_yaw_deg: 30 } } } });
    expect(helpText(CHECK_HELP.head.fix, chk("head", "warn"), turned)).toBe("Straighten and rig, under Rest pose.");
    expect(helpText(CHECK_HELP.head.why, chk("head", "warn"), turned)).toBe(
      "3D generation turned the head toward the camera, but the neck bones run straight, so head moves swing it sideways.",
    );
    expect(helpText(CHECK_HELP.slide.did, chk("slide", "warn"), doc({ template: { built: true } }))).toBe("Measured on the clips' own planted frames.");
  });

  it("fit: arms are aimed, legs get the note", () => {
    const arm = checkBody({ ...chk("fit", "warn", { bone: "UpperArm_L" }), step: "skeleton", label: "x" }, null);
    expect(arm.did).toBe("Arms are aimed at the hands when rigging.");
    expect(arm.note).toBe("");
    const leg = checkBody({ ...chk("fit", "warn", { bone: "Foot_R" }), step: "skeleton", label: "x" }, null);
    expect(leg.did).toBe("");
    expect(leg.note).toBe("Legs aren't aimed at the feet when rigging yet, so re-rigging won't move it. It may not show in the clips.");
  });

  it("the sentence only when it differs from the label; Why not when it repeats the sentence", () => {
    const c = { key: "moved", level: "warn", label: "Same", text: "Same", step: "skeleton" };
    expect(checkBody(c, null).text).toBe("");
    const r = { key: "moved", level: "warn", label: "Joints moved", text: "The skinning step moved some joints from where they were fitted!", step: "skeleton" };
    expect(checkBody(r, null)).toMatchObject({ text: r.text, why: "" });
  });
});

describe("In motion notes", () => {
  it("seam tears and the stretch on a flagged bone, then the p99 line", () => {
    const d = doc({
      bones: [{ name: "root", parent: -1 }, { name: "Shin_L", parent: 0, label: "left shin" }],
      checks: [{ key: "seam", level: "fail", text: "9 points", points: 9 }, { key: "fit", level: "warn", bone: "Shin_L", text: "x" }],
      motion: { clips: { Walk_Fast: { tears: 4, stretch_p99: 2 }, Idle: { tears: 0, stretch_p99: 1, worst_bone: "Shin_L" }, Run: { tears: 0, stretch_p99: 1, worst_bone: "Shin_L" } } },
    });
    expect(motionNotes(d)).toEqual([
      "9 points can tear. How many come apart depends on the move: up to 4 in Walk Fast.",
      "The worst stretch in Idle and Run is at the left shin, the bone flagged under Skeleton.",
      "Stretch: how much the worst 1% of edges grow or shrink. Lower is better; above 3× it starts to show, above 6× it fails.",
    ]);
  });

  it("older measurements: the single worst edge", () => {
    expect(motionNotes(doc({ motion: { clips: { Idle: { tears: 0, stretch_max: 1.2 } } } }))).toEqual([
      "Stretch: how much the edge that changes most grows or shrinks. Lower is better; knees and elbows always stretch some.",
    ]);
    expect(motionNotes(doc())).toEqual([]);
  });
});

describe("Since version N", () => {
  it("before rig checks", () => {
    expect(compareView({ n: 2, id: "t1", summary: null, current: null, sameModel: true }, doc(), false)).toEqual({
      heading: "Since version 2",
      back: "Go back to version 2",
      takeId: "t1",
      before: "Version 2 was made before rig checks.",
      rows: [],
      more: "",
    });
    expect(compareView(null, doc(), false)).toBeNull();
  });

  it("tears, head, two clips that changed most, body plan, model", () => {
    const v = compareView(
      {
        n: 1,
        id: "t0",
        summary: { tears: 5, head: { off_deg: 20, made_before: true }, class: "avian", motion: { A: { stretch_p99: 2 }, B: { stretch_p99: 2 }, C: { stretch_p99: 2 } } },
        current: { tears: 0, head: { straightened: true, before: 30, off_deg: 2 }, class: "quadruped", motion: { A: { stretch_p99: 2.1 }, B: { stretch_p99: 6 }, C: { stretch_p99: 1 } } },
        sameModel: false,
      },
      doc(),
      false,
    )!;
    expect(v.rows.map((r) => r.label)).toEqual(["Tears", "Head", "B stretch (1% of edges)", "C stretch (1% of edges)", "Body plan", "Model"]);
    expect(v.rows[0].value).toEqual({ a: "5", b: "0", aTone: "worse", bTone: "better", mark: " ▼" });
    expect(v.rows[1].value).toMatchObject({ a: "20° off", b: "straightened 30°", mark: "" });
    expect(v.rows[2].value).toMatchObject({ a: "2.0×", b: "6.0×", mark: " ▲" });
    expect(v.rows[4].value).toBe("Bird → Four legs");
    expect(v.rows[5].value).toBe("made from a different model");
    expect(v.more).toBe("All 3 clips");
    expect(compareView({ n: 1, id: "t0", summary: { motion: {} }, current: { motion: {} }, sameModel: true }, doc(), true)!.rows).toEqual([{ label: "Model", value: "same model" }]);
  });

  it("head words", () => {
    expect(headWords(null)).toEqual(["not measured", 0]);
    expect(headWords({ before: 20, off_deg: 3 })).toEqual(["kept turned 20°", 3]);
    expect(headWords({ off_deg: 4 })).toEqual(["straight", 4]);
    expect(pair(1, 1)).toEqual({ a: "1", b: "1", aTone: "", bTone: "", mark: "" });
  });
});

describe("legend, bone card, plan, fit", () => {
  const d = doc({ weights: { dropped: { total: 7 }, seam_raw: { positions_with_conflicting_top_bone: 9 } } });
  it("legend lines", () => {
    expect(legendParts("heat", { bone: "", doc: d, dropped: false, splitPoints: 0 })).toMatchObject({ lead: "Pick a bone in the list or click the model.", leadTone: "warn" });
    expect(legendParts("heat", { bone: "Head", doc: d, dropped: false, splitPoints: 0 })).toMatchObject({ lead: "Head", rest: ": red pulls fully, blue a little, dark blue not at all.", ramp: true });
    expect(legendParts("bones", { bone: "Head", doc: d, dropped: false, splitPoints: 0 }).rest).toBe("Each part takes the colour of the bone that moves it most. Head is lit, the rest dimmed.");
    expect(legendParts("dropped", { bone: "", doc: d, dropped: false, splitPoints: 0 }).rest).toBe("7 weights were dropped, but this version did not record where. Re-rig to see them.");
    expect(legendParts("dropped", { bone: "", doc: doc(), dropped: false, splitPoints: 0 }).rest).toBe("Nothing was dropped.");
    expect(legendParts("dropped", { bone: "", doc: d, dropped: true, splitPoints: 0 }).rest).toBe("Yellow to red: weight thrown away because the bone was too far. 7 in all.");
    expect(legendParts("split", { bone: "", doc: d, dropped: false, splitPoints: 3 })).toMatchObject({
      lead: "3 points have corners on different bones.",
      leadTone: "fail",
      rest: " They tear when the model moves. Re-rig to fix. Straight from the skinner it was 9.",
    });
    expect(legendParts("split", { bone: "", doc: doc(), dropped: false, splitPoints: 0 }).rest).toBe("No split points. Corners at the same point move together.");
    expect(legendParts("texture", { bone: "", doc: d, dropped: false, splitPoints: 0 }).rest).toBe("The painted model.");
  });

  it("bone card rows", () => {
    const b = doc({
      bones: [{ name: "root", parent: -1 }, { name: "neck", parent: 0, template: "Neck", ref_vs_rest_deg: 12 }],
      fit: [{ bone: "neck", median_dist_rel_h: 0.05, centroid_off_bone_rel_h: 0.2 }],
      clips: { Walk: { peak_deg: [0, 33] } },
    });
    expect(boneCardRows({ name: "neck", doc: b, owned: 120, clip: "Walk", nice: (r) => (r === "root" ? "Root" : r) })).toEqual([
      { label: "Hangs from", value: "Root" },
      { label: "Driven by", value: "template bone Neck" },
      { label: "Turned in the start pose", value: "12°" },
      { label: "Points it moves most", value: "120" },
      { label: "Usual distance to its points", value: "5% of height" },
      { label: "Its part's centre is off by", value: "20% of height", tone: "warn" },
      { label: "Biggest turn in Walk", value: "33°" },
    ]);
    expect(boneCardRows({ name: "root", doc: b, owned: 0, clip: null, nice: () => "" })[0]).toEqual({ label: "Driven by", value: "nothing", tone: "warn" });
  });

  it("body plan and skeleton fit", () => {
    expect(planRows(doc({ planner: { choice: "avian", confidence: 0.5 } }))).toEqual([
      { label: "Chosen", value: "Bird" },
      { label: "How sure", value: "50%", tone: "warn" },
    ]);
    expect(planRows(doc({ planner: { unrecorded: true } }))[1].value).toBe("Not recorded");
    expect(planRows(doc())).toEqual([
      { label: "Chosen", value: "?" },
      { label: "How sure", value: "Set by hand", tone: "" },
    ]);
    expect(fitRows(doc({ template: { rig: "rigs/quad_v2.glb", ref_clip: "Idle", fit_scale: 1.23456, fit_offset: [0, 0.5], max_joint_move_by_skinner_m: 0.004 }, head: { off_deg: -16 } }))).toEqual([
      { label: "Template", value: "quad_v2" },
      { label: "Start pose", value: "Idle, first frame" },
      { label: "Scale", value: "1.235" },
      { label: "Offset (m)", value: "0.000, 0.500" },
      { label: "Joints moved by skinning", value: "0.00 m" },
      { label: "Head vs head bone", value: "16° apart", tone: "fail" },
    ]);
    expect(fitRows(doc())[0].value).toBe("?");
    expect(fitRows(doc({ template: { built: true, plan: "spider" } }))[0].value).toBe("Built for this model (Many legs)");
    expect(stickRoles(["root", "leg", "", "leg", "tail"])).toEqual(["leg", "tail"]);
    expect(stickRoles(["", ""])).toEqual([]);
  });
});

describe("head block", () => {
  it("the four reads and their button", () => {
    expect(headBlock(null)).toBeNull();
    expect(headBlock({ measured: { turned: true, head_yaw_deg: 32 }, straightened: { head_yaw_deg: -1 } })).toEqual({
      read: "Head turned +32° → −1°",
      segments: true,
      button: { label: "Rig it as made", title: "Rigs again with the head as the model has it. This version stays in Versions.", straighten: "off" },
      kept: "Your model file is never changed.",
    });
    expect(headBlock({ measured: { turned: true, head_yaw_deg: -30 }, before_straightening: true })).toMatchObject({
      read: "Head turned −30° when it was made",
      button: { label: "Straighten and rig", title: "Rigs again with the head straightened. This version stays in Versions.", straighten: "auto" },
    });
    expect(headBlock({ measured: { turned: true, head_yaw_deg: 30 } })!.read).toBe("Head turned +30°, kept as made");
    expect(headBlock({ measured: { turned: false, head_yaw_deg: -3.4 } })).toEqual({ read: "Head straight (3°)", segments: false, button: null, kept: "" });
  });
});

describe("Show me", () => {
  const base = { names: ["root", "head", "Thigh_R"], headUI: false, dropped: false };
  it("needs the bone, the clips, the measurement or the dropped list it shows", () => {
    expect(canShow(chk("fit", "warn", { bone: "Thigh_R" }), { ...base, doc: doc() })).toBe(true);
    expect(canShow(chk("fit", "warn", { bone: "Nope" }), { ...base, doc: doc() })).toBe(false);
    expect(canShow(chk("cshape", "warn"), { ...base, doc: doc() })).toBe(false);
    expect(canShow(chk("cshape", "warn", { clips: { Walk: 1.4 } }), { ...base, doc: doc() })).toBe(true);
    expect(canShow(chk("fbx", "pass"), { ...base, doc: doc() })).toBe(false);
    expect(canShow(chk("motion", "fail"), { ...base, doc: doc({ motion: { clips: {} } }) })).toBe(false);
    expect(canShow(chk("motion", "fail", { clip: "Walk" }), { ...base, doc: doc() })).toBe(true);
    expect(canShow(chk("folded", "warn"), { ...base, doc: doc() })).toBe(false);
    expect(canShow(chk("clips", "warn"), { ...base, doc: doc() })).toBe(false);
    expect(canShow(chk("rest", "warn"), { ...base, doc: doc({ rig: { rest: { measured: { turned: true, head_yaw_deg: 30 } } } }) })).toBe(true);
    expect(canShow(chk("rest", "warn"), { ...base, doc: doc({ rig: { rest: { measured: { turned: true, head_yaw_deg: 30 }, straightened: { head_yaw_deg: 0 } } } }) })).toBe(false);
  });

  it("worst clip, right-side bones", () => {
    expect(worstClip(chk("cshape", "warn", { clips: { Walk: 1.2, Run: [1, 1.9, 1] } }))).toBe("Run");
    expect(worstClip(chk("slide", "warn", { over: ["Hop"], clips: { Walk: 9 } }))).toBe("Hop");
    expect(worstClip(chk("slide", "warn"))).toBe("");
    expect(sideOf("Thigh_R")).toEqual([-1, 0, 0]);
    expect(sideOf("RightArm")).toEqual([-1, 0, 0]);
    expect(sideOf("Thigh_L")).toBe("side");
  });
});

describe("bones list", () => {
  it("words, numbered duplicates, depth and filter", () => {
    const bones = [{ name: "root", parent: -1 }, { name: "spine_01", parent: 0, label: "spine" }, { name: "spine_02", parent: 1, label: "spine" }];
    expect(niceNames(bones.map((b) => b.name), bones)).toEqual(["Root", "Spine 01", "Spine 02"]);
    expect(boneDepth(bones, 2)).toBe(2);
    expect(boneFound("", "x", "y")).toBe(true);
    expect(boneFound("spine_0", "spine_01", "Spine 01")).toBe(true);
    expect(boneFound("SPINE", "spine_01", "Spine 01")).toBe(true);
    expect(boneFound("Spine_", "spine_01", "Spine 01")).toBe(false);
  });
});

describe("Sheet tab", () => {
  it("looking, none (with a render button), found", () => {
    expect(sheetView(SHEET0, "cow")).toEqual({ kind: "looking", button: "" });
    expect(sheetView({ ...SHEET0, stem: "cow", ok: false }, "cow")).toEqual({ kind: "none", button: "Render the sheet" });
    expect(sheetView({ ...SHEET0, stem: "cow", ok: true }, "cow")).toEqual({ kind: "found", button: "Render again" });
    expect(sheetView({ ...SHEET0, stem: "cow", ok: true, busy: true }, "cow").button).toBe("Rendering…");
    expect(sheetView({ ...SHEET0, stem: "pig", ok: true }, "cow").kind).toBe("looking");
    expect(SHEET_COPY).toEqual({
      looking: "Looking for the contact sheet…",
      none: "No contact sheet for this rig yet. Rigs made now render one; this takes under a minute on the CPU.",
      openTitle: "Open the sheet full size",
      alt: "Contact sheet: every clip of this rig",
      caption: "Rows: the rest pose, then each clip at its biggest move. Columns: side, front, three-quarter. The red line is the floor.",
      rendering: "Rendering…",
      again: "Render again",
      render: "Render the sheet",
    });
  });
});
