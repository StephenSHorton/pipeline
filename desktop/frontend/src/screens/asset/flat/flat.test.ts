import { describe, expect, it } from "vitest";
import {
  applyAudio,
  applySteps,
  audioBody,
  audioFields,
  audioPlayers,
  audioWho,
  clampIndex,
  clipLabel,
  emptyAudioAsset,
  finishFlat,
  FLAT_MODES,
  keepLocal,
  keepLocalRow,
  mergeSheet,
  partLabel,
  rereadSheet,
  savedKey,
  pictureBlock,
  REDRAW_ASK,
  reviseBody,
  revisePart,
  soundState,
  speakState,
  SPEAKERS,
  stagePicture,
  STEP_RUNNING,
} from "./flat";

describe("copy", () => {
  it("placeholders and running texts match the legacy page", () => {
    expect(FLAT_MODES).toEqual({
      picture: "Redraw from a new prompt…",
      motion: "Name the motion…",
      pieces: "Name the pieces…",
      say: "What it says…",
      sound: "The sound it makes…",
    });
    expect(STEP_RUNNING).toEqual({ pieces: "Cutting separate shapes.", motion: "Packing the motion.", say: "Speaking.", sound: "Writing the sound." });
    expect(SPEAKERS).toEqual(["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bm_george"]);
    expect(REDRAW_ASK).toBe("Draw a new picture from this prompt? It replaces the current one.");
  });

  it("Picture mode says why it cannot start", () => {
    expect(pictureBlock("", false)).toBe("Describe the new picture first.");
    expect(pictureBlock("a hen", true)).toBe("The factory is busy with another job.");
    expect(pictureBlock("a hen", false)).toBe("");
  });

  it("names a piece and a clip", () => {
    expect(partLabel({ name: "Sword" }, 1)).toBe("Sword");
    expect(partLabel({ name: "" }, 1)).toBe("Piece 2");
    expect(clipLabel([])).toBe("Clip");
    expect(clipLabel([{ name: "walk.mp4" }, { name: "b.png" }])).toBe("walk.mp4");
  });
});

describe("mergeSheet", () => {
  const sheet = {
    parts: [
      { id: "part-01", file: "parts/part-01.png", name: "" },
      { id: "part-02", file: "parts/part-02.png", name: "Sword", role: "weapon" },
    ],
    mechanics: [
      { file: "parts/part-01.png", name: "Bottle", line: "Hi", speaker: "bf_emma", voice: "voices/a.wav" },
      { id: "asset", line: "Moo", motion: "walk", sound: "sounds/s.wav" },
    ],
    center: "sheets/walk.png",
  };

  it("merges mechanics by file or id, with role other and speaker af_heart by default", () => {
    const { parts, audioAsset, center } = mergeSheet(sheet);
    expect(parts[0]).toMatchObject({ name: "Bottle", role: "other", line: "Hi", speaker: "bf_emma", voice: "voices/a.wav", sound: "" });
    expect(parts[1]).toMatchObject({ name: "Sword", role: "weapon", speaker: "af_heart", line: "" });
    expect(audioAsset).toEqual({ id: "asset", line: "Moo", speaker: "af_heart", projectile: "walk", sound: "sounds/s.wav", voice: "" });
    expect(center).toBe("sheets/walk.png");
  });

  it("a missing sheet is empty", () => {
    expect(mergeSheet(null)).toEqual({ parts: [], audioAsset: emptyAudioAsset(), center: "" });
    expect(mergeSheet({ center: 3 }).center).toBe("");
  });

  it("a re-read keeps the asset row's line and the file the panel just wrote", () => {
    const prev = { ...emptyAudioAsset(), line: "Moo", speaker: "am_adam", voice: "voices/asset.wav" };
    expect(keepLocalRow(prev, emptyAudioAsset())).toEqual({ ...prev });
    expect(keepLocalRow(prev, { ...emptyAudioAsset(), voice: "voices/new.wav" }).voice).toBe("voices/new.wav");
  });

  it("clamps the selected piece", () => {
    expect(clampIndex(-1, 0)).toBe(-1);
    expect(clampIndex(-1, 2)).toBe(0);
    expect(clampIndex(5, 2)).toBe(0);
    expect(clampIndex(1, 2)).toBe(1);
  });

  it("a re-read keeps what was typed or heard here", () => {
    const prev = mergeSheet(sheet).parts.map((p) => ({ ...p, line: p.line || "typed", sound: "sounds/x.wav" }));
    const next = mergeSheet({ parts: sheet.parts }).parts;
    const kept = keepLocal(prev, next);
    expect(kept[1].line).toBe("typed");
    expect(kept[0].sound).toBe("sounds/x.wav");
  });

  describe("rereadSheet", () => {
    const typed = { line: "Typed", effect: "A thud", speaker: "am_adam" };
    it("after this stage's own POST the typed fields and heard files stay", () => {
      const prev = mergeSheet(sheet);
      const local = { sheet: { parts: prev.parts, index: 1, center: "" }, row: prev.audioAsset, fields: typed };
      const r = rereadSheet(mergeSheet({ parts: sheet.parts }), local, true);
      expect(r.sheet.index).toBe(1);
      expect(r.sheet.parts[1]).toMatchObject({ line: "Typed", projectile: "A thud", speaker: "am_adam" });
      expect(r.sheet.parts[0].voice).toBe("voices/a.wav");
      expect(r.row.line).toBe("Moo");
    });
    it("with no pieces the fields go onto the asset row", () => {
      const local = { sheet: { parts: [], index: -1, center: "" }, row: emptyAudioAsset(), fields: typed };
      const r = rereadSheet(mergeSheet(null), local, true);
      expect(r.row).toMatchObject({ line: "Typed", projectile: "A thud", speaker: "am_adam" });
      expect(r.saved).toBe(savedKey(emptyAudioAsset()));
    });
    it("any other re-read (a pick, a finished job) takes the sheet as it comes", () => {
      const prev = mergeSheet(sheet);
      const local = { sheet: { parts: prev.parts, index: 0, center: "" }, row: prev.audioAsset, fields: typed };
      const older = mergeSheet({ parts: sheet.parts });
      const r = rereadSheet(older, local, false);
      expect(r.sheet.parts).toEqual(older.parts);
      expect(r.row).toEqual(older.audioAsset);
      expect(r.saved).not.toBe(savedKey(prev.parts[0]));
    });
    it("a new stem starts on the first piece", () => {
      expect(rereadSheet(mergeSheet(sheet), null, true).sheet.index).toBe(0);
    });
  });
});

describe("stagePicture", () => {
  const still = "out/hen/flat/source.png";
  it("concept shows the still, else nothing", () => {
    expect(stagePicture("concept", "source", still)).toEqual({ kind: "still", rel: still });
    expect(stagePicture("concept", "sheets/walk.png", "")).toEqual({ kind: "empty" });
  });
  it("2D: the strip, else the source, else the still, else nothing", () => {
    expect(stagePicture("flat", "sheets/walk-2.png", still)).toEqual({ kind: "strip", rel: "sheets/walk-2.png" });
    expect(stagePicture("flat", "source", still)).toEqual({ kind: "source" });
    expect(stagePicture("flat", "", still)).toEqual({ kind: "still", rel: still });
    expect(stagePicture("flat", "", "")).toEqual({ kind: "empty" });
  });
  it("only a safe sheets/<name>.png is a strip", () => {
    for (const bad of ["sheets/../x.png", "sheets/Walk.png", "sheets/-a.png", `sheets/${"a".repeat(42)}.png`, "parts/a.png"]) {
      expect(stagePicture("flat", bad, "")).toEqual({ kind: "empty" });
    }
    expect(stagePicture("flat", `sheets/${"a".repeat(41)}.png`, "").kind).toBe("strip");
  });
});

describe("audio panel", () => {
  it("says whose audio it is", () => {
    expect(audioWho("2d", { name: "Sword", id: "part-02" })).toBe("Sword");
    expect(audioWho("2d", { name: "", id: "part-02" })).toBe("part-02");
    expect(audioWho("2d", null)).toBe("Select a part");
    expect(audioWho("3d", null)).toBe("This asset");
  });

  it("fills the fields; an unknown speaker is af_heart", () => {
    expect(audioFields(null)).toEqual({ line: "", effect: "", speaker: "af_heart" });
    expect(audioFields({ line: "Hi", projectile: "", motion: "walk", speaker: "zz" })).toEqual({ line: "Hi", effect: "walk", speaker: "af_heart" });
    expect(audioFields({ projectile: "bolt", speaker: "bm_george" })).toEqual({ line: "", effect: "bolt", speaker: "bm_george" });
  });

  it("a piece with no wav keeps what the panel wrote on the asset row", () => {
    const row = { ...emptyAudioAsset(), voice: "voices/v.wav", sound: "sounds/old.wav" };
    expect(audioPlayers({ sound: "", voice: "" }, row, { voice: "voices/v.wav", sound: "" })).toEqual({ sound: "", voice: "voices/v.wav" });
    expect(audioPlayers({ sound: "sounds/p.wav", voice: "" }, row, { voice: "", sound: "sounds/old.wav" })).toEqual({ sound: "sounds/p.wav", voice: "" });
  });

  it("Speak and Sound titles follow /2d/status", () => {
    expect(speakState({ voice: true })).toEqual({ disabled: false, title: "Speak the line with Kokoro." });
    expect(speakState({ voice: false })).toEqual({ disabled: true, title: "Install mlx-audio to speak." });
    expect(soundState({ sfx_mode: "ready" })).toEqual({ disabled: false, title: "Render a one-shot effect." });
    expect(soundState({ sfx_mode: "no-runner" })).toEqual({ disabled: true, title: "SFX weights are installed. Set FLAT2D_SFX_CMD." });
    expect(soundState({})).toEqual({ disabled: false, title: "Writes a stand-in wav." });
    expect(soundState(null).title).toBe("Writes a stand-in wav until SFX weights and FLAT2D_SFX_CMD are set.");
  });
});

describe("/2d/steps bodies", () => {
  const part = { id: "part-02", speaker: "bf_emma" };
  it("one step per mode on the selected part", () => {
    expect(reviseBody("hen", "pieces", "wings", part)).toEqual({ stem: "hen", when: "revise", steps: ["pieces"], part: "part-02", speaker: "bf_emma", pieces: "wings" });
    expect(reviseBody("hen", "say", "hi", part).say).toBe("hi");
    expect(reviseBody("hen", "sound", "cluck", part).sound).toBe("cluck");
    expect(reviseBody("hen", "motion", "walk", null)).toEqual({ stem: "hen", when: "revise", steps: ["motion"], part: "", speaker: "af_heart", motion: "walk" });
  });
  it("a clip goes with Motion only", () => {
    expect(reviseBody("hen", "motion", "walk", part, { video: "AQID" }).video).toBe("AQID");
    expect(reviseBody("hen", "motion", "walk", part, { frames: ["a", "b"] }).frames).toEqual(["a", "b"]);
    expect(reviseBody("hen", "pieces", "x", part, { video: "AQID" }).video).toBeUndefined();
  });
  it("the target is the selected part, else the asset row", () => {
    const parts = mergeSheet({ parts: [{ id: "p1", file: "parts/p1.png" }] }).parts;
    expect(revisePart(parts, 0, null).id).toBe("p1");
    expect(revisePart(parts, -1, { ...emptyAudioAsset(), speaker: "am_adam" })).toMatchObject({ id: "asset", speaker: "am_adam" });
    expect(revisePart([], -1, null)).toEqual({ id: "asset", speaker: "af_heart" });
  });
  it("the panel's Speak and Sound write the asset row", () => {
    expect(audioBody("hen", "say", "Moo", "am_adam")).toEqual({ stem: "hen", when: "revise", steps: ["say"], part: "asset", speaker: "am_adam", say: "Moo" });
    expect(audioBody("hen", "sound", "", "")).toEqual({ stem: "hen", when: "revise", steps: ["sound"], part: "asset", speaker: "af_heart", sound: "" });
  });
});

describe("applySteps", () => {
  const sheet = { parts: mergeSheet({ parts: [{ id: "p1", file: "parts/p1.png" }, { id: "p2", file: "parts/p2.png" }] }).parts, index: 1, center: "source" };

  it("Motion: the new strip and the notes", () => {
    const r = applySteps(sheet, "motion", "walk", "p2", { center: "sheets/walk.png", notes: ["Packed 8 frames.", "Done."] });
    expect(r.center).toBe("sheets/walk.png");
    expect(r.status).toBe("Packed 8 frames. Done.");
    expect(r.index).toBe(1);
  });

  it("Pieces: new parts, the first selected", () => {
    const r = applySteps(sheet, "pieces", "wings", "p2", { parts: [{ id: "n1", file: "parts/n1.png", name: "Wing" }] });
    expect(r.parts.map((p) => p.id)).toEqual(["n1"]);
    expect(r.parts[0].speaker).toBe("af_heart");
    expect(r.index).toBe(0);
    expect(applySteps(sheet, "pieces", "x", "p2", { parts: [] }).index).toBe(-1);
  });

  it("Say and Sound put the heard file on the target part", () => {
    const say = applySteps(sheet, "say", "Cluck", "p2", { hear: "voices/p2.wav" });
    expect(say.hear).toBe("voices/p2.wav");
    expect(say.parts[1]).toMatchObject({ line: "Cluck", voice: "voices/p2.wav" });
    const snd = applySteps(sheet, "sound", "flap", "p1", { hear: "sounds/p1.wav" });
    expect(snd.parts[0]).toMatchObject({ projectile: "flap", sound: "sounds/p1.wav" });
    // The wrong kind of file still plays, but is not stored.
    const odd = applySteps(sheet, "say", "x", "p2", { hear: "sounds/z.wav" });
    expect(odd.hear).toBe("sounds/z.wav");
    expect(odd.parts[1].voice).toBe("");
  });

  it("the panel keeps only the right kind of file", () => {
    expect(applyAudio("say", { hear: "voices/a.wav", notes: ["ok"] })).toEqual({ rel: "voices/a.wav", status: "ok" });
    expect(applyAudio("sound", { hear: "voices/a.wav" })).toEqual({ rel: "", status: "" });
  });
});

describe("finishFlat", () => {
  it("a redraw shows the source and says Redrawn", () => {
    expect(finishFlat("hen", { revise: true, center: "sheets/x.png", notes: ["a", "b"] })).toEqual({
      center: "source",
      pieces: false,
      hear: "",
      status: "a b",
      toast: "Redrawn: hen",
    });
  });
  it("a sprite opens its center, Pieces and the heard file", () => {
    expect(finishFlat("hen", { center: "sheets/walk.png", pieces: [{}], hear: "voices/v.wav" })).toMatchObject({
      center: "sheets/walk.png",
      pieces: true,
      hear: "voices/v.wav",
      toast: "Sprite ready: hen",
    });
  });
});
