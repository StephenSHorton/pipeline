// The 2D sheet's pure logic (legacy index.html 3775-4211): what the view shows, the parts merged with their
// mechanics, the revise and audio bodies for POST /2d/steps, and what a /2d/steps answer changes. No React here.
import type { FlatMode } from "@/lib/flow";

/** The revise field's placeholder per mode (legacy FLAT_MODES). */
export const FLAT_MODES: Record<FlatMode, string> = {
  picture: "Redraw from a new prompt…",
  motion: "Name the motion…",
  pieces: "Name the pieces…",
  say: "What it says…",
  sound: "The sound it makes…",
  animate: "What happens, and which way it faces…",
};

export const MODE_LABEL: Record<FlatMode, string> = {
  picture: "Picture",
  motion: "Motion",
  pieces: "Pieces",
  say: "Say",
  sound: "Sound",
  animate: "Animate",
};

export const FLAT_MODE_LIST: FlatMode[] = ["picture", "motion", "pieces", "say", "sound", "animate"];

/** The modes that run one /2d/steps step at once (Picture and Animate start jobs instead). */
export type StepMode = Exclude<FlatMode, "picture" | "animate">;

/** The status line while a step runs (legacy STEP_RUNNING). */
export const STEP_RUNNING: Partial<Record<FlatMode, string>> = {
  pieces: "Cutting separate shapes.",
  motion: "Packing the motion.",
  say: "Speaking.",
  sound: "Writing the sound.",
};

export const SPEAKERS = ["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bm_george"] as const;
export const DEFAULT_SPEAKER = "af_heart";

/** The clip picker accepts a video or still frames (legacy #flat-clip). */
export const CLIP_ACCEPT = "video/mp4,video/webm,image/png,image/jpeg,image/webp";

/* ---------------------------------------------------------------- the sheet (GET /2d/<stem>) */

export interface SheetPart {
  id: string;
  file: string;
  name?: string;
  role?: string;
  w?: number;
  h?: number;
  [k: string]: unknown;
}

export interface Mechanic {
  id?: string;
  file?: string;
  name?: string;
  role?: string;
  motion?: string;
  projectile?: string;
  line?: string;
  speaker?: string;
  sound?: string;
  voice?: string;
}

export interface Sheet {
  ok?: boolean;
  parts?: SheetPart[];
  mechanics?: Mechanic[];
  note?: string;
  center?: unknown;
  [k: string]: unknown;
}

/** A part with its audio fields filled in (legacy loadAudio's merge). */
export interface FlatPart extends SheetPart {
  name: string;
  role: string;
  motion: string;
  projectile: string;
  line: string;
  speaker: string;
  sound: string;
  voice: string;
}

/** The asset-level audio row (mechanics id "asset"). */
export interface AudioRow {
  id: string;
  line: string;
  speaker: string;
  projectile: string;
  motion?: string;
  sound: string;
  voice: string;
  name?: string;
}

export const emptyAudioAsset = (): AudioRow => ({ id: "asset", line: "", speaker: DEFAULT_SPEAKER, projectile: "", sound: "", voice: "" });

/** Merge mechanics into the parts (by file, then id; role "other" and speaker af_heart by default) and read the
 * asset row. `center` is the string the sheet names ("source" or "sheets/<x>.png"), else "". */
export function mergeSheet(sheet: Sheet | null | undefined): { parts: FlatPart[]; audioAsset: AudioRow; center: string } {
  const mech = new Map<string, Mechanic>();
  for (const row of sheet?.mechanics ?? []) {
    if (row.file) mech.set(row.file, row);
    if (row.id) mech.set(row.id, row);
  }
  const parts = (sheet?.parts ?? []).map((p): FlatPart => {
    const extra = mech.get(p.file) || mech.get(p.id) || {};
    return {
      ...p,
      name: extra.name || p.name || "",
      role: extra.role || p.role || "other",
      motion: extra.motion || "",
      projectile: extra.projectile || "",
      line: extra.line || "",
      speaker: extra.speaker || DEFAULT_SPEAKER,
      sound: extra.sound || "",
      voice: extra.voice || "",
    };
  });
  const a = mech.get("asset") || {};
  const audioAsset: AudioRow = {
    id: "asset",
    line: a.line || "",
    speaker: a.speaker || DEFAULT_SPEAKER,
    projectile: a.projectile || a.motion || "",
    sound: a.sound || "",
    voice: a.voice || "",
  };
  const center = sheet && typeof sheet.center === "string" ? sheet.center : "";
  return { parts, audioAsset, center };
}

/** A re-read of the sheet (after a flow refresh) keeps what was typed or heard here and is not saved yet. */
export function keepLocal(prev: FlatPart[], next: FlatPart[]): FlatPart[] {
  const byId = new Map(prev.map((p) => [p.id, p]));
  return next.map((p) => {
    const old = byId.get(p.id);
    if (!old) return p;
    return {
      ...p,
      line: p.line || old.line,
      projectile: p.projectile || old.projectile,
      speaker: p.speaker !== DEFAULT_SPEAKER ? p.speaker : old.speaker,
      sound: p.sound || old.sound,
      voice: p.voice || old.voice,
    };
  });
}

/** The same for the asset row: a re-read does not drop the line or the file the panel just wrote. */
export function keepLocalRow(prev: AudioRow, next: AudioRow): AudioRow {
  return {
    ...next,
    line: next.line || prev.line,
    projectile: next.projectile || prev.projectile,
    speaker: next.speaker !== DEFAULT_SPEAKER ? next.speaker : prev.speaker,
    sound: next.sound || prev.sound,
    voice: next.voice || prev.voice,
  };
}

/** -1 with no parts, else the index kept in range (0 when it fell out). */
export function clampIndex(index: number, count: number): number {
  if (!count) return -1;
  return index < 0 || index >= count ? 0 : index;
}

/** The panel fields written back onto a part or the asset row (legacy readAudio). */
export const withFields = <T extends { line: string; projectile: string; speaker: string }>(p: T, f: AudioFields): T => ({
  ...p,
  line: f.line,
  projectile: f.effect,
  speaker: f.speaker,
});

/**
 * One re-read of the sheet. `local` is what this stage holds for the same stem (null on a new stem). With `keep`
 * (the re-read after this stage's own POST) the panel fields go onto the subject first and the merge keeps them
 * and what was heard here; any other re-read (a pick, a finished job) takes the sheet as it comes (legacy loadAudio).
 * `saved` is the subject's stored audio, so the caller refills the panel only when the subject or that changed.
 */
export function rereadSheet(
  next: ReturnType<typeof mergeSheet>,
  local: { sheet: SheetState; row: AudioRow; fields: AudioFields } | null,
  keep: boolean,
): { sheet: SheetState; row: AudioRow; saved: string } {
  let parts = next.parts;
  let row = next.audioAsset;
  if (local && keep) {
    const { sheet, fields } = local;
    const own = sheet.parts.map((p, k) => (k === sheet.index ? withFields(p, fields) : p));
    parts = keepLocal(own, next.parts);
    row = keepLocalRow(sheet.index < 0 ? withFields(local.row, fields) : local.row, next.audioAsset);
  }
  const index = clampIndex(local ? local.sheet.index : -1, parts.length);
  return { sheet: { parts, index, center: next.center }, row, saved: savedKey(index >= 0 ? next.parts[index] : next.audioAsset) };
}

/** What is stored for the panel's subject, as one comparable string. */
export const savedKey = (from: FlatPart | AudioRow | null | undefined) =>
  from ? JSON.stringify([from.line, from.projectile, "motion" in from ? from.motion : "", from.speaker, from.sound, from.voice]) : "";

export const partLabel =(p: { name?: string }, i: number) => p.name || `Piece ${i + 1}`;

/* ---------------------------------------------------------------- the view */

const STRIP = /^sheets\/[a-z0-9][a-z0-9-]{0,40}\.png$/;

export type StagePicture =
  | { kind: "strip"; rel: string }
  | { kind: "source" }
  | { kind: "still"; rel: string }
  | { kind: "empty" };

/**
 * What the view shows (legacy showStage). Concept: the concept still, else nothing. 2D: the motion strip when
 * the center names one, else the source when the center is "source", else the concept still (a photo asset's
 * cut-out), else nothing. `still` is root-relative (out/<stem>/...), from conceptStill.
 */
export function stagePicture(stage: "concept" | "flat", center: string, still: string): StagePicture {
  if (stage === "flat") {
    if (STRIP.test(center)) return { kind: "strip", rel: center };
    if (center === "source") return { kind: "source" };
  }
  return still ? { kind: "still", rel: still } : { kind: "empty" };
}

export const isStrip = (center: unknown): center is string => typeof center === "string" && STRIP.test(center);

/* ---------------------------------------------------------------- the audio panel */

/** The panel's subject line: the selected part in 2D ("Select a part" when none), "This asset" in 3D. */
export function audioWho(workflow: "2d" | "3d", part: { name?: string; id?: string } | null): string {
  if (workflow !== "2d") return "This asset";
  return part ? part.name || part.id || "" : "Select a part";
}

export interface AudioFields {
  line: string;
  effect: string;
  speaker: string;
}

/** The fields for a subject (legacy fillAudio): an unknown speaker falls back to af_heart. */
export function audioFields(p: Partial<FlatPart | AudioRow> | null): AudioFields {
  const want = (p && p.speaker) || DEFAULT_SPEAKER;
  return {
    line: p ? p.line || "" : "",
    effect: p ? p.projectile || p.motion || "" : "",
    speaker: (SPEAKERS as readonly string[]).includes(want) ? want : DEFAULT_SPEAKER,
  };
}

/** The two players under the panel. A part with no wav does not hide what the panel just wrote on the asset row. */
export function audioPlayers(
  p: Partial<FlatPart | AudioRow> | null,
  audioAsset: AudioRow,
  assetHear: { sound: string; voice: string },
): { sound: string; voice: string } {
  let sound = (p && p.sound) || "";
  let voice = (p && p.voice) || "";
  if (!sound && assetHear.sound && audioAsset.sound === assetHear.sound) sound = audioAsset.sound;
  if (!voice && assetHear.voice && audioAsset.voice === assetHear.voice) voice = audioAsset.voice;
  return { sound, voice };
}

export interface FlatFlags {
  voice?: boolean;
  sfx_mode?: string;
  [k: string]: unknown;
}

/** Speak: only with a voice model (GET /2d/status). Before the flags are read, the markup's default. */
export function speakState(f: FlatFlags | null): { disabled: boolean; title: string } {
  if (!f) return { disabled: true, title: "Install mlx-audio to speak." };
  return f.voice ? { disabled: false, title: "Speak the line with Kokoro." } : { disabled: true, title: "Install mlx-audio to speak." };
}

/** Sound: ready, no-runner (off), or the stand-in wav. Before the flags are read, the markup's default. */
export function soundState(f: FlatFlags | null): { disabled: boolean; title: string } {
  if (!f) return { disabled: false, title: "Writes a stand-in wav until SFX weights and FLAT2D_SFX_CMD are set." };
  const mode = f.sfx_mode || "stub";
  if (mode === "ready") return { disabled: false, title: "Render a one-shot effect." };
  if (mode === "no-runner") return { disabled: true, title: "SFX weights are installed. Set FLAT2D_SFX_CMD." };
  return { disabled: false, title: "Writes a stand-in wav." };
}

/* ---------------------------------------------------------------- POST /2d/steps bodies */

export interface StepsBody {
  stem: string;
  when: "revise";
  steps: FlatMode[];
  part: string;
  speaker: string;
  pieces?: string;
  motion?: string;
  say?: string;
  sound?: string;
  video?: string;
  frames?: string[];
}

export type ClipFields = { video: string } | { frames: string[] };

/** The revise form in Motion / Pieces / Say / Sound: one step on the selected part (or the asset row). */
export function reviseBody(
  stem: string,
  mode: StepMode,
  prompt: string,
  part: { id?: string; speaker?: string } | null,
  clip?: ClipFields | null,
): StepsBody {
  const body: StepsBody = { stem, when: "revise", steps: [mode], part: (part && part.id) || "", speaker: (part && part.speaker) || DEFAULT_SPEAKER };
  if (mode === "pieces") body.pieces = prompt;
  else if (mode === "motion") {
    body.motion = prompt;
    if (clip) Object.assign(body, clip);
  } else if (mode === "say") body.say = prompt;
  else body.sound = prompt;
  return body;
}

/** The panel's Speak and Sound buttons always write the asset row. */
export function audioBody(stem: string, step: "say" | "sound", text: string, speaker: string): StepsBody {
  const body: StepsBody = { stem, when: "revise", steps: [step], part: "asset", speaker: speaker || DEFAULT_SPEAKER };
  if (step === "say") body.say = text || "";
  else body.sound = text || "";
  return body;
}

/** The revise target: the selected part, else the asset row (legacy flatAudioPart). */
export function revisePart(parts: FlatPart[], index: number, audioAsset: AudioRow | null): { id: string; speaker: string } {
  if (index >= 0 && parts[index]) return parts[index];
  return audioAsset && audioAsset.id ? audioAsset : { id: "asset", speaker: DEFAULT_SPEAKER };
}

/** The revise form in Picture mode: why it cannot start, or "" to ask before redrawing. */
export function pictureBlock(prompt: string, anyJobRunning: boolean): string {
  if (!prompt) return "Describe the new picture first.";
  if (anyJobRunning) return "The factory is busy with another job.";
  return "";
}

export const REDRAW_ASK = "Draw a new picture from this prompt? It replaces the current one.";

/* ---------------------------------------------------------------- Animate (POST /2d/animate/<stem>) */

/** A code-made cycle for one facing (GET /2d/animate/<stem>): the strip it makes, whose body it was made for, and
 * the prompt the factory uses when none is given. */
export interface AnimateCycle {
  cycle: string;
  facing: string;
  action: string;
  skeleton: string;
  made_for?: string;
  prompt: string;
}

/** Another skeleton clip Pipeline ships (GET /2d/animate/<stem>): its name, plain words, and the strip it makes. */
export interface Skeleton {
  name: string;
  label: string;
  action: string;
}

/** What Animate offers an asset (GET /2d/animate/<stem>). */
export interface AnimateOffer {
  cycles: AnimateCycle[];
  skeletons: Skeleton[];
}

export const NO_OFFER: AnimateOffer = { cycles: [], skeletons: [] };

/** One entry of Animate's picker: its value, its words, and the cycle or skeleton it stands for. */
export interface AnimatePick {
  value: string;
  label: string;
  group: "cycle" | "skeleton";
  cycle?: AnimateCycle;
  skeleton?: Skeleton;
}

const word = (s: string) => s.slice(0, 1).toUpperCase() + s.slice(1);

/** Animate's picker: every cycle with its facing ("Walk, facing right"; "· made for this character" when it is),
 * then the other skeletons. */
export function animatePicks(offer: AnimateOffer): AnimatePick[] {
  return [
    ...offer.cycles.map((c) => ({
      value: `cycle:${c.cycle}:${c.facing}`,
      label: `${word(c.cycle)}, facing ${c.facing}${c.made_for === "this character" ? " · made for this character" : ""}`,
      group: "cycle" as const,
      cycle: c,
    })),
    ...offer.skeletons.map((k) => ({ value: `skeleton:${k.name}`, label: k.label || k.name, group: "skeleton" as const, skeleton: k })),
  ];
}

/** The pick to keep when the offer is (re)read: the one already picked if it is still offered, else the first. */
export function keepPick(picks: AnimatePick[], cur: string): string {
  return picks.some((p) => p.value === cur) ? cur : picks[0]?.value || "";
}

/** The revise field's text after Animate's pick changes: a cycle's ready prompt replaces an empty field or another
 * pick's ready prompt; anything typed is kept. A skeleton has no ready prompt, so a ready prompt left there goes. */
export function promptForPick(current: string, pick: AnimatePick | undefined, offer: AnimateOffer): string {
  const ready = new Set(offer.cycles.map((c) => c.prompt));
  const typed = current.trim() && !ready.has(current.trim());
  if (typed) return current;
  return pick?.cycle?.prompt || "";
}

export const ANIMATE_TITLE =
  "Animate a character picture with a skeleton (LTX-2.3): a code-made walk, run or attack turns from the camera to " +
  "face the way picked, and its prompt is filled in; another skeleton needs words that say what happens and which " +
  "way the character faces. Makes a sprite strip and a new 2D version.";

/** Animate's optional kit words: what the model invents and the factory erases from a first pass ("backpack"),
 * and, with it, what must stay ("chest rig and pouches"). POST /2d/animate/<stem> takes them as remove and keep. */
export interface AnimateKit {
  remove?: string;
  keep?: string;
}

/** The longest remove and keep the factory takes (scripts/factory_http.py animate_argv): a few words each. */
export const KIT_MAX = { remove: 120, keep: 200 } as const;

export const REMOVE_HINT = "Remove (optional): backpack…";
export const KEEP_HINT = "Keep (optional): chest rig…";
export const REMOVE_TITLE =
  "Optional. Kit the model invents that the picture does not have (like a backpack): Animate runs LTX-2.3 once, " +
  "erases this from that pass's frames, and animates again guided by them. Takes longer.";
export const KEEP_TITLE =
  "Optional, with Remove: kit that must stay when the removed kit is erased (asked only to remove a backpack, the " +
  "edit model can take a chest rig too).";

const words = (s: string | undefined) => (s || "").split(/\s+/).filter(Boolean).join(" ");

/** Why Animate cannot start, or "" when it can. A cycle's prompt may be empty: the factory makes one. */
export function animateBlock(prompt: string, pick: AnimatePick | undefined, anyJobRunning: boolean, kit: AnimateKit = {}): string {
  if (!pick) return "No skeletons to animate with.";
  if (!prompt && !pick.cycle) return "Say what happens and which way the character faces, for example: walks to the right, facing right.";
  if (words(kit.keep) && !words(kit.remove)) return "Keep goes with Remove: name the kit to erase too, or clear Keep.";
  if (anyJobRunning) return "The factory is busy with another job.";
  return "";
}

export type AnimateBody = ({ cycle: string; facing: string; prompt?: string } | { prompt: string; skeleton: string; action?: string }) & AnimateKit;

/** POST /2d/animate/<stem>'s body. A cycle goes as {cycle, facing}, with the prompt only when it is not the ready
 * one; the factory names its strip after the cycle and facing (run-left), so no cycle overwrites another. Another
 * skeleton goes with its prompt and its own strip name. */
export function animateBody(prompt: string, pick: AnimatePick, kit: AnimateKit = {}): AnimateBody {
  const c = pick.cycle;
  const k = pick.skeleton!;
  const body: AnimateBody = c
    ? prompt && prompt !== c.prompt
      ? { cycle: c.cycle, facing: c.facing, prompt }
      : { cycle: c.cycle, facing: c.facing }
    : k.action
      ? { prompt, skeleton: k.name, action: k.action }
      : { prompt, skeleton: k.name };
  // Remove and Keep go only when filled (Keep only with Remove), as one line of words each.
  const remove = words(kit.remove);
  const keep = words(kit.keep);
  if (remove) body.remove = remove;
  if (remove && keep) body.keep = keep;
  return body;
}

/** The strip an Animate job's body makes, for its end line. */
export function animateAction(body: unknown, offer: AnimateOffer): string {
  const b = (body || {}) as { action?: string; cycle?: string; facing?: string };
  if (b.action) return b.action;
  return offer.cycles.find((c) => c.cycle === b.cycle && c.facing === b.facing)?.action || (b.cycle ? `${b.cycle}-${b.facing || "right"}` : "");
}

/** The 2D status line when an Animate job ends here (`action`: the strip it was making). */
export function animateEndLine(state: string, error: string | undefined, action: string): string {
  if (state === "ok") return `Animated: the ${action || "new"} strip, saved as a new 2D version.`;
  if (state === "cancelled") return "Animate cancelled. The 2D files are as they were.";
  return error || "Animate failed.";
}

/* ---------------------------------------------------------------- a /2d/steps answer */

export interface StepsAnswer {
  center?: unknown;
  parts?: unknown;
  hear?: unknown;
  notes?: unknown;
}

export interface SheetState {
  parts: FlatPart[];
  index: number;
  center: string;
}

/**
 * What a revise answer changes (legacy flat-revise submit): a new center, new pieces (index 0), and the heard
 * file on the target part (voices/ after Say, sounds/ after Sound). `hear` is the file to play, `status` the notes.
 */
export function applySteps(
  prev: SheetState,
  mode: StepMode,
  prompt: string,
  targetId: string,
  data: StepsAnswer,
): SheetState & { hear: string; status: string } {
  let { parts, index, center } = prev;
  if (typeof data.center === "string" && data.center) center = data.center;
  if (mode === "pieces" && Array.isArray(data.parts)) {
    parts = mergeSheet({ parts: data.parts as SheetPart[] }).parts;
    index = parts.length ? 0 : -1;
  }
  const hear = typeof data.hear === "string" ? data.hear : data.hear ? String(data.hear) : "";
  if (hear) {
    const say = mode === "say" && hear.startsWith("voices/");
    const sound = mode === "sound" && hear.startsWith("sounds/");
    if (say || sound) {
      parts = parts.map((p) =>
        p.id !== targetId ? p : say ? { ...p, line: prompt, voice: hear } : { ...p, projectile: prompt, sound: hear },
      );
    }
  }
  return { parts, index, center, hear, status: notesLine(data.notes) };
}

export const notesLine = (notes: unknown) => (Array.isArray(notes) ? notes.map(String).join(" ") : "");

/** The Speak/Sound answer on the asset row: the new file when it is the right kind, and the notes. */
export function applyAudio(step: "say" | "sound", data: StepsAnswer): { rel: string; status: string } {
  const hear = data.hear ? String(data.hear) : "";
  const rel = hear.startsWith(step === "say" ? "voices/" : "sounds/") ? hear : "";
  return { rel, status: notesLine(data.notes) };
}

/** The clip label: the first file's name, else "Clip". */
export const clipLabel = (files: readonly { name: string }[] | null | undefined) => (files && files.length ? files[0].name : "Clip");

/** A finished 2D draw (legacy finishFlatJob apply): the center to show, whether to open Pieces, what to play, the
 * status line and the toast. */
export function finishFlat(name: string, result: { revise?: boolean; center?: unknown; pieces?: unknown; hear?: unknown; notes?: unknown } | null | undefined) {
  const r = result || {};
  return {
    center: !r.revise && typeof r.center === "string" && r.center ? r.center : "source",
    pieces: !!r.pieces,
    hear: r.hear ? String(r.hear) : "",
    status: notesLine(r.notes),
    toast: r.revise ? `Redrawn: ${name}` : `Sprite ready: ${name}`,
  };
}
