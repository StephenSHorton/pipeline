// The Export stage's words and cards, as plain data (legacy renderExport / psxCard / rigCard / rigCardFacts and the
// export branch of renderFlow, index.html 2905-2913, 3361-3507). ExportStage.tsx only lays these out.
import type { Intent } from "@/lib/api";
import { flowEntry, isLive, OUTPUT_STAGES, type FlowEntry, type FlowState } from "@/lib/flow";

/* ---------------------------------------------------------------- export/asset.json (scripts/export_asset.py) */

export interface ExportFile {
  file: string;
  /** asset.json carries more per role (from, has_uvs, mtl, facing, sampling…); only what the cards read is named. */
  [k: string]: unknown;
  faces?: number;
}

export interface ExportPsx extends ExportFile {
  obj?: string;
  mtl?: string;
  texture?: string;
  atlas?: number;
  triangles?: number;
  height?: number | string;
  size?: number[];
  axes?: { obj?: string; glb?: string };
}

export interface ExportRigged extends ExportFile {
  class?: string;
  clips?: string[];
  joints?: number;
  fbx?: string;
  fbx_texture?: string;
}

export interface ExportDoc {
  made?: number;
  concept?: string;
  "2d"?: {
    parts?: { image: string; sound?: string; voice?: string }[];
    motions?: Record<string, { frames: number; fps: number }>;
  };
  stack?: { layers: number; resolution: number; maker?: string; preview?: string };
  "3d"?: { psx?: ExportPsx; rigged?: ExportRigged; [role: string]: ExportFile | undefined };
}

/** rig/fbx_check.json (or rig.json's fbx_check): the FBX read back in Blender. */
export interface FbxCheck {
  ok?: boolean;
  actions?: unknown[] | number;
  bones?: unknown[] | number;
  max_influences?: number;
  scene_fps?: number;
  warnings?: string[];
  problems?: string[];
}

/* ---------------------------------------------------------------- copy */

export const LEAD_EMPTY =
  "Export copies what this asset made into one folder a game can load, with asset.json saying what each file is. Press Export below.";
export const EXPORT_TITLE = "Copy the picked 2D sheet, sprite stack and 3D files into one folder a game can load.";
export const NOTHING_TO_EXPORT = "Make a 2D sheet, a sprite stack or a 3D mesh first.";
export const FINDER_TITLE = "Open the export folder.";
export const finderFailed = (stem: string) => `Could not open Finder. The export is in out/${stem}/export/.`;

/** Body plan names (legacy RIG_NAMES). */
export const RIG_NAMES: Record<string, string> = {
  "": "Body plan: auto",
  humanoid: "Person",
  quadruped: "Four legs",
  multileg: "Many legs",
  avian: "Bird",
};

/** Plain names for the 3D roles; the PS1 model gets its own card. No game / proxy LODs: export no longer writes them
 * (an older export's asset.json that has them shows the bare role name until it is exported again). */
const ROLE: Record<string, string> = { model: "Model", pixel: "Pixel", rigged: "Rigged" };

const base = (p: string) => p.split("/").pop() ?? p;

/* ---------------------------------------------------------------- the action row (legacy takes strip) */

export type ExportAction =
  | { kind: "exporting"; finder: false }
  | { kind: "button"; label: "Export" | "Export again"; finder: boolean }
  | { kind: "none"; finder: boolean };

/** "Exporting…" while it runs; Export / Export again once a 2D sheet, mesh or stack is live; else the reason. Show in
 * Finder whenever the export stage holds files and nothing is running. */
export function exportAction(flow: FlowState | null | undefined, exporting: boolean): ExportAction {
  if (exporting) return { kind: "exporting", finder: false };
  const live = isLive(flow, "export");
  if ((["flat", "mesh", "stack"] as const).some((s) => isLive(flow, s))) return { kind: "button", label: live ? "Export again" : "Export", finder: live };
  return { kind: "none", finder: live };
}

/* ---------------------------------------------------------------- cards */

export interface PlainCard {
  kind: "plain";
  key: string;
  title: string;
  lines: string[];
  /** Root-relative under out/<stem>/export/, or "". */
  img: string;
  /** Dashed: nothing there yet / not in this export. */
  missing: boolean;
}

export interface PsxCard {
  kind: "psx";
  key: "psx";
  title: string;
  name: string;
  texture: string;
  atlas: string;
  triangles: string;
  height: string;
  size: string;
  axes: string;
  /** The Create Model step's last sentence: what the importer does with these axes (psxImportNote). */
  importNote: string;
}

export interface RigCard {
  kind: "rig";
  key: "rig";
  title: string;
  fbx: string;
  texture: string;
  bones: number;
  clipCount: number;
  clips: string;
}

export type ExportCard = PlainCard | PsxCard | RigCard;

const plain = (key: string, title: string, lines: string[], img = "", missing = false): PlainCard => ({ kind: "plain", key, title, lines, img, missing });

type IntentLike = Pick<Intent, "output"> | null | undefined;

/** The doc counts only when the export stage is live (a picked take holds it). */
export function showsDoc(doc: ExportDoc | null, entry: FlowEntry | null): doc is ExportDoc {
  return !!doc && !!entry?.live;
}

/** The line under the heading. `when` is the made time already formatted (toLocaleString in the page). */
export function leadText(stem: string, doc: ExportDoc | null, entry: FlowEntry | null, when: string): string {
  if (!showsDoc(doc, entry)) return LEAD_EMPTY;
  return `out/${stem}/export/ · ${when}` + (entry?.stale.length ? " · made from older picks, export again to update" : "");
}

/** The cards in their final order: Rigged hero, PS1 hero, then the rest (legacy prepends the heroes). */
export function exportCards(flow: FlowState | null | undefined, intent: IntentLike, doc: ExportDoc | null): ExportCard[] {
  const want = intent?.output ? OUTPUT_STAGES[intent.output] : undefined;
  const has = (s: "flat" | "stack" | "mesh") => isLive(flow, s);
  const wanted = (s: "flat" | "stack" | "mesh") => !want || want.includes(s) || has(s);
  const out: ExportCard[] = [];

  if (!showsDoc(doc, flowEntry(flow, "export"))) {
    if (wanted("flat")) out.push(plain("flat", "2D sheet", [has("flat") ? "Ready" : "Nothing yet"], "", !has("flat")));
    if (wanted("stack")) out.push(plain("stack", "Sprite stack", [has("stack") ? "Ready" : "Nothing yet"], "", !has("stack")));
    if (wanted("mesh")) out.push(plain("mesh", intent?.output === "ps1" ? "PS1 model" : "3D", [has("mesh") ? "Ready" : "Nothing yet"], "", !has("mesh")));
    return out;
  }

  if (doc.concept) out.push(plain("concept", "Concept", [doc.concept], doc.concept));
  const flat = doc["2d"];
  if (flat) {
    const lines = (flat.parts ?? []).map((p) => `${base(p.image)}${p.sound ? " + sound" : ""}${p.voice ? " + voice" : ""}`);
    for (const [name, m] of Object.entries(flat.motions ?? {})) lines.push(`${name}: ${m.frames} frames at ${m.fps} fps`);
    const shot = (flat.parts ?? [])[0];
    out.push(plain("flat", "2D sheet", lines.length ? lines : ["empty"], shot ? shot.image : ""));
  } else if (wanted("flat")) out.push(plain("flat", "2D sheet", ["Not in this export"], "", true));
  const stack = doc.stack;
  if (stack) {
    out.push(
      plain(
        "stack",
        "Sprite stack",
        [`${stack.layers} layers of ${stack.resolution}×${stack.resolution}`, stack.maker === "picture" ? "built from the picture" : "sliced from the mesh"],
        stack.preview ?? "",
      ),
    );
  } else if (wanted("stack")) out.push(plain("stack", "Sprite stack", ["Not in this export"], "", true));
  const mesh = doc["3d"];
  if (mesh) {
    const lines = Object.entries(mesh)
      .filter((e): e is [string, ExportFile] => e[0] !== "psx" && !!e[1])
      .map(([role, m]) => `${ROLE[role] || role}: ${base(m.file)}${m.faces ? ` · ${Number(m.faces).toLocaleString()} faces` : ""}`);
    if (lines.length) out.push(plain("mesh", mesh.psx ? "Other 3D files" : "3D", lines));
  } else if (!want || want.includes("mesh")) out.push(plain("mesh", "3D", ["Not in this export"], "", true));

  const heroes: ExportCard[] = [];
  if (mesh?.rigged?.fbx) heroes.push(rigCard(mesh.rigged));
  if (mesh?.psx) heroes.push(psxCard(mesh.psx));
  return [...heroes, ...out];
}

export function psxCard(p: ExportPsx): PsxCard {
  const name = base(p.obj || "model.obj").replace(/\.obj$/, "");
  return {
    kind: "psx",
    key: "psx",
    title: "PS1 model · ready for the game",
    name,
    texture: p.texture || "",
    atlas: String(p.atlas || 256),
    triangles: Number(p.triangles || 0).toLocaleString(),
    height: String(p.height || ""),
    size: Array.isArray(p.size) ? p.size.map((n) => Math.round(n)).join(" × ") : "",
    axes: p.axes?.obj || "z-up, +x front",
    importNote: psxImportNote(p.axes?.obj || "z-up, +x front"),
  };
}

/** s&box's .obj importer reads glTF axes and turns them to Source itself, so a y-up file needs nothing and a z-up
 * one (the old default, and what a psx.json with no axes was made with) is turned twice unless rotated at import. */
export function psxImportNote(axes: string): string {
  return /^y-up/.test(axes)
    ? "It is already in inches, Y up, facing +Z: s&box turns it upright itself."
    : "It is in inches, Z up, facing +X: set its import rotation to -90 0 -90 or it lands on its side.";
}

export function rigCard(r: ExportRigged): RigCard {
  const clips = r.clips ?? [];
  return {
    kind: "rig",
    key: "rig",
    // Legacy RIG_NAMES[r.class]: no class at all is "", not the "" key's "Body plan: auto".
    title: `Rigged model · ${(r.class != null ? RIG_NAMES[r.class] : undefined) || r.class || ""}`,
    fbx: base(r.fbx ?? ""),
    texture: r.fbx_texture ? base(r.fbx_texture) : "",
    bones: Number(r.joints || 0),
    clipCount: clips.length,
    clips: clips.map((c) => c.replace(/_/g, " ")).join(", "),
  };
}

/** The rig card's FBX line: checked (with a warning count), failed, or rig.json's fbx_note; null when there is none. */
export function fbxLine(fc: FbxCheck | null | undefined, note: unknown): { text: string; title: string; fail: boolean } | null {
  if (!fc && !note) return null;
  const count = (v: unknown[] | number | undefined) => (Array.isArray(v) ? v.length : Number(v || 0));
  const warnings = fc?.warnings ?? [];
  const text = fc?.ok
    ? `FBX checked: ${count(fc.actions)} clips, ${count(fc.bones)} bones, ${Number(fc.max_influences || 0)} weights max, ${Number(fc.scene_fps || 0)} fps`
    : fc
      ? `FBX check failed: ${(fc.problems || ["no report"]).join("; ")}`
      : String(note);
  const suffix = fc?.ok && warnings.length ? ` · ${warnings.length} warning${warnings.length === 1 ? "" : "s"}` : "";
  return { text: text + suffix, title: fc && warnings.length ? warnings.join("\n") : "", fail: !!(fc && !fc.ok) };
}
