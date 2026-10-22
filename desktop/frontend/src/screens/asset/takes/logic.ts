// The versions panel as data (legacy renderFlow's #takes half, index.html 2854-3031, and buildFromConcept's intent
// merge, 3542-3577). Pure, so logic.test.ts can pin the exact copy; VersionsPanel.tsx only draws these items. The
// legacy page drew them as a strip under the stage; the panel sits in the left column instead (Stephen: "The
// versions ui can be on the left, don't make it take up huge space on the bottom center").
import { outUrl, type AssetOutput, type Facing, type FacingSource } from "@/lib/api";
import { conceptStill, flowEntry, staleNote, takeShot, type FlowState, type FlowTake, type Stage } from "@/lib/flow";
import { displayStem } from "@/lib/jobs";
import { intentBody, lookSpec, type NestedIntent } from "@/screens/home/make";

/** Legacy OUTPUTS names (lower case: "Build sprite stack", "Building the PS1 model · …"), in the legacy key order. */
export const OUTPUT_NAMES: Record<AssetOutput, string> = {
  ps1: "PS1 model",
  model: "3D model",
  stack: "sprite stack",
  sprite: "2D sprite",
};
const OUTPUT_ORDER = Object.keys(OUTPUT_NAMES) as AssetOutput[];

export const VERSIONS_TITLE = "Each build is kept. Click one to go back to it; nothing is deleted.";
export const KEEP_TITLE = "Save what is on screen as a version you can come back to.";

/** What a panel button does; VersionsPanel maps each to its call. */
export type StripAction = { do: "keep" } | { do: "build"; output: AssetOutput } | { do: "stack" } | { do: "remake" };

export type StripItem =
  | { type: "label"; text: string; title: string }
  | { type: "take"; id: string; n: number; pressed: boolean; src: string; title: string; level?: string }
  | { type: "act"; text: string; title: string; action: StripAction; primary?: boolean }
  | { type: "note"; text: string; stale?: boolean; busy?: boolean };

export interface StripInput {
  stem: string;
  stage: Stage;
  flow: FlowState | null;
  /** The saved intent's output (out/<stem>/intent.json), if any. */
  output?: AssetOutput | null;
  /** "Stack from picture" is running. */
  stacking?: boolean;
}

/** One version's tooltip: "Version N · note · date"; a rig version names its checks' line instead. */
export function takeTitle(stage: Stage, take: FlowTake, i: number): string {
  const made = Number(take.made);
  const when = made ? new Date(made * 1000).toLocaleString() : "";
  if (stage === "rig") {
    const sum = take.summary;
    const checked = (take.files || []).includes("rig/inspect.json");
    const line = sum?.line ? sum.line : checked ? "Checked before clips were measured" : "Made before rig checks";
    return [`Version ${i + 1}`, line, take.note, when].filter(Boolean).join(" · ");
  }
  return [`Version ${i + 1}`, take.note, when].filter(Boolean).join(" · ");
}

/** The version's thumbnail (/out/<s>/takes/<stage>/<id>/<shot>), or "" for a blank tile. Takes never change: no bust. */
export function takeSrc(stem: string, stage: Stage, take: FlowTake): string {
  const shot = takeShot(stage, take.files);
  return shot ? outUrl(stem, `takes/${stage}/${take.id}/${shot}`, false) : "";
}

/** Everything the strip shows for the open stage, in order. [] hides the strip (Ideas, no flow, nothing to say). */
export function stripItems({ stem, stage, flow, output, stacking }: StripInput): StripItem[] {
  if (stage === "ideas") return [];
  const entry = flowEntry(flow, stage);
  if (!stem || !entry) return [];
  const items: StripItem[] = [];
  const act = (text: string, title: string, action: StripAction, primary?: boolean) => items.push({ type: "act", text, title, action, primary });
  const note = (text: string, extra: { stale?: boolean; busy?: boolean } = {}) => items.push({ type: "note", text, ...extra });

  // One version has nothing to switch to: the versions show from two, or from one when the screen differs from it.
  const showVersions = entry.takes.length > 1 || (entry.takes.length > 0 && entry.unsaved);
  if (showVersions) {
    items.push({ type: "label", text: "Versions", title: VERSIONS_TITLE });
    entry.takes.forEach((take, i) =>
      items.push({
        type: "take",
        id: take.id,
        n: i + 1,
        pressed: take.id === entry.pick && !entry.unsaved,
        src: takeSrc(stem, stage, take),
        title: takeTitle(stage, take, i),
        ...(stage === "rig" && take.summary?.level ? { level: String(take.summary.level) } : {}),
      }),
    );
  }
  if (entry.unsaved) act("Keep this version", KEEP_TITLE, { do: "keep" });
  const stale = staleNote(entry);
  if (stale) note(stale, { stale: true });

  const still = conceptStill(flow, stem);
  const mesh = flowEntry(flow, "mesh");
  if (stage === "export" || stage === "rig") {
    // Their own controls (Export / Show in Finder; clips, checks, body plan, Rig it) belong to those stages.
  } else if (stage === "concept" && still) {
    const want = output || "";
    const order = want ? [want, ...OUTPUT_ORDER.filter((k) => k !== want)] : OUTPUT_ORDER;
    note(want ? "Next:" : "Make from this picture:");
    for (const out of order) {
      const built = out !== "sprite" && !!mesh?.live && want === out;
      const label = out === "sprite" ? "2D sprite" : `${built ? "Rebuild" : "Build"} ${OUTPUT_NAMES[out]}`;
      const title = out === "sprite" ? "Cut this picture into pieces, add motion and sound." : `Make the ${OUTPUT_NAMES[out]} from this picture.`;
      act(label, title, { do: "build", output: out }, out === want);
    }
  } else if (stage === "mesh" || stage === "stack") {
    const want: AssetOutput = output || (stage === "stack" ? "stack" : "model");
    if (stage === "stack" && still) {
      if (stacking) note("Stacking the picture…", { busy: true });
      else act("Stack from picture", "Build the sprite stack straight from the concept picture. No 3D needed.", { do: "stack" });
    }
    if (still && (!mesh || !mesh.live || mesh.stale.length)) {
      const name = OUTPUT_NAMES[want] ?? "3D model";
      act(`${mesh?.live ? "Rebuild" : "Build"} ${name}`, `Make the ${name} from the picked concept.`, { do: "build", output: want }, true);
    } else if (!mesh || !mesh.live) {
      note("No concept picture yet.");
    }
    if (still && mesh?.live && stage === "mesh") {
      act("Change settings & remake…", "Open the Make form with this asset's settings and picture, and rebuild it with changes.", { do: "remake" });
    }
  } else if (!entry.live && !entry.takes.length) {
    note(stage === "concept" ? "No picture yet." : "Use a step below to start the 2D sheet.");
  }
  return items;
}

/* ---------------------------------------------------------------- the panel's sections */

type LabelItem = Extract<StripItem, { type: "label" }>;
type TakeItem = Extract<StripItem, { type: "take" }>;
type NoteItem = Extract<StripItem, { type: "note" }>;
type RestItem = Extract<StripItem, { type: "act" | "note" }>;

/** The panel's heading when it has no versions to show, only the next step (a Build, a note). */
export const NEXT_HEADING = "Next step";

export interface PanelGroups {
  /** The "Versions" label, when the versions show. */
  label: LabelItem | null;
  takes: TakeItem[];
  /** Keep, the stale note and the next step, in stripItems' order. */
  rest: RestItem[];
  /** The busy note (Stacking the picture…), read out by the panel's status line. */
  busy: NoteItem | null;
}

/** stripItems split into the panel's parts: the heading, the thumbnails, and the buttons and notes. */
export function groupItems(items: StripItem[]): PanelGroups {
  return {
    label: items.find((it): it is LabelItem => it.type === "label") ?? null,
    takes: items.filter((it): it is TakeItem => it.type === "take"),
    rest: items.filter((it): it is RestItem => it.type === "act" || it.type === "note"),
    busy: items.find((it): it is NoteItem => it.type === "note" && !!it.busy) ?? null,
  };
}

/** The panel heading: the versions label, else "Next step". */
export const panelHeading = (g: PanelGroups): string => g.label?.text ?? NEXT_HEADING;

/** The short line beside the heading, shown folded too: "2 of 3", or "3 kept" while the screen holds something
 * unsaved (no version is the live one). "" without versions. */
export function versionsSummary(takes: TakeItem[]): string {
  if (!takes.length) return "";
  const live = takes.find((t) => t.pressed);
  return live ? `${live.n} of ${takes.length}` : `${takes.length} kept`;
}

/* ---------------------------------------------------------------- build from concept (§8.3) */

export interface FlatKey {
  facing?: Facing;
  canonical?: boolean;
  facing_source?: FacingSource;
}

/** The intent a build uses: the saved one when it is for this output, else a fresh one carrying the saved subject,
 * kind, name, rig and facing source (and, for PS1, its psx with a height, 48 by default). */
export function buildIntent(saved: NestedIntent | null, output?: AssetOutput): NestedIntent {
  if (saved && (!output || saved.output === output)) return saved;
  const intent: NestedIntent = { output: output || "model" };
  if (saved) for (const k of ["subject", "kind", "name", "rig", "facing_source"] as const) if (saved[k]) (intent as unknown as Record<string, unknown>)[k] = saved[k];
  if (intent.output === "ps1") intent.psx = { ...(saved?.psx || {}), height: Number(saved?.psx?.height) || 48 };
  return intent;
}

/** A PS1 built from the drawn concept (flat/source.png) faces the way flat/key.json says, unless the user set it. */
export function withKeyFacing(intent: NestedIntent, still: string, key: FlatKey | null): NestedIntent {
  if (intent.output !== "ps1" || !still.endsWith("flat/source.png")) return intent;
  const userSet = intent.facing_source === "user" && !!intent.psx?.facing;
  if (!key?.facing || userSet) return intent;
  return { ...intent, psx: { ...(intent.psx || {}), facing: key.facing }, facing_source: key.canonical ? "drawn" : key.facing_source || "detected" };
}

/** POST /generate body. poly: the intent's, else "medium" (the legacy page fell back to the Home form's pick). */
export function buildSpec(stem: string, still: string, intent: NestedIntent): Record<string, unknown> {
  return { image: still, images: [still], stem, ...lookSpec(intent), poly: intent.poly || "medium", intent: intentBody(intent) };
}

export const buildTitle = (stem: string, output: AssetOutput) => `Building the ${OUTPUT_NAMES[output]} · ${displayStem(stem)}`;

/** The stage a build lands on: the stack for a sprite stack, else the mesh. */
export const buildStage = (output: AssetOutput): Stage => (output === "stack" ? "stack" : "mesh");
