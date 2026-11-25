// Pure logic for Make's two building modes (the Sprite stack kind): a new building from three drawn walls
// (POST /2d/building, a factory job) and buildings already made joined into a row or a corner (POST /2d/join, no
// model). The choices and their limits are the factory's (scripts/flat2d/building.py parse, scripts/stack_building.py,
// scripts/stack_join.py). Kept free of React so building.test.ts can pin the bodies.
import type { Asset, JobResult } from "@/lib/api";
import { slugStem } from "@/lib/names";

/** What the Sprite stack kind makes: anything (the usual Make), a building, or buildings joined. */
export type StackMode = "thing" | "building" | "join";
export const STACK_MODES: readonly { id: StackMode; label: string; hint: string }[] = [
  { id: "thing", label: "Anything", hint: "A photo or a few words, drawn from every angle" },
  { id: "building", label: "Building", hint: "A city building from three drawn walls: plain, front and back" },
  { id: "join", label: "Join buildings", hint: "Buildings you made, joined into a row or a corner" },
];
const MODE_KEY = "make-stack-mode";
export function loadStackMode(): StackMode {
  try {
    const v = localStorage.getItem(MODE_KEY);
    if (v && STACK_MODES.some((m) => m.id === v)) return v as StackMode;
  } catch {
    /* private window */
  }
  return "thing";
}
export function saveStackMode(m: StackMode): void {
  try {
    localStorage.setItem(MODE_KEY, m);
  } catch {
    /* ignore */
  }
}

/* ---------------------------------------------------------------- a new building */

export type Look = "crisp" | "smooth";
export const LOOKS: readonly { id: Look; label: string; hint: string }[] = [
  { id: "crisp", label: "Crisp", hint: "A small palette and a hard outline: pixel art" },
  { id: "smooth", label: "Smooth", hint: "Every cell its pixels' average colour: no colour limit" },
];
export type Roof = "flat" | "gable";
export const ROOFS: readonly { id: Roof; label: string; hint: string }[] = [
  { id: "flat", label: "Flat", hint: "A flat roof behind a parapet" },
  { id: "gable", label: "Gable", hint: "A pitched roof with a ridge" },
];
export type Ridge = "across" | "along";
export const RIDGES: readonly { id: Ridge; label: string; hint: string }[] = [
  { id: "across", label: "Across", hint: "The ridge runs across the front: the gable ends are on the sides" },
  { id: "along", label: "Along", hint: "The ridge runs front to back: a gable end faces the street" },
];
/** stack_building.ROOF_COLOURS, with their colours for the swatches. A custom colour is any #rrggbb. */
export const ROOF_COLOURS: readonly { id: string; label: string; hex: string }[] = [
  { id: "slate", label: "Slate", hex: "#4a5060" },
  { id: "tile", label: "Tile", hex: "#a84a34" },
  { id: "shingle", label: "Shingle", hex: "#684e3c" },
  { id: "green", label: "Green", hex: "#46705c" },
];
export const DEFAULT_ROOF_COLOUR = "slate";
/** stack_building.BAY_MAX: at most this many window bays on a face. null is "as drawn". */
export const BAY_MAX = 12;
export const BAY_CHOICES: readonly number[] = Array.from({ length: BAY_MAX }, (_, i) => i + 1);
/** flat2d/building.WORDS_MAX. */
export const WORDS_MAX = 1200;

/** The words the hand-made buildings were drawn from (2026-11-21): placeholders, so the fields say what they want. */
export const WALL_PLACEHOLDER = "a three-storey red brick building, sash windows on every floor, a stone cornice";
export const FRONT_PLACEHOLDER = "Add a shop front to the ground floor, with a dark green fabric awning across its top";
export const BACK_PLACEHOLDER = "Add a plain wooden back door to the ground floor";

export interface BuildingChoice {
  name: string;
  wall: string;
  front: string;
  back: string;
  look: Look;
  wide: number | null;
  deep: number | null;
  roof: Roof;
  ridge: Ridge;
  /** A ROOF_COLOURS id or #rrggbb. */
  roofColour: string;
}

export const DEFAULT_BUILDING: BuildingChoice = {
  name: "",
  wall: "",
  front: "",
  back: "",
  look: "crisp",
  wide: null,
  deep: null,
  roof: "flat",
  ridge: "across",
  roofColour: DEFAULT_ROOF_COLOUR,
};

const HEX = /^#[0-9a-f]{6}$/i;

/** What stops Make building, as one plain line; "" when it can go. */
export function buildingProblem(c: BuildingChoice): string {
  if (!c.wall.trim()) return "Describe the plain wall first.";
  if (!c.front.trim()) return "Say what to add to the front.";
  for (const [k, v] of [["plain wall", c.wall], ["front", c.front], ["back", c.back]] as const)
    if (v.trim().length > WORDS_MAX) return `The ${k} words are longer than ${WORDS_MAX} characters.`;
  if (c.roof === "gable" && !ROOF_COLOURS.some((r) => r.id === c.roofColour) && !HEX.test(c.roofColour)) return "The roof colour must be a colour like #8a4a34.";
  return "";
}

/** The name a building gets when you give none: from the plain wall's words, "building" when there are none. */
export function buildingName(c: Pick<BuildingChoice, "name" | "wall">): string {
  return slugStem(c.name) || slugStem(c.wall).split("-").slice(0, 4).join("-") || "building";
}

/** POST /2d/building. Width and depth only when picked (else as drawn); the ridge and the roof colour only for a
 * gable (a flat roof has neither). Seed, size and awning stay the factory's defaults. */
export function buildingBody(c: BuildingChoice, stem: string, project: string | null): Record<string, unknown> {
  const body: Record<string, unknown> = { stem, wall: c.wall.trim(), front: c.front.trim(), look: c.look, roof: c.roof, project };
  if (c.back.trim()) body.back = c.back.trim();
  if (c.wide !== null) body.wide = c.wide;
  if (c.deep !== null) body.deep = c.deep;
  if (c.roof === "gable") {
    body.ridge = c.ridge;
    body.roof_colour = c.roofColour.toLowerCase();
  }
  return body;
}

/** The Options summary: what Make will use, in a few words. */
export function buildingChips(c: BuildingChoice): string[] {
  const bays = (n: number | null) => (n === null ? "as drawn" : `${n} bay${n === 1 ? "" : "s"}`);
  const roof = c.roof === "gable" ? `Gable roof, ridge ${c.ridge}, ${ROOF_COLOURS.find((r) => r.id === c.roofColour)?.label.toLowerCase() ?? c.roofColour}` : "Flat roof";
  return [LOOKS.find((l) => l.id === c.look)!.label, `Wide ${bays(c.wide)}`, `Deep ${bays(c.deep)}`, roof];
}

/** The finish toast for a building: its size in cells and its bays. */
export function buildingReadyText(name: string, result: JobResult | null | undefined): string {
  const size = result?.stack?.size;
  const bays = result?.bays;
  const across = bays?.wide ?? bays?.drawn;
  const deep = bays?.deep ?? bays?.drawn;
  let text = `Building ready: ${name}`;
  if (size) text += ` · ${size.width} by ${size.length} cells, ${size.height} layers`;
  if (across && deep) text += ` · ${across} by ${deep} window bays`;
  return text;
}

/* ---------------------------------------------------------------- joining buildings */

export type Layout = "row" | "corner";
export const LAYOUTS: readonly { id: Layout; label: string; hint: string }[] = [
  { id: "row", label: "Row", hint: "Left to right, walls touching, fronts in one line: a terrace" },
  { id: "corner", label: "Corner", hint: "Two buildings: the main one, and a wing turned to face right behind it. An L with two street fronts" },
];
/** stack_join.MAX_PARTS. */
export const MAX_PARTS = 8;

/** The buildings a join may use: assets whose stack is a building, newest first (as /assets lists them). */
export function joinable(assets: readonly Asset[] | null | undefined): Asset[] {
  return (assets || []).filter((a) => a.has_stack && a.building);
}

/** What stops Join, as one plain line; "" when it can go. */
export function joinProblem(layout: Layout, parts: readonly string[]): string {
  if (layout === "corner") {
    if (parts.length !== 2) return "A corner is two buildings: pick the main one, then the wing.";
  } else if (parts.length < 2) return "Pick two or more buildings for the row.";
  if (parts.length > MAX_PARTS) return `Join at most ${MAX_PARTS} buildings.`;
  return "";
}

/** The joined building's name when you give none: its parts', "street" for a row and "corner" for a corner. */
export function joinName(name: string, layout: Layout, parts: readonly string[]): string {
  const given = slugStem(name);
  if (given) return given;
  const first = parts[0] ? parts[0].replace(/^city[-_]/, "") : "";
  return first ? slugStem(`${first} ${layout}`) : layout === "row" ? "street" : "corner";
}

/** POST /2d/join. Parts in order, left to right for a row; for a corner the main building, then the wing (the
 * factory turns the wing to face right). */
export function joinBody(stem: string, layout: Layout, parts: readonly string[], project: string | null): Record<string, unknown> {
  return { stem, layout, parts: [...parts], project };
}

/** The line after a join: its size, from the factory's answer. */
export function joinedText(name: string, out: { size?: { width: number; length: number; height: number }; parts?: unknown[] }): string {
  const s = out.size;
  return `Joined ${out.parts?.length ?? 2} buildings: ${name}${s ? ` · ${s.width} by ${s.length} cells, ${s.height} layers` : ""}`;
}

/* ---------------------------------------------------------------- a tower from a building (Stack stage) */

/** stack_tower.MAX_TIERS. */
export const MAX_TIERS = 5;
/** stack_building.FLOOR_MAX: at most this many floors in one tier. */
export const FLOOR_MAX = 40;
export const TIER_CHOICES: readonly number[] = Array.from({ length: MAX_TIERS }, (_, i) => i + 1);
export const FLOOR_CHOICES: readonly number[] = Array.from({ length: FLOOR_MAX }, (_, i) => i + 1);
export const DEFAULT_TIERS = 2;
export const DEFAULT_TIER_FLOORS = 4;

/** What a Version's stack.json says about a building, as far as a tower needs it. */
export interface TowerSource {
  maker?: string;
  shape?: string;
  bays?: { wide?: number; deep?: number; drawn?: number } | null;
}

/** A tower is made from a building drawn from walls (stack.json maker "walls", shape "code"): not from a joined
 * building or a tower, which have no drawn walls of their own. */
export const towerable = (meta: TowerSource | null | undefined): boolean => meta?.maker === "walls" && meta.shape === "code";

/** The building's own window bays [across, along]: as it was made (stack.json bays), null when it does not say. */
export function sourceBays(meta: TowerSource | null | undefined): [number, number] | null {
  const b = meta?.bays;
  const wide = b?.wide ?? b?.drawn;
  const deep = b?.deep ?? b?.drawn;
  return wide && deep ? [wide, deep] : null;
}

/** The tiers bottom first, [across, along] bays each: the building's own, then one window bay narrower on every
 * side (2 fewer across, 2 fewer along) each tier up. As stack_tower.stepped_tiers works them out. */
export function towerTiers(wide: number, deep: number, tiers: number): [number, number][] {
  return Array.from({ length: tiers }, (_, i) => [wide - 2 * i, deep - 2 * i] as [number, number]);
}

/** How many tiers the bays leave room for. */
export const towerRoom = (wide: number, deep: number): number => 1 + Math.floor((Math.min(wide, deep) - 1) / 2);

/** What stops Make tower, as one plain line; "" when it can go. The bays running out is said, never made fewer. */
export function towerProblem(bays: [number, number] | null, tiers: number, floors: number): string {
  if (!bays) return "This building's stack does not say its window bays: make it again to make a tower from it.";
  if (tiers < 1 || tiers > MAX_TIERS) return `A tower is 1 to ${MAX_TIERS} tiers.`;
  if (floors < 1 || floors > FLOOR_MAX) return `A tier is 1 to ${FLOOR_MAX} floors.`;
  const [w, d] = bays;
  const room = towerRoom(w, d);
  if (tiers > room)
    return `This building is ${w} window bay${w === 1 ? "" : "s"} across and ${d} along. Each tier steps in one bay on every side, so it has room for ${room} tier${room === 1 ? "" : "s"}, not ${tiers}.`;
  return "";
}

/** The tiers in a few words: "4 by 4 bays, then 2 by 2". */
export function tiersText(bays: [number, number], tiers: number): string {
  return towerTiers(bays[0], bays[1], tiers)
    .map(([w, d], i) => `${w} by ${d}${i === 0 ? " bays" : ""}`)
    .join(", then ");
}

/** The tower's name: the building's, with "tower". */
export const towerName = (source: string): string => slugStem(`${source} tower`) || "tower";

/** POST /2d/tower. The top tier's roof is the building's own (left out, the factory reads it). */
export function towerBody(stem: string, source: string, tiers: number, floors: number, project: string | null): Record<string, unknown> {
  return { stem, from: source, tiers, floors, project };
}

/** The line after a tower is made: its tiers and size, from the factory's answer. */
export function toweredText(name: string, out: { size?: { width: number; length: number; height: number }; tiers?: unknown[] }): string {
  const s = out.size;
  return `Tower of ${out.tiers?.length ?? 1} tiers: ${name}${s ? ` · ${s.width} by ${s.length} cells, ${s.height} layers` : ""}`;
}
