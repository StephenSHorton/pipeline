// Compare: an earlier version of the open stage beside the current one, two dioramas side by side (Stephen: "open
// previous versions of a model and compare them side by side when I click on a bone"). View only: the left side is
// a take's files read from out/<stem>/takes/<stage>/<id>/, and choosing it never picks it (that is a write,
// POST /flow/<s>/pick). The store is module level, like usePanels, so the versions panel (which sits outside the
// viewer) can set the left side. The rules are pure, for compare.test.ts.
import { create } from "zustand";
import type { FlowEntry, FlowTake } from "@/lib/flow";
import type { ViewerStage } from "./panels/panels";

interface CompareStore {
  /** The asset and stage compare is on for ("" when off). */
  stem: string;
  stage: ViewerStage | null;
  /** The take on the left, or null for the default (the version before the current one). */
  left: string | null;
  start: (stem: string, stage: ViewerStage) => void;
  stop: () => void;
  setLeft: (take: string) => void;
}

export const useCompare = create<CompareStore>((set) => ({
  stem: "",
  stage: null,
  left: null,
  start: (stem, stage) => set({ stem, stage, left: null }),
  stop: () => set({ stem: "", stage: null, left: null }),
  setLeft: (left) => set({ left }),
}));

/** Whether compare is on for this asset's stage. */
export const comparing = (s: { stem: string; stage: string | null }, stem: string, stage: string) => !!stem && s.stem === stem && s.stage === stage;

/** The take on screen now: the pick, unless the screen holds something not kept yet. */
export function currentTake(entry: FlowEntry | null | undefined): string | null {
  if (!entry || entry.unsaved || !entry.pick) return null;
  return entry.takes.some((t) => t.id === entry.pick) ? entry.pick : null;
}

/** The takes the left side can show: every one but the current. Compare is offered when there is one. */
export function compareTakes(entry: FlowEntry | null | undefined): FlowTake[] {
  const cur = currentTake(entry);
  return (entry?.takes ?? []).filter((t) => t.id !== cur);
}

/** The left side by default: the version before the current one; the newest kept one when the screen is unsaved; the
 * next one when the current is the first. null when there is nothing to compare with. */
export function defaultLeftTake(entry: FlowEntry | null | undefined): string | null {
  const takes = entry?.takes ?? [];
  const cur = currentTake(entry);
  if (!cur) return takes.length ? takes[takes.length - 1].id : null;
  const i = takes.findIndex((t) => t.id === cur);
  return (takes[i - 1] ?? takes[i + 1])?.id ?? null;
}

/** The chosen left take while it is still one to compare with, else the default. */
export function leftTake(entry: FlowEntry | null | undefined, chosen: string | null): string | null {
  if (chosen && compareTakes(entry).some((t) => t.id === chosen)) return chosen;
  return defaultLeftTake(entry);
}

/** The version number the versions panel shows for a take (1-based), or 0. */
export const versionNo = (entry: FlowEntry | null | undefined, id: string | null) => (id ? (entry?.takes.findIndex((t) => t.id === id) ?? -1) + 1 : 0);

/** The two side labels: "Version 2" and "Version 3 · current" ("Current · not kept" while unsaved). */
export function sideLabels(entry: FlowEntry | null | undefined, left: string | null): { left: string; right: string } {
  const n = versionNo(entry, left);
  const cur = versionNo(entry, currentTake(entry));
  return { left: n ? `Version ${n}` : "Earlier version", right: cur ? `Version ${cur} · current` : "Current · not kept" };
}

/** Rigged files a rig take may hold, in the viewer's order (candidates.ts riggedFiles). */
const RIG_FILES = ["rig/rigged_allclips.glb", "rigged.glb"];
/** A 3D take's finished models, in the order the viewer picks them (candidates.ts pickLod): psx, the photo bake. */
const MESH_FILES = ["psx.glb", "textured_photo.glb", "textured.glb", "mesh_lookdev.glb", "mesh.glb"];

/**
 * The files to try for the left side, under the take's folder: the file the right side shows (path under
 * out/<stem>/, so the two sides show the same kind of model) when the take has it, then the stage's usual order.
 */
export function takeModelFiles(stage: ViewerStage, files: string[], shown: string): string[] {
  const has = new Set(files);
  const order = [shown, ...(stage === "rig" ? RIG_FILES : MESH_FILES)];
  const out = [...new Set(order.filter((f) => f && /\.glb$/i.test(f) && has.has(f)))];
  if (out.length) return out;
  // An older take with other names: its first model outside the workbench.
  const any = files.find((f) => /\.glb$/i.test(f) && !f.startsWith("workbench/"));
  return any ? [any] : [];
}

/**
 * Bones matched by name: each left bone's index in the right skeleton, so the left heat map and bone colours use the
 * right's numbering (the bone list's). A left bone the right lacks gets an index past the right's bones.
 */
export function boneMap(left: string[], right: string[]): Int32Array {
  const at = new Map(right.map((n, i) => [n, i]));
  const out = new Int32Array(left.length);
  let extra = right.length;
  left.forEach((n, i) => (out[i] = at.get(n) ?? extra++));
  return out;
}

/** The right side's bone `bone` is not in the left skeleton (the left says so). */
export const missingBone = (left: string[], right: string[], bone: number) => bone >= 0 && bone < right.length && !left.includes(right[bone]);
