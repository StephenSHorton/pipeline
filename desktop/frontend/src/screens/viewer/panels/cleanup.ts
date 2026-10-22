// The Cleanup panel's pure logic and copy (legacy cleanup cubby, index.html 2090-2104; cleanAllowed, the Floaters
// and Carve handlers, worldCarveRadius and carveBall, 4860-4887, 5300-5370). CleanupPanel.tsx only lays these out;
// cleanupActions.ts runs them (cleanup*.ts, not actions.ts: Paint and Detail levels share this folder). No React,
// no fetch here.
import type { Lod } from "@/lib/api";

/** The hint line under the title (index.html 2096). */
export const CLEANUP_HINT = "Knock off generator scraps. Carve is a sphere delete in 3D.";

/** The Floaters button (index.html 2098). */
export const FLOATERS_LABEL = "Floaters";
export const FLOATERS_TITLE = "Delete disconnected bits. Keeps the largest solid.";

/** The Carve toggle (index.html 2099). */
export const CARVE_LABEL = "Carve";
export const CARVE_TITLE = "Click the mesh to delete a sphere of volume. Orbit with right-drag while this is on.";

/** The Radius slider (index.html 2101): a fraction of the mesh's bounding-box diagonal. */
export const RADIUS_LABEL = "Radius";
export const RADIUS_TITLE = "Carve sphere size, as a fraction of the mesh.";
export const RADIUS = { min: 0.02, max: 0.35, step: 0.01, value: 0.08 } as const;

/** The confirm's yes button (index.html 5314). */
export const CUT_A_COPY = "Cut a copy";

/** The /clean POST runs with no job, and the server has no busy check (unlike POST /stack): legacy jobOwns copy. */
export const WAIT_FOR_JOB = "Wait for the job on this asset to finish.";

/** The hover ball (legacy carveBall: 0xc45c3a, opacity 0.3, depthWrite off). */
export const CARVE_BALL = { color: "#c45c3a", opacity: 0.3 } as const;

/** The file /clean writes when the server's answer names none (legacy r.file || "mesh_clean.glb"). */
export const DEFAULT_CLEAN = "mesh_clean.glb";

/** Where a clean of `file` lands: psx* into psx_clean.glb, everything else into mesh_clean.glb (clean_mesh.py clean_name). */
export function cleanInto(file: string): string {
  return /^psx/.test(file) ? "psx_clean.glb" : DEFAULT_CLEAN;
}

/** The once-per-asset confirm (legacy cleanAllowed). `file` "" is a /glb fallback: the server picks the mesh. */
export function cleanAsk(file: string): string {
  return `Cleanup cuts a copy of ${file || "the mesh"} and saves it as ${cleanInto(file)}. The original stays.`;
}

/**
 * New copy (legacy had no such case): /clean's `from` takes a bare file in out/<stem>/, so a file under a folder
 * (the Rig stage's rig/rigged_allclips.glb) cannot be cut. Sending "" instead would cut a different mesh silently.
 */
export function nestedFileNote(file: string): string {
  return `Cleanup can't cut ${file}. Show a detail level first.`;
}

/** Why a clean can't start on `file` now, or "" when it can. */
export function cleanBlock(file: string, jobOnAsset: boolean): string {
  if (jobOnAsset) return WAIT_FOR_JOB;
  if (file.includes("/")) return nestedFileNote(file);
  return "";
}

export interface FloatersBody {
  stem: string;
  op: "floaters";
  keep: 1;
  from: string;
}

export interface SphereBody {
  stem: string;
  op: "sphere";
  center: [number, number, number];
  radius: number;
  from: string;
}

export type CleanBody = FloatersBody | SphereBody;

/** POST /clean for Floaters, exactly as legacy sent it (index.html 5326; min_faces is never sent). */
export function floatersBody(stem: string, from: string): FloatersBody {
  return { stem, op: "floaters", keep: 1, from };
}

/**
 * POST /clean for one carve (index.html 5357-5364): the hit in the hit mesh's own space, and the world radius
 * divided by that mesh's world scale, because clean_mesh.py measures in the GLB's units.
 */
export function sphereBody(stem: string, from: string, local: { x: number; y: number; z: number }, worldRadius: number, scale: number): SphereBody {
  return { stem, op: "sphere", center: [local.x, local.y, local.z], radius: worldRadius / (scale || 1), from };
}

/**
 * The carve sphere's world radius: the model's bounding-box diagonal times the slider (legacy worldCarveRadius).
 * An empty box counts as 1, as worldPaintRadius does; legacy carve sent radius 0 there.
 */
export function worldCarveRadius(diagonal: number, fraction: number): number {
  return (diagonal || 1) * fraction;
}

/** The slider readout (legacy Number(value).toFixed(2)). */
export function radiusText(value: number): string {
  return Number(value).toFixed(2);
}

/** The file the server wrote (legacy r.file || "mesh_clean.glb"). */
export function writtenFile(r: { file?: unknown } | null | undefined): string {
  return typeof r?.file === "string" && r.file ? r.file : DEFAULT_CLEAN;
}

/** The Detail levels entry that lists `file` (clean_mesh.py patch_lods: "clean" / "psx-clean"), or null. */
export function lodFor(lods: Lod[], file: string): Lod | null {
  return lods.find((l) => l.file === file) ?? null;
}
