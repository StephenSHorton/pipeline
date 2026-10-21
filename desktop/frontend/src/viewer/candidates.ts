// Pure helpers for the viewer: which GLB to load (the legacy page's order), which clip plays first,
// the rename slug and the kind names. No three, no DOM: unit-tested in candidates.test.ts.
import { meshUrls, outUrl, type AssetOutput, type Lod } from "@/lib/api";

/** The LOD id that means "the rigged model" (the legacy page's convention, ?lod=rigged). */
export const RIGGED = "rigged";

/** LODs the picker offers: lods.json minus the stack and pixel* entries (legacy load(), ix:5248). */
export function viewableLods(lods: Lod[] | null | undefined): Lod[] {
  return (lods ?? []).filter((l) => l && l.file && l.kind !== "stack" && !String(l.id).startsWith("pixel"));
}

/**
 * The initial LOD (legacy ix:5263-5279): if the preference is lookdev and a psx LOD exists, psx;
 * else the preferred id, then lookdev, pbr, hero, then the first.
 */
export function pickLod(lods: Lod[], prefer = "lookdev"): Lod | null {
  if (!lods.length) return null;
  const by = (id: string) => lods.find((l) => l.id === id);
  if (prefer === "lookdev" && by("psx")) return by("psx")!;
  return by(prefer) ?? by("lookdev") ?? by("pbr") ?? by("hero") ?? lods[0];
}

/** Rigged files in the order to try: rig/rigged_allclips.glb, rig.json's rigged file, then rigged.glb. */
export function riggedFiles(opts: { allClips: boolean; rigJsonRigged?: unknown }): string[] {
  const out: string[] = [];
  if (opts.allClips) out.push("rig/rigged_allclips.glb");
  if (typeof opts.rigJsonRigged === "string" && /\.glb$/i.test(opts.rigJsonRigged)) out.push(opts.rigJsonRigged.replace(/^\/+/, ""));
  out.push("rigged.glb");
  return [...new Set(out)];
}

export interface CandidateInput {
  stem: string;
  /** The LOD the user asked for (?lod= or the picker). "" or undefined: the default. */
  lod?: string;
  lods: Lod[];
  /** True when a rigged file is known to exist (allclips HEAD ok or rig.json names one). */
  hasRig: boolean;
  rigFiles: string[];
  t?: number;
}

export interface Candidates {
  /** The LOD id shown (RIGGED for the rigged model, "" when only the /glb chain is left). */
  lod: string;
  urls: string[];
}

/**
 * Every URL to try, in order. Explicit LOD first (?lod=rigged or the Rigged tab for the rigged model and its
 * clips); with none, the legacy pick from lods.json (psx, then lookdev, pbr, hero: ix:5263-5279), so a rigged
 * PS1 asset still opens on its PS1 model with nearest filtering. Always followed by the legacy meshUrls chain.
 */
export function modelCandidates(c: CandidateInput): Candidates {
  const t = c.t ?? Date.now();
  const lods = viewableLods(c.lods);
  const want = c.lod || "";
  if (want === RIGGED) {
    return { lod: RIGGED, urls: dedupe([...c.rigFiles.map((f) => outUrl(c.stem, f, t)), ...meshUrls(c.stem, "lookdev")]) };
  }
  const pick = want ? (lods.find((l) => l.id === want) ?? pickLod(lods, want)) : pickLod(lods, "lookdev");
  const urls = [...(pick ? [outUrl(c.stem, pick.file, t)] : []), ...meshUrls(c.stem, pick?.id || want || "lookdev")];
  return { lod: pick?.id ?? want, urls: dedupe(urls) };
}

function dedupe(urls: string[]): string[] {
  const seen = new Set<string>();
  return urls.filter((u) => {
    const k = u.replace(/[?&]t=\d+/, "");
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** The clip that plays first: Idle_A, Idle, idle, Walk, walk, else the first (legacy setupClips). */
export function preferredClip(names: string[]): string | null {
  for (const n of ["Idle_A", "Idle", "idle", "Walk", "walk"]) if (names.includes(n)) return n;
  return names[0] ?? null;
}

export { slugStem } from "@/lib/names";

/** The legacy OUTPUTS names (ix:2368). */
export const KIND_NAMES: Record<AssetOutput, string> = {
  ps1: "PS1 model",
  model: "3D model",
  stack: "sprite stack",
  sprite: "2D sprite",
};

/** The pane tag (ix:5130): Rigged model / PS1 model / Photo bake / 3D model. */
export function fileTag(file: string, lod: string): string {
  if (lod === RIGGED || /rigged/i.test(file)) return "Rigged model";
  if (/^psx/i.test(file) || lod === "psx") return "PS1 model";
  if (/photo/i.test(file) || lod === "photo") return "Photo bake";
  return "3D model";
}

/** PS1/pixel textures get nearest filtering (legacy loadFrom: pixel* LODs, psx files). */
export function isPixelFile(file: string, lod: string): boolean {
  return lod.startsWith("pixel") || lod === "psx" || /pixel/i.test(file) || /^psx/i.test(file);
}

/** The last path segment of a URL, without its query. */
export function fileOf(url: string): string {
  const path = url.split("?")[0];
  return decodeURIComponent(path.slice(path.lastIndexOf("/") + 1));
}
