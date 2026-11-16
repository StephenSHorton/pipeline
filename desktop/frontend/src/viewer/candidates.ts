// Pure helpers for the viewer: which GLB to load (the legacy page's order), which clip plays first,
// the rename slug and the kind names. No three, no DOM: unit-tested in candidates.test.ts.
import { meshUrls, outUrl, type AssetOutput, type Lod } from "@/lib/api";

/** The LOD id that means "the rigged model" (the legacy page's convention, ?lod=rigged). */
export const RIGGED = "rigged";

/** The label the finished textured model goes by, whichever file it is (scripts/lods.py's "photo" entry). */
export const PHOTO_BAKE = "Photo bake";

/**
 * Finished models the picker offers, in lods.json order: the photo bake, the PS1 model, and what the app's own tools
 * saved (Painted, Clean, PSX clean). Stephen: the Model tabs show finished models, not detail levels, so hero /
 * lookdev (steps on the way), game / proxy (no longer made) and the other intermediates (2mv, quads, PBR, AniGen …)
 * stay off even when an older lods.json lists them. An older asset with no photo entry shows its paint (the lookdev
 * entry's textured.glb, else PBR) as the Photo bake, as scripts/lods.py now writes it. Rigged is its own button.
 */
const FINISHED = new Set(["photo", "psx", "paint", "clean", "psx-clean"]);

export function viewableLods(lods: Lod[] | null | undefined): Lod[] {
  const all = (lods ?? []).filter((l) => l && l.file && l.kind !== "stack" && !String(l.id).startsWith("pixel"));
  const paint = all.some((l) => l.id === "photo")
    ? null
    : (all.find((l) => l.id === "lookdev" && /^textured\.glb$/i.test(l.file)) ?? all.find((l) => l.id === "pbr"));
  return all.flatMap((l) => (FINISHED.has(l.id) ? [l] : l === paint ? [{ ...l, label: PHOTO_BAKE }] : []));
}

/**
 * The initial LOD (legacy ix:5263-5279): if the preference is lookdev and a psx LOD exists, psx; else the preferred
 * id, then the photo bake (also an older asset's lookdev / pbr paint standing in for it), then the first.
 */
export function pickLod(lods: Lod[], prefer = "lookdev"): Lod | null {
  if (!lods.length) return null;
  const by = (id: string) => lods.find((l) => l.id === id);
  if (prefer === "lookdev" && by("psx")) return by("psx")!;
  return by(prefer) ?? by("photo") ?? by("lookdev") ?? by("pbr") ?? lods[0];
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

/**
 * The pane tag (ix:5130): Rigged model / PS1 model / Photo bake / 3D model. An older asset's lookdev / pbr paint
 * (textured.glb, pbr.glb) is tagged Photo bake too, so the tag matches the tab viewableLods gives it.
 */
export function fileTag(file: string, lod: string): string {
  if (lod === RIGGED || /rigged/i.test(file)) return "Rigged model";
  if (/^psx/i.test(file) || lod === "psx") return "PS1 model";
  if (/photo/i.test(file) || lod === "photo") return PHOTO_BAKE;
  if ((lod === "lookdev" || lod === "pbr") && /^(textured|pbr)\.glb$/i.test(file)) return PHOTO_BAKE;
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
