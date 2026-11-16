// What Floaters and a carve click do (legacy cleanAllowed and the Floaters / Carve handlers, index.html 5311-5370).
// Side effects only through POST /clean, askConfirm, toast and the injected viewer seams, so the tests mock those.
import type { Lod } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { askConfirm, toast } from "@/lib/ui";
import { jobOwns } from "@/screens/asset/flat/actions";
import type { ToolHit, ViewerTools } from "@/viewer/tools";
import { cleanAsk, cleanBlock, CUT_A_COPY, floatersBody, lodFor, sphereBody, writtenFile, type CleanBody } from "./cleanup";
import { postClean, useCleaning } from "./cleanupApi";

export interface CleanDeps {
  tools: Pick<ViewerTools, "currentFile" | "loadFile">;
  /** The Detail levels list as it stands now: loadFile re-reads lods.json, so read it after, not before. */
  lods(): Lod[];
  /** Show a LOD (the HUD picker's onPickLod). */
  pickLod(id: string): void;
  /** Told the written file before it loads, so a held Carve knows the coming model change is its own. */
  onWritten?(file: string): void;
  /** Re-read the flow; defaults to useFlow's load when the open asset is `stem`. */
  reloadFlow?(stem: string): Promise<unknown>;
}

// The asset the confirm was accepted for (legacy cleanAsked): once per asset per session, not once per LOD.
let cleanAsked = "";
// A clean between its first check and its last reload: a second click is dropped, not queued.
let pending = false;

/** Tests only: forget the accepted confirm and any clean in flight. */
export function resetCleanup(): void {
  cleanAsked = "";
  pending = false;
  useCleaning.setState({ busy: false });
}

async function cleanAllowed(stem: string, file: string): Promise<boolean> {
  if (cleanAsked === stem) return true;
  if (!(await askConfirm(cleanAsk(file), CUT_A_COPY))) return false;
  cleanAsked = stem;
  return true;
}

const flowReload = async (stem: string) => {
  if (useFlow.getState().stem === stem) await useFlow.getState().load(stem);
};

/**
 * Check, ask once, POST /clean, then show the written file. Returns the written file, or null when nothing was
 * cut (declined, blocked, busy or refused; a block or a refusal is a toast).
 */
async function clean(stem: string, body: (from: string) => CleanBody, deps: CleanDeps): Promise<string | null> {
  if (pending) return null;
  const from = deps.tools.currentFile();
  const block = cleanBlock(from, jobOwns(stem));
  if (block) {
    toast(block);
    return null;
  }
  pending = true;
  try {
    if (!(await cleanAllowed(stem, from))) return null;
    useCleaning.setState({ busy: true });
    let file: string;
    try {
      file = writtenFile(await postClean(body(from)));
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err));
      return null;
    }
    deps.onWritten?.(file);
    try {
      // Later cleans and carves cut this copy, so carves add up (legacy currentFile = the cleaned file).
      await deps.tools.loadFile(file);
      // Pin its LOD (patch_lods put "clean" / "psx-clean" first in lods.json): the flow re-read below sends the
      // viewer back to its usual candidates, which would put the uncut file back on screen.
      const lod = lodFor(deps.lods(), file);
      if (lod) deps.pickLod(lod.id);
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err));
    }
    // /clean is not one of the POSTs that mark the flow dirty, and it does no autosave: re-read it so the mesh
    // stage shows unsaved ("Keep this version"). Legacy never did, so the strip lagged until the next load.
    await (deps.reloadFlow ?? flowReload)(stem).catch(() => {});
    return file;
  } finally {
    pending = false;
    useCleaning.setState({ busy: false });
  }
}

/** The Floaters button: keep the largest connected part of the mesh on screen. */
export function floaters(stem: string, deps: CleanDeps): Promise<string | null> {
  return clean(stem, (from) => floatersBody(stem, from), deps);
}

/** A carve click that hit the model: delete a sphere of `worldRadius` around the hit (legacy confirm after the hit). */
export function carve(stem: string, hit: Pick<ToolHit, "local" | "scale">, worldRadius: number, deps: CleanDeps): Promise<string | null> {
  return clean(stem, (from) => sphereBody(stem, from, hit.local, worldRadius, hit.scale), deps);
}
