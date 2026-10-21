// The Export stage's factory calls and its small store (legacy exportDoc / exportStem / exporting, loadExport,
// exportAsset, openExport, rigCardFacts; index.html 3313-3359, 3454-3481). Kept here, not in lib/api.ts, so the
// stage owns them; the store is module-level so the versions strip can show the same Exporting… state later.
import { create } from "zustand";
import { enc, factoryJson, headOut, readOutJson, readRig, WAILS, wailsByName } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { finderFailed, type ExportDoc, type FbxCheck } from "./export";

interface ExportStore {
  /** The doc belongs to this stem; a doc for another stem is never shown. */
  stem: string;
  doc: ExportDoc | null;
  exporting: boolean;
}

export const useExport = create<ExportStore>(() => ({ stem: "", doc: null, exporting: false }));

let loadSeq = 0;
/** GET /export/<stem>: the doc, or null when it is not exported yet (404) or the read fails. */
export async function loadExport(stem: string): Promise<void> {
  const seq = ++loadSeq;
  if (useExport.getState().stem !== stem) useExport.setState({ stem, doc: null });
  const data = await factoryJson<{ ok?: boolean; asset?: ExportDoc }>(`/export/${enc(stem)}`).catch(() => null);
  // Only the newest read lands (a quick stem switch, or a pick right after another).
  if (seq !== loadSeq) return;
  useExport.setState({ stem, doc: data?.ok ? (data.asset ?? null) : null });
}

/** No doc for this stem (the flow says the export stage holds nothing); a GET still in flight is dropped. */
export function clearExport(stem: string): void {
  loadSeq++;
  useExport.setState({ stem, doc: null });
}

/** POST /export/<stem> {}: rebuild out/<stem>/export/, keep the new doc, then re-read the flow (the export stage is
 * live now, and its take changed). A refusal (nothing to export, a generate writing the asset) is a toast. */
export async function exportAsset(stem: string): Promise<void> {
  if (useExport.getState().exporting) return;
  useExport.setState({ exporting: true });
  try {
    const data = await factoryJson<{ asset?: ExportDoc }>(`/export/${enc(stem)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (useExport.getState().stem === stem) {
      loadSeq++; // a GET still in flight is older than this doc
      useExport.setState({ doc: data.asset ?? null });
    }
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err));
  } finally {
    useExport.setState({ exporting: false });
    if (useFlow.getState().stem === stem) await useFlow.getState().load(stem);
  }
}

/** Show in Finder: Wails OpenExport, else POST /open-export (only when Wails is absent, as legacy did; lib's
 * openExport would also POST after a Wails refusal). Any failure says where the folder is instead. */
export async function showInFinder(stem: string): Promise<void> {
  try {
    const via = wailsByName(WAILS.OPEN_EXPORT, stem);
    if (via) await via;
    else await factoryJson(`/open-export?stem=${enc(stem)}`, { method: "POST" });
  } catch {
    toast(finderFailed(stem));
  }
}

export interface RigFacts {
  fbxCheck: FbxCheck | null;
  fbxNote: unknown;
  /** The contact sheet is there (HEAD ok). */
  sheet: boolean;
}

/** rig.json (its fbx_check, else rig/fbx_check.json; and fbx_note), and a HEAD on the contact sheet. `files` is the
 * flow's file list: a file it does not hold is not asked for (no 404 per missing file); null asks for everything. */
export async function readRigFacts(stem: string, files: Set<string> | null = null): Promise<RigFacts> {
  const t = Date.now();
  const may = (rel: string) => !files || files.has(rel);
  const info = may("rig/rig.json") ? ((await readRig(stem)) as { fbx_check?: FbxCheck; fbx_note?: unknown } | null) : null;
  const fbxCheck = info?.fbx_check ?? (may("rig/fbx_check.json") ? await readOutJson<FbxCheck>(stem, "rig/fbx_check.json", t) : null);
  const sheet = may("rig/preview/contact_sheet.png") && (await headOut(stem, "rig/preview/contact_sheet.png"));
  return { fbxCheck: fbxCheck ?? null, fbxNote: info?.fbx_note, sheet };
}
