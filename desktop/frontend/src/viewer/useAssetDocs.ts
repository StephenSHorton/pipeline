// The out/<stem>/*.json files the viewer reads, fetched together; a miss is null (never an error).
import { useEffect, useState } from "react";
import {
  flowFiles,
  headOut,
  readDegraded,
  readInspect,
  readIntent,
  readLods,
  readOutJson,
  readRig,
  type DegradedEntry,
  type InspectDoc,
  type Intent,
  type LodsDoc,
  type RigDoc,
} from "@/lib/api";
import { riggedFiles } from "./candidates";

export interface RigSummary {
  level?: "pass" | "warn" | "fail" | string;
  line?: string;
  problems?: Array<{ level?: string; key?: string; label?: string }>;
}

export interface AssetDocs {
  lods: LodsDoc | null;
  rig: RigDoc | null;
  intent: (Intent & { made?: number }) | null;
  inspect: InspectDoc | null;
  summary: RigSummary | null;
  degraded: DegradedEntry[];
  rigFiles: string[];
  hasRig: boolean;
}

/** inspect.json's and degraded.json's fallbacks, one entry per stage/code/message. */
export function mergeDegraded(...lists: unknown[]): DegradedEntry[] {
  const seen = new Set<string>();
  const out: DegradedEntry[] = [];
  for (const list of lists) {
    if (!Array.isArray(list)) continue;
    for (const e of list as DegradedEntry[]) {
      if (!e || !(e.stage || e.code || e.message)) continue;
      const k = `${e.stage}|${e.code}|${e.message}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(e);
    }
  }
  return out;
}

export async function readAssetDocs(stem: string): Promise<AssetDocs> {
  // Optional files are read only when the flow lists them (no 404 noise); if the flow is unreadable, read them all.
  const files = await flowFiles(stem);
  const has = (rel: string) => !files || files.has(rel);
  const opt = <T,>(rel: string, read: () => Promise<T | null>): Promise<T | null> => (has(rel) ? read() : Promise.resolve(null));
  const [lods, rig, intent, inspect, summary, degradedDoc] = await Promise.all([
    opt("lods.json", () => readLods(stem)),
    opt("rig/rig.json", () => readRig(stem)),
    readIntent(stem),
    opt("inspect.json", () => readInspect(stem)),
    opt("rig/summary.json", () => readOutJson<RigSummary>(stem, "rig/summary.json")),
    opt("degraded.json", () => readDegraded(stem)),
  ]);
  const allClips = files ? files.has("rig/rigged_allclips.glb") : await headOut(stem, "rig/rigged_allclips.glb");
  const rigged = rig?.rigged;
  return {
    lods,
    rig,
    intent,
    inspect,
    summary,
    degraded: mergeDegraded(inspect?.degraded, degradedDoc?.degraded),
    rigFiles: riggedFiles({ allClips, rigJsonRigged: rigged }),
    hasRig: allClips || (typeof rigged === "string" && /\.glb$/i.test(rigged)),
  };
}

/** null while loading. Re-read when `version` changes (after a rename or a job on this asset). */
export function useAssetDocs(stem: string, version = 0): AssetDocs | null {
  const [docs, setDocs] = useState<{ key: string; docs: AssetDocs } | null>(null);
  const key = `${stem}#${version}`;
  useEffect(() => {
    let live = true;
    void readAssetDocs(stem).then((d) => {
      if (live) setDocs({ key, docs: d });
    });
    return () => {
      live = false;
    };
  }, [stem, key]);
  return docs && docs.key === key ? docs.docs : null;
}
