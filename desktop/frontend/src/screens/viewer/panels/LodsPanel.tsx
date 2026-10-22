// The Detail levels panel (legacy lods cubby, index.html 2129-2142; the list 5249-5262; fillTurnaround 4274-4296):
// the hint, Rigged (when there is a rig) and one button per LOD in lods.json order, then the turnaround figures. It
// is the viewer's only LOD switch, as legacy's list was (the HUD's bottom model tabs are gone). The clips stay in the
// HUD's clip picker. Layout only: copy and rules are in lods.ts, the views.json read in lodsApi.ts.
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TOGGLE } from "../classes";
import type { ViewerPanelProps } from "../types";
import { LODS_HINT, LODS_LIST_LABEL, TURNAROUND_LABEL, lodButtons, pressedLod } from "./lods";
import { readTurnaround, type ShownFigure } from "./lodsApi";
import { ViewerPanel } from "./ViewerPanel";

export function LodsPanel(p: ViewerPanelProps) {
  // The last LOD shown stays pressed while the next loads (legacy set aria-pressed only after a load).
  const shown = useRef("");
  if (p.load.state === "ready") shown.current = p.load.info.lod;
  const buttons = lodButtons(p.lods, !!p.docs?.hasRig);
  const pressed = pressedLod(buttons, shown.current);

  // The turnaround is re-read with the docs (a job on this asset, a version pick), while the panel is open. The
  // last figures stay up during a re-read; another stem's never show.
  const [figs, setFigs] = useState<{ stem: string; list: ShownFigure[] } | null>(null);
  const { stem, docs, open } = p;
  useEffect(() => {
    if (!open || !docs) return;
    let live = true;
    void readTurnaround(stem).then((list) => {
      if (live) setFigs({ stem, list });
    });
    return () => {
      live = false;
    };
  }, [stem, docs, open]);
  const figures = figs && figs.stem === stem ? figs.list : [];

  return (
    <ViewerPanel panel="lods" id={p.id} open={p.open} onClose={p.onClose}>
      <p className="m-0 text-[0.8rem] leading-[1.45] text-ink-3">{LODS_HINT}</p>
      {/* No LODs, no list (legacy .lods:empty); the hint still shows. */}
      {buttons.length ? (
        <div role="group" aria-label={LODS_LIST_LABEL} className="flex flex-col gap-1">
          {buttons.map((b) => (
            <Button
              key={b.id}
              type="button"
              variant="hud-ghost"
              aria-pressed={b.id === pressed}
              title={b.title}
              onClick={() => b.id !== pressed && p.onPickLod(b.id)}
              className={cn(TOGGLE, "w-full justify-start tabular-nums")}
            >
              {b.label}
            </Button>
          ))}
        </div>
      ) : null}
      {figures.length ? (
        <aside aria-label={TURNAROUND_LABEL} className="flex flex-col gap-2">
          {figures.map((f) => (
            <figure key={`${f.id}:${f.file}`} className={cn("m-0 overflow-hidden rounded-[10px] border p-1", f.shape ? "border-primary" : "border-border")}>
              <img src={f.src} alt={f.caption} className="block h-24 w-full rounded-[7px] bg-muted object-contain" />
              <figcaption className="mt-1.5 font-mono text-[10px] tracking-[0.08em] text-ink-3 uppercase">{f.caption}</figcaption>
            </figure>
          ))}
        </aside>
      ) : null}
    </ViewerPanel>
  );
}
