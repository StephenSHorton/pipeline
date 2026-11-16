// The Detail levels panel (legacy lods cubby, index.html 2129-2142; the list 5249-5262; fillTurnaround 4274-4296):
// the hint and one button per LOD in lods.json order, then the turnaround figures. It
// is the same pick as the HUD's bottom model tabs, so it uses the same segmented tray (vertical). The clips stay in
// the HUD's clip picker. Layout only: copy and rules are in lods.ts, the views.json read in lodsApi.ts.
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TRAY, TRAY_ITEM } from "../classes";
import type { ViewerPanelProps } from "../types";
import { LODS_HINT, LODS_LIST_LABEL, TURNAROUND_LABEL, lodButtons, pressedLod } from "./lods";
import { readTurnaround, type ShownFigure } from "./lodsApi";
import { ViewerPanel } from "./ViewerPanel";

export function LodsPanel(p: ViewerPanelProps) {
  // The last LOD shown stays pressed while the next loads (legacy set aria-pressed only after a load).
  const shown = useRef("");
  if (p.load.state === "ready") shown.current = p.load.info.lod;
  const buttons = lodButtons(p.lods);
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
        <div role="group" aria-label={LODS_LIST_LABEL} className={cn(TRAY, "flex-col items-stretch")}>
          {buttons.map((b) => (
            <Button
              key={b.id}
              type="button"
              variant="hud-ghost"
              aria-pressed={b.id === pressed}
              title={b.title}
              onClick={() => b.id !== pressed && p.onPickLod(b.id)}
              className={cn(TRAY_ITEM, "w-full justify-start tabular-nums")}
            >
              {b.label}
            </Button>
          ))}
        </div>
      ) : null}
      {figures.length ? (
        // A two-up grid of unboxed square views, so four views fit without scrolling. Every view is often role
        // "shape", so the accent ring marks one only when it is the single shape view.
        <aside aria-label={TURNAROUND_LABEL} className="grid grid-cols-2 gap-2">
          {figures.map((f) => (
            <figure key={`${f.id}:${f.file}`} className="m-0 flex min-w-0 flex-col gap-1">
              <img
                src={f.src}
                alt={f.caption}
                className={cn("block aspect-square w-full rounded-[8px] object-contain", f.shape && figures.filter((x) => x.shape).length === 1 && "ring-2 ring-primary")}
              />
              <figcaption className="truncate text-[0.75rem] text-ink-3 first-letter:uppercase">{f.caption}</figcaption>
            </figure>
          ))}
        </aside>
      ) : null}
    </ViewerPanel>
  );
}
