// The left rail (Cleanup, Paint, Detail levels, under the asset card) and the right dock (the Rig check, under the
// camera stack). One open panel across both (legacy openStep); every panel stays mounted and hides while closed.
// Phones have no right column (the camera stack is hidden there), so the dock's panels join the left rail.
import { useCallback, useEffect, useRef, useSyncExternalStore, type ComponentType } from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { RigPanel, RigToggle } from "@/screens/asset/rig/RigPanel";
import { PANEL } from "../classes";
import type { ViewerPanelBase, ViewerPanelProps } from "../types";
import { CleanupPanel } from "./CleanupPanel";
import { LodsPanel } from "./LodsPanel";
import { PaintPanel } from "./PaintPanel";
import { openOn, panelDomId, panelsFor, type PanelId, type PanelSide } from "./panels";
import { PanelToggle, type PanelToggleProps } from "./PanelToggle";
import { usePanels } from "./store";

const BODY: Record<PanelId, ComponentType<ViewerPanelProps>> = { cleanup: CleanupPanel, paint: PaintPanel, lods: LodsPanel, rig: RigPanel };
const TOGGLES: Partial<Record<PanelId, ComponentType<Omit<PanelToggleProps, "panel"> & { props: ViewerPanelProps }>>> = { rig: RigToggle };

const NARROW = "(max-width: 639px)";
function useNarrow(): boolean {
  return useSyncExternalStore(
    (fn) => {
      const m = matchMedia(NARROW);
      m.addEventListener("change", fn);
      return () => m.removeEventListener("change", fn);
    },
    () => matchMedia(NARROW).matches,
    () => false,
  );
}

/** The panels one side shows on this stage and screen width. */
function useSide(side: PanelSide, stage: ViewerPanelBase["stage"]): PanelId[] {
  const narrow = useNarrow();
  if (narrow) return side === "left" ? [...panelsFor(stage, "left"), ...panelsFor(stage, "right")] : [];
  return panelsFor(stage, side);
}

export function PanelRail({ side, base, className }: { side: PanelSide; base: ViewerPanelBase; className?: string }) {
  const ids = useSide(side, base.stage);
  const open = openOn(usePanels((s) => s.open), base.stage);
  const toggles = useRef<Partial<Record<PanelId, HTMLButtonElement | null>>>({});

  // A stage that does not show the open panel closes it (legacy: the Rig check closes off the Rig stage).
  const stored = usePanels((s) => s.open);
  useEffect(() => {
    if (stored && !open && side === "left") usePanels.getState().close();
  }, [stored, open, side]);

  const close = useCallback((id: PanelId) => {
    // Closing while focus is in the panel hands focus back to its toggle, not <body>.
    const inside = !!document.activeElement?.closest(`#${CSS.escape(panelDomId(id))}`);
    if (usePanels.getState().open === id) usePanels.getState().close();
    if (inside) toggles.current[id]?.focus();
  }, []);

  if (!ids.length) return null;
  return (
    <div className={cn("flex min-h-0 flex-col gap-2", side === "right" ? "items-end" : "items-stretch", className)}>
      <Card size="hud" className={cn(PANEL, "w-fit max-w-full flex-row flex-wrap items-center gap-0.5 p-1")} role="group" aria-label={side === "left" ? "Panels" : "Dock"}>
        {ids.map((id) => {
          const Toggle = TOGGLES[id];
          const tp = {
            open: open === id,
            controls: panelDomId(id),
            onToggle: () => usePanels.getState().toggle(id),
            ref: (el: HTMLButtonElement | null) => {
              toggles.current[id] = el;
            },
          };
          return Toggle ? <Toggle key={id} {...tp} props={props(base, id, open, close)} /> : <PanelToggle key={id} panel={id} {...tp} />;
        })}
      </Card>
      {ids.map((id) => {
        const Body = BODY[id];
        return <Body key={id} {...props(base, id, open, close)} />;
      })}
    </div>
  );
}

function props(base: ViewerPanelBase, id: PanelId, open: PanelId | null, close: (id: PanelId) => void): ViewerPanelProps {
  return { ...base, id: panelDomId(id), open: open === id, onClose: () => close(id) };
}
