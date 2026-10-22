// A rail or dock toggle: the legacy cubby header / step button for one panel (openStep, index.html 5817-5829).
// aria-expanded, not aria-pressed: it shows and hides a region (the ▾/▸ text is not ported).
import type { ReactNode, Ref } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TOGGLE } from "../classes";
import { PANEL_TITLES, type PanelId } from "./panels";

export interface PanelToggleProps {
  panel: PanelId;
  open: boolean;
  /** The panel's DOM id (aria-controls). */
  controls: string;
  onToggle: () => void;
  ref?: Ref<HTMLButtonElement>;
  title?: string;
  className?: string;
  /** Defaults to the panel title. */
  children?: ReactNode;
}

export function PanelToggle({ panel, open, controls, onToggle, ref, title, className, children }: PanelToggleProps) {
  return (
    <Button
      ref={ref}
      type="button"
      variant="hud-ghost"
      aria-expanded={open}
      aria-controls={controls}
      title={title}
      onClick={onToggle}
      className={cn(TOGGLE, "aria-expanded:bg-accent aria-expanded:text-accent-foreground", className)}
    >
      {children ?? PANEL_TITLES[panel]}
    </Button>
  );
}
