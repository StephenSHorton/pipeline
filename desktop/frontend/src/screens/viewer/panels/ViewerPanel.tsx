// The shell every viewer panel draws in: a HUD card (no backdrop blur, see Hud.tsx), its title and a close
// button, hidden while closed. The rail toggle is the legacy cubby header; the × here is the Audio panel's.
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { PANEL } from "../classes";
import { closeLabel, closeTitle, PANEL_TITLES, type PanelId } from "./panels";

export function ViewerPanel({
  panel,
  id,
  open,
  onClose,
  className,
  children,
}: {
  panel: PanelId;
  id: string;
  open: boolean;
  onClose: () => void;
  className?: string;
  children?: ReactNode;
}) {
  const title = PANEL_TITLES[panel];
  return (
    <Card
      id={id}
      size="hud"
      role="region"
      aria-label={title}
      hidden={!open}
      className={cn(PANEL, "min-h-0 w-full gap-2.5 overflow-y-auto px-4 pt-3 pb-3.5 [&[hidden]]:hidden", className)}
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="m-0 text-[0.95rem] font-semibold text-ink">{title}</h2>
        <Button type="button" variant="ghost" size="icon-sm" aria-label={closeLabel(panel)} title={closeTitle(panel)} onClick={onClose} className="hit-44 text-ink-3">
          <X />
        </Button>
      </div>
      {children}
    </Card>
  );
}
