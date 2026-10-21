// Settings (legacy #settings, index.html 1987-2010, 6696-6837): the version and updates, and the factory
// folder. One glass card on the clay ground, like Home. The Updates row reads the shared update store
// (update.ts), so a check or an install started elsewhere shows here too.
import { useEffect, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { openFolder, refreshDataDir, useDataDir } from "./dataDir";
import { checkForUpdates, startUpdateWatch, updateView, useUpdate, type HintTone } from "./update";

// The legacy data-state colours: ok olive → success text, error red → danger text, running gold → the
// signal dot beside ink text (signal yellow is too light for small text). The dot says it without colour too.
const TONE_TEXT: Record<HintTone, string> = {
  "": "text-ink-3",
  running: "text-ink",
  ok: "text-success-text",
  error: "text-danger-text",
};
const TONE_DOT: Record<HintTone, string> = {
  "": "bg-hair",
  running: "bg-signal animate-pulse motion-reduce:animate-none",
  ok: "bg-success",
  error: "bg-destructive",
};

export function SettingsScreen() {
  const update = useUpdate();
  const view = updateView(update);
  const dir = useDataDir();

  // Every open re-reads the folder (legacy showSettings → refreshSettingsPanel). The watch normally starts at
  // boot from App; this makes sure it ran.
  useEffect(() => {
    startUpdateWatch();
    void refreshDataDir();
  }, []);

  return (
    <div className="absolute inset-0 overflow-x-hidden overflow-y-auto bg-background text-ink">
      <div aria-hidden="true" className="pointer-events-none absolute top-[262px] left-1/2 -ml-[550px] h-[520px] w-[1100px] rounded-[50%] bg-ground-ellipse" />
      <div aria-hidden="true" className="pointer-events-none absolute top-[312px] left-1/2 -ml-[380px] h-[360px] w-[760px] rounded-[50%] bg-clay-grass opacity-70" />
      <div className="relative flex min-h-full flex-col items-center px-4 pt-[clamp(2rem,8vh,4.5rem)] pb-12">
        <Card className="w-full max-w-[38rem] gap-0 p-0">
          <div className="px-6 pt-6 pb-5 max-sm:px-4">
            <div className="text-[0.75rem] font-semibold tracking-wide text-ink-3 uppercase">Pipeline</div>
            <h1 className="mt-1 text-[1.6rem] leading-tight font-semibold tracking-tight text-ink">Settings</h1>
            <p className="mt-1.5 text-sm text-ink-3">Version, updates, and the factory folder. That’s it.</p>
          </div>
          <Separator />
          <Row
            label="Updates"
            labelId="settings-updates"
            hint={
              <span id="settings-updates-hint" role="status" aria-live="polite" className={cn("flex items-start gap-2", TONE_TEXT[view.tone])}>
                <span aria-hidden="true" className={cn("mt-[0.45em] size-2 shrink-0 rounded-full", TONE_DOT[view.tone])} />
                <span data-state={view.tone || undefined}>{view.hint}</span>
              </span>
            }
          >
            <Button
              variant="outline"
              className="min-h-11 min-w-[11rem] px-4"
              aria-describedby="settings-updates-hint"
              disabled={view.buttonDisabled}
              onClick={() => void checkForUpdates(true)}
            >
              {view.buttonLabel}
            </Button>
          </Row>
          <Separator />
          <Row
            label="Factory folder"
            labelId="settings-folder"
            hint={
              <span
                id="settings-folder-hint"
                role={dir.tone === "error" ? "alert" : undefined}
                data-state={dir.tone || undefined}
                className={cn("block font-mono text-[0.8rem] [overflow-wrap:anywhere]", dir.tone === "error" ? "text-danger-text" : "text-ink-3")}
              >
                {dir.text}
              </span>
            }
          >
            <Button variant="outline" className="min-h-11 min-w-[5.5rem] px-4" aria-describedby="settings-folder-hint" onClick={() => void openFolder()}>
              Open
            </Button>
          </Row>
        </Card>
      </div>
    </div>
  );
}

/** A settings row: label and hint left, the action right; stacked at phone width. */
function Row({ label, labelId, hint, children }: { label: string; labelId: string; hint: ReactNode; children: ReactNode }) {
  return (
    <section aria-labelledby={`${labelId}-label`} className="flex items-center gap-4 px-6 py-4 max-sm:flex-col max-sm:items-stretch max-sm:gap-3 max-sm:px-4">
      <div className="min-w-0 flex-1">
        <h2 id={`${labelId}-label`} className="text-sm font-semibold text-ink">
          {label}
        </h2>
        <div className="mt-1 text-sm">{hint}</div>
      </div>
      <div className="flex shrink-0 max-sm:justify-start">{children}</div>
    </section>
  );
}
