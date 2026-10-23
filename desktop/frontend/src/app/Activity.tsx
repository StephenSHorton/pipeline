// The activity chip in the top bar and its drawer: what Claude and the factory are doing, as it happens
// (lib/activity.ts). The chip pulses with the action going on now ("Claude · Measuring mlg sheep · 3.1 s"); the
// drawer lists every step newest first, with the numbers it found, the picture it made (a look sheet, the drawing
// as it forms) and a link to the asset, plus the Follow switch (app/follow.ts).
import { X } from "lucide-react";
import { useLayoutEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { collapseRuns, currentEntry, detailText, entryTime, imageUrl, useActivity, type ActivityEntry } from "@/lib/activity";
import { displayStem, useJobs } from "@/lib/jobs";
import { cn } from "@/lib/utils";
import { PANEL, TOGGLE } from "@/screens/viewer/classes";
import { navigate } from "./router";

function who(e: ActivityEntry): string {
  return e.source === "job" ? "Factory" : "Claude";
}

function Dot({ state, className }: { state?: string; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-2 shrink-0 rounded-full",
        state === "running"
          ? "animate-pulse bg-signal motion-reduce:animate-none"
          : state === "ok"
            ? "bg-success"
            : state === "error"
              ? "bg-destructive"
              : "bg-[var(--muted-2)]",
        className,
      )}
    />
  );
}

export function ActivityChip() {
  const entries = useActivity((s) => s.entries);
  const now = useActivity((s) => s.now);
  const open = useActivity((s) => s.open);
  const setOpen = useActivity((s) => s.setOpen);
  const jobId = useJobs((s) => (s.job?.state === "running" ? s.job.id : null));
  if (!entries.length) return null;
  // The job chip already shows the app's job with its stage: this chip then just opens the log.
  const now0 = currentEntry(entries);
  const cur = now0 && !(now0.source === "job" && now0.job_id === jobId) ? now0 : null;
  const text = cur ? `${who(cur)} · ${displayStem(cur.title)}${entryTime(cur, now) ? ` · ${entryTime(cur, now)}` : ""}` : "Activity";
  return (
    <Button
      variant="secondary"
      aria-expanded={open}
      aria-controls="activity-drawer"
      onClick={() => setOpen(!open)}
      title={cur ? text : "What Claude and the factory did"}
      className="hit-44 h-auto min-h-9 min-w-0 max-w-[20rem] justify-start gap-2 rounded-full bg-[var(--glass-bg)] px-3 text-[0.8rem] font-medium text-ink shadow-[var(--glass-shadow)] max-sm:max-w-[8rem]"
    >
      {/* Idle, the dot would mean nothing: only a running step shows one. */}
      {cur ? <Dot state="running" /> : null}
      <span className="truncate">{text}</span>
    </Button>
  );
}

export function ActivityDrawer() {
  const open = useActivity((s) => s.open);
  const entries = useActivity((s) => s.entries);
  const now = useActivity((s) => s.now);
  const follow = useActivity((s) => s.follow);
  const setFollow = useActivity((s) => s.setFollow);
  const setOpen = useActivity((s) => s.setOpen);
  // Just under the top bar, which is one row or two (the stage rail wraps below 960 px).
  const [top, setTop] = useState(80);
  useLayoutEffect(() => {
    if (!open) return;
    const bar = document.querySelector<HTMLElement>("[data-slot=topbar]");
    if (!bar) return;
    const place = () => setTop(Math.round(bar.getBoundingClientRect().bottom) + 8);
    place();
    const ro = new ResizeObserver(place);
    ro.observe(bar);
    return () => ro.disconnect();
  }, [open]);
  if (!open) return null;
  const cur = currentEntry(entries);
  const curImage = cur ? imageUrl(cur) : null;
  return (
    <div
      style={{ top, maxHeight: `calc(100vh - ${top + 16}px)` }}
      className="pointer-events-none fixed right-[clamp(12px,2vw,20px)] z-30 flex w-[min(24rem,calc(100vw-32px))] flex-col"
    >
      <Card
        id="activity-drawer"
        size="hud"
        role="region"
        aria-label="Activity"
        // The card colour is 90% white and a HUD panel has no blur: over the page colour it reads as solid, so
        // the viewer's toolbar underneath never shows through the list.
        className={cn(PANEL, "min-h-0 flex-1 rounded-[20px] p-0 text-ink")}
        style={{ background: "linear-gradient(var(--card), var(--card)), var(--background)" }}
      >
        <div className="flex items-center gap-2 border-b border-[var(--topbar-border)] py-2 pr-2 pl-4">
          <h2 className="flex-1 text-[0.95rem] font-semibold">Activity</h2>
          <Button
            variant="hud-ghost"
            aria-pressed={follow}
            onClick={() => setFollow(!follow)}
            title="Open the asset Claude is working on and show each model as it lands"
            className={cn(TOGGLE, "text-[0.8rem]")}
          >
            Follow Claude
          </Button>
          <Button variant="ghost" size="icon" aria-label="Close activity" onClick={() => setOpen(false)} className="size-9">
            <X className="size-4" />
          </Button>
        </div>
        {cur ? (
          <section aria-label="Now" className="border-b border-[var(--topbar-border)] px-4 py-3">
            <p className="text-[0.7rem] font-semibold tracking-wide text-ink-3 uppercase">Now · {who(cur)}</p>
            <p className="mt-0.5 font-medium">{displayStem(cur.title)}</p>
            <p className="text-[0.8rem] text-ink-3">
              {[detailText(cur.detail), entryTime(cur, now)].filter(Boolean).join(" · ")}
            </p>
            {curImage ? <img src={curImage} alt="" className="mt-2 max-h-56 w-full rounded-[10px] bg-muted object-contain" /> : null}
          </section>
        ) : null}
        <ol className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {entries.length === 0 ? (
            <li className="px-2 py-3 text-[0.85rem] text-ink-3">Nothing yet. When Claude works on an asset, each step shows here.</li>
          ) : (
            collapseRuns(entries).map(({ e, n }) => <Row key={e.id} e={e} n={n} now={now} />)
          )}
        </ol>
      </Card>
    </div>
  );
}

function Row({ e, n, now }: { e: ActivityEntry; n: number; now: number }) {
  const img = imageUrl(e);
  const line = e.state === "error" ? e.error : detailText(e.detail);
  return (
    <li className="flex gap-2.5 rounded-[10px] px-2 py-2 hover:bg-muted/60">
      <Dot state={e.state} className="mt-1.5" />
      <div className="min-w-0 flex-1">
        <p className="text-[0.85rem] leading-snug font-medium">
          {displayStem(e.title)}
          {n > 1 ? <span className="ml-1.5 text-[0.75rem] font-normal text-ink-3 tabular-nums">×{n}</span> : null}
        </p>
        {line ? (
          <p className={cn("text-[0.78rem] leading-snug", e.state === "error" ? "text-danger-text" : "text-ink-3")}>{line}</p>
        ) : null}
        <p className="mt-0.5 flex items-center gap-2 text-[0.72rem] text-ink-3">
          <span>{who(e)}</span>
          {entryTime(e, now) ? <span>{entryTime(e, now)}</span> : null}
          {e.stem ? (
            <button
              type="button"
              className="rounded font-medium text-primary underline underline-offset-2 hover:text-ink focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              onClick={() => navigate({ name: "asset", stem: e.stem! })}
            >
              Open {displayStem(e.stem)}
            </button>
          ) : null}
        </p>
      </div>
      {img ? <img src={img} alt="" loading="lazy" className="size-12 shrink-0 rounded-[8px] bg-muted object-contain" /> : null}
    </li>
  );
}
