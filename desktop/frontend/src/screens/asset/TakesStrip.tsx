// The versions strip under every stage except Ideas (legacy #takes, index.html 2854-3031): the stage's saved
// versions, "Keep this version", the stale note, and the next step from the concept (Build / Stack from picture).
// The items come from ./takes/logic.ts. It is a glass bar in its own row under the stage, not over it, so the
// viewer's bottom bar (model and clip pickers, Hud.tsx) always sits above it; it is gone when it has nothing to say.
import { useLayoutEffect, useRef, useState } from "react";
import { navigate } from "@/app/router";
import { Button } from "@/components/ui/button";
import { conceptStill, pickTake, snapshotTake, useFlow, type Stage } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import type { NestedIntent } from "@/screens/home/make";
import { buildFromConcept, stackFromPicture, useStacking } from "./takes/api";
import { stripItems, type StripAction, type StripItem } from "./takes/logic";
import type { StageProps } from "./types";

const ACT = "min-h-11 rounded-[10px] px-3.5 text-[length:calc(14px*var(--fs))] font-semibold";
// A rig version's check: pass green, warn the signal yellow, fail red (the rail and the HUD use the same three).
const DOT: Record<string, string> = {
  pass: "bg-success",
  ok: "bg-success",
  warn: "bg-signal",
  fail: "bg-destructive",
};

export function TakesStrip({ stem, stage, flow, intent }: StageProps) {
  // Module level (takes/api.ts): survives a remount, and only this stem's strip shows it.
  const stacking = useStacking((s) => s.stem === stem);
  // The stem whose pick or snapshot is in flight: its version buttons hold until the flow answers. Keyed by stem
  // so a request still running on the last asset never holds this one's buttons.
  const [pendingStem, setPendingStem] = useState("");
  const pending = pendingStem === stem;
  const settle = () => setPendingStem((s) => (s === stem ? "" : s));
  const row = useRef<HTMLDivElement>(null);
  const items = stripItems({
    stem,
    stage,
    flow,
    output: intent?.output,
    stacking,
  });
  const shown = items.length > 0;

  // The toasts sit above the strip, not over its buttons: --strip-h is the row's height while it shows.
  useLayoutEffect(() => {
    const el = row.current;
    if (!el) return;
    const root = document.documentElement;
    const write = () => root.style.setProperty("--strip-h", `${el.offsetHeight}px`);
    write();
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(write);
    ro?.observe(el);
    return () => {
      ro?.disconnect();
      root.style.removeProperty("--strip-h");
    };
  }, [shown]);

  if (!shown) return null;

  const pick = async (take: string) => {
    if (pending) return;
    setPendingStem(stem);
    try {
      // The response is the whole flow; the store bumps its version and every stage re-reads its files.
      await pickTake(stem, stage, take);
    } catch (err) {
      toast((err as Error)?.message || String(err));
    } finally {
      settle();
    }
  };

  const run = async (a: StripAction) => {
    if (a.do === "keep") {
      if (pending) return;
      setPendingStem(stem);
      try {
        await snapshotTake(stem, stage);
      } catch (err) {
        toast((err as Error)?.message || String(err));
        // Legacy re-reads after a failed snapshot too, so the strip shows what the factory really holds.
        if (useFlow.getState().stem === stem) void useFlow.getState().load(stem);
      } finally {
        settle();
      }
    } else if (a.do === "build") {
      await buildFromConcept(stem, conceptStill(flow, stem), a.output, intent as NestedIntent | null);
    } else if (a.do === "stack") {
      if (useStacking.getState().stem === stem) return;
      await stackFromPicture(stem);
    } else {
      // The Make form's remake (settings and picture prefilled, the job reusing this stem) is not on Home yet.
      navigate({ name: "home", section: "make" });
    }
  };

  const versions = items.filter((it) => it.type === "label" || it.type === "take");
  const rest = items.filter((it) => it.type !== "label" && it.type !== "take");
  const busy = items.find((it) => it.type === "note" && it.busy);
  return (
    // relative: on 3D and Rig the viewer's clay runs on under this row (ViewerScreen), and the row sits above it.
    <div ref={row} className="relative flex shrink-0 justify-center px-3 pt-1 pb-3">
      {/* Always mounted, so a screen reader hears the busy note when its words arrive. */}
      <span role="status" className="sr-only">
        {busy?.type === "note" ? busy.text : ""}
      </span>
      <div className="glass-panel flex max-w-full flex-wrap items-center justify-center gap-x-3 gap-y-2 rounded-[14px] px-2 py-1.5 text-ink tracking-[var(--track)]">
        {versions.length ? (
          <div role="group" aria-label="Versions" className="flex max-w-full items-center gap-1 overflow-x-auto [scrollbar-width:thin]">
            {versions.map((it) =>
              it.type === "label" ? (
                <Label key="label" item={it} />
              ) : (
                <Take key={it.type === "take" ? it.id : ""} item={it} stage={stage} disabled={pending} onPick={pick} />
              ),
            )}
          </div>
        ) : null}
        {rest.length ? (
          <div className="flex max-w-full flex-wrap items-center justify-center gap-1.5">
            {rest.map((it, i) =>
              it.type === "act" ? (
                <Button
                  key={i}
                  type="button"
                  variant={it.primary ? "default" : "outline"}
                  title={it.title}
                  disabled={it.action.do === "keep" && pending}
                  onClick={() => void run(it.action)}
                  className={ACT}
                >
                  {it.text}
                </Button>
              ) : it.type === "note" ? (
                <Note key={i} item={it} />
              ) : null,
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Label({ item }: { item: Extract<StripItem, { type: "label" }> }) {
  return (
    <span title={item.title} className="shrink-0 cursor-help px-1.5 text-[length:calc(11.5px*var(--fs))] font-semibold tracking-[0.1em] text-ink-3 uppercase">
      {item.text}
    </span>
  );
}

function Take({
  item,
  stage,
  disabled,
  onPick,
}: {
  item: Extract<StripItem, { type: "take" }>;
  stage: Stage;
  disabled: boolean;
  onPick: (id: string) => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={item.pressed}
      aria-label={item.title}
      title={item.title}
      // aria-disabled, not disabled: the button just pressed keeps keyboard focus while the pick runs.
      aria-disabled={disabled || undefined}
      onClick={() => onPick(item.id)}
      className={cn(
        "group/take relative flex shrink-0 flex-col items-center gap-0.5 rounded-[10px] border border-transparent p-1 outline-none",
        "text-[length:calc(12.5px*var(--fs))] font-medium text-ink-3 tabular-nums hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring aria-disabled:cursor-wait aria-disabled:opacity-60",
        "aria-pressed:border-primary aria-pressed:bg-background aria-pressed:font-semibold aria-pressed:text-primary aria-pressed:shadow-[var(--glass-shadow)]",
      )}
    >
      {item.src ? (
        <img src={item.src} alt="" loading="lazy" className="size-[52px] rounded-[7px] bg-muted object-contain" />
      ) : (
        <span aria-hidden="true" className="size-[52px] rounded-[7px] bg-muted" />
      )}
      <span className="leading-none">{item.n}</span>
      {stage === "rig" && item.level ? (
        <span aria-hidden="true" className={cn("absolute top-1.5 right-1.5 size-2 rounded-full ring-2 ring-background", DOT[item.level] ?? "bg-signal")} />
      ) : null}
    </button>
  );
}

function Note({ item }: { item: Extract<StripItem, { type: "note" }> }) {
  if (item.stale) {
    // Stale: the rail's amber dot, with the words in ink (amber text fails contrast on the glass).
    return (
      <span className="inline-flex items-center gap-1.5 rounded-[10px] bg-warning/10 px-2.5 py-1.5 text-[length:calc(13.5px*var(--fs))] text-ink-2">
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-warning ring-2 ring-warning/30" />
        {item.text}
      </span>
    );
  }
  return (
    <span aria-hidden={item.busy || undefined} className={cn("px-1.5 text-[length:calc(13.5px*var(--fs))] text-ink-3", item.busy && "animate-pulse")}>
      {item.text}
    </span>
  );
}
