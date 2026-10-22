// The versions panel on every stage except Ideas (legacy #takes, index.html 2854-3031): the stage's saved versions
// as small thumbnails, "Keep this version", the stale note, and the next step from the concept (Build / Stack from
// picture). The items come from ./takes/logic.ts. It used to be a wide strip at the bottom centre; Stephen: "The
// versions ui can be on the left, don't make it take up huge space on the bottom center". So it is a compact glass
// card on the left: the last thing in the viewer's left column (under the asset card and the panel rail, passed in
// through ViewerScreen), and at the left edge on the other stages (AssetScreen places it). The heading folds it;
// it starts folded on phones, where the screen has no room to spare. It is gone when it has nothing to say.
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { navigate } from "@/app/router";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { conceptStill, pickTake, snapshotTake, useFlow, type Stage } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { useNarrow } from "@/lib/useNarrow";
import { cn } from "@/lib/utils";
import type { NestedIntent } from "@/screens/home/make";
import { PANEL } from "@/screens/viewer/classes";
import { buildFromConcept, stackFromPicture, useStacking } from "./takes/api";
import { groupItems, panelHeading, stripItems, versionsSummary, type StripAction, type StripItem } from "./takes/logic";
import type { StageProps } from "./types";

const ACT =
  "hit-44 h-auto min-h-8 w-full justify-start rounded-[9px] px-2.5 py-1.5 text-left text-[length:calc(13px*var(--fs))] font-semibold whitespace-normal";
// A rig version's check: pass green, warn the signal yellow, fail red (the rail and the HUD use the same three).
const DOT: Record<string, string> = {
  pass: "bg-success",
  ok: "bg-success",
  warn: "bg-signal",
  fail: "bg-destructive",
};

export function VersionsPanel({ stem, stage, flow, intent, className }: StageProps & { className?: string }) {
  // Module level (takes/api.ts): survives a remount, and only this stem's panel shows it.
  const stacking = useStacking((s) => s.stem === stem);
  // The stem whose pick or snapshot is in flight: its version buttons hold until the flow answers. Keyed by stem
  // so a request still running on the last asset never holds this one's buttons.
  const [pendingStem, setPendingStem] = useState("");
  const pending = pendingStem === stem;
  const settle = () => setPendingStem((s) => (s === stem ? "" : s));
  // null: the default for the screen width (open wide, folded on phones) until the heading is pressed.
  const narrow = useNarrow();
  const [unfolded, setUnfolded] = useState<boolean | null>(null);
  const open = unfolded ?? !narrow;
  const items = stripItems({
    stem,
    stage,
    flow,
    output: intent?.output,
    stacking,
  });

  if (!items.length) return null;

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
        // Legacy re-reads after a failed snapshot too, so the panel shows what the factory really holds.
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

  const g = groupItems(items);
  const summary = versionsSummary(g.takes);
  const bodyId = `versions-${stem}`;
  return (
    <Card size="hud" className={cn(PANEL, "w-[13.5rem] max-w-full shrink-0 gap-1.5 p-1.5 text-ink tracking-[var(--track)]", className)}>
      {/* Always mounted, so a screen reader hears the busy note when its words arrive. */}
      <span role="status" className="sr-only">
        {g.busy?.text ?? ""}
      </span>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        title={g.label?.title}
        onClick={() => setUnfolded(!open)}
        className="hit-44 flex min-h-7 w-full items-center gap-1.5 rounded-[8px] px-1.5 text-left outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="text-[length:calc(11.5px*var(--fs))] font-semibold tracking-[0.1em] text-ink-3 uppercase">{panelHeading(g)}</span>
        {summary ? <span className="text-[length:calc(12px*var(--fs))] text-ink-3 tabular-nums">{summary}</span> : null}
        <ChevronDown aria-hidden="true" size={15} strokeWidth={2.2} className={cn("ml-auto shrink-0 text-ink-3 transition-transform", !open && "-rotate-90")} />
      </button>
      <div id={bodyId} className={cn("min-h-0 flex-col gap-1.5", open ? "flex" : "hidden")}>
        {g.takes.length ? (
          <div
            role="group"
            aria-label="Versions"
            className="grid max-h-[min(11.5rem,30vh)] grid-cols-4 gap-1 overflow-y-auto [scrollbar-width:thin]"
          >
            {g.takes.map((it) => (
              <Take key={it.id} item={it} stage={stage} disabled={pending} onPick={pick} />
            ))}
          </div>
        ) : null}
        {g.rest.length ? (
          <div className="flex flex-col gap-1">
            {g.rest.map((it, i) =>
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
              ) : (
                <Note key={i} item={it} />
              ),
            )}
          </div>
        ) : null}
      </div>
    </Card>
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
        "group/take relative flex flex-col items-center gap-0.5 rounded-[9px] border border-transparent p-0.5 outline-none",
        "text-[length:calc(11.5px*var(--fs))] font-medium text-ink-3 tabular-nums hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring aria-disabled:cursor-wait aria-disabled:opacity-60",
        "aria-pressed:border-primary aria-pressed:bg-background aria-pressed:font-semibold aria-pressed:text-primary aria-pressed:shadow-[var(--glass-shadow)]",
      )}
    >
      {item.src ? (
        <img src={item.src} alt="" loading="lazy" className="aspect-square w-full rounded-[6px] bg-muted object-contain" />
      ) : (
        <span aria-hidden="true" className="aspect-square w-full rounded-[6px] bg-muted" />
      )}
      <span className="leading-none">{item.n}</span>
      {stage === "rig" && item.level ? (
        <span aria-hidden="true" className={cn("absolute top-1 right-1 size-2 rounded-full ring-2 ring-background", DOT[item.level] ?? "bg-signal")} />
      ) : null}
    </button>
  );
}

function Note({ item }: { item: Extract<StripItem, { type: "note" }> }) {
  if (item.stale) {
    // Stale: the rail's amber dot, with the words in ink (amber text fails contrast on the glass).
    return (
      <span className="flex items-center gap-1.5 rounded-[9px] bg-warning/10 px-2 py-1.5 text-[length:calc(12.5px*var(--fs))] leading-snug text-ink-2">
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-warning ring-2 ring-warning/30" />
        {item.text}
      </span>
    );
  }
  return (
    <span aria-hidden={item.busy || undefined} className={cn("px-1.5 text-[length:calc(12.5px*var(--fs))] leading-snug text-ink-3", item.busy && "animate-pulse")}>
      {item.text}
    </span>
  );
}
