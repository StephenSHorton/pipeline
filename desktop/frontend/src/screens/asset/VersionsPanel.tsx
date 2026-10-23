// The versions panel on every stage except Ideas (legacy #takes, index.html 2854-3031): the stage's saved versions
// as small thumbnails, "Keep this version", the stale note, and the next step from the concept (Build / Stack from
// picture). The items come from ./takes/logic.ts. It used to be a wide strip at the bottom centre; Stephen: "The
// versions ui can be on the left, don't make it take up huge space on the bottom center". So it is a compact glass
// card on the left: the last thing in the viewer's left column (under the asset card and the panel rail, passed in
// through ViewerScreen), and at the left edge on the other stages (AssetScreen places it). The heading folds it;
// it starts folded on phones, where the screen has no room to spare. It is gone when it has nothing to say.
// While the viewer compares (screens/viewer/compare.ts), a click shows that version on the left instead of picking it:
// nothing is written.
import { ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { navigate } from "@/app/router";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { conceptStill, flowEntry, pickTake, snapshotTake, useFlow, type Stage } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { useNarrow } from "@/lib/useNarrow";
import { cn } from "@/lib/utils";
import type { NestedIntent } from "@/screens/home/make";
import { MUTE, PANEL } from "@/screens/viewer/classes";
import { comparing, currentTake, leftTake, useCompare } from "@/screens/viewer/compare";
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
  // The current version may sit below the grid's fold (16 versions, 12 shown): scroll the grid, only the grid, to it.
  const grid = useRef<HTMLDivElement>(null);
  const current = items.find((it) => it.type === "take" && it.pressed);
  const currentId = current && current.type === "take" ? current.id : "";
  useEffect(() => {
    const g = grid.current;
    const tile = g?.querySelector<HTMLElement>("[aria-pressed=true]");
    if (!g || !tile) return;
    const top = tile.offsetTop - g.offsetTop;
    if (top < g.scrollTop || top + tile.offsetHeight > g.scrollTop + g.clientHeight) g.scrollTop = top + tile.offsetHeight - g.clientHeight;
  }, [currentId, open, stem]);

  // Comparing: the version on the left, which a click changes.
  const compare = useCompare((s) => comparing(s, stem, stage));
  const chosenLeft = useCompare((s) => s.left);
  const entry = flowEntry(flow, stage);
  const leftId = compare ? leftTake(entry, chosenLeft) : null;

  if (!items.length) return null;

  const pick = async (take: string) => {
    if (compare) {
      // View only: the current version is already on the right.
      if (take !== currentTake(entry)) useCompare.getState().setLeft(take);
      return;
    }
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
      {/* overflow-y-auto: in the viewer's left column the panel may shrink (AssetScreen), and its body scrolls then. */}
      <div id={bodyId} className={cn("min-h-0 flex-col gap-1.5 overflow-y-auto [scrollbar-width:thin]", open ? "flex" : "hidden")}>
        {g.takes.length ? (
          <div
            ref={grid}
            role="group"
            aria-label="Versions"
            className="grid min-h-12 max-h-[min(11.5rem,30vh)] grid-cols-4 gap-1 overflow-y-auto [scrollbar-gutter:stable] [scrollbar-width:thin]"
          >
            {g.takes.map((it) => (
              <Take key={it.id} item={it} stage={stage} disabled={pending && !compare} compare={compare} left={it.id === leftId} onPick={pick} />
            ))}
          </div>
        ) : null}
        {compare ? <p className={cn(MUTE, "px-1.5")}>Comparing. Click a version to show it on the left.</p> : null}
        {g.rest.length ? (
          <div className="flex flex-col gap-1">
            {g.rest.map((it, i) =>
              it.type === "act" ? (
                <Button
                  key={i}
                  type="button"
                  variant="outline"
                  title={it.title}
                  disabled={it.action.do === "keep" && pending}
                  onClick={() => void run(it.action)}
                  className={cn(ACT, it.primary && "border-primary text-primary hover:text-primary")}
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
  compare,
  left,
  onPick,
}: {
  item: Extract<StripItem, { type: "take" }>;
  stage: Stage;
  disabled: boolean;
  /** The viewer compares: a click shows this version on the left. */
  compare: boolean;
  /** It is the one on the left now. */
  left: boolean;
  onPick: (id: string) => void;
}) {
  const title = !compare ? item.title : item.pressed ? `${item.title} · on the right` : left ? `${item.title} · on the left` : `Show on the left: ${item.title}`;
  return (
    <button
      type="button"
      aria-pressed={item.pressed}
      aria-current={left || undefined}
      aria-label={title}
      title={title}
      // aria-disabled, not disabled: the button just pressed keeps keyboard focus while the pick runs.
      aria-disabled={disabled || undefined}
      onClick={() => onPick(item.id)}
      className={cn(
        "group/take relative flex flex-col items-center gap-0.5 rounded-[9px] border border-transparent p-0.5 outline-none",
        "text-[length:calc(11.5px*var(--fs))] font-medium text-ink-3 tabular-nums hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring aria-disabled:cursor-wait aria-disabled:opacity-60",
        "aria-pressed:border-primary aria-pressed:bg-background aria-pressed:font-semibold aria-pressed:text-primary aria-pressed:shadow-[var(--glass-shadow)]",
        // The left side of a compare: a dashed outline, so it never reads as the picked one.
        "aria-[current=true]:border-dashed aria-[current=true]:border-ink-3 aria-[current=true]:bg-background aria-[current=true]:font-semibold aria-[current=true]:text-ink",
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
