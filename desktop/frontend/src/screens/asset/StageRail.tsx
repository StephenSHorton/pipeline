// The stage rail (legacy #stage-rail, index.html 1784-1792 and renderFlow 2812-2827): Ideas / Concept / 2D / 3D /
// Rig / Stack / Export. A stage the output does not need and that holds nothing is hidden; the pressed one is
// open; a dot says it holds files, an amber ring that it is stale, red that its check failed.
import { flowEntry, railHidden, railTitle, STAGE_LABEL, STAGES, useFlow } from "@/lib/flow";
import { cn } from "@/lib/utils";
import { useStage } from "./stage";

export function StageRail({ stem }: { stem: string }) {
  const stage = useStage((s) => (s.stem === stem ? s.stage : null));
  const output = useStage((s) => (s.stem === stem ? s.intent?.output : undefined));
  const enter = useStage((s) => s.enter);
  const flow = useFlow((s) => (s.stem === stem ? s.flow : null));
  if (!stage) return null;
  return (
    <nav aria-label="Stages" className="flex min-w-0 gap-0.5 overflow-x-auto rounded-[10px] bg-muted/60 p-0.5 [scrollbar-width:none]">
      {STAGES.filter((s) => !railHidden(flow, output, s, stage)).map((s) => {
        const entry = flowEntry(flow, s);
        const problem = !!(entry?.live && entry.summary?.level === "fail");
        return (
          <button
            key={s}
            type="button"
            aria-pressed={s === stage}
            title={railTitle(entry)}
            onClick={() => enter(s)}
            className={cn(
              "relative flex min-h-10 shrink-0 items-center gap-1.5 rounded-[8px] px-2.5 text-[0.8rem] font-medium text-ink-3 outline-none hover:text-ink focus-visible:ring-2 focus-visible:ring-ring",
              "aria-pressed:bg-background aria-pressed:font-semibold aria-pressed:text-ink aria-pressed:shadow-[var(--glass-shadow)]",
            )}
          >
            {STAGE_LABEL[s]}
            {entry?.live ? (
              <span
                aria-hidden="true"
                className={cn(
                  "size-1.5 rounded-full",
                  problem ? "bg-destructive" : entry.stale.length ? "bg-warning ring-2 ring-warning/30" : "bg-success",
                )}
              />
            ) : null}
          </button>
        );
      })}
    </nav>
  );
}
