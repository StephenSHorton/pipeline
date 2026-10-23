// The stage rail (legacy #stage-rail, index.html 1784-1792 and renderFlow 2812-2827): Ideas / Concept / 2D / 3D /
// Rig / Stack / Export. A stage the output does not need and that holds nothing is hidden; the pressed one is
// open; a dot says it holds files, an amber ring that it is stale, red that its check failed.
import { flowEntry, railHidden, railTitle, STAGE_LABEL, STAGES, useFlow } from "@/lib/flow";
import { cn } from "@/lib/utils";
import { TRAY, TRAY_ITEM } from "@/screens/viewer/classes";
import { useStage } from "./stage";

export function StageRail({ stem }: { stem: string }) {
  const stage = useStage((s) => (s.stem === stem ? s.stage : null));
  const output = useStage((s) => (s.stem === stem ? s.intent?.output : undefined));
  const enter = useStage((s) => s.enter);
  const flow = useFlow((s) => (s.stem === stem ? s.flow : null));
  if (!stage) return null;
  return (
    // The shared segmented tray (the open stage a white pill), one row that never wraps. It gets its own row in the
    // top bar before it would clip (TopBar); if it still overflows, the thin scrollbar shows there is more.
    <nav aria-label="Stages" className={cn(TRAY, "flex-nowrap overflow-x-auto [scrollbar-width:thin]")}>
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
            className={cn(TRAY_ITEM, "relative flex h-[30px] shrink-0 items-center gap-1.5 outline-none focus-visible:ring-2 focus-visible:ring-ring")}
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
