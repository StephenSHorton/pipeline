// The job card (legacy #gen-overlay / drawJob) as a Mokei glass card in the Make column: title, the stages
// (done / now / next), the bar with time left, the step line, the one-line Degraded note and the actions.
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Stepper, type StepperStep } from "@/components/ui/stepper";
import {
  cancelJob,
  degradedLine,
  displayStem,
  jobDoing,
  jobLines,
  jobVerb,
  overlayPct,
  overlayPctText,
  stageSubText,
  startJob,
  useJobs,
  type Job,
} from "@/lib/jobs";
import { cn } from "@/lib/utils";

export function JobCard({ job }: { job: Job }) {
  const setWatching = useJobs((s) => s.setWatching);
  const dismiss = useJobs((s) => s.dismiss);
  const [cancelError, setCancelError] = useState("");
  const running = job.state === "running";

  // The 1 s ticker: stage seconds, the bar and time left move between polls.
  const [, tick] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [running]);

  // Never let the bar go backwards within one job.
  const [floor, setFloor] = useState<{ id: string; pct: number }>({ id: "", pct: 0 });
  const raw = overlayPct(job.progress, job.state, job);
  const pct = floor.id === job.id ? Math.max(floor.pct, raw) : raw;
  useEffect(() => {
    if (floor.id !== job.id || pct > floor.pct) setFloor({ id: job.id, pct });
  }, [job.id, pct, floor]);

  const name = displayStem(job.stem);
  const title = job.next
    ? job.next
    : running
    ? job.title || jobDoing(job)
    : job.state === "ok"
      ? `Opening ${name}`
      : job.state === "cancelled"
        ? `Stopped ${name}`
        : `Couldn't ${jobVerb(job)} ${name}`;
  const kicker = running || job.next ? "Working" : job.state === "ok" ? "Done" : job.state === "cancelled" ? "Cancelled" : "Error";
  const { step, log } = jobLines(job);
  const degraded = degradedLine(job.degraded);
  const stages = job.progress?.stages || [];
  const loading = running && job.progress?.stages[job.progress.stage_index]?.key === "load";
  const steps: StepperStep[] = stages.map((st, i) => {
    const idx = job.progress!.stage_index;
    const stopped = !running && job.state !== "ok" && i === idx;
    return { label: st.label, time: stageSubText(job.progress!, i, job.state, job) || undefined, ...(stopped ? { state: "stopped" as const } : {}) };
  });

  // Details follows the log's end while it runs (legacy shows the last 30 lines then).
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  const cancel = async () => {
    setCancelError("");
    const err = await cancelJob();
    if (err) setCancelError(err);
  };

  return (
    <Card size="hud" role="region" aria-label="Job" data-state={job.state} className="w-full gap-0 rounded-[20px] p-4 text-ink">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <Badge
            size="hud-sm"
            className="h-6 rounded-lg px-[0.55rem] text-[0.72rem] font-bold"
            variant={running || job.next ? "signal" : job.state === "ok" ? "success" : job.state === "error" ? "danger" : "slate"}
          >
            {running ? "Running" : kicker}
          </Badge>
          <h2 className="mt-2 truncate text-[1.05rem] font-semibold tracking-[-0.02em]">{title}</h2>
          <p className="mt-0.5 line-clamp-2 text-[0.82rem] text-ink-3" aria-live="polite">
            {cancelError || step}
          </p>
        </div>
        {job.preview_url ? (
          <img src={job.preview_url} alt="The picture so far" className="size-16 shrink-0 rounded-[11px] bg-clay-sky object-cover" />
        ) : null}
      </div>

      {degraded ? (
        <p role="status" className="mt-2 line-clamp-2 text-[0.78rem] text-ink-3" title={degraded}>
          {degraded}
        </p>
      ) : null}

      {steps.length ? <Stepper size="sm" aria-label="Stages" steps={steps} current={job.progress!.stage_index} done={job.state === "ok"} className="mt-3" /> : null}

      <div className="mt-3">
        <Progress value={pct} tone="blue" size="hud" aria-label="Progress" className={cn(loading && !job.stage_est && "animate-pulse motion-reduce:animate-none")} />
        <p className="mt-1.5 text-[0.78rem] text-ink-3 tabular-nums" aria-live="polite">
          {overlayPctText(job.progress, job.state, pct, job)}
        </p>
      </div>

      {log ? (
        <details className="mt-2 text-[0.75rem]" open={job.state === "error" || undefined}>
          <summary className="inline-flex min-h-11 cursor-pointer items-center rounded-md text-ink-3 outline-none focus-visible:ring-2 focus-visible:ring-ring">Details</summary>
          <pre ref={logRef} className="mt-1 max-h-48 overflow-auto rounded-lg bg-muted p-2 text-[0.7rem] whitespace-pre-wrap text-ink-2">{log}</pre>
        </details>
      ) : null}

      <div className="mt-3 flex flex-wrap justify-end gap-2">
        {running ? (
          <>
            <Button variant="ghost" className="min-h-11 rounded-xl px-4" onClick={() => setWatching(false)}>
              Run in background
            </Button>
            <Button variant="outline" className="min-h-11 rounded-xl px-4" disabled={job.stopping} onClick={() => void cancel()}>
              {job.stopping ? "Stopping…" : loading ? "Stops after loading" : "Cancel"}
            </Button>
          </>
        ) : job.state === "ok" || job.next ? null : (
          <>
            <Button variant="ghost" className="min-h-11 rounded-xl px-4" onClick={dismiss}>
              Close
            </Button>
            {job.spec ? (
              <Button className="min-h-11 rounded-xl px-4" onClick={() => void startJob(job.spec!, job.opts)}>
                Try again
              </Button>
            ) : null}
          </>
        )}
      </div>
    </Card>
  );
}
