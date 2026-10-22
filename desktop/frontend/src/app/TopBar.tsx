import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { getHealth, type Health } from "@/lib/api";
import { healthLabel, jobChipText, shapeMvNote, useJobs } from "@/lib/jobs";
import { cn } from "@/lib/utils";
import { StageRail } from "@/screens/asset/StageRail";
import { ActivityChip } from "./Activity";
import { onJobChip } from "./open";
import { navigate, type Route } from "./router";

// The glass top bar from the approved Make-panel design: brand left, Make / Library / Settings right; on an
// asset the stage rail sits in the middle (legacy #stage-rail). At phone width the factory line hides (its tooltip
// text is in the job chip and on Home), the brand word goes below 400 px and the nav tightens.
export function TopBar({ route }: { route: Route }) {
  const active = route.name === "home" ? (route.section === "library" ? "library" : "make") : route.name === "settings" ? "settings" : null;
  return (
    <header
      data-slot="topbar"
      className="relative z-10 mx-[clamp(12px,2vw,20px)] mt-3 flex min-w-0 items-center justify-between gap-2 rounded-[14px] border border-[var(--topbar-border)] bg-[var(--topbar-bg)] py-1.5 pr-2 pl-3.5 shadow-[var(--topbar-shadow)] [-webkit-backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))] [backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))] sm:gap-4 max-sm:flex-wrap max-sm:pl-2.5"
    >
      <a
        href="/"
        onClick={(e) => {
          e.preventDefault();
          navigate({ name: "home" });
        }}
        aria-label="Pipeline home"
        className="flex min-h-11 shrink-0 items-center gap-[0.55rem] rounded-[10px] text-[0.95rem] font-bold text-ink no-underline outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span aria-hidden="true" className="flex size-[22px] items-center justify-center rounded-[7px] bg-primary">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="stroke-primary-foreground" strokeWidth="2.2" strokeLinejoin="round">
            <path d="M12 3 20 7.5v9L12 21 4 16.5v-9z" />
            <path d="M4 7.5 12 12l8-4.5M12 12v9" />
          </svg>
        </span>
        <span className="max-[400px]:sr-only">Pipeline</span>
      </a>
      {route.name === "asset" ? (
        // At phone width the rail takes its own row under the brand and the nav.
        <div className="flex min-w-0 flex-1 justify-center max-sm:order-last max-sm:basis-full max-sm:pb-0.5">
          <StageRail stem={route.stem} />
        </div>
      ) : null}
      <div className="flex min-w-0 items-center gap-2">
        <ActivityChip />
        <JobChip />
        <FactoryPill />
        <nav aria-label="Main" className="flex shrink-0 gap-1 max-sm:gap-0">
          <NavButton current={active === "make"} onClick={() => navigate({ name: "home", section: "make" })}>Make</NavButton>
          <NavButton current={active === "library"} onClick={() => navigate({ name: "home", section: "library" })}>Library</NavButton>
          <NavButton current={active === "settings"} onClick={() => navigate({ name: "settings" })}>Settings</NavButton>
        </nav>
      </div>
    </header>
  );
}

const navClass = (current: boolean) =>
  cn(
    "h-auto min-h-11 rounded-[10px] px-[0.85rem] text-[0.85rem] no-underline max-sm:px-2",
    current ? "font-semibold" : "font-medium text-ink-3 hover:text-ink",
  );

function NavButton({ current, onClick, children }: { current: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <Button variant={current ? "secondary" : "ghost"} aria-current={current ? "page" : undefined} className={navClass(current)} onClick={onClick}>
      {children}
    </Button>
  );
}

/**
 * The job, from any screen (legacy renderJobChip): running → "… · Show", then "<name> is ready · Open",
 * "<name> cancelled · Dismiss" or "<name> failed · Details". Hidden only while you watch a running job.
 */
function JobChip() {
  const job = useJobs((s) => s.job);
  const watching = useJobs((s) => s.watching);
  if (!job || (watching && (job.state === "running" || !!job.next))) return null;
  const running = job.state === "running" || !!job.next;
  const text = jobChipText(job);
  return (
    <Button
      variant="secondary"
      onClick={onJobChip}
      data-state={job.state}
      title={text}
      className="h-auto min-h-11 min-w-0 max-w-[16rem] justify-start gap-2 rounded-full bg-[var(--glass-bg)] px-3 text-[0.8rem] font-medium text-ink shadow-[var(--glass-shadow)] max-sm:max-w-[7.5rem]"
    >
      <span
        aria-hidden="true"
        className={cn(
          "size-2 shrink-0 rounded-full",
          running ? "animate-pulse bg-signal motion-reduce:animate-none" : job.state === "ok" ? "bg-success" : job.state === "error" ? "bg-destructive" : "bg-[var(--muted-2)]",
        )}
      />
      <span className="truncate">{text}</span>
    </Button>
  );
}

/** "Factory: <label><multi-view note>" on hover; the line says Starting / Ready / busy / the start error. */
function FactoryPill() {
  const [h, setH] = useState<Health | null>(null);
  const text = useJobs((s) => s.statusText);
  const state = useJobs((s) => s.statusState);
  const running = useJobs((s) => s.job?.state === "running");
  useEffect(() => {
    let live = true;
    getHealth().then((x) => live && setH(x)).catch(() => {});
    return () => {
      live = false;
    };
  }, [running]);
  if (!text && !h) return null;
  return (
    <span
      role="status"
      title={`Factory: ${healthLabel(h)}${shapeMvNote(h?.features)}`}
      className={cn("max-w-[22rem] truncate px-2 text-[0.8rem] max-sm:hidden", state === "error" ? "text-danger-text" : "text-ink-3")}
    >
      {text || "Ready."}
    </span>
  );
}
