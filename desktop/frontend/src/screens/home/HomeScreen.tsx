import { useEffect, useRef } from "react";
import type { Route } from "@/app/router";
import { useJobs } from "@/lib/jobs";
import { JobCard } from "./JobCard";
import { Library, useAssets } from "./Library";
import { MakePanel } from "./MakePanel";
import "./chain"; // the draw → 3D chain and the finish toast live at module scope

// Home: the Make panel (or the job card while you watch a job) and the recent grid; ?view=library is the full
// library with the kind filter and search. Layout and colours follow the approved design (Mokei.dc.html).
export function HomeScreen({ section }: { section?: Extract<Route, { name: "home" }>["section"] }) {
  const job = useJobs((s) => s.job);
  const watching = useJobs((s) => s.watching);
  const setWatching = useJobs((s) => s.setWatching);
  const dismiss = useJobs((s) => s.dismiss);
  const { assets, error, refresh } = useAssets();
  const scroller = useRef<HTMLDivElement>(null);
  const library = section === "library";
  const showJob = watching && !!job;

  useEffect(() => {
    scroller.current?.scrollTo({ top: 0 });
  }, [section]);

  // Escape: a running job goes to the background (the chip keeps it); a finished one is dropped.
  useEffect(() => {
    if (!showJob) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      if (useJobs.getState().job?.state === "running") setWatching(false);
      else dismiss();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [showJob, setWatching, dismiss]);

  return (
    <div ref={scroller} className="absolute inset-0 overflow-x-hidden overflow-y-auto bg-background text-ink">
      <div aria-hidden="true" className="pointer-events-none absolute top-[262px] left-1/2 -ml-[550px] h-[520px] w-[1100px] rounded-[50%] bg-ground-ellipse" />
      <div aria-hidden="true" className="pointer-events-none absolute top-[312px] left-1/2 -ml-[380px] h-[360px] w-[760px] rounded-[50%] bg-clay-grass opacity-70" />
      <div className="relative flex min-h-full flex-col items-center px-4 pt-[clamp(2rem,8vh,4.5rem)] pb-12">
        {showJob ? (
          <section aria-label="Job" className="w-full max-w-[38rem]">
            <JobCard job={job!} />
          </section>
        ) : null}
        {/* Kept mounted under the job card: the typed words and the photo are still there after "Run in background". */}
        {library ? null : (
          <div hidden={showJob} className="w-full max-w-[38rem]">
            <MakePanel />
          </div>
        )}
        <div className={library && !showJob ? "w-full max-w-[52rem]" : "mt-[clamp(2.5rem,7vh,4rem)] w-full max-w-[52rem]"}>
          <Library full={library} assets={assets} error={error} refresh={refresh} />
        </div>
      </div>
    </div>
  );
}
