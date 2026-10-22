import { useEffect, useRef } from "react";
import type { Route } from "@/app/router";
import { useJobs } from "@/lib/jobs";
import { JobCard } from "./JobCard";
import { Library, useAssets } from "./Library";
import { MakePanel } from "./MakePanel";
import "./chain"; // the draw → 3D chain and the finish toast live at module scope

// Home: the Make panel (or the job card while you watch a job); ?view=library is the full library with the kind
// filter and search. Make is only the create area, centred on its clay ground, with no recent grid under it
// (Stephen: "Why does make show the library if we have a library nav? Library should be a separate thing from the
// create area."): the top bar's Library is where the assets live. Layout and colours follow the approved design
// (Mokei.dc.html).
export function HomeScreen({ section }: { section?: Extract<Route, { name: "home" }>["section"] }) {
  const job = useJobs((s) => s.job);
  const watching = useJobs((s) => s.watching);
  const setWatching = useJobs((s) => s.setWatching);
  const dismiss = useJobs((s) => s.dismiss);
  const scroller = useRef<HTMLDivElement>(null);
  const library = section === "library";
  const showJob = watching && !!job;
  // The make card sits in the middle of the screen; the job card and the library keep the top-down layout.
  const centred = !library && !showJob;

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
      {centred ? null : <Ground top={262} />}
      <div className="relative flex min-h-full flex-col items-center px-4 pt-[clamp(2rem,8vh,4.5rem)] pb-12">
        {showJob ? (
          <section aria-label="Job" className="w-full max-w-[38rem]">
            <JobCard job={job!} />
          </section>
        ) : null}
        {/* Kept mounted under the job card: the typed words and the photo are still there after "Run in background".
            my-auto centres it in the column; the ground rides with it, at the offset it had under the top padding. */}
        {library ? null : (
          <div hidden={showJob} className="relative my-auto w-full max-w-[38rem]">
            {centred ? <Ground top={190} /> : null}
            <div className="relative">
              <MakePanel />
            </div>
          </div>
        )}
        {library ? <LibrarySection below={showJob} /> : null}
      </div>
    </div>
  );
}

/** The clay ground ellipses behind the card (the approved design's soft pad), `top` px down from their anchor. */
function Ground({ top }: { top: number }) {
  return (
    <>
      <div aria-hidden="true" className="pointer-events-none absolute left-1/2 -ml-[550px] h-[520px] w-[1100px] rounded-[50%] bg-ground-ellipse" style={{ top }} />
      <div aria-hidden="true" className="pointer-events-none absolute left-1/2 -ml-[380px] h-[360px] w-[760px] rounded-[50%] bg-clay-grass opacity-70" style={{ top: top + 50 }} />
    </>
  );
}

/** The library page: its own component, so Make never asks GET /assets. `below`: under a watched job's card. */
function LibrarySection({ below }: { below: boolean }) {
  const { assets, error, refresh } = useAssets();
  return (
    <div className={below ? "mt-[clamp(2.5rem,7vh,4rem)] w-full max-w-[52rem]" : "w-full max-w-[52rem]"}>
      <Library assets={assets} error={error} refresh={refresh} />
    </div>
  );
}
