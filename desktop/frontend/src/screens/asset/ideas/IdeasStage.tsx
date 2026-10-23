// The Ideas stage (legacy #ideas-stage, renderIdeas and the #ideas-more form, index.html 2148-2150, 2253-2256,
// 3585-3735): a full page of drawn ideas, each with Keep and Use, the job's live slot and its waiting slots, then
// the More ideas bar (prompt, Clear unkept, More ideas) at the bottom. No versions panel here.
// A picture opens full size in an in-app lightbox: window.open / target=_blank is unreliable in the Wails
// WKWebView (gaps B6). Kept is the primary border, In use the primary ring, the live slot the signal accent.
import { Dialog } from "@base-ui/react/dialog";
import { Check, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useFlow } from "@/lib/flow";
import { cancelJob, onJobEnded, useJobs } from "@/lib/jobs";
import { askConfirm } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { useStage } from "../stage";
import type { StageProps } from "../types";
import { clearIdeas, getIdeas, keepIdea, makeConcept, type IdeaItem, type IdeasDoc } from "./api";
import {
  clearConfirmText,
  draftFor,
  endedStatus,
  ideasJobHere,
  ideasLoading,
  ideasPending,
  IDEAS_COPY,
  ideaUrl,
  jobItems,
  liveSlotText,
  loadStatus,
  setIdeasDraft,
  unkeptCount,
} from "./ideas";
import { startMoreIdeas } from "./more";

const msg = (err: unknown) => (err instanceof Error ? err.message : String(err));

// Kept shows on the Keep button only (the toggle chip's on look); In use shows as the card's ring only.
const PRESSED = "aria-pressed:border-[#bcd2fb] aria-pressed:bg-accent aria-pressed:text-accent-foreground aria-pressed:font-semibold";

export function IdeasStage(props: StageProps) {
  // Keyed by stem: the grid, the prompt and the status line start over for another asset.
  return <IdeasBody key={props.stem} {...props} />;
}

function IdeasBody({ stem, flowVersion, enter }: StageProps) {
  const job = useJobs((s) => s.job);
  const starting = useJobs((s) => s.busy);
  const [doc, setDoc] = useState<IdeasDoc | null>(null);
  const [status, setStatus] = useState("");
  const [prompt, setPromptState] = useState(() => draftFor(stem));
  const [full, setFull] = useState<IdeaItem | null>(null);
  const seq = useRef(0);

  const setPrompt = useCallback(
    (text: string) => {
      setIdeasDraft(stem, text);
      setPromptState(text);
    },
    [stem],
  );

  const here = ideasJobHere(job, stem);
  const pending = ideasPending(job, stem);
  const anyRunning = job?.state === "running";

  // Legacy renderIdeas: an empty grid with nothing drawing says so (over whatever the line said before).
  const settle = useCallback(
    (d: IdeasDoc | null) => {
      if (!(d?.items ?? []).length && !ideasPending(useJobs.getState().job, stem)) setStatus(IDEAS_COPY.empty);
    },
    [stem],
  );

  const load = useCallback(async () => {
    const n = ++seq.current;
    const data = await getIdeas(stem).catch(() => null);
    if (n !== seq.current) return;
    setDoc(data);
    if (data?.prompt && !draftFor(stem)) setPrompt(data.prompt);
    const say = loadStatus(data, ideasJobHere(useJobs.getState().job, stem));
    if (say) setStatus(say);
    settle(data);
  }, [stem, setPrompt, settle]);

  // A load still in flight when this asset closes must not write its prompt over the next asset's draft.
  useEffect(
    () => () => {
      seq.current++;
    },
    [],
  );

  // On entry, and whenever the flow is re-read (a keep, a use, a finished job).
  useEffect(() => {
    void load();
  }, [load, flowVersion]);

  // Each picture the job here writes reloads the grid; the live slot repaints between them from the poll.
  const written = here ? jobItems(job) : 0;
  useEffect(() => {
    if (written) void load();
  }, [written, load]);

  // Legacy jobEnded / finishIdeasJob: an Ideas job on this asset ended. Stopped or failed says so; then reload.
  useEffect(
    () =>
      onJobEnded((j) => {
        if (j.kind !== "ideas" || j.stem !== stem) return;
        setStatus(j.state === "ok" ? "" : endedStatus(j));
        void load();
      }),
    [stem, load],
  );

  // Keep and Clear return the fresher doc: a GET already in flight must not put the old one back.
  const toggleKeep = async (item: IdeaItem) => {
    seq.current++;
    try {
      const d = await keepIdea(stem, item.file, !item.keep);
      setDoc(d);
      settle(d);
    } catch (err) {
      setStatus(msg(err));
    }
  };

  const use = async (item: IdeaItem) => {
    try {
      await makeConcept(stem, item.file);
      // Concept reads its picture from the flow, so it is re-read before the stage changes. The library re-reads
      // /assets when Home mounts.
      await useFlow.getState().load(stem);
      // You may have opened another asset while those ran: only this one goes to Concept.
      if (useStage.getState().stem !== stem) return;
      enter("concept");
    } catch (err) {
      setStatus(msg(err));
    }
  };

  const more = (e: FormEvent) => {
    e.preventDefault();
    const text = prompt.trim();
    if (!text) return;
    if (useJobs.getState().job?.state === "running" || useJobs.getState().busy) {
      setStatus(IDEAS_COPY.busy);
      return;
    }
    setStatus("");
    void startMoreIdeas(stem, text).catch((err) => setStatus(msg(err)));
  };

  const clear = async () => {
    const n = unkeptCount(doc?.items);
    if (!n) {
      setStatus(IDEAS_COPY.allKept);
      return;
    }
    if (!(await askConfirm(clearConfirmText(n), "Delete"))) return;
    seq.current++;
    try {
      const d = await clearIdeas(stem);
      setDoc(d);
      settle(d);
    } catch (err) {
      setStatus(msg(err));
    }
  };

  const stop = async () => {
    const err = await cancelJob();
    if (err) setStatus(err);
  };

  const items = doc?.items ?? [];
  return (
    <section aria-label="Ideas" className="absolute inset-0 flex flex-col text-ink">
      <div className="min-h-0 flex-1 overflow-y-auto px-[clamp(12px,2vw,20px)] pt-4 pb-3">
        {items.length || pending ? (
          <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3 p-0">
            {items.map((item) => (
              <IdeaCard
                key={item.file}
                stem={stem}
                item={item}
                busyHere={here}
                onOpen={() => setFull(item)}
                onKeep={() => void toggleKeep(item)}
                onUse={() => void use(item)}
              />
            ))}
            {Array.from({ length: pending }, (_, i) =>
              i === 0 && here ? <LiveSlot key="live" job={job} pending={pending} onStop={() => void stop()} /> : <WaitingSlot key={`wait-${i}`} />,
            )}
          </ul>
        ) : null}
        <p role="status" aria-live="polite" className={cn("m-0 min-h-5 px-1 text-[0.85rem] text-ink-3", status && (items.length || pending) && "mt-3")}>
          {status}
        </p>
      </div>

      <form
        onSubmit={more}
        autoComplete="off"
        aria-label="More ideas"
        // As wide as the Concept bar (56rem), not the whole window.
        className="glass-panel mx-auto mb-3 flex w-[calc(100%-24px)] max-w-[56rem] flex-wrap items-center gap-2 rounded-[14px] p-2"
      >
        <label className="min-w-0 flex-[1_1_16rem]">
          <span className="sr-only">Prompt</span>
          <Input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={IDEAS_COPY.placeholder}
            title={IDEAS_COPY.promptTitle}
            spellCheck={false}
            className="hit-44 h-9 rounded-xl bg-white/90 px-3"
          />
        </label>
        <Tip title={IDEAS_COPY.clearTitle} className="max-sm:flex-1">
          <Button type="button" variant="outline" title={IDEAS_COPY.clearTitle} disabled={here} onClick={() => void clear()} className="hit-44 min-h-9 w-full rounded-xl px-4">
            Clear unkept
          </Button>
        </Tip>
        <Tip title={IDEAS_COPY.moreTitle} className="max-sm:flex-1">
          <Button type="submit" title={IDEAS_COPY.moreTitle} disabled={anyRunning || starting} className="hit-44 min-h-9 w-full rounded-xl px-4">
            More ideas
          </Button>
        </Tip>
      </form>

      <Lightbox stem={stem} item={full} onClose={() => setFull(null)} />
    </section>
  );
}

/** A title that still shows while the control inside is disabled (disabled buttons get no pointer events). */
function Tip({ title, className, children }: { title: string; className?: string; children: ReactNode }) {
  return (
    <span title={title} className={cn("inline-flex", className)}>
      {children}
    </span>
  );
}

function IdeaCard({
  stem,
  item,
  busyHere,
  onOpen,
  onKeep,
  onUse,
}: {
  stem: string;
  item: IdeaItem;
  busyHere: boolean;
  onOpen: () => void;
  onKeep: () => void;
  onUse: () => void;
}) {
  const useTitle = busyHere ? IDEAS_COPY.useWait : IDEAS_COPY.useTitle;
  return (
    <li
      data-kept={item.keep || undefined}
      data-used={item.used || undefined}
      className={cn(
        "flex flex-col gap-2 rounded-2xl border border-[var(--glass-border)] bg-[var(--glass-bg)] p-2 shadow-[var(--glass-shadow)]",
        item.used && "ring-2 ring-primary",
      )}
    >
      <button
        type="button"
        aria-haspopup="dialog"
        aria-label={`Open ${item.prompt || item.file} full size, seed ${item.seed}`}
        onClick={onOpen}
        className="block aspect-square w-full cursor-zoom-in overflow-hidden rounded-xl border-0 bg-clay-sky p-0 outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <img src={ideaUrl(stem, item)} alt={item.prompt || item.file} title={`Seed ${item.seed}`} loading="lazy" className="size-full object-contain" />
      </button>
      <div className="flex gap-2">
        <Button type="button" variant="outline" aria-pressed={!!item.keep} onClick={onKeep} className={cn("hit-44 min-h-8 flex-1 rounded-xl", PRESSED)}>
          {item.keep ? <Check aria-hidden="true" /> : null}
          {item.keep ? "Kept" : "Keep"}
        </Button>
        <Tip title={useTitle} className="flex-1">
          <Button
            type="button"
            variant="outline"
            title={useTitle}
            disabled={busyHere}
            onClick={onUse}
            className={cn("hit-44 min-h-8 w-full rounded-xl", item.used && "font-semibold text-primary")}
          >
            {item.used ? <Check aria-hidden="true" /> : null}
            {item.used ? "In use" : "Use"}
          </Button>
        </Tip>
      </div>
    </li>
  );
}

/** The first waiting slot: the picture as it draws, which one it is and the step, and Stop. */
function LiveSlot({ job, pending, onStop }: { job: NonNullable<ReturnType<typeof useJobs.getState>["job"]>; pending: number; onStop: () => void }) {
  const loading = ideasLoading(job);
  return (
    <li data-live className="relative aspect-square overflow-hidden rounded-2xl border-2 border-signal bg-clay-sky shadow-[var(--glass-shadow)]">
      {job.preview_url ? <img src={job.preview_url} alt="" className="absolute inset-0 size-full object-contain" /> : null}
      <span className="absolute inset-x-2 bottom-2 rounded-lg bg-white/92 px-2.5 py-1.5 text-[0.78rem] text-ink tabular-nums shadow-[var(--glass-shadow)]">
        {liveSlotText(job, pending)}
      </span>
      <Tip title={loading ? IDEAS_COPY.stopsAfterLoading : ""} className="absolute top-2 right-2">
        <Button type="button" variant="outline" size="sm" title={loading ? IDEAS_COPY.stopsAfterLoading : undefined} disabled={!!job.stopping} onClick={onStop} className="hit-44 rounded-lg bg-white/92">
          Stop
        </Button>
      </Tip>
    </li>
  );
}

function WaitingSlot() {
  return (
    <li className="grid aspect-square place-items-center rounded-2xl border border-dashed border-[color-mix(in_srgb,var(--ink-3)_35%,transparent)] text-[0.8rem] text-ink-3">
      {IDEAS_COPY.waiting}
    </li>
  );
}

/** One idea full size, over the page. Escape, the backdrop or Close shuts it. */
function Lightbox({ stem, item, onClose }: { stem: string; item: IdeaItem | null; onClose: () => void }) {
  // Keep the last picture while the dialog animates out.
  const [shown, setShown] = useState<IdeaItem | null>(item);
  useEffect(() => {
    if (item) setShown(item);
  }, [item]);
  return (
    <Dialog.Root open={!!item} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-[rgba(15,23,42,0.4)] data-open:animate-in data-open:fade-in-0" />
        <Dialog.Popup className="fixed top-1/2 left-1/2 z-50 flex max-h-[calc(100dvh-2rem)] w-[min(56rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 flex-col gap-2 rounded-xl border border-[var(--glass-border)] bg-background p-3 text-ink shadow-[var(--glass-shadow)] outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95">
          <div className="flex items-center gap-2 pl-1">
            <Dialog.Title className="m-0 min-w-0 flex-1 truncate text-sm font-semibold">{shown?.prompt || shown?.file}</Dialog.Title>
            <Dialog.Close render={<Button variant="ghost" size="icon-sm" aria-label="Close" title="Close" className="hit-44" />}>
              <X />
            </Dialog.Close>
          </div>
          {shown ? (
            <img
              src={ideaUrl(stem, shown)}
              alt={shown.prompt || shown.file}
              className="min-h-0 w-full flex-1 rounded-lg bg-clay-sky object-contain"
              style={{ maxHeight: "calc(100dvh - 7rem)" }}
            />
          ) : null}
          <Dialog.Description className="m-0 pl-1 text-[0.78rem] text-ink-3 tabular-nums">{shown ? `Seed ${shown.seed}` : ""}</Dialog.Description>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
