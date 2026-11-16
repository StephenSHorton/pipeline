// The Export stage (legacy #export-stage, index.html 2152-2156, 3313-3507): a full page of cards saying what
// out/<stem>/export/ holds, the PS1 and Rigged hero cards with their s&box steps, and the Export / Show in Finder row.
// The legacy row lived in the versions strip; here it sits at the foot of the page (the versions panel has its own
// column at the left), so the lead's "Press Export below." still reads true.
import { FolderOpen, LoaderCircle } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { outUrl } from "@/lib/api";
import { flowEntry } from "@/lib/flow";
import { cn } from "@/lib/utils";
import type { StageProps } from "../types";
import { clearExport, exportAsset, loadExport, readRigFacts, showInFinder, useExport, type RigFacts } from "./api";
import {
  exportAction,
  exportCards,
  EXPORT_TITLE,
  fbxLine,
  FINDER_TITLE,
  leadText,
  NOTHING_TO_EXPORT,
  showsDoc,
  type PlainCard,
  type PsxCard,
  type RigCard,
} from "./export";

const H3 = "m-0 text-[0.75rem] font-semibold tracking-[0.06em] text-ink-3 uppercase";
const MONO = "font-mono text-[0.78rem] leading-[1.5]";
// Card's glass-panel sets its own border, fill and shadow, and size=sm sets py and bg under a data- selector that
// outranks plain utilities, so the overrides below are marked important (!).
// Hero cards span the grid; figure + text side by side above 760 px, one column at or below it (legacy ix:818).
const HERO = "col-span-full grid items-start gap-x-[18px] gap-y-1.5 border-primary/30! px-4 py-3.5!";
const HERO_COLS = "min-[761px]:grid-cols-[minmax(180px,260px)_minmax(0,1fr)]";

export function ExportStage({ stem, flow, flowVersion, intent }: StageProps) {
  const doc = useExport((s) => (s.stem === stem ? s.doc : null));
  const exporting = useExport((s) => s.exporting);
  const entry = flowEntry(flow, "export");

  // Re-read on entry and after every flow load (a pick of an older export, a finished export or job). When the flow
  // says the export stage holds nothing, the doc would not show anyway, so it is not asked for (no 404).
  const exportHolds = flow ? !!entry?.live : null;
  useEffect(() => {
    if (exportHolds === false) clearExport(stem);
    else void loadExport(stem);
  }, [stem, flowVersion, exportHolds]);

  const live = showsDoc(doc, entry);
  const made = live ? doc.made || 0 : 0;
  const when = made ? new Date(made * 1000).toLocaleString() : "";
  // With nothing to export there is no Export button, so the lead does not point at one.
  const nothing = exportAction(flow, exporting).kind === "none";
  const leadFull = leadText(stem, doc, entry, when);
  const lead = nothing ? leadFull.replace(" Press Export below.", "") : leadFull;
  const staleAt = lead.indexOf(" · made from older picks");
  const cards = exportCards(flow, intent, doc);
  // Every file the flow lists, so the rig card asks only for the rig files that are there (null: flow not read yet).
  const files = useMemo(() => (flow ? new Set(flow.stages.flatMap((s) => s.files)) : null), [flow]);
  const file = (rel: string) => `${outUrl(stem, `export/${rel}`, false)}?v=${made}`;

  return (
    <section aria-label="Export" className="absolute inset-0 overflow-x-hidden overflow-y-auto bg-background text-ink">
      <div className="mx-auto flex min-h-full w-full max-w-[76rem] flex-col px-4 pt-5 sm:px-5">
        <h2 className="m-0 mb-1.5 text-[0.95rem] font-semibold text-ink">Export</h2>
        <p className="m-0 max-w-[48rem] text-[0.85rem] break-words text-ink-3">
          {staleAt < 0 ? (
            lead
          ) : (
            <>
              {lead.slice(0, staleAt)}
              {/* Stale: accent2 dot, dark ink text (no yellow text on white). */}
              <span className="font-medium text-ink">
                {" · "}
                <span aria-hidden="true" className="mr-1 inline-block size-2 rounded-full bg-signal align-middle" />
                {lead.slice(staleAt + 3)}
              </span>
            </>
          )}
        </p>

        <div className="mt-3 grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
          {cards.map((c) =>
            c.kind === "rig" ? (
              <RigHero key={c.key} stem={stem} card={c} made={made} files={files} />
            ) : c.kind === "psx" ? (
              <PsxHero key={c.key} card={c} src={c.texture ? file(c.texture) : ""} />
            ) : (
              <Plain key={c.key} card={c} src={c.img ? file(c.img) : ""} />
            ),
          )}
        </div>

        <ActionRow stem={stem} exporting={exporting} flow={flow} />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ the action row (legacy takes strip, export branch) */

function ActionRow({ stem, exporting, flow }: { stem: string; exporting: boolean; flow: StageProps["flow"] }) {
  const act = exportAction(flow, exporting);
  // Right under the cards, left-aligned with them, as plain buttons; sticky, so it only floats once the page scrolls.
  // With nothing to export it is a plain line, not a pill that looks like a button.
  if (act.kind === "none")
    return (
      <div className="pt-3 pb-4">
        <p className="m-0 px-1 text-[0.85rem] text-ink-3">{NOTHING_TO_EXPORT}</p>
      </div>
    );
  return (
    <div className="pointer-events-none sticky bottom-0 pt-3 pb-4">
      <div className="pointer-events-auto flex w-fit max-w-full flex-wrap items-center gap-2">
        {/* One live region, mounted for good, so the start and the end of an export are both announced. */}
        <span role="status" aria-live="polite" className="sr-only">
          {act.kind === "exporting" ? "Exporting…" : ""}
        </span>
        {/* The button stays mounted while exporting (disabled, "Exporting…"), so focus does not drop to the body. */}
        <Button className="hit-44 min-h-9 px-4 disabled:opacity-80" title={EXPORT_TITLE} disabled={act.kind === "exporting"} onClick={() => void exportAsset(stem)}>
          {act.kind === "exporting" ? (
            <>
              <LoaderCircle aria-hidden="true" className="animate-spin" />
              Exporting…
            </>
          ) : (
            act.label
          )}
        </Button>
        {act.finder ? (
          <Button variant="outline" className="hit-44 min-h-9 px-3.5" title={FINDER_TITLE} onClick={() => void showInFinder(stem)}>
            <FolderOpen aria-hidden="true" />
            Show in Finder
          </Button>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ cards */

function Plain({ card, src }: { card: PlainCard; src: string }) {
  return (
    <Card
      size="sm"
      className={cn(
        "gap-2 self-start px-3 py-2.5!",
        // Missing: dashed and quiet, the glass dropped so it reads as a slot, not a file.
        card.missing && "border-dashed! border-ink-3/45! bg-white/35! text-ink-3 shadow-none!",
      )}
    >
      <h3 className={H3}>{card.title}</h3>
      {src ? <img alt="" src={src} className="max-h-[180px] w-full rounded-lg bg-muted object-contain" /> : null}
      <ul className={cn("m-0 list-none p-0 text-[0.8rem] leading-[1.5] break-words", card.missing ? "text-ink-3" : "text-ink")}>
        {/* Mono for file names only; "Ready" and "Nothing yet" are words. */}
        {card.lines.map((l, i) => (
          <li key={i} className={cn(/[./]/.test(l) && MONO)}>
            {l}
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Steps({ children }: { children: ReactNode }) {
  return (
    <div>
      <strong className="text-[0.85rem] font-semibold text-ink">Into s&amp;box</strong>
      <ol className="mt-1 mb-0 list-decimal pl-[18px] marker:text-ink-3 text-[0.8rem] leading-[1.55] text-ink-3 [&_b]:font-medium [&_b]:text-ink [&_code]:font-mono [&_code]:text-[0.75rem] [&_code]:text-ink">
        {children}
      </ol>
    </div>
  );
}

const FACTS = "m-0 text-[0.85rem] break-words text-ink [&_b]:font-semibold";

// The PS1 model: the files, the facts and how to bring it into s&box (legacy psxCard).
function PsxHero({ card: p, src }: { card: PsxCard; src: string }) {
  return (
    <Card size="sm" className={cn(HERO, HERO_COLS)}>
      <h3 className={cn(H3, "col-span-full text-primary")}>{p.title}</h3>
      <figure className="m-0 min-w-0">
        {src ? (
          <img alt="The model's texture" src={src} className="max-h-[220px] w-full rounded-lg bg-muted object-contain [image-rendering:pixelated]" />
        ) : (
          // No texture in the doc: a quiet slot, not a broken image.
          <div aria-hidden="true" className="aspect-square max-h-[220px] w-full rounded-lg bg-muted" />
        )}
        <figcaption className="mt-1 font-mono text-[0.7rem] text-ink-3">
          {p.name}.png · the {p.atlas} px texture
        </figcaption>
      </figure>
      <div className="grid min-w-0 content-start gap-1.5">
        <p className={FACTS}>
          <b>{p.name}.obj</b> + <b>{p.name}.png</b> + {p.name}.mtl · {p.triangles} triangles · {p.atlas} px texture
        </p>
        <p className={FACTS}>
          {p.height} in tall{p.size ? ` (${p.size})` : ""} · {p.axes} · nearest sampling, no mips
        </p>
        <Steps>
          <li>
            Show in Finder and copy <code>3d/psx/</code> into your project's Assets folder.
          </li>
          <li>
            Right-click <b>{p.name}.obj</b> → <b>Create Model</b>. {p.importNote}
          </li>
          <li>
            Give it a material on <b>psx.shader</b> with <b>{p.name}.png</b> as TextureColor, point filtering, no mips.
          </li>
        </Steps>
      </div>
    </Card>
  );
}

// The rigged model: what it is and how its clips get into s&box (legacy rigCard + rigCardFacts). The FBX line and the
// contact sheet come from out/<stem>/rig/, read once per export.
function RigHero({ stem, card: r, made, files }: { stem: string; card: RigCard; made: number; files: Set<string> | null }) {
  const [facts, setFacts] = useState<(RigFacts & { stem: string; t: number }) | null>(null);
  useEffect(() => {
    let on = true;
    void readRigFacts(stem, files).then((f) => on && setFacts({ ...f, stem, t: Date.now() }));
    return () => {
      on = false;
    };
  }, [stem, made, files]);
  const mine = facts?.stem === stem ? facts : null;
  const line = mine ? fbxLine(mine.fbxCheck, mine.fbxNote) : null;
  const sheet = mine?.sheet ? `${outUrl(stem, "rig/preview/contact_sheet.png", mine.t)}` : "";

  return (
    <Card size="sm" className={cn(HERO, sheet && HERO_COLS)}>
      <h3 className={cn(H3, "col-span-full text-primary")}>{r.title}</h3>
      {sheet ? (
        <figure className="m-0 min-w-0">
          <a
            href={sheet}
            target="_blank"
            rel="noopener"
            title="Every clip, from the side, front and three-quarter."
            className="block rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <img alt="The rig's contact sheet" src={sheet} className="max-h-[420px] w-full rounded-lg bg-muted object-contain" />
          </a>
          <figcaption className="mt-1 font-mono text-[0.7rem] text-ink-3">contact sheet · every clip</figcaption>
        </figure>
      ) : null}
      <div className="grid min-w-0 content-start gap-1.5">
        <p className={FACTS}>
          <b>{r.fbx}</b>
          {r.texture ? (
            <>
              {" "}
              + <b>{r.texture}</b>
            </>
          ) : null}{" "}
          · {r.bones} bones · {r.clipCount} clips
        </p>
        <p className={FACTS}>{r.clips}</p>
        {line ? (
          <p className={cn(FACTS, line.fail && "font-medium text-danger-text")} title={line.title || undefined}>
            {line.text}
          </p>
        ) : null}
        <Steps>
          <li>
            Copy <code>3d/rig/</code> into your project's Assets folder.
          </li>
          <li>
            Right-click <b>{r.fbx}</b> → <b>Create Model</b>. Its clips come in as animations; in ModelDoc use <b>Add Simple Animations</b> if
            any are missing.
          </li>
          <li>Give it the same psx material as the static model. Each vertex has at most 4 bone weights, as s&amp;box requires.</li>
        </Steps>
      </div>
    </Card>
  );
}
