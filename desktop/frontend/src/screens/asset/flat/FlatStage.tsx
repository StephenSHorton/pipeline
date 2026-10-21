// Concept and the 2D sheet (legacy #flat-stage, index.html 2157-2166, and the 2D half of the workspace bar,
// 2247-2265). The picture fills the middle on the clay ground; the status line and the result player sit above
// it; Pieces shows the parts strip; the bar at the bottom picks the mode (Picture / Motion / Pieces / Say / Sound),
// opens the Audio panel, and holds the revise form. Concept shows the concept still; 2D shows the motion strip,
// the source or the still (showStage).
import { AudioLines, Film } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { flatUrl, rootUrl } from "@/lib/api";
import { conceptStill, type FlatMode } from "@/lib/flow";
import { displayStem, onJobEnded, onJobFinished, useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { useStage } from "../stage";
import type { StageProps } from "../types";
import { flatSpecs, panelAudio, redraw, reviseStep } from "./actions";
import { getFlatFlags, getSheet } from "./api";
import { AudioPanel, PANEL, type Bound } from "./AudioPanel";
import {
  audioFields,
  audioPlayers,
  audioWho,
  CLIP_ACCEPT,
  clipLabel,
  emptyAudioAsset,
  finishFlat,
  FLAT_MODE_LIST,
  FLAT_MODES,
  mergeSheet,
  MODE_LABEL,
  partLabel,
  rereadSheet,
  savedKey,
  soundState,
  speakState,
  stagePicture,
  type AudioFields,
  type AudioRow,
  type FlatFlags,
  type SheetState,
} from "./flat";

const TOGGLE =
  "hit-44 h-[29px] rounded-[8px] px-[11px] text-[length:calc(14.2px*var(--fs))] font-semibold text-ink-3 hover:bg-muted aria-pressed:bg-accent aria-pressed:text-accent-foreground aria-expanded:bg-accent aria-expanded:text-accent-foreground";
const NONE: Bound = { rel: "", t: 0 };
const bind = (rel: string): Bound => ({ rel, t: rel ? Date.now() : 0 });

export function FlatStage({ stem, stage, flow, flowVersion, enter }: StageProps) {
  const flatMode = useStage((s) => s.flatMode);
  const setFlatMode = useStage((s) => s.setFlatMode);
  const job = useJobs((s) => s.job);

  const [sheet, setSheet] = useState<SheetState>({ parts: [], index: -1, center: "" });
  const [audioAsset, setAudioAsset] = useState<AudioRow>(emptyAudioAsset);
  const assetHear = useRef({ sound: "", voice: "" });
  // Bumped after a revise answer, so the same sheets/<x>.png path re-reads.
  const [nonce, setNonce] = useState(0);
  const [status, setStatus] = useState("");
  const [hear, setHear] = useState<Bound>(NONE);
  const [flags, setFlags] = useState<FlatFlags | null>(null);
  const [audioOpen, setAudioOpen] = useState(false);
  const [fields, setFields] = useState<AudioFields>(() => audioFields(null));
  const [players, setPlayers] = useState({ voice: NONE, sfx: NONE });
  const [audioStatus, setAudioStatus] = useState("");
  const [prompt, setPrompt] = useState("");
  const [clip, setClip] = useState<File[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [generating, setGenerating] = useState(false);

  const subject = sheet.index >= 0 ? sheet.parts[sheet.index] ?? null : null;

  // The stem, piece and stored audio the panel was last filled from.
  const filled = useRef<{ stem: string; index: number; saved: string } | null>(null);

  /** Fill the panel from a subject (legacy fillAudio). */
  const fill = useCallback((p: SheetState["parts"][number] | null, row: AudioRow, at?: { stem: string; index: number }) => {
    if (at) filled.current = { ...at, saved: savedKey(p || row) };
    setFields(audioFields(p));
    const rel = audioPlayers(p, row, assetHear.current);
    setPlayers({ voice: bind(rel.voice), sfx: bind(rel.sound) });
  }, []);

  // The sheet (GET /2d/<stem>), again after every flow re-read (a pick, a 2D write, a finished job). The selected
  // piece is kept. The re-read after this stage's own POST (keepNext) keeps what was typed and heard here; any other
  // takes the sheet as it comes. The panel refills only when its subject, or what is saved for it, changed, so a
  // plain refresh leaves the typed fields and a playing player alone.
  const sheetRef = useRef(sheet);
  sheetRef.current = sheet;
  const rowRef = useRef(audioAsset);
  rowRef.current = audioAsset;
  const fieldsRef = useRef(fields);
  fieldsRef.current = fields;
  const keepNext = useRef(false);
  useEffect(() => {
    let live = true;
    void getSheet(stem).then((doc) => {
      if (!live) return;
      const keep = keepNext.current;
      keepNext.current = false;
      const was = filled.current;
      const same = was?.stem === stem;
      const local = same ? { sheet: sheetRef.current, row: rowRef.current, fields: fieldsRef.current } : null;
      const r = rereadSheet(mergeSheet(doc), local, keep);
      if (!keep) assetHear.current = { sound: "", voice: "" };
      setAudioAsset(r.row);
      setSheet(r.sheet);
      const moved = !same || was.index !== r.sheet.index;
      if (moved || (!keep && was.saved !== r.saved)) fill(r.sheet.index >= 0 ? r.sheet.parts[r.sheet.index] : null, r.row);
      filled.current = { stem, index: r.sheet.index, saved: r.saved };
    });
    return () => {
      live = false;
    };
  }, [stem, flowVersion, fill]);

  // Speak and Sound follow GET /2d/status, read at once and after every job (legacy refreshAudioFlags on jobEnded).
  useEffect(() => {
    void getFlatFlags().then(setFlags);
    return onJobEnded((j) => {
      setDrawing(false);
      // A redraw that would not start, or failed, says why on the 2D status line too (not only the top-bar chip).
      if (j.state === "error" && j.spec && flatSpecs.has(j.spec)) {
        keepNext.current = false;
        setStatus(j.error || "Failed.");
      }
      void getFlatFlags().then(setFlags);
    });
  }, []);

  // The prompt that started this asset's job goes into an empty revise field (legacy copied the Home prompt).
  useEffect(() => {
    const j = useJobs.getState().job;
    const p = j && j.stem === stem ? String((j.spec as { prompt?: string } | undefined)?.prompt || j.prompt || "") : "";
    if (p) setPrompt((cur) => cur || p);
  }, [stem]);

  // A 2D draw on this asset finished (legacy finishFlatJob apply). The asset screen re-reads the flow, which
  // re-reads the sheet; a redraw started here also says so in the toast.
  useEffect(
    () =>
      onJobFinished((j) => {
        if (j.kind !== "flat2d" || j.stem !== stem) return;
        const f = finishFlat(displayStem(stem), j.result);
        setSheet((s) => ({ ...s, center: f.center }));
        if (f.pieces) setFlatMode("pieces");
        if (f.hear) setHear(bind(f.hear));
        setStatus(f.status);
        if (j.spec && flatSpecs.has(j.spec)) {
          // You are on the asset already: no "<name> is ready · Open" chip (legacy finishFlatJob job = null).
          useJobs.getState().dismiss();
          toast(f.toast, "Export →", () => enter("export"));
        }
      }),
    [stem, enter, setFlatMode],
  );

  // Closing the panel while focus is in it hands focus back to the Audio button, not <body>.
  const toggleRef = useRef<HTMLButtonElement>(null);
  const closeAudio = useCallback(() => {
    const inside = !!document.activeElement?.closest(`#${CSS.escape(`flat-audio-${stem}`)}`);
    setAudioOpen(false);
    if (inside) toggleRef.current?.focus();
  }, [stem]);

  // A mode button collapses every panel (legacy setFlatMode).
  useEffect(() => closeAudio(), [flatMode, closeAudio]);

  /** A piece was clicked: the panel's fields go back to the one it leaves (legacy readAudio), then it fills. */
  const selectPart = (i: number) => {
    const parts = sheet.parts.map((p, k) => (k === sheet.index ? { ...p, line: fields.line, projectile: fields.effect, speaker: fields.speaker } : p));
    setSheet({ ...sheet, parts, index: i });
    fill(parts[i] ?? null, audioAsset, { stem, index: i });
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const text = prompt.trim();
    if (flatMode === "picture") {
      const r = await redraw(stem, text);
      if (r.status) setStatus(r.status);
      else if (r.started) {
        // The re-read after POST /2d/draw changes no audio: keep the panel as typed.
        keepNext.current = true;
        setStatus("");
        setDrawing(true);
      }
      return;
    }
    setGenerating(true);
    const r = await reviseStep(stem, flatMode, text, sheet, audioAsset, clip, setStatus);
    setGenerating(false);
    setStatus(r.status);
    if (!r.ok) return;
    keepNext.current = true;
    setSheet({ parts: r.parts, index: r.index, center: r.center });
    setNonce((n) => n + 1);
    if (r.hear) {
      fill(r.index >= 0 ? r.parts[r.index] : null, audioAsset, { stem, index: r.index });
      setHear(bind(r.hear));
    }
  };

  const onAudio = async (step: "say" | "sound") => {
    // The asset row takes the panel's fields (legacy), then the step writes it.
    const row = { ...audioAsset, line: fields.line, projectile: fields.effect, speaker: fields.speaker };
    setAudioAsset(row);
    const r = await panelAudio(stem, step, step === "say" ? fields.line : fields.effect, fields.speaker, setAudioStatus);
    setAudioStatus(r.status);
    if (!r.rel) return;
    keepNext.current = true;
    if (step === "say") {
      assetHear.current.voice = r.rel;
      setAudioAsset((a) => ({ ...a, voice: r.rel }));
      setPlayers((p) => ({ ...p, voice: bind(r.rel) }));
    } else {
      assetHear.current.sound = r.rel;
      setAudioAsset((a) => ({ ...a, sound: r.rel }));
      setPlayers((p) => ({ ...p, sfx: bind(r.rel) }));
    }
  };

  // A redraw started here holds the button until its job ends (also after coming back to this stage).
  const redrawing = drawing || (job?.state === "running" && !!job.spec && flatSpecs.has(job.spec));
  const busyLabel = redrawing ? "Drawing…" : generating ? "Generating…" : "";
  const still = conceptStill(flow, stem);
  const pic = stagePicture(stage === "concept" ? "concept" : "flat", sheet.center, still);
  const t = `${flowVersion}-${nonce}`;
  const showParts = flatMode === "pieces" && sheet.parts.length > 0;
  const panelId = `flat-audio-${stem}`;

  return (
    <section aria-label={stage === "concept" ? "Concept" : "2D sheet"} className="absolute inset-0 flex flex-col">
      <div className="relative min-h-0 flex-1">
        <div className="absolute inset-0 flex items-center justify-center px-3 pt-16 pb-3 sm:px-8">
          {pic.kind === "empty" ? (
            <Card size="hud" className={cn(PANEL, "px-4 py-2.5 text-ink-3")}>
              No picture yet.
            </Card>
          ) : (
            <img
              key={pic.kind}
              alt={stage === "concept" ? "The concept picture" : "The 2D sheet"}
              src={pic.kind === "strip" ? flatUrl(stem, pic.rel, t) : pic.kind === "source" ? flatUrl(stem, "source", t) : `${rootUrl(pic.rel)}?t=${flowVersion}`}
              className="max-h-full max-w-full rounded-[14px] bg-white object-contain shadow-[var(--glass-shadow)]"
            />
          )}
        </div>
        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
          {/* #flat-status: every flatSay line, not clickable. */}
          <p role="status" aria-live="polite" className={cn("m-0 min-w-0 text-[0.85rem] leading-snug text-ink-3", status && "glass-panel rounded-[12px] px-3 py-2")}>
            {status}
          </p>
          <FlatHear stem={stem} bound={hear} />
        </div>
        {audioOpen ? (
          <AudioPanel
            id={panelId}
            stem={stem}
            who={audioWho("2d", subject)}
            fields={fields}
            onFields={setFields}
            speak={speakState(flags)}
            sound={soundState(flags)}
            onSpeak={() => void onAudio("say")}
            onSound={() => void onAudio("sound")}
            status={audioStatus}
            voice={players.voice}
            sfx={players.sfx}
            onClose={closeAudio}
          />
        ) : null}
      </div>

      {showParts ? <PartsStrip stem={stem} parts={sheet.parts} index={sheet.index} t={t} onPick={selectPart} /> : null}

      <div className="flex justify-center px-3 pb-3">
        <Card size="hud" className={cn(PANEL, "w-full max-w-[56rem] flex-row flex-wrap items-center gap-1.5 p-1.5")}>
          <div role="group" aria-label="2D mode" className="flex flex-wrap items-center gap-0.5">
            {FLAT_MODE_LIST.map((m) => (
              <ModeButton key={m} mode={m} on={flatMode === m} onPick={setFlatMode} />
            ))}
            <span aria-hidden="true" className="mx-0.5 h-5 w-px bg-hair" />
            <Button
              ref={toggleRef}
              type="button"
              variant="hud-ghost"
              aria-expanded={audioOpen}
              aria-controls={audioOpen ? panelId : undefined}
              title="What it says and the sound it makes."
              onClick={() => setAudioOpen((o) => !o)}
              aria-label="Audio"
              className={cn(TOGGLE, "gap-1 max-sm:px-2")}
            >
              <AudioLines size={15} strokeWidth={2.2} aria-hidden="true" />
              {/* Phones: the icon only, so the six buttons fit one row. */}
              <span className="max-sm:sr-only">Audio</span>
            </Button>
          </div>
          <form onSubmit={(e) => void onSubmit(e)} autoComplete="off" className="flex min-w-[min(100%,18rem)] flex-1 items-center gap-1.5">
            <label htmlFor="flat-revise-prompt" className="sr-only">
              {MODE_LABEL[flatMode]}
            </label>
            <Input
              id="flat-revise-prompt"
              type="text"
              spellCheck={false}
              placeholder={FLAT_MODES[flatMode]}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="h-11 min-w-0 flex-1 rounded-[10px] border-input bg-white px-3 text-[0.9rem] md:text-[0.9rem]"
            />
            {flatMode === "motion" ? (
              <label
                title="A video or frames the motion follows."
                className="hit-44 inline-flex h-11 max-w-[10rem] shrink-0 cursor-pointer items-center gap-1.5 rounded-[10px] border border-input bg-white px-3 text-[0.85rem] font-medium text-ink-2 focus-within:ring-3 focus-within:ring-ring/50 hover:text-ink"
              >
                <Film size={15} strokeWidth={2.1} aria-hidden="true" className="shrink-0" />
                <span className="truncate">{clipLabel(clip)}</span>
                <input
                  type="file"
                  multiple
                  accept={CLIP_ACCEPT}
                  className="sr-only"
                  onChange={(e) => setClip(Array.from(e.target.files || []))}
                />
              </label>
            ) : null}
            <Button type="submit" disabled={!!busyLabel} className="h-11 shrink-0 rounded-[10px] px-4 font-semibold">
              {busyLabel || "Generate"}
            </Button>
          </form>
        </Card>
      </div>
    </section>
  );
}

function ModeButton({ mode, on, onPick }: { mode: FlatMode; on: boolean; onPick: (m: FlatMode) => void }) {
  return (
    <Button type="button" variant="hud-ghost" aria-pressed={on} onClick={() => onPick(mode)} className={cn(TOGGLE, "max-sm:px-2")}>
      {MODE_LABEL[mode]}
    </Button>
  );
}

/** #flat-hear: the result of the last step or draw, shown and played at once. */
function FlatHear({ stem, bound }: { stem: string; bound: Bound }) {
  const ref = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    if (bound.rel) ref.current?.play().catch(() => {});
  }, [bound]);
  if (!bound.rel) return null;
  return (
    <Card size="hud" className={cn(PANEL, "shrink-0 p-1.5")}>
      <audio ref={ref} aria-label="The result" controls src={flatUrl(stem, bound.rel, bound.t)} className="h-10 w-[min(300px,50vw)]" />
    </Card>
  );
}

/** The parts strip (Pieces mode): one card per piece; the pressed one is the Audio panel's subject. */
function PartsStrip({ stem, parts, index, t, onPick }: { stem: string; parts: SheetState["parts"]; index: number; t: string; onPick: (i: number) => void }) {
  return (
    <div className="overflow-x-auto px-3 pt-1 pb-2 [scrollbar-width:thin]">
      <div role="group" aria-label="Pieces" className="mx-auto flex w-max gap-2 p-0.5">
        {parts.map((p, i) => (
          <button
            key={p.id || i}
            type="button"
            aria-pressed={i === index}
            onClick={() => onPick(i)}
            className={cn(
              "glass-panel flex w-[120px] shrink-0 flex-col gap-1 rounded-[12px] p-1.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              "aria-pressed:ring-2 aria-pressed:ring-primary",
            )}
          >
            <img src={flatUrl(stem, p.file, t)} alt={partLabel(p, i)} className="h-[72px] w-full rounded-[8px] bg-white object-contain" />
            <span className={cn("truncate px-0.5 text-[0.75rem] text-ink-2", i === index && "font-semibold text-ink")}>{partLabel(p, i)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
