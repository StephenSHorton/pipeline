// The Stack stage (legacy #stack-pane and its dock, index.html 2167-2236; the stack paint and palette host code,
// 5384-5642; stack-lab.js load, 1704-1740). The canvas fills the stage (the versions panel floats bottom left,
// AssetScreen); the stats card sits top left, the dock top right: one toggle per legacy cubby (Sprite stack, Stack,
// Palette, plus Paint, whose controls legacy kept in the hidden mesh pane), one open at a time, and the shared
// Game look's Pixel. The stack re-reads on every flow version; "Generating…" shows while a build runs on it.
import { X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useJobs } from "@/lib/jobs";
import { cn } from "@/lib/utils";
import { writePixelLook } from "@/viewer/pixelLook";
import { ACTION, COMMIT, PANEL, TOGGLE, TRAY, TRAY_ITEM } from "@/screens/viewer/classes";
import type { StageProps } from "../types";
import { saveStackPaint } from "./actions";
import { readStackMeta, sheetUrl } from "./api";
import { readGameLook, type GameLookState } from "./gameLook";
import {
  buildingStack,
  CUBBY_DEFAULT,
  CUBBY_LABELS,
  defaultsFor,
  defaultsFromSearch,
  GAME_TOGGLE,
  LOOK_BUTTONS,
  openCubby,
  PAINT_COPY,
  PALETTE_COPY,
  PBR_SLIDERS,
  pbrReadout,
  phaseText,
  radiusReadout,
  readout,
  resetState,
  rgbToHex,
  sliderMax,
  SLIDERS,
  sliderValue,
  STACK_COPY,
  stateForMeta,
  statsParts,
  TOGGLE_BUTTONS,
  TOGGLE_RESET,
  type PaletteEntry,
  type PbrKey,
  type SliderKey,
  type StackCubby,
  type StackMeta,
  type StackPhase,
  type StackState,
} from "./stack";
import { createStackView, loadImage, type PaintMode, type StackView } from "./view";

const CUBBIES: StackCubby[] = ["stack-look", "stack-sliders", "stack-palette", "paint"];
const cubbyId = (stem: string, c: StackCubby) => `stack-${c === "stack-look" ? "look" : c.replace("stack-", "")}-${stem}`;
const PAINT0: PaintMode = { on: false, drop: false, erase: false, color: PAINT_COPY.color.value, radius: PAINT_COPY.radius.value };
const PBR0: Record<PbrKey, number> = { metal: 0, rough: 0.7, emit: 0 };
/** A flow re-read this soon after our own Save is the one it queued (factoryJson's /paint/ hook, ~300 ms later):
 * the canvas already shows what was saved. Short, so a version pick right after a Save still reloads. */
const OWN_SAVE_MS = 1500;

function readShared(): GameLookState {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem("pipeline.pixelLook");
  } catch {
    /* blocked storage: the defaults */
  }
  return readGameLook(raw, location.search);
}

export function StackStage(props: StageProps) {
  // Keyed by stem: a new asset starts with a fresh canvas, dock and palette.
  return <StackBody key={props.stem} {...props} />;
}

function StackBody({ stem, flowVersion }: StageProps) {
  const base = useMemo(() => defaultsFromSearch(location.search), []);
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<StackView | null>(null);
  const [state, setState] = useState<StackState>(base);
  const stateRef = useRef(state);
  const [defaults, setDefaults] = useState<StackState>(base);
  const [meta, setMeta] = useState<StackMeta | null>(null);
  const [phase, setPhase] = useState<StackPhase>("slicing");
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState<StackCubby | null>(CUBBY_DEFAULT);
  const [paint, setPaint] = useState<PaintMode>(PAINT0);
  const [radiusMoved, setRadiusMoved] = useState(false);
  const [game, setGame] = useState<GameLookState>(readShared);
  const [palette, setPalette] = useState<PaletteEntry[] | null>(null);
  const [palSel, setPalSel] = useState(0);
  const [palCount, setPalCount] = useState<number>(PALETTE_COPY.count.value);
  const [pbr, setPbr] = useState(PBR0);
  const [saving, setSaving] = useState(false);
  const ownSave = useRef(0);
  const toggles = useRef<Partial<Record<StackCubby, HTMLButtonElement | null>>>({});
  const building = useJobs((s) => buildingStack(s.job, stem));

  // One canvas per mount; dispose frees its WebGL context (legacy never did, so every remount would leak one).
  useEffect(() => {
    const v = createStackView(host.current!, stateRef.current, readShared(), {
      onSample: (rgb) => setPaint((p) => ({ ...p, color: rgbToHex(rgb[0], rgb[1], rgb[2]), drop: false })),
    });
    view.current = v;
    return () => {
      v.dispose();
      view.current = null;
    };
  }, []);

  // stack.json + the sheet, again on every flow re-read (a pick, a finished job, Stack from picture). A running
  // build on this asset empties the canvas instead (legacy clearStage); when it ends, for any reason, this reloads.
  useEffect(() => {
    const v = view.current;
    if (!v) return;
    if (!building && ownSave.current && Date.now() - ownSave.current < OWN_SAVE_MS) {
      ownSave.current = 0;
      return;
    }
    let live = true;
    v.clear();
    setPalette(null);
    setPalSel(0);
    setPbr(PBR0);
    setFailed(false);
    if (building) {
      setPhase("generating");
      return;
    }
    setPhase("slicing");
    const t = Date.now();
    void (async () => {
      const m = await readStackMeta(stem, t);
      const img = m ? await loadImage(sheetUrl(stem, t)).catch(() => null) : null;
      if (!live) return;
      if (!m || !img) {
        setMeta(null);
        setFailed(true);
        setPhase("empty");
        return;
      }
      const d = defaultsFor(base, m);
      const next = stateForMeta(stateRef.current, d, m);
      stateRef.current = next;
      setState(next);
      setDefaults(d);
      setMeta(m);
      v.load(m, img, next);
      setPhase("ready");
    })();
    return () => {
      live = false;
    };
  }, [stem, flowVersion, building, base]);

  useEffect(() => view.current?.setPaint(paint), [paint]);

  const update = useCallback((patch: Partial<StackState>) => {
    const next = { ...stateRef.current, ...patch };
    stateRef.current = next;
    setState(next);
    view.current?.set(patch);
  }, []);

  const toggleGame = () => {
    const next = { ...game, on: !game.on };
    setGame(next);
    view.current?.setGame({ on: next.on });
    writePixelLook({ on: next.on, pixelSize: next.pixelSize, lines: next.lines });
  };

  const reduce = () => {
    const pal = view.current?.quantize(palCount) ?? [];
    setPalette(pal.length ? [...pal] : null);
    setPalSel(0);
    if (pal[0]) {
      setPaint((p) => ({ ...p, color: rgbToHex(pal[0].r, pal[0].g, pal[0].b) }));
      // Legacy left the sliders on the last swatch's values; swatch 0 is the selected one now.
      setPbr({ metal: pal[0].metal, rough: pal[0].rough, emit: pal[0].emit });
    }
  };

  const pickSwatch = (i: number) => {
    const e = palette?.[i];
    if (!e) return;
    view.current?.selectPalette(i);
    setPalSel(i);
    setPaint((p) => ({ ...p, color: rgbToHex(e.r, e.g, e.b) }));
    setPbr({ metal: e.metal, rough: e.rough, emit: e.emit });
  };

  const setPbrValue = (key: PbrKey, v: number) => {
    setPbr((p) => ({ ...p, [key]: v }));
    view.current?.setPaletteEntry(palSel, { [key]: v });
  };

  const save = async () => {
    const v = view.current;
    if (!v) return;
    setSaving(true);
    const ok = await saveStackPaint(stem, () => v.getSheetBlob());
    if (ok) ownSave.current = Date.now();
    setSaving(false);
  };

  // Closing a panel while focus is in it hands focus back to its dock toggle, not <body>.
  const closeCubby = (c: StackCubby) => {
    const inside = !!document.activeElement?.closest(`#${CSS.escape(cubbyId(stem, c))}`);
    setOpen((o) => (o === c ? null : o));
    if (inside) toggles.current[c]?.focus();
  };

  const ready = phase === "ready";
  const empty = phaseText(phase);

  return (
    <section aria-label={STACK_COPY.pane} className="absolute inset-0 overflow-hidden bg-[#0c0d0b]">
      <div ref={host} className={cn("absolute inset-0", paint.on && "cursor-crosshair")} />
      <div className="pointer-events-none absolute inset-0">
        <Card size="hud" className={cn(PANEL, "absolute top-3 left-3 max-w-[min(420px,calc(100%-390px))] gap-1 px-4 pt-2.5 pb-3 max-md:right-3 max-md:max-w-none")}>
          <h2 className="m-0 text-[0.95rem] font-semibold text-ink">{STACK_COPY.pane}</h2>
          <p className="m-0 text-[0.8rem] leading-snug text-ink-3 empty:hidden">
            {ready && meta ? statsParts(state, meta).map((p, i) => (p.b ? <b key={i} className="font-semibold text-ink">{p.text}</b> : <span key={i}>{p.text}</span>)) : failed ? STACK_COPY.noStack : null}
          </p>
        </Card>

        {empty ? (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <Card size="hud" role="status" aria-live="polite" className={cn(PANEL, "max-w-sm px-4 py-2.5 text-center text-ink-3")}>
              {empty}
            </Card>
          </div>
        ) : null}

        {/* The dock sizes to its buttons (one row); only the panels under it are capped at 342 px. Below md the
            column moves to the bottom, clear of the title card. */}
        <div className="absolute top-3 right-3 flex max-h-[calc(100%-24px)] max-w-[calc(100%-24px)] flex-col items-end gap-2 max-md:top-auto max-md:bottom-3 max-md:max-h-[60%]">
          <Card size="hud" role="group" aria-label="Dock" className={cn(PANEL, "w-max max-w-full flex-row flex-wrap items-center justify-end gap-1 p-1")}>
            {ready
              ? CUBBIES.map((c) => (
                  <Button
                    key={c}
                    ref={(el) => {
                      toggles.current[c] = el;
                    }}
                    type="button"
                    variant="hud-ghost"
                    aria-expanded={open === c}
                    aria-controls={cubbyId(stem, c)}
                    onClick={() => setOpen((o) => openCubby(o, c))}
                    className={TOGGLE}
                  >
                    {CUBBY_LABELS[c]}
                  </Button>
                ))
              : null}
            {ready ? <Separator orientation="vertical" className="mx-0.5 h-5 w-px bg-hair" /> : null}
            <Button type="button" variant="hud-ghost" aria-pressed={game.on} title={GAME_TOGGLE.title} onClick={toggleGame} className={TOGGLE}>
              {GAME_TOGGLE.label}
            </Button>
          </Card>

          {/* The dock is hidden while the pane is empty (legacy setEmpty: sliders.root.hidden). */}
          {ready && meta ? (
            <div className="flex min-h-0 w-[min(342px,calc(100vw-24px))] flex-col overflow-y-auto">
              <Cubby id={cubbyId(stem, "stack-look")} cubby="stack-look" open={open === "stack-look"} onClose={() => closeCubby("stack-look")}>
                <p className="m-0 text-[0.8rem] leading-snug text-ink-3">{STACK_COPY.lookBlurb}</p>
                <div role="group" aria-label="Look" className={cn(TRAY, "w-fit")}>
                  {LOOK_BUTTONS.map((b) => (
                    <Button key={b.look} type="button" variant="hud-ghost" aria-pressed={state.look === b.look} title={b.title} onClick={() => state.look !== b.look && update({ look: b.look })} className={TRAY_ITEM}>
                      {b.label}
                    </Button>
                  ))}
                </div>
                <Group label="Stack">
                  {TOGGLE_BUTTONS.map((b) => (
                    <Toggle key={b.key} on={state[b.key]} title={b.title} onClick={() => update({ [b.key]: !state[b.key] })}>
                      {b.label}
                    </Toggle>
                  ))}
                </Group>
                {/* Reset is a one-shot, so it is a button on its own row, not a fifth switch. */}
                <Button type="button" variant="outline" title={TOGGLE_RESET.title} onClick={() => update(resetState(stateRef.current, defaults, meta))} className={cn(ACTION, "self-start")}>
                  {TOGGLE_RESET.label}
                </Button>
              </Cubby>

              <Cubby id={cubbyId(stem, "stack-sliders")} cubby="stack-sliders" open={open === "stack-sliders"} onClose={() => closeCubby("stack-sliders")}>
                {/* One column of 32 px rows: the panel's own gap between nine sliders made it ~450 px tall. */}
                <div className="flex flex-col">
                  {SLIDERS.map((s) => (
                    <Slider
                      key={s.key}
                      label={s.label}
                      title={s.title}
                      min={s.min}
                      max={sliderMax(s, meta)}
                      step={s.step}
                      value={state[s.key]}
                      text={readout(s, state[s.key])}
                      onChange={(v) => update({ [s.key]: sliderValue(s.key as SliderKey, v) })}
                    />
                  ))}
                </div>
              </Cubby>

              <Cubby id={cubbyId(stem, "stack-palette")} cubby="stack-palette" open={open === "stack-palette"} onClose={() => closeCubby("stack-palette")}>
                <p className="m-0 text-[0.8rem] leading-snug text-ink-3">{STACK_COPY.paletteHint}</p>
                <Slider
                  label={PALETTE_COPY.count.label}
                  title={PALETTE_COPY.count.title}
                  min={PALETTE_COPY.count.min}
                  max={PALETTE_COPY.count.max}
                  step={PALETTE_COPY.count.step}
                  value={palCount}
                  text={String(palCount)}
                  onChange={setPalCount}
                />
                <Button type="button" variant="outline" title={PALETTE_COPY.reduce.title} onClick={reduce} className={cn(ACTION, "self-start")}>
                  {PALETTE_COPY.reduce.label}
                </Button>
                {palette ? (
                  <div role="group" aria-label={PALETTE_COPY.label} className="flex flex-wrap gap-1.5">
                    {palette.map((e, i) => (
                      <button
                        key={i}
                        type="button"
                        aria-pressed={i === palSel}
                        title={PALETTE_COPY.swatchTitle(i)}
                        aria-label={PALETTE_COPY.swatchTitle(i)}
                        onClick={() => pickSwatch(i)}
                        style={{ background: `rgb(${e.r},${e.g},${e.b})` }}
                        className="hit-44 size-7 rounded-[8px] border border-black/15 outline-none focus-visible:ring-3 focus-visible:ring-ring/50 aria-pressed:ring-2 aria-pressed:ring-primary aria-pressed:ring-offset-2"
                      />
                    ))}
                  </div>
                ) : null}
                {PBR_SLIDERS.map((s) => (
                  <Slider key={s.key} label={s.label} title={s.title} min={s.min} max={s.max} step={s.step} value={pbr[s.key]} text={pbrReadout(pbr[s.key])} onChange={(v) => setPbrValue(s.key, v)} />
                ))}
              </Cubby>

              <Cubby id={cubbyId(stem, "paint")} cubby="paint" open={open === "paint"} onClose={() => closeCubby("paint")}>
                <p className="m-0 text-[0.8rem] leading-snug text-ink-3">{STACK_COPY.paintHint}</p>
                <Group label={CUBBY_LABELS.paint}>
                  <Toggle on={paint.on} title={PAINT_COPY.paint.title} onClick={() => setPaint((p) => ({ ...p, on: !p.on }))}>
                    {PAINT_COPY.paint.label}
                  </Toggle>
                  <Toggle on={paint.drop} title={PAINT_COPY.drop.title} onClick={() => setPaint((p) => ({ ...p, drop: !p.drop }))}>
                    {PAINT_COPY.drop.label}
                  </Toggle>
                  <Toggle on={paint.erase} title={PAINT_COPY.erase.title} onClick={() => setPaint((p) => ({ ...p, erase: !p.erase }))}>
                    {PAINT_COPY.erase.label}
                  </Toggle>
                </Group>
                <div className="flex flex-wrap gap-1">
                  <Button type="button" variant="outline" title={PAINT_COPY.undo.title} onClick={() => view.current?.undoStroke()} className={ACTION}>
                    {PAINT_COPY.undo.label}
                  </Button>
                  {/* aria-disabled, not disabled: the button just pressed keeps keyboard focus while the save runs. */}
                  <Button
                    type="button"
                    variant="default"
                    title={PAINT_COPY.save.title}
                    aria-disabled={saving}
                    onClick={() => !saving && void save()}
                    className={cn(COMMIT, "aria-disabled:opacity-50")}
                  >
                    {PAINT_COPY.save.label}
                  </Button>
                </div>
                <label className="flex min-h-8 items-center gap-3">
                  <span className="w-16 shrink-0 text-[0.78rem] font-semibold text-ink-3">{PAINT_COPY.color.label}</span>
                  <input
                    type="color"
                    value={paint.color}
                    title={PAINT_COPY.color.title}
                    onChange={(e) => setPaint((p) => ({ ...p, color: e.target.value }))}
                    className="hit-44 size-[29px] cursor-pointer rounded-[8px] border border-input bg-white p-0.5"
                  />
                </label>
                <Slider
                  label={PAINT_COPY.radius.label}
                  title={PAINT_COPY.radius.title}
                  min={PAINT_COPY.radius.min}
                  max={PAINT_COPY.radius.max}
                  step={PAINT_COPY.radius.step}
                  value={paint.radius}
                  text={radiusReadout(paint.radius, radiusMoved)}
                  onChange={(v) => {
                    setRadiusMoved(true);
                    setPaint((p) => ({ ...p, radius: v }));
                  }}
                />
              </Cubby>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

/** One legacy cubby as a dock panel: its bar label as the title, a ×, hidden while closed. */
function Cubby({ id, cubby, open, onClose, children }: { id: string; cubby: StackCubby; open: boolean; onClose: () => void; children: ReactNode }) {
  const title = CUBBY_LABELS[cubby];
  return (
    <Card
      id={id}
      size="hud"
      role="region"
      aria-label={title}
      hidden={!open}
      className={cn(PANEL, "w-full shrink-0 gap-2.5 px-4 pt-3 pb-3.5 [&[hidden]]:hidden")}
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="m-0 text-[0.95rem] font-semibold text-ink">{title}</h2>
        <Button type="button" variant="ghost" size="icon-sm" aria-label={`Close ${title}`} title={`Close ${title}`} onClick={onClose} className="hit-44 text-ink-3">
          <X />
        </Button>
      </div>
      {children}
    </Card>
  );
}

function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap items-center gap-1">
      {children}
    </div>
  );
}

function Toggle({ on, title, onClick, children }: { on: boolean; title: string; onClick: () => void; children: ReactNode }) {
  return (
    <Button type="button" variant="hud-ghost" aria-pressed={on} title={title} onClick={onClick} className={TOGGLE}>
      {children}
    </Button>
  );
}

function Slider(p: { label: string; title: string; min: number; max: number; step: number; value: number; text: string; onChange: (v: number) => void }) {
  return (
    <label title={p.title} className="grid min-h-8 grid-cols-[4.75rem_minmax(0,1fr)_3rem] items-center gap-2">
      <span className="text-[0.78rem] font-semibold text-ink-3">{p.label}</span>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.value}
        onChange={(e) => p.onChange(Number(e.target.value))}
        className="hit-44 h-6 w-full cursor-pointer accent-primary"
      />
      <strong className="text-right text-[0.8rem] font-semibold text-ink tabular-nums">{p.text}</strong>
    </label>
  );
}
