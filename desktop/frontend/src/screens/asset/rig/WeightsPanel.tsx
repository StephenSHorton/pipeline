// The Weights panel (Rig stage dock): how much of the mesh each bone moves, seen as a heat map and changed by hand
// with a Grab slider per bone and a brush (Add, Erase, Smooth). Edits go straight onto the model on screen, so any
// clip played meanwhile shows them; Save writes them into the rig (POST /rig/<stem>/weights). Layout only: the
// logic is ./weights.ts, the model side ./weightsView.ts, the store and Save ./weightsApi.ts.
import { useEffect, useMemo, useRef, useState } from "react";
import { Mesh, MeshBasicMaterial, SphereGeometry } from "three";
import { Button } from "@/components/ui/button";
import { flowEntry } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { TOGGLE } from "@/screens/viewer/classes";
import { PanelToggle, type PanelToggleProps } from "@/screens/viewer/panels/PanelToggle";
import { ViewerPanel } from "@/screens/viewer/panels/ViewerPanel";
import type { ViewerPanelProps } from "@/screens/viewer/types";
import { useViewerOverlay } from "@/viewer/ToolBridge";
import { useViewerTools, type ViewerTool } from "@/viewer/tools";
import { useRig } from "./api";
import { BRUSH_SIZE, BRUSH_STRENGTH, boneChains, boneRgb, changedText, GRAB, WEIGHTS_COPY as C, type BrushMode } from "./weights";
import { saveWeights, setWeightsSession, useWeights, weightsFor, weightsSession } from "./weightsApi";
import { WeightsSession } from "./weightsView";

const LABEL = "text-[length:calc(11px*var(--fs))] font-semibold tracking-[0.08em] text-ink-3 uppercase";
const MUTE = "m-0 text-[length:calc(12.5px*var(--fs))] leading-snug text-ink-3";
const SLIDER = "hit-44 min-w-0 flex-1 accent-primary";
const hex = (rgb: [number, number, number]) => `rgb(${rgb.map((x) => Math.round(x * 255)).join(",")})`;

export function WeightsToggle(p: Omit<PanelToggleProps, "panel"> & { props: ViewerPanelProps }) {
  if (!flowEntry(p.props.flow, "rig")?.live) return null;
  const { props: _props, ...rest } = p;
  return <PanelToggle panel="weights" title="How much of the mesh each bone moves: see it, and paint it" {...rest} />;
}

export function WeightsPanel(p: ViewerPanelProps) {
  const { stem, open, view, tools } = p;
  const rigged = !!flowEntry(p.flow, "rig")?.live;
  const model = useViewerTools((s) => s.model);
  const st = useWeights();
  const rigSt = useRig();
  const doc = rigSt.check && rigSt.check.stem === stem ? rigSt.check.doc : null;
  const riggedFile = (rigSt.check?.stem === stem && (rigSt.check.info?.rigged as string | undefined)) || "rigged.glb";
  const [s, setS] = useState<WeightsSession | null>(null);
  const sRef = useRef(s);
  sRef.current = s;

  useEffect(() => weightsFor(stem), [stem]);

  // The edit follows the model on screen: bound while open, and carried onto a reload of the same mesh even while
  // closed (an unsaved edit keeps showing in the clips). Not skinned (an LOD on screen): load the rigged model.
  const triedLoad = useRef("");
  useEffect(() => {
    if (!model || !rigged) return;
    const prev = weightsSession();
    if (prev && prev.stem === stem && prev.isOn(model.root)) {
      setS(prev);
      return;
    }
    if (!open && !(prev && prev.stem === stem)) return;
    if (!model.skinned) {
      const key = `${stem}|${riggedFile}`;
      if (open && triedLoad.current !== key) {
        triedLoad.current = key;
        tools.loadFile(riggedFile).catch((err: unknown) => toast(err instanceof Error ? err.message : String(err)));
      }
      return;
    }
    const next = WeightsSession.bind(model.root, stem, prev && prev.stem === stem ? prev : null);
    setWeightsSession(next);
    setS(next);
  }, [model, open, stem, rigged, riggedFile, tools]);

  // Every edit: the numbers follow, once a frame at most.
  useEffect(() => {
    if (!s) return;
    let raf = 0;
    s.onChange = () => {
      if (!raf) raf = requestAnimationFrame(() => {
        raf = 0;
        useWeights.setState((x) => ({ rev: x.rev + 1 }));
      });
    };
    return () => {
      cancelAnimationFrame(raf);
      s.onChange = () => {};
    };
  }, [s]);

  // The colours, while open.
  useEffect(() => {
    if (s && open) s.show({ mode: st.show, bone: st.bone });
  }, [s, open, st.show, st.bone]);
  useEffect(() => {
    if (!s || !open) return;
    let raf = requestAnimationFrame(function tick() {
      s.keep();
      raf = requestAnimationFrame(tick);
    });
    return () => {
      cancelAnimationFrame(raf);
      s.release();
    };
  }, [s, open]);

  // Opening stops Spin (painting a turning model is no fun); closing gives it back.
  const viewRef = useRef(view);
  viewRef.current = view;
  useEffect(() => {
    if (!open) return;
    const spin = viewRef.current.spin;
    if (spin) viewRef.current.onSpin(false);
    return () => {
      if (spin) viewRef.current.onSpin(true);
    };
  }, [open]);

  // Clay and Wire swap the model's materials: close, as the Rig check does.
  const looks = `${view.clay}|${view.wire}`;
  const seenLooks = useRef(looks);
  useEffect(() => {
    if (seenLooks.current === looks) return;
    seenLooks.current = looks;
    if (open) p.onClose();
  }, [looks, open, p]);

  /* ---------------------------------------------------------------- the brush */

  const ball = useMemo(() => {
    const m = new Mesh(new SphereGeometry(1, 20, 14), new MeshBasicMaterial({ color: 0xff6a55, transparent: true, opacity: 0.22, depthWrite: false }));
    m.visible = false;
    return m;
  }, []);
  useEffect(
    () => () => {
      ball.geometry.dispose();
      ball.material.dispose();
    },
    [ball],
  );
  const ballNode = useMemo(() => <primitive object={ball} />, [ball]);
  useViewerOverlay("weights-ball", open ? ballNode : null);

  const tool = useMemo<ViewerTool>(() => {
    let stroking = false;
    let last: { x: number; y: number; z: number } | null = null;
    const radius = () => useWeights.getState().size * tools.rootDiagonal();
    const dabAt = (point: { x: number; y: number; z: number; clone(): never }, force: boolean) => {
      const ss = sRef.current;
      const w = useWeights.getState();
      if (!ss || !w.brush || (w.brush !== "smooth" && w.bone < 0)) return;
      const r = radius();
      // Spaced dabs: a slow drag doesn't pile up strength where the pointer rests.
      if (!force && last && Math.hypot(point.x - last.x, point.y - last.y, point.z - last.z) < r * 0.3) return;
      last = { x: point.x, y: point.y, z: point.z };
      ss.dab(w.brush, w.bone, point as never, r, w.strength);
    };
    const finish = () => {
      if (stroking) sRef.current?.end();
      stroking = false;
      last = null;
    };
    return {
      cursor: "crosshair",
      onMove(e) {
        const hit = e.hit();
        if (!hit) ball.visible = false;
        else {
          ball.position.copy(hit.point);
          ball.scale.setScalar(radius());
          ball.visible = true;
        }
        if (hit && stroking && e.buttons & 1) dabAt(hit.point as never, false);
      },
      onDown(e) {
        const hit = e.hit();
        const ss = sRef.current;
        if (!hit || !ss) return;
        if (e.altKey) {
          const b = ss.boneAt(hit);
          if (b >= 0) useWeights.setState({ bone: b });
          return;
        }
        if (useWeights.getState().brush !== "smooth" && useWeights.getState().bone < 0) {
          toast("Pick a bone first: the brush paints the chosen bone.");
          return;
        }
        ss.begin();
        stroking = true;
        dabAt(hit.point as never, true);
      },
      onUp: finish,
      onLeave() {
        ball.visible = false;
      },
      onDisarm() {
        finish();
        ball.visible = false;
        if (useWeights.getState().brush) useWeights.setState({ brush: null });
      },
    };
  }, [tools, ball]);

  const armed = !!st.brush;
  useEffect(() => {
    if (!open || !armed || !s) return;
    return tools.armTool(tool);
  }, [open, armed, s, tool, tools]);
  useEffect(() => {
    if (!open && useWeights.getState().brush) useWeights.setState({ brush: null });
  }, [open]);

  // No brush: a click on the model (not a drag to turn it) picks the bone that moves that spot most.
  useEffect(() => {
    if (!open || !s || armed) return;
    let down: [number, number] | null = null;
    const onView = (e: Event) => {
      const canvas = tools.three()?.gl.domElement;
      const t = e.target as Node | null;
      return !!canvas && !!t && (t === canvas || (t as Element).contains?.(canvas));
    };
    const pd = (e: PointerEvent) => {
      down = onView(e) && e.button === 0 ? [e.clientX, e.clientY] : null;
    };
    const pu = (e: PointerEvent) => {
      const was = down;
      down = null;
      if (!was || Math.hypot(e.clientX - was[0], e.clientY - was[1]) > 4) return;
      const b = s.boneAt(tools.raycastAt(e.clientX, e.clientY));
      if (b >= 0) useWeights.setState({ bone: b });
    };
    document.addEventListener("pointerdown", pd);
    document.addEventListener("pointerup", pu);
    return () => {
      document.removeEventListener("pointerdown", pd);
      document.removeEventListener("pointerup", pu);
    };
  }, [open, s, armed, tools]);

  // Cmd/Ctrl+Z undoes, with Shift redoes, while the panel is open (not while typing).
  useEffect(() => {
    if (!open || !s) return;
    const key = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "z") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && (t as HTMLInputElement).type === "text") || t.isContentEditable)) return;
      e.preventDefault();
      if (e.shiftKey) s.redo();
      else s.undo();
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [open, s]);

  /* ---------------------------------------------------------------- what the panel shows */

  const labels = useMemo(() => Object.fromEntries((doc?.bones ?? []).map((b) => [b.name, String(b.label || "")]).filter(([, l]) => l)), [doc]);
  const roles = useMemo(() => Object.fromEntries((doc?.bones ?? []).map((b) => [b.name, String(b.role || "")]).filter(([, r]) => r)), [doc]);
  const chains = useMemo(() => (s ? boneChains(s.names, s.parents, { labels, roles, weighted: (i) => s.weighted(i) }) : []), [s, labels, roles]);
  const rev = st.rev;
  const grab = useMemo(() => (s ? s.grabAll() : []), [s, rev]);
  const changed = useMemo(() => (s && open ? s.changedPoints() : 0), [s, rev, open]);
  const boneName = (i: number) => {
    for (const c of chains) {
      const b = c.bones.find((x) => x.i === i);
      if (b) return `${b.name} (${c.label.toLowerCase()})`;
    }
    return s?.names[i] ?? "";
  };

  // The Grab slider: the bone's grab now, or where the drag is while it moves.
  const [drag, setDrag] = useState<number | null>(null);
  const dragging = useRef(false);
  const grabNow = st.bone >= 0 ? Math.round(grab[st.bone] ?? 100) : 100;
  const startGrab = () => {
    if (!s || st.bone < 0 || dragging.current) return;
    dragging.current = true;
    s.grabBegin(st.bone);
  };
  const endGrab = () => {
    if (!dragging.current) return;
    dragging.current = false;
    s?.end();
    setDrag(null);
  };

  const pickBrush = (m: BrushMode) => useWeights.setState({ brush: st.brush === m ? null : m });

  let body;
  if (!rigged) body = <p className={MUTE}>{C.noRig}</p>;
  else if (!s) body = <p className={MUTE}>{C.noSkin}</p>;
  else
    body = (
      <>
        <p className={MUTE}>{st.brush ? C.armedHint : C.hint}</p>
        <div className="flex flex-wrap items-center gap-0.5" role="group" aria-label="Show">
          <Toggle on={st.show === "heat"} title={C.heat.title} onClick={() => useWeights.setState({ show: "heat" })}>
            {C.heat.label}
          </Toggle>
          <Toggle on={st.show === "parts"} title={C.parts.title} onClick={() => useWeights.setState({ show: "parts" })}>
            {C.parts.label}
          </Toggle>
        </div>

        <div className={LABEL}>{C.bones}</div>
        <div className="-mx-1 max-h-[min(34vh,260px)] shrink-0 overflow-y-auto border-y border-hair px-1 py-1" role="listbox" aria-label="Bones">
          {chains.map((c) => (
            <div key={c.key} className="mb-1">
              <div className="pt-0.5 text-[length:calc(11.5px*var(--fs))] font-semibold text-ink-2">{c.label}</div>
              <div className="flex flex-wrap gap-0.5">
                {c.bones.map((b) => {
                  const g = Math.round(grab[b.i] ?? 100);
                  return (
                    <button
                      key={b.i}
                      type="button"
                      role="option"
                      aria-selected={st.bone === b.i}
                      title={`${b.name}: ${s.names[b.i]}${g !== 100 ? `, grab ${g}%` : ""}`}
                      onClick={() => useWeights.setState({ bone: st.bone === b.i ? -1 : b.i })}
                      className="hit-44 inline-flex h-[22px] items-center gap-1 rounded-[6px] border border-hair bg-white/70 px-1.5 text-[length:calc(11.5px*var(--fs))] text-ink hover:bg-muted aria-selected:border-accent aria-selected:bg-accent aria-selected:text-accent-foreground"
                    >
                      <span aria-hidden className="size-2 shrink-0 rounded-[2px]" style={{ background: hex(boneRgb(b.i)) }} />
                      {b.name}
                      {g !== 100 ? <span className="font-mono text-[10px] tabular-nums opacity-70">{g}%</span> : null}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {st.bone >= 0 ? (
          <label title={C.grab.title} className="flex items-center gap-2 text-[length:calc(12.5px*var(--fs))] font-semibold text-ink-3">
            <span className="min-w-0 truncate" title={boneName(st.bone)}>
              {C.grab.label}
            </span>
            <input
              type="range"
              min={GRAB.min}
              max={GRAB.max}
              step={GRAB.step}
              value={drag ?? Math.min(GRAB.max, grabNow)}
              aria-label={`${C.grab.label} for ${boneName(st.bone)}`}
              onPointerDown={startGrab}
              onPointerUp={endGrab}
              onPointerCancel={endGrab}
              onBlur={endGrab}
              onKeyUp={endGrab}
              onChange={(e) => {
                const v = Number(e.target.value);
                startGrab();
                setDrag(v);
                s.grabTo(st.bone, v);
              }}
              className={SLIDER}
            />
            <strong className="w-[3.4em] text-right font-mono text-ink tabular-nums">{drag ?? grabNow}%</strong>
          </label>
        ) : null}
        {st.bone >= 0 ? <p className={cn(MUTE, "-mt-1 truncate")}>{boneName(st.bone)}</p> : null}

        <div className={LABEL}>Brush</div>
        <div className="flex flex-wrap items-center gap-0.5" role="group" aria-label="Brush">
          {(["add", "erase", "smooth"] as const).map((m) => (
            <Toggle key={m} on={st.brush === m} title={C[m].title} onClick={() => pickBrush(m)}>
              {C[m].label}
            </Toggle>
          ))}
        </div>
        <Slider label={C.size.label} title={C.size.title} min={BRUSH_SIZE.min} max={BRUSH_SIZE.max} step={BRUSH_SIZE.step} value={st.size} text={`${Math.round(st.size * 100)}%`} onChange={(v) => useWeights.setState({ size: v })} />
        <Slider label={C.strength.label} title={C.strength.title} min={BRUSH_STRENGTH.min} max={BRUSH_STRENGTH.max} step={BRUSH_STRENGTH.step} value={st.strength} text={`${Math.round(st.strength * 100)}%`} onChange={(v) => useWeights.setState({ strength: v })} />

        <div className="flex flex-wrap items-center gap-0.5 border-t border-hair pt-2">
          <Act title={C.undo.title} off={!s.canUndo} onClick={() => s.undo()}>
            {C.undo.label}
          </Act>
          <Act title={C.redo.title} off={!s.canRedo} onClick={() => s.redo()}>
            {C.redo.label}
          </Act>
          <Act title={C.reset.title} off={!changed} onClick={() => s.reset()}>
            {C.reset.label}
          </Act>
          <Button
            type="button"
            variant={changed ? "default" : "outline"}
            aria-disabled={st.saving || !changed}
            title={C.save.title}
            onClick={() => {
              if (!st.saving && changed) void saveWeights(s);
            }}
            className="hit-44 ml-auto h-[29px] rounded-[8px] px-3 text-[length:calc(13px*var(--fs))] font-semibold aria-disabled:opacity-50"
          >
            {st.saving ? C.saving : C.save.label}
          </Button>
        </div>
        <p className={MUTE} aria-live="polite">
          {changedText(changed)}
        </p>
      </>
    );

  return (
    <ViewerPanel panel="weights" id={p.id} open={open} onClose={p.onClose} className="max-w-[340px]">
      {body}
    </ViewerPanel>
  );
}

function Toggle({ on, title, onClick, children }: { on: boolean; title: string; onClick: () => void; children: string }) {
  return (
    <Button type="button" variant="hud-ghost" aria-pressed={on} title={title} onClick={onClick} className={TOGGLE}>
      {children}
    </Button>
  );
}

function Act({ title, off, onClick, children }: { title: string; off: boolean; onClick: () => void; children: string }) {
  return (
    <Button type="button" variant="hud-ghost" aria-disabled={off} title={title} onClick={() => !off && onClick()} className={cn(TOGGLE, "aria-disabled:opacity-40")}>
      {children}
    </Button>
  );
}

function Slider(p: { label: string; title: string; min: number; max: number; step: number; value: number; text: string; onChange: (v: number) => void }) {
  return (
    <label title={p.title} className="flex items-center gap-2 text-[length:calc(12.5px*var(--fs))] font-semibold text-ink-3">
      <span className="w-[4.6em]">{p.label}</span>
      <input type="range" min={p.min} max={p.max} step={p.step} value={p.value} onChange={(e) => p.onChange(Number(e.target.value))} className={SLIDER} />
      <strong className="w-[3.4em] text-right font-mono text-ink tabular-nums">{p.text}</strong>
    </label>
  );
}
