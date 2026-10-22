// The Paint panel (legacy paint cubby, index.html 2106-2128, 5379-5562): the 3D brush on the mesh's texture and
// Save through POST /paint/stack and /paint/mesh. Layout only: the copy and maths are in ./paint/paint.ts, the
// brush in ./paint/brush.ts and ./paint/actions.ts. The brush stays armed while the panel is closed, as legacy
// kept it with the cubby collapsed.
import { useEffect, useMemo, useState } from "react";
import { Mesh, MeshBasicMaterial, SphereGeometry, type Vector3 } from "three";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useViewerOverlay } from "@/viewer/ToolBridge";
import { useViewerTools } from "@/viewer/tools";
import { TOGGLE } from "../classes";
import type { ViewerPanelProps } from "../types";
import { paintTool, saveDeps, savePaint, setBrushRadius, setPaintMode, toggleDrop, toggleErase, undoPaint, type BrushBall, type PaintEnv } from "./paint/actions";
import { MeshBrush } from "./paint/brush";
import { BALL, BRUSH, PAINT_COPY, PAINT_HINT, PAINTED_LOD } from "./paint/paint";
import { usePaint } from "./paint/store";
import { ViewerPanel } from "./ViewerPanel";

export function PaintPanel(p: ViewerPanelProps) {
  const { tools } = p;
  const paint = usePaint();
  const [brush] = useState(() => new MeshBrush());
  const ball = useMemo(() => {
    const m = new Mesh(
      new SphereGeometry(1, BALL.widthSegments, BALL.heightSegments),
      new MeshBasicMaterial({ color: BALL.color, transparent: true, opacity: BALL.opacity, depthWrite: false }),
    );
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
  const overlay = useMemo(() => <primitive object={ball} />, [ball]);
  useViewerOverlay("paint-ball", overlay);

  const env = useMemo<PaintEnv>(() => {
    const handle: BrushBall = {
      show(point: Vector3, radius: number) {
        ball.position.copy(point);
        ball.scale.setScalar(radius);
        ball.visible = true;
      },
      hide() {
        ball.visible = false;
      },
    };
    return { tools, mesh: brush, ball: handle };
  }, [tools, brush, ball]);
  const tool = useMemo(() => paintTool(env), [env]);

  // A new model (a LOD, a reload, a version pick) takes a fresh texture on the next stroke; the plumbing disarms.
  const model = useViewerTools((s) => s.model);
  useEffect(() => () => brush.forget(), [model, brush]);

  const save = async () => {
    const done = await savePaint(p.stem, saveDeps(brush));
    // The flow refresh after a /paint/ POST reloads the viewer, which would put the unpainted file back on screen:
    // show what Save wrote instead (paint_save.py lists it first in lods.json as "Painted"). The Rig stage keeps
    // its rigged model, which Save does not change.
    if (done.mesh && p.stage === "mesh") p.onPickLod(PAINTED_LOD);
  };

  return (
    <ViewerPanel panel="paint" id={p.id} open={p.open} onClose={p.onClose}>
      <span className="text-[0.8rem] text-ink-3">{PAINT_HINT}</span>
      <div className="flex flex-wrap gap-0.5">
        <Toggle on={paint.on} title={PAINT_COPY.paint.title} onClick={() => setPaintMode(env, tool, !paint.on)}>
          {PAINT_COPY.paint.label}
        </Toggle>
        <Toggle on={paint.drop} title={PAINT_COPY.drop.title} onClick={toggleDrop}>
          {PAINT_COPY.drop.label}
        </Toggle>
        <Toggle on={paint.erase} title={PAINT_COPY.erase.title} onClick={toggleErase}>
          {PAINT_COPY.erase.label}
        </Toggle>
        <Button type="button" variant="hud-ghost" title={PAINT_COPY.undo.title} onClick={() => undoPaint(env)} className={TOGGLE}>
          {PAINT_COPY.undo.label}
        </Button>
        {/* aria-disabled, not disabled: the button just pressed keeps keyboard focus while Save runs. */}
        <Button
          type="button"
          variant="hud-ghost"
          aria-disabled={paint.saving}
          title={PAINT_COPY.save.title}
          onClick={() => void save()}
          className={cn(TOGGLE, "aria-disabled:opacity-50")}
        >
          {PAINT_COPY.save.label}
        </Button>
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-[0.78rem] font-semibold text-ink-3">{PAINT_COPY.color.label}</span>
        <input
          type="color"
          value={paint.color}
          title={PAINT_COPY.color.title}
          onChange={(e) => usePaint.setState({ color: e.target.value })}
          className="hit-44 h-7 w-full cursor-pointer rounded-md border border-input bg-white p-0.5"
        />
      </label>
      <label title={PAINT_COPY.brush.title} className="flex items-center gap-2 text-[0.78rem] font-semibold text-ink-3">
        <span>{PAINT_COPY.brush.label}</span>
        <input
          type="range"
          min={BRUSH.min}
          max={BRUSH.max}
          step={BRUSH.step}
          value={paint.radius}
          onChange={(e) => setBrushRadius(e.target.value)}
          className="hit-44 min-w-0 flex-1 accent-primary"
        />
        <strong className="w-[3.2em] text-right font-mono text-ink tabular-nums">{paint.radiusText}</strong>
      </label>
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
