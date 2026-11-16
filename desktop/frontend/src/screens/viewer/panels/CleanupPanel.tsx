// The Cleanup panel (legacy cleanup cubby, index.html 2090-2104, 4860-4898, 5300-5370): Floaters and Carve through
// POST /clean. Layout and the Carve tool's wiring only; the copy and bodies are in cleanup.ts, the run in
// cleanupActions.ts.
import { useFrame } from "@react-three/fiber";
import { LoaderCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Vector3, type Mesh } from "three";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useViewerOverlay } from "@/viewer/ToolBridge";
import { useViewerTools, type ViewerTool } from "@/viewer/tools";
import { ACTION, TOGGLE } from "../classes";
import type { ViewerPanelProps } from "../types";
import {
  CARVE_BALL,
  CARVE_LABEL,
  CARVE_TITLE,
  CLEANUP_HINT,
  FLOATERS_LABEL,
  FLOATERS_TITLE,
  RADIUS,
  RADIUS_LABEL,
  RADIUS_TITLE,
  radiusText,
  worldCarveRadius,
} from "./cleanup";
import { carve, floaters, type CleanDeps } from "./cleanupActions";
import { useCleaning } from "./cleanupApi";
import { ViewerPanel } from "./ViewerPanel";

// The hover ball's place, written at pointer rate by the tool and read every frame by <CarveBall/>, so hovering
// never re-renders the panel. One viewer, one ball.
const ball = { on: false, point: new Vector3(), radius: 1 };

function CarveBall() {
  const ref = useRef<Mesh>(null);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    m.visible = ball.on;
    if (ball.on) {
      m.position.copy(ball.point);
      m.scale.setScalar(ball.radius);
    }
  });
  return (
    <mesh ref={ref} visible={false} renderOrder={10}>
      <sphereGeometry args={[1, 24, 16]} />
      <meshBasicMaterial color={CARVE_BALL.color} transparent opacity={CARVE_BALL.opacity} depthWrite={false} />
    </mesh>
  );
}
const BALL = <CarveBall />;

export function CleanupPanel(p: ViewerPanelProps) {
  const { stem, tools, onPickLod } = p;
  const busy = useCleaning((s) => s.busy);
  const model = useViewerTools((s) => s.model);
  const [carveOn, setCarveOn] = useState(false);
  const [fraction, setFraction] = useState<number>(RADIUS.value);
  const fractionRef = useRef(fraction);
  fractionRef.current = fraction;
  const lodsRef = useRef(p.lods);
  lodsRef.current = p.lods;
  // The file the last clean wrote: Carve stays on across the reloads of its own output (carves add up), and only
  // those. Any other model change (another LOD, a version pick, a finished job) turns it off.
  const keep = useRef("");

  const deps: CleanDeps = useMemo(
    () => ({
      tools,
      lods: () => lodsRef.current,
      pickLod: onPickLod,
      onWritten: (file) => {
        keep.current = file;
      },
    }),
    [tools, onPickLod],
  );

  const depsRef = useRef(deps);
  depsRef.current = deps;
  const stemRef = useRef(stem);
  stemRef.current = stem;

  // One tool for the panel's life: re-arming a new object would read as "another tool took over" in onDisarm.
  const tool = useMemo<ViewerTool>(() => {
    const t: ViewerTool = {
      onMove(e) {
        const hit = e.hit();
        ball.on = !!hit;
        if (!hit) return;
        ball.point.copy(hit.point);
        ball.radius = worldCarveRadius(tools.rootDiagonal(), fractionRef.current);
      },
      onLeave() {
        ball.on = false;
      },
      onDown(e) {
        if (e.button !== 0) return;
        const hit = e.hit();
        if (!hit) return;
        void carve(stemRef.current, hit, worldCarveRadius(tools.rootDiagonal(), fractionRef.current), depsRef.current);
      },
      onDisarm() {
        ball.on = false;
        // Another tool (Paint) took over: unpress. A model change: keep Carve only when the new model is the
        // file it just wrote (setViewerModel swaps the model right after this, hence the microtask).
        if (tools.armed()) return setCarveOn(false);
        const before = tools.model();
        queueMicrotask(() => {
          if (tools.model() === before || !keep.current || tools.currentFile() !== keep.current) setCarveOn(false);
        });
      },
    };
    return t;
  }, [tools]);

  // Armed while on; re-armed after each reload of its own output (the model change disarmed it).
  useEffect(() => {
    if (!carveOn) return;
    return tools.armTool(tool);
  }, [carveOn, tool, tools, model]);
  // Legacy left Carve armed with the cubby collapsed; an armed delete tool with no toggle in sight is a trap.
  useEffect(() => {
    if (!p.open) setCarveOn(false);
  }, [p.open]);
  useViewerOverlay("carve-ball", carveOn ? BALL : null);

  const toggleCarve = useCallback(() => {
    keep.current = "";
    ball.on = false;
    setCarveOn((on) => !on);
  }, []);

  const onFloaters = useCallback(() => {
    if (useCleaning.getState().busy) return;
    void floaters(stem, deps);
  }, [stem, deps]);

  return (
    <ViewerPanel panel="cleanup" id={p.id} open={p.open} onClose={p.onClose}>
      <span className="text-[0.8rem] text-ink-3">{CLEANUP_HINT}</span>
      {/* Floaters runs once (an action); Carve stays on (a toggle): the shared looks keep them apart. */}
      <div className="flex flex-wrap gap-1">
        {/* aria-disabled, not disabled: the button just pressed keeps keyboard focus while the clean runs. */}
        <Button type="button" variant="outline" title={FLOATERS_TITLE} aria-disabled={busy} aria-busy={busy} onClick={onFloaters} className={cn(ACTION, "gap-1.5", busy && "opacity-60")}>
          {busy ? <LoaderCircle className="animate-spin" aria-hidden /> : null}
          {FLOATERS_LABEL}
        </Button>
        <Button type="button" variant="hud-ghost" title={CARVE_TITLE} aria-pressed={carveOn} onClick={toggleCarve} className={TOGGLE}>
          {CARVE_LABEL}
        </Button>
      </div>
      <label title={RADIUS_TITLE} className="flex items-center gap-2.5 text-[0.8rem] font-semibold text-ink-3">
        <span>{RADIUS_LABEL}</span>
        <input
          type="range"
          min={RADIUS.min}
          max={RADIUS.max}
          step={RADIUS.step}
          value={fraction}
          onChange={(e) => setFraction(Number(e.target.value))}
          className="hit-44 min-w-0 flex-1 accent-primary"
        />
        <strong className="w-9 text-right font-semibold text-ink tabular-nums">{radiusText(fraction)}</strong>
      </label>
    </ViewerPanel>
  );
}
