// What the Paint panel's controls and the brush do (legacy setPaintMode, applyPaintAt, the paint toggles, Undo,
// Save, paintHover and the mesh pointer handlers; index.html 5384-5399, 5453-5562). Side effects only through
// the viewer tool API, the MeshBrush, the registered stack brush and the injected Save calls, so the tests fake them.
import type { Vector2, Vector3 } from "three";
import { toast } from "@/lib/ui";
import { jobOwns } from "@/screens/asset/flat/actions";
import type { ToolPointer, ViewerTool, ViewerTools } from "@/viewer/tools";
import { postPaint } from "./api";
import type { MeshBrush } from "./brush";
import { brushText, hexToRgb, PAINT_BUSY, worldPaintRadius, type PaintKind } from "./paint";
import { stackBrush, usePaint } from "./store";

/** The hover ball (legacy paintBall), drawn by the panel inside the viewer. */
export interface BrushBall {
  show(point: Vector3, radius: number): void;
  hide(): void;
}

/** What the brush works on. */
export interface PaintEnv {
  tools: Pick<ViewerTools, "root" | "rootDiagonal" | "armTool" | "armed">;
  mesh: Pick<MeshBrush, "begin" | "stamp" | "undo" | "sample">;
  ball: BrushBall;
}

/** The brush radius in world units now (legacy worldPaintRadius). */
export function paintRadius(env: PaintEnv): number {
  return worldPaintRadius(env.tools.rootDiagonal(), usePaint.getState().radius);
}

/** A new stroke: one Undo snapshot on the mesh and the stack (legacy beginPaintStroke). */
export function beginPaintStroke(env: PaintEnv): void {
  env.mesh.begin(env.tools.root());
  stackBrush()?.beginStroke();
}

/** One dab at a world point: the mesh through its uv (none off the mesh), the stack in the same world space. */
export function applyPaintAt(env: PaintEnv, point: Vector3, uv: Vector2 | null): void {
  const { color, erase } = usePaint.getState();
  const radius = paintRadius(env);
  const rgb = hexToRgb(color);
  if (uv) env.mesh.stamp(env.tools.root(), uv, radius, env.tools.rootDiagonal(), rgb, erase);
  stackBrush()?.paintSphere(point, radius, rgb, erase);
}

/** The ball follows the pointer over the mesh while Paint is on (legacy paintHover from the mesh). */
function hover(env: PaintEnv, point: Vector3 | null): void {
  stackBrush()?.setBrush?.(null);
  if (!usePaint.getState().on || !point) env.ball.hide();
  else env.ball.show(point, paintRadius(env));
}

/**
 * The brush as a viewer tool. Left down paints (or with Drop on, samples the colour once and turns Drop off);
 * left-drag keeps painting; right-drag orbits (the tool API). Disarming it (another tool, a model change, the
 * viewer closing) turns Paint off.
 */
export function paintTool(env: PaintEnv): ViewerTool {
  // Legacy painted a drag that started off the mesh without an Undo snapshot, and a drag right after a drop
  // pick: a stroke begins on its first dab here, and a drop click paints nothing until the button comes up.
  let stroking = false;
  let sampled = false;
  const tool: ViewerTool = {
    cursor: "crosshair",
    onMove(e: ToolPointer) {
      const hit = e.hit();
      hover(env, hit ? hit.point : null);
      if (!hit || !(e.buttons & 1) || usePaint.getState().drop || sampled) return;
      if (!stroking) {
        beginPaintStroke(env);
        stroking = true;
      }
      applyPaintAt(env, hit.point, hit.uv);
    },
    onDown(e: ToolPointer) {
      const hit = e.hit();
      if (!hit) return;
      if (usePaint.getState().drop) {
        const hex = env.mesh.sample(env.tools.root(), hit.uv);
        usePaint.setState(hex ? { color: hex, drop: false } : { drop: false });
        sampled = true;
        return;
      }
      beginPaintStroke(env);
      stroking = true;
      applyPaintAt(env, hit.point, hit.uv);
    },
    onUp() {
      stroking = false;
      sampled = false;
    },
    onLeave() {
      env.ball.hide();
    },
    onDisarm() {
      stroking = false;
      sampled = false;
      usePaint.setState({ on: false });
      env.ball.hide();
      stackBrush()?.setBrush?.(null);
    },
  };
  return tool;
}

/** The Paint toggle (legacy setPaintMode): arming the brush disarms Carve; turning it off disarms only the brush. */
export function setPaintMode(env: PaintEnv, tool: ViewerTool, on: boolean): void {
  env.ball.hide();
  stackBrush()?.setBrush?.(null);
  if (on) {
    env.tools.armTool(tool);
    usePaint.setState({ on: true });
    return;
  }
  if (env.tools.armed() === tool) env.tools.armTool(null);
  usePaint.setState({ on: false });
}

/** Drop and Erase flip on their own; neither turns Paint on (legacy). */
export function toggleDrop(): void {
  usePaint.setState((s) => ({ drop: !s.drop }));
}
export function toggleErase(): void {
  usePaint.setState((s) => ({ erase: !s.erase }));
}

/** The slider: the value and its toFixed(3) readout. */
export function setBrushRadius(value: string | number): void {
  usePaint.setState({ radius: Number(value), radiusText: brushText(value) });
}

/** Undo the last stroke on the mesh and the stack. */
export function undoPaint(env: PaintEnv): void {
  env.mesh.undo();
  stackBrush()?.undoStroke();
}

export interface SaveDeps {
  busy(stem: string): boolean;
  /** The stack sheet, or null with no stack loaded. */
  stackSheet(): Promise<Blob | null>;
  /** The painted mesh texture, or null when no stroke or drop took it. */
  meshPng(): Promise<Blob | null>;
  post(kind: PaintKind, stem: string, png: Blob): Promise<unknown>;
  toast(text: string): void;
}

/** The real Save calls for a mesh brush. */
export function saveDeps(mesh: Pick<MeshBrush, "toBlob">): SaveDeps {
  return {
    busy: jobOwns,
    stackSheet: () => stackBrush()?.getSheetBlob() ?? Promise.resolve(null),
    meshPng: () => mesh.toBlob(),
    post: postPaint,
    toast: (text) => toast(text),
  };
}

/**
 * Save (legacy paint-save click): POST /paint/stack with the stack sheet when a stack is loaded, then POST
 * /paint/mesh with the texture when a stroke took it. A failure toasts its message and stops there (a stack
 * failure skips the mesh). No success message, as in legacy. Returns what was written.
 */
export async function savePaint(stem: string, deps: SaveDeps): Promise<{ stack: boolean; mesh: boolean }> {
  const done = { stack: false, mesh: false };
  if (usePaint.getState().saving) return done;
  if (deps.busy(stem)) {
    deps.toast(PAINT_BUSY);
    return done;
  }
  usePaint.setState({ saving: true });
  try {
    const sheet = await deps.stackSheet();
    if (sheet) {
      await deps.post("stack", stem, sheet);
      done.stack = true;
    }
    const png = await deps.meshPng();
    if (png) {
      await deps.post("mesh", stem, png);
      done.mesh = true;
    }
  } catch (err) {
    deps.toast((err as Error)?.message || String(err));
  } finally {
    usePaint.setState({ saving: false });
  }
  return done;
}
