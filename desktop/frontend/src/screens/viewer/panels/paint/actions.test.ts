// The brush as a viewer tool and Save, against fakes: no canvas, no Canvas, no factory (legacy index.html 5384-5562).
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Vector2, Vector3 } from "three";
import type { ToolHit, ToolPointer, ViewerTool } from "@/viewer/tools";
import { applyPaintAt, paintTool, savePaint, setBrushRadius, setPaintMode, toggleDrop, toggleErase, undoPaint, type PaintEnv, type SaveDeps } from "./actions";
import { PAINT0, setStackBrush, usePaint, type StackBrush } from "./store";

let log: string[] = [];
let armed: ViewerTool | null = null;

function env(): PaintEnv {
  return {
    tools: {
      root: () => null,
      rootDiagonal: () => 10,
      armTool(t) {
        if (armed && armed !== t) armed.onDisarm?.();
        armed = t;
        log.push(t ? "arm" : "disarm");
        return () => {};
      },
      armed: () => armed,
    },
    mesh: {
      begin: () => log.push("begin"),
      stamp: (_root, uv, r, _d, rgb, erase) => log.push(`stamp ${uv?.x} ${r.toFixed(2)} ${rgb.join(",")}${erase ? " erase" : ""}`),
      undo: () => log.push("undo"),
      sample: (_root, uv) => (uv ? "#123456" : null),
    },
    ball: { show: (p, r) => log.push(`ball ${p.x} ${r.toFixed(2)}`), hide: () => log.push("hide") },
  };
}

const hitAt = (x: number, uv: Vector2 | null = new Vector2(x, x)): ToolHit =>
  ({ point: new Vector3(x, 0, 0), local: new Vector3(), scale: 1, normal: null, object: {} as never, faceIndex: 0, uv, distance: 1 }) as ToolHit;
const ptr = (hit: ToolHit | null, buttons = 0, button = -1): ToolPointer => ({
  clientX: 0,
  clientY: 0,
  button,
  buttons,
  shiftKey: false,
  altKey: false,
  pointerId: 1,
  hit: () => hit,
});

beforeEach(() => {
  log = [];
  armed = null;
  usePaint.setState({ ...PAINT0 });
  setStackBrush(null);
});
afterEach(() => setStackBrush(null));

describe("the toggles", () => {
  it("Paint arms the brush and disarming it turns Paint off; Drop and Erase flip alone", () => {
    const e = env();
    const tool = paintTool(e);
    setPaintMode(e, tool, true);
    expect(usePaint.getState().on).toBe(true);
    expect(armed).toBe(tool);
    toggleDrop();
    toggleErase();
    expect(usePaint.getState()).toMatchObject({ on: true, drop: true, erase: true });
    setPaintMode(e, tool, false);
    expect(armed).toBeNull();
    expect(usePaint.getState().on).toBe(false);
    // Another tool (Carve) taking over unpresses Paint.
    setPaintMode(e, tool, true);
    e.tools.armTool({});
    expect(usePaint.getState().on).toBe(false);
  });

  it("the slider sets the fraction and a toFixed(3) readout", () => {
    expect(usePaint.getState().radiusText).toBe("0.04");
    setBrushRadius("0.045");
    expect(usePaint.getState()).toMatchObject({ radius: 0.045, radiusText: "0.045" });
  });
});

describe("the brush tool", () => {
  it("hover shows the ball at the hit, sized diagonal x fraction; no hit hides it", () => {
    const e = env();
    const tool = paintTool(e);
    setPaintMode(e, tool, true);
    log = [];
    tool.onMove!(ptr(hitAt(2)));
    tool.onMove!(ptr(null));
    expect(log).toEqual(["ball 2 0.40", "hide"]);
  });

  it("a down on the mesh begins a stroke and dabs; a drag keeps dabbing; up ends it", () => {
    const e = env();
    const tool = paintTool(e);
    setPaintMode(e, tool, true);
    log = [];
    tool.onDown!(ptr(hitAt(0.5), 1, 0));
    tool.onMove!(ptr(hitAt(0.6), 1));
    tool.onUp!(ptr(null, 0, 0));
    expect(log).toEqual(["begin", "stamp 0.5 0.40 46,204,113", "ball 0.6 0.40", "stamp 0.6 0.40 46,204,113"]);
  });

  it("a drag that starts off the mesh begins its own stroke, so Undo takes it back", () => {
    const e = env();
    const tool = paintTool(e);
    setPaintMode(e, tool, true);
    log = [];
    tool.onDown!(ptr(null, 1, 0));
    tool.onMove!(ptr(hitAt(0.3), 1));
    tool.onMove!(ptr(hitAt(0.4), 1));
    expect(log.filter((l) => l === "begin")).toHaveLength(1);
    expect(log[1]).toBe("begin");
  });

  it("Drop samples once, turns itself off and paints nothing on that click", () => {
    const e = env();
    const tool = paintTool(e);
    setPaintMode(e, tool, true);
    toggleDrop();
    log = [];
    tool.onDown!(ptr(hitAt(0.5), 1, 0));
    tool.onMove!(ptr(hitAt(0.6), 1));
    tool.onUp!(ptr(null, 0, 0));
    expect(usePaint.getState()).toMatchObject({ color: "#123456", drop: false });
    expect(log.some((l) => l.startsWith("stamp") || l === "begin")).toBe(false);
    // A drop off any uv keeps the colour and still turns Drop off.
    toggleDrop();
    tool.onDown!(ptr(hitAt(0.5, null), 1, 0));
    expect(usePaint.getState()).toMatchObject({ color: "#123456", drop: false });
  });

  it("Erase dabs the original back; a hit with no uv paints only the stack", () => {
    const e = env();
    toggleErase();
    applyPaintAt(e, new Vector3(1, 0, 0), new Vector2(0.2, 0.2));
    applyPaintAt(e, new Vector3(1, 0, 0), null);
    expect(log).toEqual(["stamp 0.2 0.40 46,204,113 erase"]);
  });

  it("the stack, when one registers, gets the same strokes, Undo and hover reset", () => {
    const calls: string[] = [];
    const stack: StackBrush = {
      beginStroke: () => calls.push("begin"),
      paintSphere: (p, r, rgb, erase) => calls.push(`sphere ${p.x} ${r.toFixed(2)} ${rgb.join(",")} ${erase}`),
      undoStroke: () => calls.push("undo"),
      getSheetBlob: async () => null,
      setBrush: (p) => calls.push(`brush ${p}`),
    };
    setStackBrush(stack);
    const e = env();
    const tool = paintTool(e);
    tool.onDown!(ptr(hitAt(0.5), 1, 0));
    undoPaint(e);
    expect(calls).toEqual(["begin", "sphere 0.5 0.40 46,204,113 false", "undo"]);
  });
});

describe("Save", () => {
  const png = (name: string) => new Blob([name], { type: "image/png" });
  function deps(over: Partial<SaveDeps> = {}) {
    const posted: string[] = [];
    const toasts: string[] = [];
    const d: SaveDeps = {
      busy: () => false,
      stackSheet: async () => png("sheet"),
      meshPng: async () => png("mesh"),
      post: async (kind, stem) => {
        posted.push(`${kind} ${stem}`);
        return { ok: true };
      },
      toast: (t) => toasts.push(t),
      ...over,
    };
    return { d, posted, toasts };
  }

  it("posts the stack sheet first, then the mesh, with no success message", async () => {
    const { d, posted, toasts } = deps();
    expect(await savePaint("hen", d)).toEqual({ stack: true, mesh: true });
    expect(posted).toEqual(["stack hen", "mesh hen"]);
    expect(toasts).toEqual([]);
    expect(usePaint.getState().saving).toBe(false);
  });

  it("with no stack and no stroke it does nothing and says nothing", async () => {
    const { d, posted, toasts } = deps({ stackSheet: async () => null, meshPng: async () => null });
    expect(await savePaint("hen", d)).toEqual({ stack: false, mesh: false });
    expect(posted).toEqual([]);
    expect(toasts).toEqual([]);
  });

  it("a stack failure toasts its message and skips the mesh", async () => {
    const { d, posted, toasts } = deps({
      post: async (kind) => {
        posted.push(kind);
        throw new Error("empty body");
      },
    });
    expect(await savePaint("hen", d)).toEqual({ stack: false, mesh: false });
    expect(posted).toEqual(["stack"]);
    expect(toasts).toEqual(["empty body"]);
    expect(usePaint.getState().saving).toBe(false);
  });

  it("waits for a job on this asset (the server has no busy check), and ignores a second click while saving", async () => {
    const busy = deps({ busy: (s) => s === "hen" });
    expect(await savePaint("hen", busy.d)).toEqual({ stack: false, mesh: false });
    expect(busy.toasts).toEqual(["Wait for the job on this asset to finish."]);
    expect(busy.posted).toEqual([]);

    let release!: () => void;
    const slow = deps({ stackSheet: () => new Promise((r) => (release = () => r(png("s")))) });
    const first = savePaint("hen", slow.d);
    expect(await savePaint("hen", slow.d)).toEqual({ stack: false, mesh: false });
    release();
    await first;
    expect(slow.posted).toEqual(["stack hen", "mesh hen"]);
  });
});
