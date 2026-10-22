// The viewer tool API: how a 3D panel (Cleanup's Carve, Paint's brush, the Rig check) reaches the model on screen.
// Legacy kept these as page globals: root, renderer, currentFile, loadFrom, carvePick (index.html 4763-4765,
// 4884-4898, 5074-5192), and each tool set controls.enableRotate itself (5303-5308, 5383-5399).
//
// - The R3F state (camera, gl, scene) is captured inside the Canvas by <ToolBridge/> (./ToolBridge.tsx). The
//   camera the user sees is always R3F's default camera: PixelPass renders that same camera into a small target
//   and blits it, so raycastAt is right with the Pixel and PS1 240p looks on too.
// - raycastAt hits the model the legacy way (carvePick: first mesh hit). `local` is hit.object.worldToLocal(point)
//   and `scale` hit.object's world scale x (or 1), exactly what legacy sent to /clean (centre, radius / scale).
// - currentFile is the GLB on screen as a path under out/<stem>/ ("mesh.glb", "rig/rigged_allclips.glb"), or ""
//   for the /glb/<stem> fallbacks. /clean's `from` takes a bare file name only, so check for a "/" before sending.
// - loadFile(file) shows out/<stem>/<file> next (cache-busted) and re-reads the asset's docs (a new lods.json
//   entry shows at once). reload() reloads what is on screen. The override lasts until the LOD picker, a version
//   pick or a finished job on this asset sends the viewer back to its usual candidates.
// - armTool(tool): while armed, LEFT-button down/move/up on the orbit surface go to the tool (every move, hover
//   included, with `buttons`), and RIGHT-drag orbits ("Orbit with right-drag while this is on"). Arming another
//   tool, a model change and leaving the viewer disarm it (legacy left Carve and Paint armed across loads).
// - Mesh overlays (a brush ball) render inside the Canvas through useViewerOverlay (./ToolBridge.tsx), in world
//   space: put them at hit.point.
//
// One viewer is mounted at a time, so this is one module-level store; `viewerTools` is the stable handle the
// panels get in their props.
import { Box3, Raycaster, Vector2, Vector3, type Camera, type Intersection, type Mesh, type Object3D, type WebGLRenderer, type Scene } from "three";
import { create } from "zustand";
import type { ReactNode } from "react";
import type { PreparedModel } from "./prepareModel";

export interface ToolHit {
  /** World space (where an overlay goes). */
  point: Vector3;
  /** hit.object-local: what legacy sent to /clean as `center`. */
  local: Vector3;
  /** hit.object's world scale x, or 1: divide a world radius by it for /clean (legacy radius: worldCarveRadius() / scale). */
  scale: number;
  /** World-space face normal, or null when the hit has no face. */
  normal: Vector3 | null;
  object: Mesh;
  faceIndex: number | null;
  /** The texture coordinate under the pointer (Paint's stampUV), or null without UVs. */
  uv: Vector2 | null;
  distance: number;
}

export interface ToolPointer {
  clientX: number;
  clientY: number;
  /** PointerEvent.button: 0 on a left down/up; -1 on a plain move. */
  button: number;
  /** PointerEvent.buttons: bit 1 is the left button held (legacy paint drag: e.buttons & 1). */
  buttons: number;
  shiftKey: boolean;
  altKey: boolean;
  pointerId: number;
  /** The model under the pointer, raycast on first call (hover moves that ignore it cost nothing). */
  hit(): ToolHit | null;
}

export interface ViewerTool {
  /** Left button down on the model view. */
  onDown?(e: ToolPointer): void;
  /** Every pointer move while armed: hover (buttons 0) and left-drag (buttons & 1). Not during a right-drag orbit. */
  onMove?(e: ToolPointer): void;
  /** Left button up (or the pointer cancelled). */
  onUp?(e: ToolPointer): void;
  /** The pointer left the view: hide a hover cursor. */
  onLeave?(): void;
  /** Disarmed: by armTool(null), another tool, a model change or the viewer closing. Unpress your toggle here. */
  onDisarm?(): void;
  /** CSS cursor over the view while armed (default "crosshair"). */
  cursor?: string;
}

export interface ThreeHandles {
  camera: Camera;
  gl: WebGLRenderer;
  scene: Scene;
}

export interface ViewerTools {
  /** The wrapper group the model sits in (scaled and placed by prepareModel), or null while nothing is loaded. */
  root(): Object3D | null;
  model(): PreparedModel | null;
  three(): ThreeHandles | null;
  raycastAt(clientX: number, clientY: number): ToolHit | null;
  /** The world-space diagonal of the model's bounding box (legacy worldCarveRadius / worldPaintRadius base); 0 when empty. */
  rootDiagonal(): number;
  /** The GLB on screen, under out/<stem>/, no query string; "" when it came from /glb/<stem>. */
  currentFile(): string;
  /** Show out/<stem>/<file> next. Rejects with the load error; the model on screen then stays. */
  loadFile(file: string): Promise<void>;
  /** Reload what is on screen (and re-read the docs). */
  reload(): Promise<void>;
  /** Arm a tool (or null to disarm). Returns a disarm that only disarms this tool, for a useEffect cleanup. */
  armTool(tool: ViewerTool | null): () => void;
  armed(): ViewerTool | null;
}

/** What the viewer registers so loadFile and reload reach its loader. */
export interface ViewerLoader {
  loadFile(file: string): Promise<void>;
  reload(): Promise<void>;
}

interface ToolState {
  model: PreparedModel | null;
  file: string;
  tool: ViewerTool | null;
  overlays: Record<string, ReactNode>;
  three: (() => ThreeHandles) | null;
  loader: ViewerLoader | null;
}

const empty: ToolState = { model: null, file: "", tool: null, overlays: {}, three: null, loader: null };

export const useViewerTools = create<ToolState>(() => ({ ...empty }));

function disarm(next: ViewerTool | null = null): void {
  const prev = useViewerTools.getState().tool;
  if (prev === next) return;
  useViewerTools.setState({ tool: next });
  prev?.onDisarm?.();
}

/** The viewer calls this after every load: a new model disarms the tool. */
export function setViewerModel(model: PreparedModel | null, file: string): void {
  const st = useViewerTools.getState();
  if (st.model !== model) disarm(null);
  useViewerTools.setState({ model, file });
}

export function setViewerLoader(loader: ViewerLoader | null): void {
  useViewerTools.setState({ loader });
}

export function setViewerThree(get: (() => ThreeHandles) | null): void {
  useViewerTools.setState({ three: get });
}

/** Leaving the viewer: disarm (the tool hears onDisarm) and forget the model. */
export function resetViewerTools(): void {
  disarm(null);
  useViewerTools.setState({ ...empty });
}

export function setOverlay(key: string, node: ReactNode | null): void {
  const overlays = { ...useViewerTools.getState().overlays };
  if (node == null) delete overlays[key];
  else overlays[key] = node;
  useViewerTools.setState({ overlays });
}

/* ------------------------------------------------------------------ pure helpers (tools.test.ts) */

/** Pointer to normalised device coordinates over the canvas rect (legacy carvePick). */
export function clientToNdc(clientX: number, clientY: number, rect: { left: number; top: number; width: number; height: number }): Vector2 {
  return new Vector2(((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1, -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1);
}

/** The first mesh hit as a ToolHit (legacy: hits.find(h => h.object.isMesh)). */
export function toToolHit(hits: Intersection[]): ToolHit | null {
  const h = hits.find((x) => (x.object as Mesh).isMesh);
  if (!h) return null;
  const object = h.object as Mesh;
  const local = object.worldToLocal(h.point.clone());
  const scale = object.getWorldScale(new Vector3()).x || 1;
  const normal = h.face ? h.face.normal.clone().transformDirection(object.matrixWorld) : null;
  return { point: h.point.clone(), local, scale, normal, object, faceIndex: h.faceIndex ?? null, uv: h.uv ? h.uv.clone() : null, distance: h.distance };
}

/** Raycast `root` from `camera` at NDC `ndc`. */
export function raycastModel(root: Object3D, camera: Camera, ndc: Vector2): ToolHit | null {
  const ray = new Raycaster();
  ray.setFromCamera(ndc, camera);
  return toToolHit(ray.intersectObject(root, true));
}

/** The world bounding-box diagonal of `root`; 0 when it is empty. */
export function diagonalOf(root: Object3D | null): number {
  if (!root) return 0;
  const box = new Box3().setFromObject(root);
  return box.isEmpty() ? 0 : box.getSize(new Vector3()).length();
}

/** A loaded URL as a path under out/<stem>/ with no query ("" when it is not a GLB there, e.g. /glb/<stem>?lod=). */
export function outRelFile(url: string, stem: string): string {
  const path = url.split(/[?#]/)[0];
  const prefix = `/out/${encodeURIComponent(stem)}/`;
  let rel: string;
  if (path.startsWith(prefix)) rel = path.slice(prefix.length);
  else if (path.startsWith(`/out/${stem}/`)) rel = path.slice(`/out/${stem}/`.length);
  else return "";
  rel = rel.split("/").map((s) => decodeURIComponent(s)).join("/");
  return /\.glb$/i.test(rel) ? rel : "";
}

/* ------------------------------------------------------------------ the handle panels get */

export const viewerTools: ViewerTools = {
  root: () => useViewerTools.getState().model?.root ?? null,
  model: () => useViewerTools.getState().model,
  three: () => useViewerTools.getState().three?.() ?? null,
  raycastAt(clientX, clientY) {
    const st = useViewerTools.getState();
    const three = st.three?.();
    const root = st.model?.root;
    if (!three || !root) return null;
    return raycastModel(root, three.camera, clientToNdc(clientX, clientY, three.gl.domElement.getBoundingClientRect()));
  },
  rootDiagonal: () => diagonalOf(useViewerTools.getState().model?.root ?? null),
  currentFile: () => useViewerTools.getState().file,
  loadFile(file) {
    const loader = useViewerTools.getState().loader;
    return loader ? loader.loadFile(file) : Promise.reject(new Error("no glb loaded"));
  },
  reload() {
    const loader = useViewerTools.getState().loader;
    return loader ? loader.reload() : Promise.reject(new Error("no glb loaded"));
  },
  armTool(tool) {
    disarm(tool);
    return () => {
      if (tool && useViewerTools.getState().tool === tool) disarm(null);
    };
  },
  armed: () => useViewerTools.getState().tool,
};
