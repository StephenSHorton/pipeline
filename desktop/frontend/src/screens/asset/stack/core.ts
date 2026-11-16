// The sprite stack's pure core, ported from public/stack-lab.js and kept word for word where it could be:
// cacheNative's pixel loop (378-407), the Main island mask (409-501), the Round surface (503-687), marching
// squares and the contour shapes (716-911), OkLab quantize and the palette (1229-1391), the paint brush and the
// colour sampler (1393-1483) and the sheet layout getSheetBlob writes (1496-1519). No DOM and no renderer: the
// closures' shared variables live on one `Core` object, so vitest's node env can run all of it. view.ts draws.
import { BufferAttribute, BufferGeometry, Path, Shape } from "three";
import type { PaletteEntry, StackMeta, StackState } from "./stack";

/** One slab's opaque texels at native resolution. rgb0 is what the sheet had: Erase brings it back. */
export interface Native {
  n: number;
  ix: Uint16Array;
  iz: Uint16Array;
  rgb: Uint8Array;
  rgb0: Uint8Array;
  /** Palette index per texel, once Reduce ran. */
  pal?: Uint8Array;
}

export interface Slice {
  index: number;
  native: Native | null;
}

/** What the legacy closure shared between these functions. */
export interface Core {
  meta: StackMeta;
  slices: Slice[];
  state: StackState;
  islandMask: Uint8Array | null;
  islandCrunch: number;
  palette: PaletteEntry[] | null;
  palSelected: number;
  strokeUndo: (Uint8Array | null)[] | null;
}

export function createCore(meta: StackMeta, state: StackState): Core {
  const slices: Slice[] = [];
  for (let i = 0; i < meta.layers; i++) slices.push({ index: i, native: null });
  return { meta, slices, state, islandMask: null, islandCrunch: 0, palette: null, palSelected: 0, strokeUndo: null };
}

/** The pixel loop of legacy cacheNative: texels with alpha >= 24 from one res×res RGBA slab. */
export function nativeFromPixels(data: ArrayLike<number>, res: number): Native {
  const ix: number[] = [];
  const iz: number[] = [];
  const rgb: number[] = [];
  for (let z = 0; z < res; z++) {
    for (let x = 0; x < res; x++) {
      const p = (z * res + x) * 4;
      if (data[p + 3] < 24) continue;
      ix.push(x);
      iz.push(z);
      rgb.push(data[p], data[p + 1], data[p + 2]);
    }
  }
  return {
    n: ix.length,
    ix: new Uint16Array(ix),
    iz: new Uint16Array(iz),
    rgb: new Uint8Array(rgb),
    rgb0: new Uint8Array(rgb),
  };
}

export function crunchSize(core: Core): number {
  return Math.max(8, Math.min(core.state.texels, core.meta.resolution));
}

export function layerShade(core: Core, index: number): number {
  return core.state.shade ? 0.52 + 0.48 * (index / Math.max(1, core.meta.layers - 1)) : 1;
}

/** How many slabs are shown (Layers, capped at the stack's own count). */
export const usedLayers = (core: Core) => Math.max(1, Math.min(core.state.layers, core.meta.layers));

export function occupancyGrid(core: Core, rec: Slice, crunch: number): Uint8Array {
  const native = rec.native;
  const res = core.meta.resolution;
  const occ = new Uint8Array(crunch * crunch);
  if (!native || !native.n) return occ;
  const scale = crunch / res;
  for (let i = 0; i < native.n; i++) {
    const gx = crunch >= res ? native.ix[i] : Math.min(crunch - 1, (native.ix[i] * scale) | 0);
    const gz = crunch >= res ? native.iz[i] : Math.min(crunch - 1, (native.iz[i] * scale) | 0);
    if (!cellKept(core, rec.index, gx, gz, crunch)) continue;
    occ[gz * crunch + gx] = 1;
  }
  return occ;
}

export function cellKept(core: Core, y: number, gx: number, gz: number, n: number): boolean {
  const { state, islandMask, islandCrunch, meta } = core;
  if (!state.main || !islandMask || n !== islandCrunch) return true;
  const ny = Math.max(1, Math.min(state.layers, meta.layers));
  if (y < 0 || y >= ny || gx < 0 || gz < 0 || gx >= n || gz >= n) return false;
  return islandMask[(y * n + gz) * n + gx] === 1;
}

/** Main: keep only the largest 6-connected piece of the shown slabs, at the Texels grid (union-find). */
export function rebuildIslandMask(core: Core, crunch: number): void {
  const { state, meta } = core;
  core.islandCrunch = crunch;
  const ny = Math.max(1, Math.min(state.layers, meta.layers));
  const n = crunch;
  const N = ny * n * n;
  const occ = new Uint8Array(N);
  const res = meta.resolution;
  const scale = n / res;
  for (let y = 0; y < ny; y++) {
    const native = core.slices[y] && core.slices[y].native;
    if (!native) continue;
    for (let i = 0; i < native.n; i++) {
      const gx = n >= res ? native.ix[i] : Math.min(n - 1, (native.ix[i] * scale) | 0);
      const gz = n >= res ? native.iz[i] : Math.min(n - 1, (native.iz[i] * scale) | 0);
      occ[(y * n + gz) * n + gx] = 1;
    }
  }
  if (!state.main) {
    core.islandMask = occ;
    return;
  }
  const parent = new Int32Array(N);
  for (let i = 0; i < N; i++) parent[i] = occ[i] ? i : -1;
  const find = (a: number) => {
    let x = a;
    while (parent[x] !== x) x = parent[x];
    let y = a;
    while (y !== x) {
      const p = parent[y];
      parent[y] = x;
      y = p;
    }
    return x;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  const at = (x: number, y: number, z: number) => (y * n + z) * n + x;
  for (let y = 0; y < ny; y++) {
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        const i = at(x, y, z);
        if (!occ[i]) continue;
        if (x + 1 < n && occ[at(x + 1, y, z)]) union(i, at(x + 1, y, z));
        if (z + 1 < n && occ[at(x, y, z + 1)]) union(i, at(x, y, z + 1));
        if (y + 1 < ny && occ[at(x, y + 1, z)]) union(i, at(x, y + 1, z));
      }
    }
  }
  const sizes = new Map<number, number>();
  let maxSize = 0;
  for (let i = 0; i < N; i++) {
    if (!occ[i]) continue;
    const r = find(i);
    const s = (sizes.get(r) || 0) + 1;
    sizes.set(r, s);
    if (s > maxSize) maxSize = s;
  }
  const mask = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (!occ[i]) continue;
    if ((sizes.get(find(i)) ?? 0) >= maxSize) mask[i] = 1;
  }
  core.islandMask = mask;
}

/* ---------------------------------------------------------------- Round: one welded, smoothed voxel surface */

const FACE_DIR = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
];
const FACE_CORNERS = [
  [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
  [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
  [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
  [[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]],
  [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
  [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
];

export function laplacianSmooth(pos: Float32Array, adj: number[][], lambda: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(pos.length);
  for (let i = 0; i < adj.length; i++) {
    const nbs = adj[i];
    const o = i * 3;
    if (!nbs.length) {
      out[o] = pos[o];
      out[o + 1] = pos[o + 1];
      out[o + 2] = pos[o + 2];
      continue;
    }
    let ax = 0, ay = 0, az = 0;
    for (let k = 0; k < nbs.length; k++) {
      const j = nbs[k] * 3;
      ax += pos[j];
      ay += pos[j + 1];
      az += pos[j + 2];
    }
    const inv = 1 / nbs.length;
    out[o] = pos[o] + lambda * (ax * inv - pos[o]);
    out[o + 1] = pos[o + 1] + lambda * (ay * inv - pos[o + 1]);
    out[o + 2] = pos[o + 2] + lambda * (az * inv - pos[o + 2]);
  }
  return out;
}

/** The exposed faces of the shown voxels, welded and Taubin-smoothed by `amount` (squared: 0.45 is a light blend). */
export function buildMergedVoxelSurface(core: Core, crunch: number, amount: number): BufferGeometry | null {
  const { state, meta } = core;
  const ny = Math.max(1, Math.min(state.layers, meta.layers));
  const n = crunch;
  const occ = new Uint8Array(ny * n * n);
  const col = new Uint8Array(ny * n * n * 3);
  const res = meta.resolution;
  const scale = n / res;
  for (let y = 0; y < ny; y++) {
    const native = core.slices[y] && core.slices[y].native;
    if (!native) continue;
    const shade = layerShade(core, y);
    for (let i = 0; i < native.n; i++) {
      const gx = n >= res ? native.ix[i] : Math.min(n - 1, (native.ix[i] * scale) | 0);
      const gz = n >= res ? native.iz[i] : Math.min(n - 1, (native.iz[i] * scale) | 0);
      const idx = (y * n + gz) * n + gx;
      if (!cellKept(core, y, gx, gz, n)) continue;
      occ[idx] = 1;
      col[idx * 3] = Math.round(native.rgb[i * 3] * shade);
      col[idx * 3 + 1] = Math.round(native.rgb[i * 3 + 1] * shade);
      col[idx * 3 + 2] = Math.round(native.rgb[i * 3 + 2] * shade);
    }
  }
  const filled = (x: number, y: number, z: number) => {
    if (x < 0 || y < 0 || z < 0 || x >= n || y >= ny || z >= n) return 0;
    return occ[(y * n + z) * n + x];
  };
  const world = meta.resolution * meta.pitch;
  const cellH = meta.pitch * Math.max(0.55, state.spacing);
  const vertId = new Map<number, number>();
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const triFaceN: number[] = [];
  const vert = (x: number, y: number, z: number, r: number, g: number, b: number) => {
    const k = x + n * (z + n * y);
    let id = vertId.get(k);
    if (id != null) return id;
    id = positions.length / 3;
    vertId.set(k, id);
    positions.push((x / n - 0.5) * world, y * cellH, (z / n - 0.5) * world);
    colors.push(r / 255, g / 255, b / 255);
    return id;
  };
  for (let y = 0; y < ny; y++) {
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        if (!filled(x, y, z)) continue;
        const idx = (y * n + z) * n + x;
        const r = col[idx * 3];
        const g = col[idx * 3 + 1];
        const b = col[idx * 3 + 2];
        for (let f = 0; f < 6; f++) {
          const d = FACE_DIR[f];
          if (filled(x + d[0], y + d[1], z + d[2])) continue;
          const c = FACE_CORNERS[f];
          const ia = vert(x + c[0][0], y + c[0][1], z + c[0][2], r, g, b);
          const ib = vert(x + c[1][0], y + c[1][1], z + c[1][2], r, g, b);
          const ic = vert(x + c[2][0], y + c[2][1], z + c[2][2], r, g, b);
          const idv = vert(x + c[3][0], y + c[3][1], z + c[3][2], r, g, b);
          indices.push(ia, ic, ib, ia, idv, ic);
          for (let k = 0; k < 6; k++) triFaceN.push(d[0], d[1], d[2]);
        }
      }
    }
  }
  if (!indices.length) return null;
  const vcount = positions.length / 3;
  const adj: number[][] = Array.from({ length: vcount }, () => []);
  const seenE = new Set<string>();
  const addE = (i: number, j: number) => {
    const a = i < j ? `${i},${j}` : `${j},${i}`;
    if (seenE.has(a)) return;
    seenE.add(a);
    adj[i].push(j);
    adj[j].push(i);
  };
  for (let i = 0; i < indices.length; i += 3) {
    addE(indices[i], indices[i + 1]);
    addE(indices[i + 1], indices[i + 2]);
    addE(indices[i + 2], indices[i]);
  }
  let pos = new Float32Array(positions);
  const ease = amount * amount;
  const raw = ease * 14;
  const full = Math.floor(raw);
  const frac = raw - full;
  for (let p = 0; p < full; p++) {
    pos = laplacianSmooth(pos, adj, 0.5);
    pos = laplacianSmooth(pos, adj, -0.53);
  }
  if (frac > 1e-6) {
    let next = laplacianSmooth(pos, adj, 0.5);
    next = laplacianSmooth(next, adj, -0.53);
    for (let i = 0; i < pos.length; i++) pos[i] += (next[i] - pos[i]) * frac;
  }
  const welded = new BufferGeometry();
  welded.setAttribute("position", new BufferAttribute(pos, 3));
  welded.setIndex(indices);
  welded.computeVertexNormals();
  const smoothN = welded.getAttribute("normal");
  const nMix = Math.min(1, ease * 6);
  const outP = new Float32Array(indices.length * 3);
  const outN = new Float32Array(indices.length * 3);
  const outC = new Float32Array(indices.length * 3);
  for (let t = 0; t < indices.length; t++) {
    const vi = indices[t];
    const o = t * 3;
    outP[o] = pos[vi * 3];
    outP[o + 1] = pos[vi * 3 + 1];
    outP[o + 2] = pos[vi * 3 + 2];
    const fx = triFaceN[t * 3];
    const fy = triFaceN[t * 3 + 1];
    const fz = triFaceN[t * 3 + 2];
    const nx = smoothN.getX(vi);
    const ny = smoothN.getY(vi);
    const nz = smoothN.getZ(vi);
    const mx = fx + (nx - fx) * nMix;
    const my = fy + (ny - fy) * nMix;
    const mz = fz + (nz - fz) * nMix;
    const len = Math.hypot(mx, my, mz) || 1;
    outN[o] = mx / len;
    outN[o + 1] = my / len;
    outN[o + 2] = mz / len;
    outC[o] = colors[vi * 3];
    outC[o + 1] = colors[vi * 3 + 1];
    outC[o + 2] = colors[vi * 3 + 2];
  }
  welded.dispose();
  const geo = new BufferGeometry();
  geo.setAttribute("position", new BufferAttribute(outP, 3));
  geo.setAttribute("normal", new BufferAttribute(outN, 3));
  geo.setAttribute("color", new BufferAttribute(outC, 3));
  return geo;
}

/* ---------------------------------------------------------------- Contour: marching squares → extrudable shapes */

type Pt = [number, number];

function ptKey(p: Pt): string {
  return `${p[0].toFixed(4)},${p[1].toFixed(4)}`;
}

export function stitchLoops(segs: [Pt, Pt][]): Pt[][] {
  const adj = new Map<string, string[]>();
  const coord = new Map<string, Pt>();
  const addEdge = (a: Pt, b: Pt) => {
    const ka = ptKey(a);
    const kb = ptKey(b);
    if (ka === kb) return;
    coord.set(ka, a);
    coord.set(kb, b);
    if (!adj.has(ka)) adj.set(ka, []);
    if (!adj.has(kb)) adj.set(kb, []);
    adj.get(ka)!.push(kb);
    adj.get(kb)!.push(ka);
  };
  for (const [a, b] of segs) addEdge(a, b);
  const edgeId = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const used = new Set<string>();
  const loops: Pt[][] = [];
  for (const start of adj.keys()) {
    const startNbs = adj.get(start) || [];
    for (const nb of startNbs) {
      const e0 = edgeId(start, nb);
      if (used.has(e0)) continue;
      const loop = [start];
      let prev = start;
      let cur = nb;
      used.add(e0);
      let guard = 0;
      while (cur !== start && guard++ < 200000) {
        loop.push(cur);
        const nbs = adj.get(cur) || [];
        let next: string | null = null;
        for (const cand of nbs) {
          if (cand === prev) continue;
          const e = edgeId(cur, cand);
          if (used.has(e)) continue;
          next = cand;
          used.add(e);
          break;
        }
        if (next == null) break;
        prev = cur;
        cur = next;
      }
      if (cur === start && loop.length >= 3) {
        loops.push(loop.map((k) => coord.get(k)!));
      }
    }
  }
  return loops;
}

/** Closed outlines of an n×n occupancy grid, in texel coordinates (edge midpoints). */
export function marchingSquares(occ: ArrayLike<number>, n: number): Pt[][] {
  const sample = (x: number, z: number) => {
    if (x < 0 || z < 0 || x >= n || z >= n) return 0;
    return occ[z * n + x] ? 1 : 0;
  };
  const segs: [Pt, Pt][] = [];
  const add = (a: Pt, b: Pt) => segs.push([a, b]);
  for (let z = -1; z < n; z++) {
    for (let x = -1; x < n; x++) {
      const v0 = sample(x, z);
      const v1 = sample(x + 1, z);
      const v2 = sample(x + 1, z + 1);
      const v3 = sample(x, z + 1);
      const idx = v0 | (v1 << 1) | (v2 << 2) | (v3 << 3);
      if (idx === 0 || idx === 15) continue;
      const t: Pt = [x + 0.5, z];
      const r: Pt = [x + 1, z + 0.5];
      const b: Pt = [x + 0.5, z + 1];
      const l: Pt = [x, z + 0.5];
      switch (idx) {
        case 1: add(l, t); break;
        case 2: add(t, r); break;
        case 3: add(l, r); break;
        case 4: add(r, b); break;
        case 5: add(l, t); add(r, b); break;
        case 6: add(t, b); break;
        case 7: add(l, b); break;
        case 8: add(b, l); break;
        case 9: add(t, b); break;
        case 10: add(t, r); add(b, l); break;
        case 11: add(r, b); break;
        case 12: add(r, l); break;
        case 13: add(t, r); break;
        case 14: add(t, l); break;
        default: break;
      }
    }
  }
  return stitchLoops(segs);
}

export function cleanLoop(pts: Pt[]): Pt[] {
  const out: Pt[] = [];
  for (const p of pts) {
    const last = out[out.length - 1];
    if (!last || Math.abs(last[0] - p[0]) > 1e-8 || Math.abs(last[1] - p[1]) > 1e-8) out.push(p);
  }
  if (out.length > 1) {
    const a = out[0];
    const b = out[out.length - 1];
    if (Math.abs(a[0] - b[0]) < 1e-8 && Math.abs(a[1] - b[1]) < 1e-8) out.pop();
  }
  return out;
}

export function signedArea(pts: Pt[]): number {
  let area = 0;
  for (let i = 0, n = pts.length; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area * 0.5;
}

export function pointInLoop(x: number, z: number, pts: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0];
    const zi = pts[i][1];
    const xj = pts[j][0];
    const zj = pts[j][1];
    const hit = zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi || 1e-15) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

function pixToWorld(px: number, pz: number, n: number, world: number): Pt {
  const wx = ((px + 0.5) / n - 0.5) * world;
  const wz = ((pz + 0.5) / n - 0.5) * world;
  // Shape Y is -Z so rotateX(-PI/2) maps extrusion to +Y without mirroring XZ.
  return [wx, -wz];
}

interface LoopMeta {
  pts: Pt[];
  area: number;
  abs: number;
  cx: number;
  cz: number;
  holes: LoopMeta[];
}

/** Outer loops become shapes, the loops inside them their holes; slivers under 2.5 texels are dropped. */
export function loopsToShapes(loops: Pt[][], n: number, world: number): Shape[] {
  const metas: LoopMeta[] = [];
  for (const raw of loops) {
    const cleaned = cleanLoop(raw.map((p) => pixToWorld(p[0], p[1], n, world)));
    if (cleaned.length < 3) continue;
    const area = signedArea(cleaned);
    const minArea = (world / n) * (world / n) * 2.5;
    if (Math.abs(area) < minArea) continue;
    const cx = cleaned.reduce((s, p) => s + p[0], 0) / cleaned.length;
    const cz = cleaned.reduce((s, p) => s + p[1], 0) / cleaned.length;
    metas.push({ pts: cleaned, area, abs: Math.abs(area), cx, cz, holes: [] });
  }
  metas.sort((a, b) => b.abs - a.abs);
  const isOuter = new Array<boolean>(metas.length).fill(true);
  const parentOf = new Array<number>(metas.length).fill(-1);
  for (let i = 0; i < metas.length; i++) {
    let best = -1;
    for (let j = 0; j < i; j++) {
      if (pointInLoop(metas[i].cx, metas[i].cz, metas[j].pts)) {
        if (best < 0 || metas[j].abs < metas[best].abs) best = j;
      }
    }
    parentOf[i] = best;
    if (best >= 0) isOuter[i] = !isOuter[best];
  }
  for (let i = 0; i < metas.length; i++) {
    if (isOuter[i]) continue;
    let p = parentOf[i];
    while (p >= 0 && !isOuter[p]) p = parentOf[p];
    if (p >= 0) metas[p].holes.push(metas[i]);
  }
  const shapes: Shape[] = [];
  for (let i = 0; i < metas.length; i++) {
    if (!isOuter[i]) continue;
    const m = metas[i];
    let pts = m.pts;
    if (m.area < 0) pts = pts.slice().reverse();
    const shape = new Shape();
    shape.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < pts.length; k++) shape.lineTo(pts[k][0], pts[k][1]);
    shape.closePath();
    for (const h of m.holes) {
      let hp = h.pts;
      if (h.area > 0) hp = hp.slice().reverse();
      if (hp.length < 3) continue;
      const path = new Path();
      path.moveTo(hp[0][0], hp[0][1]);
      for (let k = 1; k < hp.length; k++) path.lineTo(hp[k][0], hp[k][1]);
      path.closePath();
      shape.holes.push(path);
    }
    shapes.push(shape);
  }
  return shapes;
}

/* ---------------------------------------------------------------- palette: OkLab, chroma-weighted unique hues */

export function srgbToOkLab(r: number, g: number, b: number): [number, number, number] {
  const lin = (c: number) => {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const R = lin(r), G = lin(g), B = lin(b);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function labDist2(a: number[], b: number[]): number {
  const dL = a[0] - b[0], da = a[1] - b[1], db = a[2] - b[2];
  return dL * dL + da * da + db * db;
}

interface Cand {
  r: number;
  g: number;
  b: number;
  lab: [number, number, number];
  chroma: number;
  score: number;
  n: number;
}

/**
 * Reduce: k distinct hues from the original colours (rgb0), every texel recoloured to its nearest one and its index
 * kept in `pal`. It writes rgb, so Save bakes the reduced colours into the sheet; Erase still brings back rgb0.
 * Returns the palette; every slab changed, so the view refills them all.
 */
export function quantize(core: Core, k: number): PaletteEntry[] {
  k = Math.max(2, Math.min(32, k | 0));
  const pts: number[] = [];
  for (const rec of core.slices) {
    const n = rec.native;
    if (!n) continue;
    const src = n.rgb0 || n.rgb;
    for (let i = 0; i < n.n; i++) pts.push(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
  }
  const N = pts.length / 3;
  if (!N) return [];
  k = Math.min(k, N);

  const B = 24;
  const bins = new Map<number, { n: number; r: number; g: number; b: number; peakC: number; pr: number; pg: number; pb: number }>();
  for (let i = 0; i < N; i++) {
    const r = pts[i * 3], g = pts[i * 3 + 1], b = pts[i * 3 + 2];
    const key = (((r * B) / 256) | 0) << 16 | (((g * B) / 256) | 0) << 8 | (((b * B) / 256) | 0);
    let bin = bins.get(key);
    if (!bin) {
      bin = { n: 0, r: 0, g: 0, b: 0, peakC: -1, pr: r, pg: g, pb: b };
      bins.set(key, bin);
    }
    bin.n++;
    bin.r += r;
    bin.g += g;
    bin.b += b;
    const lab = srgbToOkLab(r, g, b);
    const c = Math.hypot(lab[1], lab[2]);
    if (c > bin.peakC) {
      bin.peakC = c;
      bin.pr = r;
      bin.pg = g;
      bin.pb = b;
    }
  }
  const cands: Cand[] = [];
  for (const bin of bins.values()) {
    const mr = bin.r / bin.n, mg = bin.g / bin.n, mb = bin.b / bin.n;
    const usePeak = bin.peakC > 0.08;
    const r = usePeak ? bin.pr : mr;
    const g = usePeak ? bin.pg : mg;
    const b = usePeak ? bin.pb : mb;
    const lab = srgbToOkLab(r, g, b);
    const chroma = Math.hypot(lab[1], lab[2]);
    const L = lab[0];
    const score = Math.pow(bin.n, 0.28) * (0.12 + chroma) * (0.35 + L);
    cands.push({ r, g, b, lab, chroma, score, n: bin.n });
  }
  cands.sort((a, b) => b.score - a.score);
  const chromaCands = cands.filter((c) => c.chroma >= 0.07).sort((a, b) => b.chroma - a.chroma || b.score - a.score);

  const picked: Cand[] = [];
  const farEnough = (c: Cand, minD2: number) => {
    for (const p of picked) {
      if (labDist2(c.lab, p.lab) < minD2) return false;
    }
    return true;
  };
  const take = (list: Cand[], limit: number, minD2: number) => {
    for (const c of list) {
      if (picked.length >= limit) break;
      if (picked.includes(c)) continue;
      if (farEnough(c, minD2)) picked.push(c);
    }
  };
  const nChroma = Math.min(chromaCands.length, Math.max(1, Math.round(k * 0.4)));
  take(chromaCands, nChroma, 0.028);
  take(cands, k, 0.04);
  if (picked.length < k) take(cands, k, 0.016);
  if (picked.length < k) take(cands, k, 0.006);
  if (picked.length < k) {
    for (const c of cands) {
      if (picked.length >= k) break;
      if (!picked.includes(c)) picked.push(c);
    }
  }

  const palLab = picked.map((c) => c.lab);
  const assign = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const lab = srgbToOkLab(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    let best = 0, bestD = 1e15;
    for (let c = 0; c < picked.length; c++) {
      const d = labDist2(lab, palLab[c]);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    assign[i] = best;
  }

  const palette = picked.map((c) => ({
    r: Math.round(c.r),
    g: Math.round(c.g),
    b: Math.round(c.b),
    metal: 0,
    rough: 0.7,
    emit: c.chroma > 0.12 && c.lab[0] > 0.55 ? 0.35 : 0,
  }));
  core.palette = palette;
  core.palSelected = 0;
  let i = 0;
  for (const rec of core.slices) {
    const n = rec.native;
    if (!n) continue;
    n.pal = new Uint8Array(n.n);
    for (let j = 0; j < n.n; j++, i++) {
      const a = assign[i];
      n.pal[j] = a;
      n.rgb[j * 3] = palette[a].r;
      n.rgb[j * 3 + 1] = palette[a].g;
      n.rgb[j * 3 + 2] = palette[a].b;
    }
  }
  return palette;
}

/** Change one swatch (Metal / Rough / Glow, or its colour): the texels using it follow. Returns the slices touched. */
export function applyPaletteEntry(core: Core, index: number, patch: Partial<PaletteEntry>): number[] {
  const palette = core.palette;
  if (!palette || !palette[index]) return [];
  Object.assign(palette[index], patch);
  const e = palette[index];
  const dirtyList: number[] = [];
  for (const rec of core.slices) {
    const n = rec.native;
    if (!n || !n.pal) continue;
    let dirty = false;
    for (let j = 0; j < n.n; j++) {
      if (n.pal[j] !== index) continue;
      n.rgb[j * 3] = e.r;
      n.rgb[j * 3 + 1] = e.g;
      n.rgb[j * 3 + 2] = e.b;
      dirty = true;
    }
    if (dirty) dirtyList.push(rec.index);
  }
  return dirtyList;
}

/** Metal / rough / glow of one texel: its swatch's after Reduce, else the material's own (0, 0.7, 0). */
export function pbrOf(core: Core, native: Native, i: number): [number, number, number] {
  const palette = core.palette;
  if (palette && native.pal && native.pal[i] < palette.length) {
    const e = palette[native.pal[i]];
    return [e.metal, e.rough, e.emit];
  }
  return [0, 0.7, 0];
}

/* ---------------------------------------------------------------- paint */

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

/** A stroke starts: remember every slab's colours for Undo. */
export function beginStroke(core: Core): void {
  core.strokeUndo = core.slices.map((rec) => (rec.native ? new Uint8Array(rec.native.rgb) : null));
}

/** Undo the last stroke. Returns false when there is none (one level, as legacy). */
export function undoStroke(core: Core): boolean {
  const undo = core.strokeUndo;
  if (!undo) return false;
  core.slices.forEach((rec, i) => {
    if (!rec.native || !undo[i]) return;
    rec.native.rgb.set(undo[i]!);
  });
  core.strokeUndo = null;
  return true;
}

/**
 * Colour every texel within `radius` of `center` (world space, before Spacing pulls slabs apart it is the slab's own
 * height), or with `erase` put back its rgb0. After Reduce the brush uses the selected swatch. Returns the count
 * painted and the slices to refill.
 */
export function paintSphere(core: Core, center: Vec3Like, radius: number, color: number[], erase: boolean): { painted: number; dirty: number[] } {
  const { meta, state } = core;
  if (!meta || !core.slices.length) return { painted: 0, dirty: [] };
  const gap = meta.pitch * Math.max(0.55, state.spacing);
  const world = meta.resolution * meta.pitch;
  const r2 = radius * radius;
  const palette = core.palette;
  let painted = 0;
  const dirtyList: number[] = [];
  for (const rec of core.slices) {
    const native = rec.native;
    if (!native) continue;
    const y = rec.index * gap;
    let dirty = false;
    for (let i = 0; i < native.n; i++) {
      const x = ((native.ix[i] + 0.5) / meta.resolution - 0.5) * world;
      const z = ((native.iz[i] + 0.5) / meta.resolution - 0.5) * world;
      const dx = x - center.x;
      const dy = y - center.y;
      const dz = z - center.z;
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      if (erase && native.rgb0) {
        native.rgb[i * 3] = native.rgb0[i * 3];
        native.rgb[i * 3 + 1] = native.rgb0[i * 3 + 1];
        native.rgb[i * 3 + 2] = native.rgb0[i * 3 + 2];
      } else if (!erase) {
        let rgb = color;
        if (palette && palette[core.palSelected]) {
          const e = palette[core.palSelected];
          rgb = [e.r, e.g, e.b];
          if (native.pal) native.pal[i] = core.palSelected;
        }
        native.rgb[i * 3] = rgb[0];
        native.rgb[i * 3 + 1] = rgb[1];
        native.rgb[i * 3 + 2] = rgb[2];
      }
      dirty = true;
      painted++;
    }
    if (dirty) dirtyList.push(rec.index);
  }
  return { painted, dirty: dirtyList };
}

/** Drop: the colour of the texel nearest `center`. */
export function sampleVoxel(core: Core, center: Vec3Like): [number, number, number] | null {
  const { meta, state } = core;
  if (!meta) return null;
  const gap = meta.pitch * Math.max(0.55, state.spacing);
  const world = meta.resolution * meta.pitch;
  let best: [number, number, number] | null = null;
  let bestD = Infinity;
  for (const rec of core.slices) {
    const native = rec.native;
    if (!native) continue;
    const y = rec.index * gap;
    for (let i = 0; i < native.n; i++) {
      const x = ((native.ix[i] + 0.5) / meta.resolution - 0.5) * world;
      const z = ((native.iz[i] + 0.5) / meta.resolution - 0.5) * world;
      const dx = x - center.x;
      const dy = y - center.y;
      const dz = z - center.z;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = [native.rgb[i * 3], native.rgb[i * 3 + 1], native.rgb[i * 3 + 2]];
      }
    }
  }
  return best;
}

/**
 * The sheet Save writes (legacy getSheetBlob): (res*layers)×res RGBA, slab i at x = i*res, bottom slab first, every
 * texel at native resolution and full alpha. Main, Texels and Layers only change the view, so none of them apply.
 */
export function sheetPixels(core: Core): { width: number; height: number; data: Uint8ClampedArray<ArrayBuffer> } {
  const res = core.meta.resolution;
  const width = res * core.meta.layers;
  const data = new Uint8ClampedArray(width * res * 4);
  for (const rec of core.slices) {
    const native = rec.native;
    if (!native) continue;
    for (let i = 0; i < native.n; i++) {
      const p = (native.iz[i] * width + rec.index * res + native.ix[i]) * 4;
      data[p] = native.rgb[i * 3];
      data[p + 1] = native.rgb[i * 3 + 1];
      data[p + 2] = native.rgb[i * 3 + 2];
      data[p + 3] = 255;
    }
  }
  return { width, height: res, data };
}

/** The world box of the whole stack at rest, for the brush radius (diagonal) and the framing. */
export function restBox(meta: StackMeta): { min: [number, number, number]; max: [number, number, number]; diagonal: number } {
  const worldX = meta.extents[0];
  const worldZ = meta.extents[2];
  const restH = meta.layers * meta.pitch;
  const min: [number, number, number] = [-worldX * 0.5, 0, -worldZ * 0.5];
  const max: [number, number, number] = [worldX * 0.5, restH, worldZ * 0.5];
  return { min, max, diagonal: Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) };
}
