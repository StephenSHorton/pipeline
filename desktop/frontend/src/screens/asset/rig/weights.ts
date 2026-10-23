// The Weights panel's logic: how much of the mesh each bone moves (skin weights), edited by hand. No three, no DOM:
// typed arrays in, typed arrays out, so weights.test.ts runs it bare.
//
// A skinned mesh keeps four bones per point (skinIndex / skinWeight, glTF JOINTS_0 / WEIGHTS_0), adding up to 1.
// WeightEdit works on those two arrays in place (the panel hands it the mesh's own attribute arrays, so a clip
// playing shows every change at once) and keeps every point's row normalised and at most four bones:
// - grab: scale one bone's weights by a factor (the Grab slider, 0-200%), the other bones of each point share the
//   rest in proportion. A point the bone moved alone hands what it loses to the bone's parent (there is no one else).
// - brush: Add pulls a bone's influence in, Erase pushes it out (to the point's other bones, else the parent),
//   Smooth evens the weights out with the neighbouring points. Each dab falls off from the centre.
// - a bone a point doesn't have yet replaces the point's smallest of its four.
// - corners at the same position are one point (flat-shaded copies, UV seams): every edit gives them the same row,
//   so the mesh never tears there.
// - undo / redo per stroke or slider drag, and Reset back to the weights the panel started from.
//
// Bone names and the chains the list groups them in come from the skeleton (names and parents); inspect.json's
// labels are used when they are there.

/* ------------------------------------------------------------------ copy */

export const WEIGHTS_COPY = {
  hint: "Pick a bone, or click the model on the part you want. Warm is moved a lot by the bone, blue not at all.",
  heat: { label: "Heat", title: "How much the chosen bone moves each part: warm moves a lot, blue not at all" },
  parts: { label: "All bones", title: "Every part in the colour of the bone that moves it most" },
  grab: { label: "Grab", title: "How much of the mesh this bone moves, against the saved rig. Below 100% it lets go, above it pulls more" },
  add: { label: "Add", title: "Paint on the model to make the chosen bone move more of it" },
  erase: { label: "Erase", title: "Paint on the model to make the chosen bone move less of it" },
  smooth: { label: "Smooth", title: "Paint on the model to even out the weights between neighbouring parts" },
  armedHint: "Drag on the model to paint. Right-drag turns the view. Alt-click picks the bone under the pointer.",
  size: { label: "Size", title: "How wide the brush is" },
  strength: { label: "Strength", title: "How much one pass of the brush changes" },
  undo: { label: "Undo", title: "Undo the last stroke or slider change" },
  redo: { label: "Redo", title: "Redo what Undo took back" },
  reset: { label: "Reset", title: "Back to the saved weights" },
  save: { label: "Save", title: "Save these weights into the rig (a new rig version) and export the FBX again" },
  saving: "Saving…",
  noRig: "Rig the model first: the weights come with the rig.",
  noSkin: "Loading the rigged model…",
  saved: "Saved",
  bones: "Bones",
} as const;

export function changedText(points: number): string {
  if (!points) return "No changes";
  return `${points} point${points === 1 ? "" : "s"} changed, not saved`;
}

export function savedToast(out: { changed?: number; fbx_note?: string; blender?: boolean }): string {
  const n = out.changed ?? 0;
  const head = `Weights saved (${n} point${n === 1 ? "" : "s"} changed)`;
  if (out.blender) return `${head}. The FBX was exported again.`;
  return out.fbx_note ? `${head}. ${out.fbx_note}` : `${head}.`;
}

export type BrushMode = "add" | "erase" | "smooth";
export type ShowMode = "heat" | "parts";

export const BRUSH_SIZE = { min: 0.02, max: 0.5, step: 0.01, value: 0.12 } as const; // of the model's size
export const BRUSH_STRENGTH = { min: 0.05, max: 1, step: 0.05, value: 0.35 } as const;
export const GRAB = { min: 0, max: 200, step: 5 } as const;
/** A weight smaller than this is dropped from a point after an edit (and the rest renormalised). */
export const MIN_WEIGHT = 1e-3;

/* ------------------------------------------------------------------ the brush falloff */

/** How much a dab at distance d of a brush of radius r reaches: 1 at the centre, 0 at the rim, smooth both ends. */
export function falloff(d: number, r: number): number {
  if (!(r > 0) || d >= r) return 0;
  const t = 1 - (d * d) / (r * r);
  return t * t;
}

/* ------------------------------------------------------------------ rows */

/** One point's four bones and weights, scratch space for an edit. */
type Row = { j: number[]; w: number[] };

function readRow(idx: ArrayLike<number>, w: ArrayLike<number>, v: number): Row {
  const o = v * 4;
  return { j: [idx[o], idx[o + 1], idx[o + 2], idx[o + 3]], w: [w[o], w[o + 1], w[o + 2], w[o + 3]] };
}

/** A row as {bone: weight}, zero weights left out, the same bone twice summed. */
function toMap(r: Row): Map<number, number> {
  const m = new Map<number, number>();
  for (let k = 0; k < 4; k++) if (r.w[k] > 0) m.set(r.j[k], (m.get(r.j[k]) || 0) + r.w[k]);
  return m;
}

/**
 * The four biggest of `m`, renormalised to 1, as a row. `keep` (the bone being painted) is never the one dropped:
 * a bone new to a full point replaces the smallest other. Weights under MIN_WEIGHT go. An empty map stays as `fallback`.
 */
export function topFour(m: Map<number, number>, keep = -1, fallback?: Row): Row {
  let e = [...m.entries()].filter(([, x]) => x >= MIN_WEIGHT && Number.isFinite(x));
  e.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  if (e.length > 4) {
    const k = keep >= 0 ? e.findIndex(([j]) => j === keep) : -1;
    if (k >= 4) e = [...e.slice(0, 3), e[k]];
    else e = e.slice(0, 4);
  }
  const s = e.reduce((a, [, x]) => a + x, 0);
  if (!(s > 0)) return fallback ? { j: [...fallback.j], w: [...fallback.w] } : { j: [0, 0, 0, 0], w: [1, 0, 0, 0] };
  // The painted bone keeps the share it was given; what a dropped bone held goes to the other three.
  const kept = keep >= 0 ? Math.min(1, e.find(([j]) => j === keep)?.[1] ?? 0) : 0;
  const others = s - kept;
  const out: Row = { j: [0, 0, 0, 0], w: [0, 0, 0, 0] };
  e.forEach(([j, x], k) => {
    out.j[k] = j;
    out.w[k] = j === keep && kept > 0 ? (others > 1e-9 ? kept : 1) : others > 1e-9 && kept > 0 ? (x * (1 - kept)) / others : x / s;
  });
  return out;
}

/**
 * Bone j's weight at a point becomes `target` (0..1); the point's other bones share 1 - target in proportion.
 * When nothing else holds the point and j lets go, the rest goes to `parent` (no parent: the point stays as it is).
 */
export function setShare(m: Map<number, number>, j: number, target: number, parent: number): Map<number, number> {
  target = Math.min(1, Math.max(0, target));
  const cur = m.get(j) || 0;
  const out = new Map(m);
  const rest = [...m.entries()].reduce((a, [b, x]) => a + (b === j ? 0 : x), 0);
  if (rest > 1e-9) {
    const f = (1 - target) / rest;
    for (const [b, x] of m) if (b !== j) out.set(b, x * f);
  } else if (target < cur) {
    if (parent < 0 || parent === j) return out;
    out.set(parent, 1 - target);
  }
  if (target > 0) out.set(j, target);
  else out.delete(j);
  return out;
}

/* ------------------------------------------------------------------ corners and neighbours */

/** Points at the same position (to 1e-5) as one: group[v] is v's point, members[g] its corners. */
export function cornerGroups(pos: ArrayLike<number>, n: number): { group: Int32Array; members: number[][] } {
  const group = new Int32Array(n);
  const members: number[][] = [];
  const seen = new Map<string, number>();
  for (let v = 0; v < n; v++) {
    const k = `${Math.round(pos[v * 3] * 1e5)},${Math.round(pos[v * 3 + 1] * 1e5)},${Math.round(pos[v * 3 + 2] * 1e5)}`;
    let g = seen.get(k);
    if (g === undefined) {
      g = members.length;
      seen.set(k, g);
      members.push([]);
    }
    group[v] = g;
    members[g].push(v);
  }
  return { group, members };
}

/** Which points share an edge (across corner copies), from the triangles (an index buffer, or every 3 in order). */
export function neighbours(group: Int32Array, groups: number, index: ArrayLike<number> | null, n: number): number[][] {
  const sets: Set<number>[] = Array.from({ length: groups }, () => new Set<number>());
  const tri = index ? index.length : n - (n % 3);
  const at = (k: number) => (index ? index[k] : k);
  for (let t = 0; t + 2 < tri; t += 3) {
    const a = group[at(t)];
    const b = group[at(t + 1)];
    const c = group[at(t + 2)];
    sets[a].add(b).add(c);
    sets[b].add(a).add(c);
    sets[c].add(a).add(b);
  }
  return sets.map((s, g) => [...s].filter((x) => x !== g));
}

/* ------------------------------------------------------------------ the edit */

interface Patch {
  verts: Int32Array;
  before: { j: Uint16Array; w: Float32Array };
  after: { j: Uint16Array; w: Float32Array };
}

export interface WeightMesh {
  /** skinIndex, 4 per point (edited in place). */
  idx: Uint16Array;
  /** skinWeight, 4 per point (edited in place). */
  w: Float32Array;
  /** Bind-pose positions, 3 per point: corners at the same position are one point. */
  pos: ArrayLike<number>;
  /** The triangles (for Smooth's neighbours), or null for a non-indexed mesh. */
  index: ArrayLike<number> | null;
}

/** One stroke, slider drag or reset being recorded: each point's row before it was first touched. */
class Recorder {
  private seen = new Map<number, number[]>();
  constructor(private e: WeightEdit) {}
  touch(v: number): void {
    if (this.seen.has(v)) return;
    const o = v * 4;
    const { idx, w } = this.e;
    this.seen.set(v, [idx[o], idx[o + 1], idx[o + 2], idx[o + 3], w[o], w[o + 1], w[o + 2], w[o + 3]]);
  }
  patch(): Patch | null {
    const { idx, w } = this.e;
    const verts: number[] = [];
    for (const [v, b] of this.seen) {
      const o = v * 4;
      let same = true;
      for (let k = 0; k < 4 && same; k++) same = b[k] === idx[o + k] && Math.abs(b[4 + k] - w[o + k]) < 1e-7;
      if (!same) verts.push(v);
    }
    if (!verts.length) return null;
    const p: Patch = {
      verts: Int32Array.from(verts),
      before: { j: new Uint16Array(verts.length * 4), w: new Float32Array(verts.length * 4) },
      after: { j: new Uint16Array(verts.length * 4), w: new Float32Array(verts.length * 4) },
    };
    verts.forEach((v, i) => {
      const b = this.seen.get(v)!;
      for (let k = 0; k < 4; k++) {
        p.before.j[i * 4 + k] = b[k];
        p.before.w[i * 4 + k] = b[4 + k];
        p.after.j[i * 4 + k] = idx[v * 4 + k];
        p.after.w[i * 4 + k] = w[v * 4 + k];
      }
    });
    return p;
  }
}

export class WeightEdit {
  idx: Uint16Array;
  w: Float32Array;
  readonly n: number;
  readonly group: Int32Array;
  readonly members: number[][];
  readonly near: number[][];
  /** Each bone's parent (-1 for the root), for the weight a lone bone lets go of. */
  readonly parents: number[];
  private saved: { j: Uint16Array; w: Float32Array };
  private savedTotals: Float64Array;
  private undoStack: Patch[] = [];
  private redoStack: Patch[] = [];
  private rec: Recorder | null = null;
  private snap: { j: Uint16Array; w: Float32Array } | null = null;
  /** Called after every change with nothing (the panel repaints and flags the attributes). */
  onChange: () => void = () => {};

  private bind: ArrayLike<number>;

  constructor(mesh: WeightMesh, parents: number[]) {
    this.bind = mesh.pos;
    this.idx = mesh.idx;
    this.w = mesh.w;
    this.n = mesh.w.length / 4;
    this.parents = parents;
    const { group, members } = cornerGroups(mesh.pos, this.n);
    this.group = group;
    this.members = members;
    this.near = neighbours(group, members.length, mesh.index, this.n);
    this.saved = { j: this.idx.slice(), w: this.w.slice() };
    this.savedTotals = this.totals(this.saved.j, this.saved.w);
  }

  /* ---------------------------------------------------------------- reading */

  /** The bind-pose positions (3 per point): the brush's space when nothing is playing. */
  pos(): ArrayLike<number> {
    return this.bind;
  }

  weightOf(v: number, j: number): number {
    let s = 0;
    for (let k = 0; k < 4; k++) if (this.idx[v * 4 + k] === j) s += this.w[v * 4 + k];
    return s;
  }

  /** The bone that moves a point most. */
  topBone(v: number): number {
    let b = this.idx[v * 4];
    let x = -1;
    for (let k = 0; k < 4; k++) {
      if (this.w[v * 4 + k] > x) {
        x = this.w[v * 4 + k];
        b = this.idx[v * 4 + k];
      }
    }
    return b;
  }

  private totals(j: ArrayLike<number>, w: ArrayLike<number>): Float64Array {
    let max = 0;
    for (let i = 0; i < j.length; i++) max = Math.max(max, j[i]);
    for (const p of this.parents.keys()) max = Math.max(max, p);
    const t = new Float64Array(max + 1);
    for (let i = 0; i < j.length; i++) t[j[i]] += w[i];
    return t;
  }

  /** Each bone's grab against the saved weights, in percent (100 = as saved; a bone that had none: 0, or 100 once painted in). */
  grabPercent(j: number): number {
    const now = this.totals(this.idx, this.w)[j] || 0;
    const was = this.savedTotals[j] || 0;
    if (was < 1e-9) return now > 1e-9 ? 100 : 0;
    return (100 * now) / was;
  }

  /** Every bone's total weight as saved, indexed by bone. */
  savedTotalsByBone(): Float64Array {
    return this.savedTotals;
  }

  /** Every bone's total weight now (the bone list's "has any"), indexed by bone. */
  boneTotals(): Float64Array {
    return this.totals(this.idx, this.w);
  }

  /** How many points differ from the saved weights. */
  changedPoints(): number {
    let n = 0;
    for (let v = 0; v < this.n; v++) {
      const a = toMap(readRow(this.idx, this.w, v));
      const b = toMap(readRow(this.saved.j, this.saved.w, v));
      let diff = a.size !== b.size;
      for (const [j, x] of a) if (diff || Math.abs((b.get(j) ?? -1) - x) > 1e-4) diff = true;
      if (diff) n++;
    }
    return n;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  /* ---------------------------------------------------------------- writing */

  private write(v: number, r: Row): void {
    this.rec?.touch(v);
    const o = v * 4;
    for (let k = 0; k < 4; k++) {
      this.idx[o + k] = r.j[k];
      this.w[o + k] = r.w[k];
    }
  }

  /** A corner point's row: the mean of its copies (they should be the same; a split one is mended). */
  private pointMap(g: number, src?: { j: ArrayLike<number>; w: ArrayLike<number> }): Map<number, number> {
    const vs = this.members[g];
    const m = new Map<number, number>();
    for (const v of vs) for (const [j, x] of toMap(readRow(src?.j ?? this.idx, src?.w ?? this.w, v))) m.set(j, (m.get(j) || 0) + x / vs.length);
    return m;
  }

  private writePoint(g: number, m: Map<number, number>, keep = -1): void {
    const vs = this.members[g];
    const r = topFour(m, keep, readRow(this.idx, this.w, vs[0]));
    for (const v of vs) this.write(v, r);
  }

  /** Start recording one undo step (a stroke, a slider drag). Ends with end(). */
  begin(): void {
    if (!this.rec) this.rec = new Recorder(this);
  }

  /** Close the step: an undo entry when anything changed. */
  end(): boolean {
    const p = this.rec?.patch() ?? null;
    this.rec = null;
    this.snap = null;
    if (!p) return false;
    this.undoStack.push(p);
    if (this.undoStack.length > 100) this.undoStack.shift();
    this.redoStack = [];
    return true;
  }

  /**
   * One brush dab: every point within `radius` of `center` (in the same space as `posed`, 3 per point: where the
   * points are on screen now) changes by strength x falloff. Add and Erase change bone `bone`; Smooth all bones.
   */
  dab(mode: BrushMode, bone: number, posed: ArrayLike<number>, center: readonly [number, number, number], radius: number, strength: number): number {
    const own = !this.rec;
    if (own) this.begin();
    const reach: Array<[number, number]> = [];
    for (let g = 0; g < this.members.length; g++) {
      const v = this.members[g][0];
      const d = Math.hypot(posed[v * 3] - center[0], posed[v * 3 + 1] - center[1], posed[v * 3 + 2] - center[2]);
      const a = falloff(d, radius) * strength;
      if (a > 1e-4) reach.push([g, Math.min(1, a)]);
    }
    if (mode === "smooth") {
      // From the weights before this dab, so the order the points are visited in doesn't matter.
      const before = reach.map(([g]) => this.pointMap(g));
      const avg = reach.map(([g]) => {
        const m = new Map<number, number>();
        const ring = [g, ...this.near[g]];
        for (const h of ring) for (const [j, x] of this.pointMap(h)) m.set(j, (m.get(j) || 0) + x / ring.length);
        return m;
      });
      reach.forEach(([g, a], i) => {
        const m = new Map<number, number>();
        for (const j of new Set([...before[i].keys(), ...avg[i].keys()])) m.set(j, (1 - a) * (before[i].get(j) || 0) + a * (avg[i].get(j) || 0));
        this.writePoint(g, m);
      });
    } else {
      const parent = this.parents[bone] ?? -1;
      for (const [g, a] of reach) {
        const m = this.pointMap(g);
        const cur = m.get(bone) || 0;
        const target = mode === "add" ? cur + a * (1 - cur) : cur * (1 - a);
        if (mode === "erase" && cur === 0) continue;
        this.writePoint(g, setShare(m, bone, target, parent), mode === "add" ? bone : -1);
      }
    }
    if (own) this.end();
    if (reach.length) this.onChange();
    return reach.length;
  }

  /**
   * The Grab slider: bone j's weight everywhere it has some times `factor`, from the weights when the drag began
   * (so dragging back and forth lands where the slider is, not compounded). Call begin() at the drag's start and
   * end() at its release; between them call grab() as often as the slider moves.
   */
  grab(j: number, factor: number): void {
    const own = !this.rec;
    if (own) this.begin();
    if (!this.snap) this.snap = { j: this.idx.slice(), w: this.w.slice() };
    const src = this.snap;
    const parent = this.parents[j] ?? -1;
    for (let g = 0; g < this.members.length; g++) {
      const vs = this.members[g];
      let has = false;
      for (const v of vs) for (let k = 0; k < 4; k++) if (src.j[v * 4 + k] === j && src.w[v * 4 + k] > 0) has = true;
      if (!has) {
        // Back to how it was at the drag's start, in case an earlier tick of this drag changed it.
        for (const v of vs) {
          let same = true;
          for (let k = 0; k < 4; k++) same &&= this.idx[v * 4 + k] === src.j[v * 4 + k] && this.w[v * 4 + k] === src.w[v * 4 + k];
          if (!same) this.write(v, readRow(src.j, src.w, v));
        }
        continue;
      }
      const m = this.pointMap(g, src);
      this.writePoint(g, setShare(m, j, (m.get(j) || 0) * Math.max(0, factor), parent), j);
    }
    if (own) this.end();
    this.onChange();
  }

  /** Grab to a percentage against the saved weights (the slider's value), from the drag's starting weights. */
  grabTo(j: number, percent: number, startPercent: number): void {
    const f = startPercent > 1e-6 ? percent / startPercent : percent > 0 ? 1 : 0;
    this.grab(j, f);
  }

  private apply(p: Patch, side: "before" | "after"): void {
    const s = p[side];
    p.verts.forEach((v, i) => {
      for (let k = 0; k < 4; k++) {
        this.idx[v * 4 + k] = s.j[i * 4 + k];
        this.w[v * 4 + k] = s.w[i * 4 + k];
      }
    });
    this.onChange();
  }

  undo(): boolean {
    this.rec = null;
    const p = this.undoStack.pop();
    if (!p) return false;
    this.apply(p, "before");
    this.redoStack.push(p);
    return true;
  }

  redo(): boolean {
    const p = this.redoStack.pop();
    if (!p) return false;
    this.apply(p, "after");
    this.undoStack.push(p);
    return true;
  }

  /** Back to the saved weights (one undo step, so Undo brings the edit back). */
  reset(): boolean {
    this.begin();
    for (let v = 0; v < this.n; v++) {
      const o = v * 4;
      let same = true;
      for (let k = 0; k < 4; k++) same &&= this.idx[o + k] === this.saved.j[o + k] && this.w[o + k] === this.saved.w[o + k];
      if (!same) this.write(v, readRow(this.saved.j, this.saved.w, v));
    }
    const any = this.end();
    if (any) this.onChange();
    return any;
  }

  /** The weights now are what is saved (after Save). Undo stays. */
  markSaved(): void {
    this.saved = { j: this.idx.slice(), w: this.w.slice() };
    this.savedTotals = this.totals(this.saved.j, this.saved.w);
  }

  /** A reloaded model of the same mesh: carry the edit over into its arrays and edit those from now on. */
  adopt(idx: Uint16Array, w: Float32Array): void {
    idx.set(this.idx);
    w.set(this.w);
    this.idx = idx;
    this.w = w;
    this.onChange();
  }
}

/* ------------------------------------------------------------------ bones: chains and plain names */

export interface BoneRow {
  i: number;
  name: string;
}

export interface BoneChain {
  key: string;
  label: string;
  bones: BoneRow[];
}

const CHAIN_ORDER = ["spine", "neck", "tail", "fl", "fr", "bl", "br", "al", "ar", "ll", "lr", "wl", "wr", "other"];
const CHAIN_LABEL: Record<string, string> = {
  spine: "Spine", neck: "Neck & head", tail: "Tail", fl: "Front left leg", fr: "Front right leg", bl: "Back left leg",
  br: "Back right leg", al: "Left arm", ar: "Right arm", ll: "Left leg", lr: "Right leg", wl: "Left wing", wr: "Right wing", other: "Other",
};
const HEADISH = new Set(["head", "neck", "ear", "nose", "mouth", "jaw", "eye", "beak", "horn", "tongue", "skull", "snout", "chin", "lip", "brow", "antler", "mane"]);
const ARMISH = new Set(["arm", "hand", "shoulder", "clavicle", "elbow", "wrist", "finger", "thumb", "forearm", "upperarm"]);
const LEGISH = new Set(["leg", "foot", "toe", "thigh", "shin", "calf", "knee", "hoof", "ankle", "femur", "tibia", "humerus", "scapula", "pelvis", "paw", "heel"]);
const FRONT = new Set(["front", "fore", "forelimb", "foreleg"]);
const BACK = new Set(["back", "hind", "rear", "hindlimb", "hindleg"]);
const LEAF = /leaf|tip|_end$|end$/i;
const ADJ = new Set(["upper", "lower", "mid", "middle", "inner", "outer", "top", "bottom"]);

/** A bone name in lower-case words: "back_leg_upper_l" -> back leg upper l; "UpperLeg_L" -> upper leg l. */
export function boneWords(name: string): string[] {
  return name
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .split(/[\s_.\-:|]+/)
    .map((x) => x.toLowerCase())
    .filter(Boolean);
}

function sideOfWords(ws: string[]): "l" | "r" | "" {
  if (ws.some((x) => x === "l" || x === "left")) return "l";
  if (ws.some((x) => x === "r" || x === "right")) return "r";
  return "";
}

const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/**
 * The bone list: bones grouped by chain (Spine, Neck & head, Tail, Front left leg, …) with plain names. `labels`
 * (inspect.json bones[].label by bone name) is used when it is there. A spine bone with the head below it and no
 * limb is the neck. Leaf bones (the ends, which move nothing) are left out unless `weighted` says they hold weight.
 */
export function boneChains(names: string[], parents: number[], opts: { labels?: Record<string, string>; roles?: Record<string, string>; weighted?: (i: number) => boolean } = {}): BoneChain[] {
  const words = names.map(boneWords);
  const role = (i: number) => opts.roles?.[names[i]] || "";
  const isHead = (i: number) => role(i) === "head" || words[i].some((x) => HEADISH.has(x) && x !== "neck") || words[i].includes("neck");
  const isLimb = (i: number) => role(i) === "leg" || role(i) === "wing" || words[i].some((x) => LEGISH.has(x) || ARMISH.has(x) || x === "wing");
  const kids: number[][] = names.map(() => []);
  parents.forEach((p, i) => {
    if (p >= 0 && p < names.length) kids[p].push(i);
  });
  const below = (i: number, test: (k: number) => boolean): boolean => kids[i].some((k) => test(k) || below(k, test));
  const chainOf = (i: number): string => {
    const ws = words[i];
    const r = role(i);
    if (r === "tail" || ws.includes("tail")) return "tail";
    if (r === "head" || isHead(i)) return "neck";
    const side = sideOfWords(ws);
    if (r === "wing" || ws.includes("wing")) return side ? `w${side}` : "other";
    if (r === "leg" || ws.some((x) => LEGISH.has(x) || ARMISH.has(x))) {
      if (!side) return "other";
      if (ws.some((x) => FRONT.has(x))) return `f${side}`;
      if (ws.some((x) => BACK.has(x))) return `b${side}`;
      if (ws.some((x) => ARMISH.has(x))) return `a${side}`;
      return `l${side}`;
    }
    if (r === "spine" || r === "root" || /spine|hips|pelvis|chest|ribcage|torso|body|root|waist|belly/.test(ws.join(" "))) {
      // The horse template's last spine bones are the neck: the head hangs below them and no leg does.
      if (/spine|neck/.test(ws.join(" ")) && below(i, isHead) && !below(i, isLimb)) return "neck";
      return "spine";
    }
    return "other";
  };
  const plain = (i: number, chain: string): string => {
    const lab = opts.labels?.[names[i]];
    let ws = (lab ? lab.toLowerCase().split(/\s+/) : words[i]).filter((x) => !/^\d+$/.test(x));
    const side = sideOfWords(ws);
    // In a side's chain the side and front/back are in the chain's name already; the head keeps "left ear".
    if (chain !== "neck" || !side) ws = ws.filter((x) => !["l", "r", "left", "right"].includes(x));
    else ws = [side === "l" ? "left" : "right", ...ws.filter((x) => !["l", "r", "left", "right"].includes(x))];
    if (/^[fb][lr]$/.test(chain)) ws = ws.filter((x) => !FRONT.has(x) && !BACK.has(x));
    if (/^[fblaw][lr]$/.test(chain) && ws.length > 1) ws = ws.filter((x) => x !== "leg" || ws.indexOf("upper") >= 0 || ws.indexOf("lower") >= 0);
    // "leg upper" (the template's word order) reads "upper leg".
    const adj = ws.filter((x) => ADJ.has(x));
    if (adj.length && adj.length < ws.length) ws = [...adj, ...ws.filter((x) => !ADJ.has(x))];
    if (chain === "neck" && ws.join(" ") === "spine") ws = ["neck"];
    if (chain === "tail" && !ws.length) ws = ["tail"];
    return cap(ws.join(" ") || names[i]);
  };
  const byChain = new Map<string, BoneRow[]>();
  names.forEach((n, i) => {
    if (LEAF.test(n) && !(opts.weighted?.(i) ?? false)) return;
    if ((role(i) === "root" || /^root$/i.test(n)) && !(opts.weighted?.(i) ?? false)) return;
    const c = chainOf(i);
    if (!byChain.has(c)) byChain.set(c, []);
    byChain.get(c)!.push({ i, name: plain(i, c) });
  });
  const out: BoneChain[] = [];
  for (const key of CHAIN_ORDER) {
    const rows = byChain.get(key);
    if (!rows?.length) continue;
    // The same plain name twice in a chain (Spine, Spine, Spine): number them in skeleton order.
    const count: Record<string, number> = {};
    for (const r of rows) count[r.name] = (count[r.name] || 0) + 1;
    const seen: Record<string, number> = {};
    for (const r of rows) if (count[r.name] > 1) r.name = `${r.name} ${(seen[r.name] = (seen[r.name] || 0) + 1)}`;
    out.push({ key, label: CHAIN_LABEL[key], bones: rows });
  }
  return out;
}

/* ------------------------------------------------------------------ colours */

/** The heat ramp: 0 a cold blue, then blue -> cyan -> green -> yellow -> red at 1 (as the Rig check's heat). */
export function heatRgb(w: number): [number, number, number] {
  if (!(w > 0)) return [0.05, 0.05, 0.35];
  const t = Math.min(1, w);
  return hslToRgb((1 - t) * 0.6667, 1, 0.5);
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** Bone colours, golden-ratio hues: the same as the Rig check's bonePalette (overlay.ts). */
export function boneRgb(i: number): [number, number, number] {
  return hslToRgb((i * 0.618034) % 1, 0.62, 0.55);
}

/** Fill `out` (3 per point) with the chosen bone's heat, or every point in its top bone's colour. */
export function paintColours(e: WeightEdit, mode: ShowMode, bone: number, out: Float32Array): void {
  for (let v = 0; v < e.n; v++) {
    let c: [number, number, number];
    if (mode === "parts") {
      const t = e.topBone(v);
      c = boneRgb(t);
      if (bone >= 0 && t !== bone) c = [c[0] * 0.55 + 0.12, c[1] * 0.55 + 0.12, c[2] * 0.55 + 0.13];
    } else c = bone >= 0 ? heatRgb(e.weightOf(v, bone)) : [0.27, 0.27, 0.3];
    out[v * 3] = c[0];
    out[v * 3 + 1] = c[1];
    out[v * 3 + 2] = c[2];
  }
}

/* ------------------------------------------------------------------ Save's body */

export interface WeightPart {
  mesh: number;
  primitive: number;
  idx: Uint16Array;
  w: Float32Array;
}

function b64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Little-endian bytes of a typed array (every platform the app runs on is little-endian; this makes sure). */
function le(a: Uint16Array | Float32Array): Uint8Array {
  const out = new Uint8Array(a.length * a.BYTES_PER_ELEMENT);
  const dv = new DataView(out.buffer);
  if (a instanceof Uint16Array) a.forEach((x, i) => dv.setUint16(i * 2, x, true));
  else a.forEach((x, i) => dv.setFloat32(i * 4, x, true));
  return out;
}

/** POST /rig/<stem>/weights's body (scripts/rig_weights.py): every point's four bones and weights, base64. */
export function weightsBody(parts: WeightPart[], note = ""): { primitives: Array<{ mesh: number; primitive: number; count: number; joints: string; weights: string }>; note: string } {
  return {
    primitives: parts.map((p) => ({ mesh: p.mesh, primitive: p.primitive, count: p.w.length / 4, joints: b64(le(p.idx)), weights: b64(le(p.w)) })),
    note,
  };
}

export const weightsPath = (stem: string) => `/rig/${encodeURIComponent(stem)}/weights`;
