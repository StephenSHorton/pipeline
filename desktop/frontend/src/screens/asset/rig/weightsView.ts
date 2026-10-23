// The Weights panel on the model: the skinned mesh(es) on screen, their skin weights edited in place (./weights.ts
// WeightEdit on the mesh's own skinIndex / skinWeight arrays, so a playing clip shows each change at once), the heat
// or territory colours, the brush's reach in posed (on-screen) positions, and Save's body.
//
// The session outlives the panel: closing it takes the colours off but keeps the edit on the mesh (play a clip and
// look), and a reload of the same mesh (a flow refresh, the all-clips file) carries an unsaved edit over.
import {
  BufferAttribute,
  DoubleSide,
  Float32BufferAttribute,
  MeshLambertMaterial,
  Uint16BufferAttribute,
  Vector3,
  type Material,
  type Object3D,
  type SkinnedMesh,
} from "three";
import type { ToolHit } from "@/viewer/tools";
import { paintColours, WeightEdit, weightsBody, type BrushMode, type ShowMode } from "./weights";

export interface WeightPartView {
  mesh: SkinnedMesh;
  edit: WeightEdit;
  /** The glTF mesh and primitive it was loaded from (loadGlb tags userData.gltfPrimitive). */
  ref: [number, number];
  saved: { material: Material | Material[]; color: BufferAttribute | null };
  colors: BufferAttribute;
}

const v3 = new Vector3();

/** skinIndex as Uint16 and skinWeight as plain float32, so the edit can write them in place. */
function editableArrays(mesh: SkinnedMesh): { idx: Uint16Array; w: Float32Array } {
  const geo = mesh.geometry;
  const si = geo.getAttribute("skinIndex") as BufferAttribute;
  const sw = geo.getAttribute("skinWeight") as BufferAttribute;
  const n = geo.attributes.position.count;
  let idx = si.array as Uint16Array;
  if (!(si.array instanceof Uint16Array) || (si as { isInterleavedBufferAttribute?: boolean }).isInterleavedBufferAttribute) {
    idx = new Uint16Array(n * 4);
    for (let v = 0; v < n; v++) for (let k = 0; k < 4; k++) idx[v * 4 + k] = si.getComponent(v, k);
    geo.setAttribute("skinIndex", new Uint16BufferAttribute(idx, 4));
  }
  let w = sw.array as Float32Array;
  if (!(sw.array instanceof Float32Array) || sw.normalized || (sw as { isInterleavedBufferAttribute?: boolean }).isInterleavedBufferAttribute) {
    w = new Float32Array(n * 4);
    for (let v = 0; v < n; v++) for (let k = 0; k < 4; k++) w[v * 4 + k] = sw.getComponent(v, k);
    geo.setAttribute("skinWeight", new Float32BufferAttribute(w, 4));
  }
  return { idx, w };
}

export class WeightsSession {
  readonly stem: string;
  parts: WeightPartView[];
  readonly names: string[];
  readonly parents: number[];
  /** Which parts each undo step changed (a stroke over two meshes is one step). */
  private history: WeightPartView[][] = [];
  private future: WeightPartView[][] = [];
  private colMat = new MeshLambertMaterial({ vertexColors: true, side: DoubleSide, flatShading: true });
  private shown: { mode: ShowMode; bone: number } | null = null;
  private wasShown = false;
  private grabStart = 0;
  /** After every change (the panel re-renders, the colours follow). */
  onChange: () => void = () => {};

  private constructor(stem: string, meshes: SkinnedMesh[], prev: WeightsSession | null) {
    this.stem = stem;
    const bones = meshes[0].skeleton.bones;
    this.names = bones.map((b) => b.name);
    this.parents = bones.map((b) => bones.indexOf(b.parent as never));
    const carry = prev && prev.stem === stem && prev.sameShape(meshes) ? prev : null;
    this.parts = meshes.map((mesh, k) => {
      const geo = mesh.geometry;
      const { idx, w } = editableArrays(mesh);
      let edit: WeightEdit;
      if (carry) {
        edit = carry.parts[k].edit;
        edit.adopt(idx, w);
        geo.attributes.skinIndex.needsUpdate = geo.attributes.skinWeight.needsUpdate = true;
      } else {
        edit = new WeightEdit({ idx, w, pos: geo.attributes.position.array, index: geo.index ? geo.index.array : null }, this.parents);
      }
      const ref = (mesh.userData.gltfPrimitive as [number, number] | undefined) ?? [k, 0];
      return {
        mesh,
        edit,
        ref,
        saved: { material: mesh.material, color: (geo.getAttribute("color") as BufferAttribute) || null },
        colors: new BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3),
      };
    });
    if (carry) {
      this.history = carry.history.map((step) => step.map((p) => this.parts[carry.parts.indexOf(p)]).filter(Boolean));
      this.future = carry.future.map((step) => step.map((p) => this.parts[carry.parts.indexOf(p)]).filter(Boolean));
    }
    for (const p of this.parts) p.edit.onChange = () => this.changed(p);
  }

  /** The skinned meshes under the model, or null when there are none (an LOD on screen). */
  static bind(root: Object3D, stem: string, prev: WeightsSession | null): WeightsSession | null {
    const meshes: SkinnedMesh[] = [];
    root.traverse((o) => {
      if ((o as SkinnedMesh).isSkinnedMesh) meshes.push(o as SkinnedMesh);
    });
    if (!meshes.length) return null;
    return new WeightsSession(stem, meshes, prev);
  }

  private sameShape(meshes: SkinnedMesh[]): boolean {
    return meshes.length === this.parts.length && meshes.every((m, k) => m.geometry.attributes.position.count === this.parts[k].edit.n && m.skeleton.bones.length === this.names.length);
  }

  /** Whether this session is on the model on screen. */
  isOn(root: Object3D | null): boolean {
    if (!root) return false;
    let on = true;
    for (const p of this.parts) {
      let found = false;
      root.traverse((o) => {
        if (o === p.mesh) found = true;
      });
      on &&= found;
    }
    return on;
  }

  private changed(p: WeightPartView): void {
    const g = p.mesh.geometry;
    g.attributes.skinIndex.needsUpdate = true;
    g.attributes.skinWeight.needsUpdate = true;
    if (this.shown) this.paintPart(p);
    this.onChange();
  }

  /* ---------------------------------------------------------------- colours */

  /** Heat for a bone, or every bone's territory; null takes the colours off (the material comes back). */
  show(next: { mode: ShowMode; bone: number } | null): void {
    this.wasShown = !!this.shown;
    this.shown = next;
    for (const p of this.parts) {
      const geo = p.mesh.geometry;
      if (!next) {
        if (p.mesh.material === this.colMat) p.mesh.material = p.saved.material;
        if (p.saved.color) geo.setAttribute("color", p.saved.color);
        else if (geo.getAttribute("color") === p.colors) geo.deleteAttribute("color");
        continue;
      }
      // What to give back: the material on the mesh as the colours go on (the Rig check may have just given its back).
      if (!this.wasShown) p.saved = { material: p.mesh.material, color: geo.getAttribute("color") === p.colors ? p.saved.color : (geo.getAttribute("color") as BufferAttribute) || null };
      this.paintPart(p);
      // Clay or Wire swapped the material since: theirs stays (the panel closes on either).
      if (p.mesh.material === p.saved.material || p.mesh.material === this.colMat) p.mesh.material = this.colMat;
    }
  }

  private paintPart(p: WeightPartView): void {
    if (!this.shown) return;
    paintColours(p.edit, this.shown.mode, this.shown.bone, p.colors.array as Float32Array);
    p.colors.needsUpdate = true;
    p.mesh.geometry.setAttribute("color", p.colors);
  }

  /* ---------------------------------------------------------------- picking and the brush */

  /** The point of the hit triangle nearest the hit, and its mesh's part. */
  private hitPoint(hit: ToolHit): { part: WeightPartView; v: number } | null {
    const part = this.parts.find((p) => p.mesh === hit.object);
    if (!part || hit.faceIndex == null) return null;
    const geo = part.mesh.geometry;
    const f = hit.faceIndex * 3;
    const tri = geo.index ? [geo.index.getX(f), geo.index.getX(f + 1), geo.index.getX(f + 2)] : [f, f + 1, f + 2];
    let best = tri[0];
    let bd = Infinity;
    for (const v of tri) {
      part.mesh.getVertexPosition(v, v3);
      part.mesh.localToWorld(v3);
      const d = v3.distanceTo(hit.point);
      if (d < bd) {
        bd = d;
        best = v;
      }
    }
    return { part, v: best };
  }

  /** The bone that moves the clicked spot most, or -1. */
  boneAt(hit: ToolHit | null): number {
    const at = hit ? this.hitPoint(hit) : null;
    return at ? at.part.edit.topBone(at.v) : -1;
  }

  /** Where every point of a part is on screen now (world space, 3 per point), the clip's pose included. */
  private posed(p: WeightPartView): Float32Array {
    const n = p.edit.n;
    const out = new Float32Array(n * 3);
    p.mesh.updateMatrixWorld(true);
    // Only the first corner of each point is measured: the brush reads one per point.
    for (const vs of p.edit.members) {
      const v = vs[0];
      p.mesh.getVertexPosition(v, v3);
      p.mesh.localToWorld(v3);
      out[v * 3] = v3.x;
      out[v * 3 + 1] = v3.y;
      out[v * 3 + 2] = v3.z;
    }
    return out;
  }

  begin(): void {
    for (const p of this.parts) p.edit.begin();
  }

  end(): void {
    const step = this.parts.filter((p) => p.edit.end());
    if (step.length) {
      this.history.push(step);
      if (this.history.length > 100) this.history.shift();
      this.future = [];
    }
    this.onChange();
  }

  /** One dab at a world point (the stroke is open: begin() / end() around it). */
  dab(mode: BrushMode, bone: number, point: Vector3, radius: number, strength: number): void {
    for (const p of this.parts) p.edit.dab(mode, bone, this.posed(p), [point.x, point.y, point.z], radius, strength);
  }

  /** The Grab slider's drag: start, move to a percentage, release (one undo step). */
  grabBegin(bone: number): void {
    this.begin();
    this.grabStart = this.grabPercent(bone);
  }

  grabTo(bone: number, percent: number): void {
    for (const p of this.parts) p.edit.grabTo(bone, percent, this.grabStart);
  }

  grabPercent(bone: number): number {
    // Over every part, weighted by how much each holds (one mesh is the usual case).
    let now = 0;
    let n = 0;
    for (const p of this.parts) {
      const t = p.edit.boneTotals()[bone] || 0;
      now += p.edit.grabPercent(bone) * t;
      n += t;
    }
    return n > 0 ? now / n : this.parts[0]?.edit.grabPercent(bone) ?? 0;
  }

  /** Every bone's grab against the saved rig in percent, from one pass (the bone list). */
  grabAll(): number[] {
    const now: number[] = [];
    const was: number[] = [];
    for (const p of this.parts) {
      p.edit.boneTotals().forEach((x, j) => (now[j] = (now[j] || 0) + x));
      p.edit.savedTotalsByBone().forEach((x, j) => (was[j] = (was[j] || 0) + x));
    }
    return this.names.map((_, j) => ((was[j] || 0) < 1e-9 ? ((now[j] || 0) > 1e-9 ? 100 : 0) : (100 * (now[j] || 0)) / was[j]));
  }

  /** Bones that move anything now. */
  weighted(bone: number): boolean {
    return this.parts.some((p) => (p.edit.boneTotals()[bone] || 0) > 1e-6);
  }

  undo(): boolean {
    const step = this.history.pop();
    if (!step) return false;
    for (const p of step) p.edit.undo();
    this.future.push(step);
    this.onChange();
    return true;
  }

  redo(): boolean {
    const step = this.future.pop();
    if (!step) return false;
    for (const p of step) p.edit.redo();
    this.history.push(step);
    this.onChange();
    return true;
  }

  get canUndo(): boolean {
    return this.history.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }

  reset(): void {
    const step = this.parts.filter((p) => p.edit.reset());
    if (step.length) {
      this.history.push(step);
      this.future = [];
    }
    this.onChange();
  }

  changedPoints(): number {
    return this.parts.reduce((a, p) => a + p.edit.changedPoints(), 0);
  }

  /** POST /rig/<stem>/weights's body for every part. */
  body(note = "") {
    return weightsBody(
      this.parts.map((p) => ({ mesh: p.ref[0], primitive: p.ref[1], idx: p.edit.idx, w: p.edit.w })),
      note,
    );
  }

  markSaved(): void {
    for (const p of this.parts) p.edit.markSaved();
    this.onChange();
  }

  /** Colours off and the material back; the edit stays on the mesh. */
  release(): void {
    this.show(null);
  }
}
