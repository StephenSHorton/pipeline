// The Rig check on the model (legacy public/rig-inspect.js createRigInspect, 195-560, 596-760, 1030-1123): the
// colour modes on the skinned mesh, the skeleton with joint dots and decluttered bone names, the hover card, a click
// that picks the point's bone, the original model as a ghost, and the camera framing a bone or a view.
//
// Differences from legacy, on purpose:
// - The camera is Mokei's orthographic ClayCameraRig driven by the useLook store (azimuth, elevation, zoom, pan on
//   the ground), not a perspective camera with OrbitControls. A view becomes an azimuth and elevation, a fitted box
//   becomes a zoom and a pan. The rig always aims at height 0 (the diorama is lowered so the model's middle sits
//   there), so a box above or below the middle is centred across, and as near as the view allows up and down.
// - Spin turns the model, not the camera, so a view is turned by the model's own rotation first.
// - The colour material goes back only while it is still on the mesh: Clay and Wire swap materials by reference
//   (prepareModel), and the panel closes on either, in whichever order those effects run.
// - Pointer events come from the document (the orbit surface captures the pointer), filtered to the model view.
import { useLook } from "mokei/clay";
import {
  BufferAttribute,
  BufferGeometry,
  Box3,
  Color,
  DoubleSide,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshLambertMaterial,
  Points,
  PointsMaterial,
  Quaternion,
  Raycaster,
  SkeletonHelper,
  Vector2,
  Vector3,
  WireframeGeometry,
  type Material,
  type Object3D,
  type SkinnedMesh,
} from "three";
import type { ViewerTools } from "@/viewer/tools";
import { MINOR, ROLE_RGB, VIEWS, niceNames, tipDropped, tipPick, tipSplit, tipTitle, type RigInspectDoc, type RigMode, type RigViewId } from "./inspect";

export type ViewArg = RigViewId | readonly [number, number, number] | null;

/** What the overlay draws, set by the panel on every change. */
export interface OverlayState {
  mode: RigMode;
  bone: number;
  skel: boolean;
}

const grey = new Color(0.27, 0.27, 0.3);
const cold = new Color(0.05, 0.05, 0.35);
const heat = (t: number) => new Color().setHSL((1 - Math.min(1, Math.max(0, t))) * 0.6667, 1, 0.5);
const roleCol: Record<string, Color> = Object.fromEntries(Object.entries(ROLE_RGB).map(([k, [r, g, b]]) => [k, new Color(r / 255, g / 255, b / 255)]));
const wp = new Vector3();

export class RigOverlay {
  readonly mesh: SkinnedMesh;
  readonly names: string[];
  readonly nice: string[];
  readonly roles: string[];
  readonly pal: Color[];
  readonly owned: number[];
  readonly dropped: Float32Array | null;
  splitPoints = 0;
  /** Called when a click on the model picks a bone. */
  onPick: (i: number) => void = () => {};
  /** The Rig check panel, so framing keeps the model left of it. */
  panel: () => Element | null = () => null;

  private doc: RigInspectDoc;
  private tools: ViewerTools;
  private NV: number;
  private top: Int32Array;
  private split: Float32Array;
  private saved: { material: Material | Material[]; color: BufferAttribute | null };
  private colors: BufferAttribute;
  private colMat = new MeshLambertMaterial({ vertexColors: true, side: DoubleSide, flatShading: true });
  private helper: SkeletonHelper;
  private dots: Points;
  private labels: HTMLDivElement;
  private spans: HTMLSpanElement[];
  private tip: HTMLDivElement;
  private state: OverlayState = { mode: "bones", bone: -1, skel: true };
  private mouse: [number, number] | null = null;
  private down: [number, number] | null = null;
  private raf = 0;
  private ghost: Group | null = null;
  private ghostTok = 0;
  private lineMat = new LineBasicMaterial({ color: 0xf0eadc, transparent: true, opacity: 0.22, depthTest: false });
  private off: Array<() => void> = [];

  /** Bind to the first skinned mesh under `root`; null when there is none (legacy open() returned false). */
  static bind(root: Object3D, doc: RigInspectDoc, tools: ViewerTools): RigOverlay | null {
    let mesh: SkinnedMesh | null = null;
    root.traverse((o) => {
      if ((o as SkinnedMesh).isSkinnedMesh && !mesh) mesh = o as SkinnedMesh;
    });
    const three = tools.three();
    return mesh && three ? new RigOverlay(mesh, doc, tools) : null;
  }

  private constructor(mesh: SkinnedMesh, doc: RigInspectDoc, tools: ViewerTools) {
    this.mesh = mesh;
    this.doc = doc;
    this.tools = tools;
    const bones = mesh.skeleton.bones;
    const want = doc.bones.map((b) => b.name);
    if (bones.length !== want.length || bones.some((b, i) => b.name !== want[i])) console.warn("rig check: skin joints do not match inspect.json bones");
    this.names = bones.map((b) => b.name);
    this.nice = niceNames(this.names, doc.bones);
    this.roles = this.names.map((n) => {
      const k = doc.bones.findIndex((b) => b.name === n);
      return (k >= 0 && (doc.bones[k].role || (doc.roles || [])[k])) || "";
    });
    this.pal = bonePalette(this.names.length);
    const NV = (this.NV = mesh.geometry.attributes.position.count);
    const tf = doc.weights?.top_final;
    this.top = new Int32Array(NV);
    for (let v = 0; v < NV; v++) this.top[v] = tf && tf.length === NV ? tf[v] : (this.infl(v)[0] || [0])[0];
    this.owned = this.names.map(() => 0);
    for (let v = 0; v < NV; v++) this.owned[this.top[v]]++;
    const list = doc.weights?.dropped?.list || [];
    let dropped: Float32Array | null = null;
    if (list.length) {
      dropped = new Float32Array(NV);
      for (const [v, , w] of list) if (v < NV) dropped[v] += w;
    }
    this.dropped = dropped;
    this.split = new Float32Array(NV);
    this.cornerSplit();
    const geo = mesh.geometry;
    this.saved = { material: mesh.material, color: (geo.getAttribute("color") as BufferAttribute) || null };
    this.colors = new BufferAttribute(new Float32Array(NV * 3), 3);

    const scene = this.tools.three()!.scene;
    this.helper = new SkeletonHelper(this.tools.root() ?? mesh);
    const hm = this.helper.material as LineBasicMaterial;
    hm.depthTest = false;
    hm.transparent = true;
    this.helper.renderOrder = 9;
    scene.add(this.helper);
    const dg = new BufferGeometry();
    dg.setAttribute("position", new BufferAttribute(new Float32Array(this.names.length * 3), 3));
    dg.setAttribute("color", new BufferAttribute(new Float32Array(this.names.length * 3), 3));
    this.dots = new Points(dg, new PointsMaterial({ size: 7, sizeAttenuation: false, vertexColors: true, depthTest: false, transparent: true }));
    this.dots.renderOrder = 10;
    this.dots.frustumCulled = false;
    scene.add(this.dots);

    // Names and the hover card sit over the canvas, fixed to the page: the orbit surface is under the HUD.
    this.labels = document.createElement("div");
    this.labels.className = "pointer-events-none fixed z-[4] overflow-hidden";
    this.spans = this.nice.map((n) => {
      const s = document.createElement("span");
      s.className = "absolute -translate-y-1/2 translate-x-[5px] whitespace-nowrap font-mono text-[10px] text-white [text-shadow:0_0_3px_#000,0_0_3px_#000] data-[picked]:text-[11px] data-[picked]:text-[#ff6a55]";
      s.textContent = n;
      this.labels.appendChild(s);
      return s;
    });
    this.tip = document.createElement("div");
    this.tip.className = "pointer-events-none fixed z-[7] min-w-[170px] rounded-[10px] border border-hair bg-white/95 px-2 py-1.5 text-[11px] leading-normal text-ink shadow-md";
    this.tip.hidden = true;
    document.body.append(this.labels, this.tip);
    this.listen();
    this.paint();
    this.colorSkeleton();
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      this.tick();
    };
    this.raf = requestAnimationFrame(tick);
  }

  /* ---------------------------------------------------------------- weights */

  private infl(v: number): [number, number][] {
    const SI = this.mesh.geometry.attributes.skinIndex;
    const SW = this.mesh.geometry.attributes.skinWeight;
    const out: [number, number][] = [];
    for (let k = 0; k < 4; k++) {
      const w = SW.getComponent(v, k);
      if (w > 0) out.push([SI.getComponent(v, k), w]);
    }
    return out.sort((a, b) => b[1] - a[1]);
  }

  private weightOf(v: number, j: number): number {
    let w = 0;
    for (const [b, x] of this.infl(v)) if (b === j) w += x;
    return w;
  }

  // Copies of one point (flat-shaded corners) should follow the same bones; how far each one is off the mean.
  private cornerSplit(): void {
    const P = this.mesh.geometry.attributes.position;
    const groups = new Map<string, number[]>();
    for (let v = 0; v < this.NV; v++) {
      const k = `${Math.round(P.getX(v) * 1e5)},${Math.round(P.getY(v) * 1e5)},${Math.round(P.getZ(v) * 1e5)}`;
      const g = groups.get(k);
      if (g) g.push(v);
      else groups.set(k, [v]);
    }
    this.splitPoints = 0;
    for (const vs of groups.values()) {
      if (vs.length < 2) continue;
      const mean = new Map<number, number>();
      for (const v of vs) for (const [j, w] of this.infl(v)) mean.set(j, (mean.get(j) || 0) + w / vs.length);
      for (const v of vs) {
        let d = 0;
        for (const j of new Set([...mean.keys(), ...this.infl(v).map((x) => x[0])])) d += Math.abs(this.weightOf(v, j) - (mean.get(j) || 0));
        this.split[v] = d / 2;
      }
      if (new Set(vs.map((v) => this.top[v])).size > 1) this.splitPoints++;
    }
  }

  /** The top bone of the point under the pointer, or -1. */
  topBoneAt(clientX: number, clientY: number): number {
    const v = this.hitVertex(clientX, clientY);
    return v < 0 ? -1 : (this.infl(v)[0] || [-1])[0];
  }

  /* ---------------------------------------------------------------- drawing */

  /** The panel's mode, bone and skeleton switch changed. */
  set(next: OverlayState): void {
    const repaint = next.mode !== this.state.mode || next.bone !== this.state.bone;
    this.state = { ...next };
    if (repaint) {
      this.paint();
      this.colorSkeleton();
    }
  }

  private paint(): void {
    const { mesh } = this;
    const geo = mesh.geometry;
    const { mode, bone } = this.state;
    if (mode === "texture") {
      if (mesh.material === this.colMat) mesh.material = this.saved.material;
      if (this.saved.color) geo.setAttribute("color", this.saved.color);
      else if (geo.getAttribute("color") === this.colors) geo.deleteAttribute("color");
      return;
    }
    const c = new Color();
    for (let v = 0; v < this.NV; v++) {
      if (mode === "bones") {
        c.copy(this.pal[this.top[v]] || grey);
        if (bone >= 0 && this.top[v] !== bone) c.lerp(grey, 0.85);
      } else if (mode === "heat") {
        const w = bone >= 0 ? this.weightOf(v, bone) : 0;
        c.copy(bone < 0 ? grey : w > 0 ? heat(w) : cold);
      } else if (mode === "dropped") {
        c.copy(this.dropped && this.dropped[v] ? new Color(1, 1 - Math.min(1, this.dropped[v] * 3), 0) : grey);
      } else {
        c.copy(this.split[v] > 0.05 ? heat(Math.min(1, this.split[v] * 2)) : grey);
      }
      this.colors.setXYZ(v, c.r, c.g, c.b);
    }
    this.colors.needsUpdate = true;
    geo.setAttribute("color", this.colors);
    // Clay or Wire swapped the mesh's material since: keep theirs, the panel is closing.
    if (mesh.material === this.saved.material || mesh.material === this.colMat) mesh.material = this.colMat;
  }

  private colorSkeleton(): void {
    const ca = this.helper.geometry.attributes.color as BufferAttribute;
    const hot = new Color(1, 0.35, 0.3);
    const dim = new Color(0.45, 0.45, 0.48);
    const ink = new Color(0.94, 0.92, 0.86);
    let i = 0;
    for (const b of this.helper.bones) {
      if (!b.parent || !(b.parent as { isBone?: boolean }).isBone) continue;
      const k = this.names.indexOf(b.name);
      const pk = this.names.indexOf(b.parent.name);
      const role = this.roles[k];
      const col = k === this.state.bone || pk === this.state.bone ? hot : b.parent.name === "root" ? dim : role && roleCol[role] ? roleCol[role] : ink;
      ca.setXYZ(i++, col.r, col.g, col.b);
      ca.setXYZ(i++, col.r, col.g, col.b);
    }
    ca.needsUpdate = true;
  }

  private canvasRect(): DOMRect | null {
    return this.tools.three()?.gl.domElement.getBoundingClientRect() ?? null;
  }

  private tick(): void {
    if (!this.mesh.parent) return;
    this.drawSkeleton();
    if (this.mouse && !this.down) this.showTip();
    else this.tip.hidden = true;
  }

  private drawSkeleton(): void {
    const on = this.state.skel;
    this.helper.visible = this.dots.visible = on;
    this.labels.hidden = !on;
    if (!on) return;
    const three = this.tools.three();
    const rect = this.canvasRect();
    if (!three || !rect) return;
    Object.assign(this.labels.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
    const pa = this.dots.geometry.attributes.position as BufferAttribute;
    const ca = this.dots.geometry.attributes.color as BufferAttribute;
    const bones = this.mesh.skeleton.bones;
    const { bone } = this.state;
    bones.forEach((b, i) => {
      b.getWorldPosition(wp);
      pa.setXYZ(i, wp.x, wp.y, wp.z);
      const col = i === bone ? new Color(1, 0.25, 0.2) : this.pal[i];
      ca.setXYZ(i, col.r, col.g, col.b);
    });
    pa.needsUpdate = ca.needsUpdate = true;
    // Names: the picked bone first, then skip any that would land on one already placed.
    const placed: [number, number, number][] = [];
    const order = [...this.names.keys()].sort((a, b) => Number(b === bone) - Number(a === bone));
    const docBones = this.doc.bones;
    const near = (i: number) => bone < 0 || i === bone || docBones[i]?.parent === bone || docBones[bone]?.parent === i;
    for (const i of order) {
      const s = this.spans[i];
      let show = (!MINOR.test(this.names[i]) && near(i)) || i === bone;
      let x = 0;
      let y = 0;
      if (show) {
        bones[i].getWorldPosition(wp);
        wp.project(three.camera);
        x = ((wp.x + 1) / 2) * rect.width;
        y = ((1 - wp.y) / 2) * rect.height;
        show = wp.z < 1;
      }
      const w = this.nice[i].length * 5.6 + 6;
      show = show && (i === bone || !placed.some(([px, py, pw]) => x < px + pw && px < x + w && Math.abs(py - y) < 11));
      s.hidden = !show;
      if (!show) continue;
      placed.push([x, y, w]);
      s.style.left = `${x}px`;
      s.style.top = `${y}px`;
      s.toggleAttribute("data-picked", i === bone);
    }
  }

  private hitVertex(clientX: number, clientY: number): number {
    const three = this.tools.three();
    const rect = this.canvasRect();
    if (!three || !rect || !this.mesh.visible) return -1;
    const ndc = new Vector2(((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1, -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1);
    const ray = new Raycaster();
    ray.setFromCamera(ndc, three.camera);
    const h = ray.intersectObject(this.mesh, false)[0];
    if (!h || !h.face) return -1;
    const p = new Vector3();
    let best = h.face.a;
    let bd = Infinity;
    for (const v of [h.face.a, h.face.b, h.face.c]) {
      this.mesh.getVertexPosition(v, p);
      this.mesh.localToWorld(p);
      const d = p.distanceTo(h.point);
      if (d < bd) {
        bd = d;
        best = v;
      }
    }
    return best;
  }

  private showTip(): void {
    const [mx, my] = this.mouse!;
    const v = this.hitVertex(mx, my);
    if (v < 0) {
      this.tip.hidden = true;
      return;
    }
    const tip = this.tip;
    tip.replaceChildren();
    const line = (text: string, cls = "") => {
      const d = document.createElement("div");
      d.textContent = text;
      if (cls) d.className = cls;
      tip.appendChild(d);
      return d;
    };
    line(tipTitle(v), "font-mono text-[10px] tracking-[0.08em] uppercase text-ink-3");
    const inf = this.infl(v);
    for (const [j, w] of inf) {
      const row = line("", "flex items-center gap-1.5");
      const sw = document.createElement("span");
      sw.className = "inline-block size-2.5 shrink-0 rounded-[2px]";
      sw.style.background = `#${this.pal[j].getHexString()}`;
      const nm = document.createElement("span");
      nm.className = "flex-1";
      nm.textContent = this.nice[j];
      const b = document.createElement("b");
      b.className = "font-semibold tabular-nums";
      b.textContent = w.toFixed(2);
      row.append(sw, nm, b);
    }
    if (this.dropped && this.dropped[v]) line(tipDropped(this.dropped[v]), "w-fit rounded-[4px] bg-signal px-1 font-medium text-signal-foreground");
    if (this.split[v] > 0.1) line(tipSplit(this.split[v]), "font-medium text-danger-text");
    if (inf[0]) line(tipPick(this.nice[inf[0][0]]), "text-ink-3");
    tip.style.left = `${Math.min(mx + 14, innerWidth - 200)}px`;
    tip.style.top = `${my + 14}px`;
    tip.hidden = false;
  }

  /** A press on the model view (the canvas, or the orbit surface holding the pointer), not on the HUD. */
  private onView(e: Event): boolean {
    const canvas = this.tools.three()?.gl.domElement;
    const t = e.target as Node | null;
    return !!canvas && !!t && (t === canvas || (t as Element).contains?.(canvas));
  }

  private listen(): void {
    const move = (e: PointerEvent) => {
      if (!this.onView(e)) {
        this.mouse = null;
        return;
      }
      if (!this.down) this.mouse = [e.clientX, e.clientY];
    };
    const down = (e: PointerEvent) => {
      if (!this.onView(e) || e.button !== 0) return;
      this.down = [e.clientX, e.clientY];
      this.tip.hidden = true;
    };
    const up = (e: PointerEvent) => {
      const was = this.down;
      this.down = null;
      if (!was || !this.mesh.visible || Math.hypot(e.clientX - was[0], e.clientY - was[1]) > 4) return;
      const i = this.topBoneAt(e.clientX, e.clientY);
      if (i >= 0) this.onPick(i);
    };
    const leave = () => {
      this.mouse = null;
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerdown", down);
    document.addEventListener("pointerup", up);
    document.documentElement.addEventListener("pointerleave", leave);
    this.off.push(() => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerdown", down);
      document.removeEventListener("pointerup", up);
      document.documentElement.removeEventListener("pointerleave", leave);
    });
  }

  /* ---------------------------------------------------------------- the original model (head ghost) */

  /** Show the original model: `original` instead of the rig, `both` as a wireframe over it; `rigged` drops it. */
  async setGhost(mode: "rigged" | "original" | "both", load: () => Promise<Object3D>): Promise<boolean> {
    this.clearGhost();
    if (mode === "rigged") return true;
    const tok = this.ghostTok;
    try {
      const scene = await load();
      if (tok !== this.ghostTok || !this.mesh.parent) return false;
      const ghost = new Group();
      ghost.position.copy(this.mesh.position);
      ghost.quaternion.copy(this.mesh.quaternion);
      ghost.scale.copy(this.mesh.scale);
      scene.updateMatrixWorld(true);
      scene.traverse((o) => {
        const m = o as Mesh;
        if (!m.isMesh) return;
        const geo = m.geometry.clone().applyMatrix4(m.matrixWorld);
        if (mode === "original") {
          // Mokei's lights are flat and not tone mapped: a Lambert copy of the painted look, as prepareModel does.
          const src = (Array.isArray(m.material) ? m.material[0] : m.material) as Material & { map?: unknown; color?: Color };
          ghost.add(new Mesh(geo, new MeshLambertMaterial({ map: (src.map as never) ?? null, color: src.color ? src.color.clone() : 0xffffff })));
          return;
        }
        const w = new LineSegments(new WireframeGeometry(geo), this.lineMat);
        w.renderOrder = 8;
        ghost.add(w);
        geo.dispose();
      });
      this.mesh.parent.add(ghost);
      this.ghost = ghost;
      this.mesh.visible = mode !== "original";
      return true;
    } catch (err) {
      console.warn("rig check: could not load the original model", err);
      return false;
    }
  }

  clearGhost(): void {
    this.ghostTok++;
    if (this.ghost) {
      this.ghost.parent?.remove(this.ghost);
      this.ghost.traverse((o) => (o as Mesh).geometry?.dispose());
      this.ghost = null;
    }
    this.mesh.visible = true;
  }

  /** The bind pose (after restPose stopped the clips). */
  pose(): void {
    this.mesh.skeleton.pose();
  }

  /* ---------------------------------------------------------------- framing */

  /** Points of the mesh as it is posed now, in world space (all of them, or those a bone moves most). */
  posedBox(bone: number): Box3 {
    this.tools.root()?.updateMatrixWorld(true);
    const box = new Box3();
    const p = new Vector3();
    const step = bone >= 0 ? 1 : Math.max(1, Math.floor(this.NV / 4000));
    for (let v = 0; v < this.NV; v += step) {
      if (bone >= 0 && this.top[v] !== bone) continue;
      this.mesh.getVertexPosition(v, p);
      box.expandByPoint(this.mesh.localToWorld(p));
    }
    return box;
  }

  /** A bone's part with room around it to see where it sits on the model. */
  boneBox(i: number): Box3 {
    const box = this.posedBox(i);
    for (const k of [i, ...this.doc.bones.map((b, c) => (b.parent === i ? c : -1)).filter((c) => c >= 0)]) {
      this.mesh.skeleton.bones[k]?.getWorldPosition(wp);
      box.expandByPoint(wp);
    }
    // At least 3/4 of the model across, kept on the model rather than centred on empty floor.
    const all = this.posedBox(-1);
    const as = all.getSize(new Vector3());
    const min = 0.75 * Math.max(as.x, as.y, as.z);
    const out = box.clone();
    for (const ax of ["x", "y", "z"] as const) {
      const want = Math.min(Math.max(box.max[ax] - box.min[ax], min), as[ax]);
      if (box.max[ax] - box.min[ax] >= want) continue;
      const c = Math.min(Math.max((box.min[ax] + box.max[ax]) / 2, all.min[ax] + want / 2), all.max[ax] - want / 2);
      out.min[ax] = c - want / 2;
      out.max[ax] = c + want / 2;
    }
    return out;
  }

  /** A view of the whole model. */
  setView(id: ViewArg): void {
    this.frameBox(this.posedBox(-1), id);
  }

  /** Look from a view (or the current direction) and fit `box` into the free part of the view: left of the panel. */
  frameBox(box: Box3, id: ViewArg): void {
    if (box.isEmpty()) return;
    const look = useLook.getState();
    let az = (look.cameraAzimuth * Math.PI) / 180;
    let el = (look.cameraElevation * Math.PI) / 180;
    if (id) {
      const v = typeof id === "string" ? VIEWS.find((x) => x[0] === id)![2] : id;
      // The view is in the model's own space; Spin and the wrapper turn the model, so turn the view with it.
      const q = this.mesh.parent ? this.mesh.parent.getWorldQuaternion(new Quaternion()) : new Quaternion();
      const dir = new Vector3(v[0], v[1], v[2]).normalize().applyQuaternion(q);
      el = Math.asin(Math.max(-1, Math.min(1, dir.y)));
      el = Math.max((-88 * Math.PI) / 180, Math.min((88 * Math.PI) / 180, el));
      az = Math.atan2(dir.x, dir.z);
    }
    const right = new Vector3(Math.cos(az), 0, -Math.sin(az));
    const up = new Vector3(-Math.sin(el) * Math.sin(az), Math.cos(el), -Math.sin(el) * Math.cos(az));
    const c = box.getCenter(new Vector3());
    let hx = 0;
    let hy = 0;
    for (let k = 0; k < 8; k++) {
      const q = new Vector3(k & 1 ? box.max.x : box.min.x, k & 2 ? box.max.y : box.min.y, k & 4 ? box.max.z : box.min.z).sub(c);
      hx = Math.max(hx, Math.abs(q.dot(right)));
      hy = Math.max(hy, Math.abs(q.dot(up)));
    }
    const rect = this.canvasRect();
    if (!rect) return;
    const free = freeArea(rect, this.panel());
    const m = 0.86;
    const zoom = Math.max(1e-3, Math.min(free.w / (2 * Math.max(hx, 1e-3)), free.h / (2 * Math.max(hy, 1e-3))) * m);
    // Where the box's centre should land, from the view's centre, in world units (up the screen is +y).
    const fx = (free.cx - (rect.left + rect.width / 2)) / zoom;
    const fy = -(free.cy - (rect.top + rect.height / 2)) / zoom;
    // Solve the ground pan T = (tx, 0, tz) so right·(c - T) = fx and, where the elevation allows, up·(c - T) = fy.
    const s = Math.sin(el);
    const depth = Math.abs(s) > 0.05 ? (Math.cos(el) * c.y - fy) / s : 0;
    const a = Math.cos(az) * fx + Math.sin(az) * depth;
    const b = -Math.sin(az) * fx + Math.cos(az) * depth;
    look.setLook({ cameraAzimuth: (az * 180) / Math.PI, cameraElevation: (el * 180) / Math.PI, cameraZoom: zoom, panX: c.x - a, panZ: c.z - b });
  }

  /** Put the material, colour attribute and DOM back, and stop listening. */
  dispose(): void {
    cancelAnimationFrame(this.raf);
    for (const f of this.off) f();
    this.off = [];
    this.clearGhost();
    const geo = this.mesh.geometry;
    if (this.mesh.material === this.colMat) this.mesh.material = this.saved.material;
    if (this.saved.color) geo.setAttribute("color", this.saved.color);
    else if (geo.getAttribute("color") === this.colors) geo.deleteAttribute("color");
    for (const o of [this.helper, this.dots]) {
      o.parent?.remove(o);
      o.geometry.dispose();
      (o.material as Material).dispose();
    }
    this.colMat.dispose();
    this.lineMat.dispose();
    this.labels.remove();
    this.tip.remove();
  }
}

/** Bone colours, golden-ratio hues (legacy pal); the panel's swatches use the same with getHexString. */
export function bonePalette(n: number): Color[] {
  return Array.from({ length: n }, (_, i) => new Color().setHSL((i * 0.618034) % 1, 0.62, 0.55));
}

/** The part of the view the model can be seen in: the canvas less the HUD's strip, and left of a panel on the right. */
export function freeArea(rect: { left: number; top: number; width: number; height: number }, panel: { getBoundingClientRect(): { left: number; width: number } } | null, hudBottom?: number) {
  let left = rect.left;
  let right = rect.left + rect.width;
  const top = rect.top;
  let bottom = rect.top + rect.height;
  const hud = hudBottom ?? (typeof document !== "undefined" ? document.querySelector("[data-hud]")?.getBoundingClientRect().bottom : undefined);
  if (hud && hud > top + 80) bottom = Math.min(bottom, hud);
  if (panel) {
    const r = panel.getBoundingClientRect();
    if (r.width && r.left > left + (right - left) / 2) right = Math.min(right, r.left);
  }
  right = Math.max(left + 80, right);
  left = Math.min(left, right - 80);
  return { cx: (left + right) / 2, cy: (top + bottom) / 2, w: right - left, h: bottom - top };
}

