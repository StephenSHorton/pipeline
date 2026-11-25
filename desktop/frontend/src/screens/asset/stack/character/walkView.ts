// The walking character in the Stack stage's scene (pipeline#2): one instanced mesh of cubes per part (the torso, a
// mesh per slice so it can twist), placed every frame from the rig's pose. It walks a closed lap of whole strides
// on a grid, so a planted foot can be seen to stay on its square. Turning the view is the stage's own camera.
// The sprite-stack look (pipeline#6) draws each part slice as one flat picture instead of cubes (sliceLayers).
import {
  BoxGeometry,
  DataTexture,
  DoubleSide,
  Group,
  InstancedBufferAttribute,
  InstancedMesh,
  type Material,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  NearestFilter,
  PlaneGeometry,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
  Vector3,
} from "three";
import { createWalker, lapPoint, place, pose, step, type Pose, type Walker } from "./rig";
import { bodyFromRig, cycleFrame, makeCycle, rigPose, type CycleName } from "./skeleton";
import { partCells, partKind, placeParts, type CharacterDoc, type CharacterPart, type PartCells, type Placement } from "./parts";

export interface WalkViewOptions {
  /** The group it draws into (the stage's subject, so the Game look and the camera see it). */
  parent: Group;
  doc: CharacterDoc;
  /** Each part's strip as RGBA, by part name. */
  strips: Map<string, ArrayLike<number>>;
  /** World size of one cell. */
  pitch: number;
  material: Material;
  /** A cube's colour channel (0-1) from a byte and its body layer, as the stage colours its own cubes. */
  color: (byte: number, layer: number) => number;
  /** "slices": each part slice one flat picture (the sprite-stack look), not cubes. */
  look?: "cubes" | "slices";
  /** Play a code-made skeleton cycle (pipeline#14) round the lap instead of the rig's own walk. */
  cycle?: CycleName;
}

/** A slice's darkening in the sprite-stack look: the lowest layer at 0.62 of its colour, the top at full. Darker
 * than that and dark boots and jeans sink into a dark ground. */
export const SLICE_SHADE_LOW = 0.62;
export const sliceShade = (layer: number, top: number) =>
  SLICE_SHADE_LOW + (1 - SLICE_SHADE_LOW) * Math.min(1, Math.max(0, layer / Math.max(1, top)));

/** One part slice as an RGBA picture (cw x ch; x forward, y left), each colour darkened by its body layer. */
export function slicePixels(part: CharacterPart, rgba: ArrayLike<number>, s: number, shade: number): Uint8Array {
  const { cw, ch, n } = part;
  const width = cw * n;
  const out = new Uint8Array(cw * ch * 4);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const i = (y * width + s * cw + x) * 4;
      const o = (y * cw + x) * 4;
      if (rgba[i + 3] < 128) continue;
      out[o] = Math.round(rgba[i] * shade);
      out[o + 1] = Math.round(rgba[i + 1] * shade);
      out[o + 2] = Math.round(rgba[i + 2] * shade);
      out[o + 3] = 255;
    }
  return out;
}

export interface WalkView {
  /** Advance the walk by dt seconds (fixed steps of 1/fps). */
  tick(dt: number): void;
  /** The character's place on the ground, in world units (three: x, z). */
  where(): Vector3;
  /** The body's height in world units. */
  height: number;
  /** The lap's radius in world units. */
  radius: number;
  setMaterial(m: Material): void;
  recolor(): void;
  dispose(): void;
}

interface Piece {
  part: string;
  slice: number | null;
  group: Group;
  /** The cubes look; null in the slices look, where the group holds one flat picture per slice. */
  mesh: InstancedMesh | null;
  cells: PartCells;
  first: number;
  count: number;
}

export function createWalkView(o: WalkViewOptions): WalkView {
  const { doc, pitch } = o;
  const fps = doc.walk.fps || 60;
  const strides = doc.walk.lap.strides;
  const lapFrames = doc.walk.lap.frames;
  // Rig frame (x forward, y left, z up, in cells) -> three (y up): turn -90 degrees about x, scale to world.
  const root = new Group();
  root.rotation.x = -Math.PI / 2;
  root.scale.setScalar(pitch);
  o.parent.add(root);
  const geo = new BoxGeometry(1, 1, 1);
  const pieces: Piece[] = [];
  let material = o.material;
  const slices = o.look === "slices";
  const flats: { mesh: Mesh; tex: DataTexture; mat: MeshBasicMaterial }[] = [];
  const top = Math.max(1, doc.height - 1);

  for (const part of doc.parts) {
    const rgba = o.strips.get(part.name);
    if (!rgba) continue;
    const cells = partCells(part, rgba);
    const perSlice = partKind(part.name) === "torso";
    if (slices) {
      // One flat picture per slice, in the part's frame (x forward, y left, z up from its joint), at the slice's
      // middle height. A plane's u runs with x and, unflipped, its v with y, as the strip's image x and y do.
      const groups: Group[] = [];
      for (let s = 0; s < part.n; s++) {
        const px = slicePixels(part, rgba, s, sliceShade(part.z0 + s, top));
        let any = false;
        for (let i = 3; i < px.length; i += 4) if (px[i]) { any = true; break; }
        if (!any) continue;
        const tex = new DataTexture(px, part.cw, part.ch, RGBAFormat, UnsignedByteType);
        tex.magFilter = NearestFilter;
        tex.minFilter = NearestFilter;
        tex.generateMipmaps = false;
        tex.colorSpace = SRGBColorSpace;
        tex.needsUpdate = true;
        const mat = new MeshBasicMaterial({ map: tex, alphaTest: 0.5, side: DoubleSide, toneMapped: false });
        const mesh = new Mesh(new PlaneGeometry(part.cw, part.ch), mat);
        mesh.position.z = part.z0 + s + 0.5 - part.joint_layer;
        mesh.frustumCulled = false;
        flats.push({ mesh, tex, mat });
        if (perSlice || !groups.length) {
          const group = new Group();
          group.matrixAutoUpdate = false;
          root.add(group);
          groups.push(group);
          pieces.push({ part: part.name, slice: perSlice ? s : null, group, mesh: null, cells, first: 0, count: 0 });
        }
        groups[groups.length - 1].add(mesh);
      }
      continue;
    }
    const runs: { slice: number | null; first: number; count: number }[] = [];
    if (perSlice) {
      let i = 0;
      while (i < cells.count) {
        const s = cells.slice[i];
        let j = i;
        while (j < cells.count && cells.slice[j] === s) j++;
        runs.push({ slice: s, first: i, count: j - i });
        i = j;
      }
    } else runs.push({ slice: null, first: 0, count: cells.count });
    for (const r of runs) {
      if (!r.count) continue;
      const mesh = new InstancedMesh(geo, material, r.count);
      mesh.frustumCulled = false;
      mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(r.count * 3), 3);
      const m = new Matrix4();
      for (let k = 0; k < r.count; k++) {
        const i = r.first + k;
        m.makeTranslation(cells.pos[i * 3], cells.pos[i * 3 + 1], cells.pos[i * 3 + 2]);
        mesh.setMatrixAt(k, m);
      }
      const group = new Group();
      group.matrixAutoUpdate = false;
      group.add(mesh);
      root.add(group);
      pieces.push({ part: part.name, slice: r.slice, group, mesh, cells, first: r.first, count: r.count });
    }
  }

  function recolor() {
    for (const p of pieces) {
      if (!p.mesh) continue;
      const ca = p.mesh.instanceColor!.array as Float32Array;
      for (let k = 0; k < p.count; k++) {
        const i = p.first + k;
        const layer = p.cells.layer[i];
        ca[k * 3] = o.color(p.cells.rgb[i * 3], layer);
        ca[k * 3 + 1] = o.color(p.cells.rgb[i * 3 + 1], layer);
        ca[k * 3 + 2] = o.color(p.cells.rgb[i * 3 + 2], layer);
      }
      p.mesh.instanceColor!.needsUpdate = true;
    }
  }
  recolor();

  const walker: Walker = createWalker(doc.rig);
  let frame = 0;
  const start = lapPoint(0, walker.c, strides, lapFrames);
  place(walker, start.x, start.y, start.h, true);
  // One lap first, unseen, so the loop the stage shows is the steady one (the frame after the last is the first).
  for (let f = 1; f <= lapFrames; f++) {
    const p = lapPoint(f, walker.c, strides, lapFrames);
    step(walker, p.x, p.y, p.h, 1 / fps);
  }
  frame = lapFrames;

  // A skeleton cycle round the same lap: its frames at its own rate, the body moved along the circle by the cycle's
  // own travel and turned along it (on the spot for the attack, which does not travel).
  const body = bodyFromRig(doc.rig, doc.height);
  const cyc = o.cycle ? makeCycle(o.cycle, body) : null;
  let cycFrame = 0;
  let cycAcc = 0;
  const cycWhere = { x: start.x, y: start.y, h: start.h };
  function cyclePose(): Pose {
    const c = cyc!;
    const f = cycleFrame(c, cycFrame % c.frames);
    const own = ((cycFrame % c.frames) / c.frames) * c.travel; // the travel, not the pelvis (it sways)
    const circ = strides * walker.c.stride;
    const d = ((cycFrame * c.travel) / c.frames) % circ;
    const a = (d / circ) * Math.PI * 2;
    const R = start.radius;
    cycWhere.x = R * Math.cos(a);
    cycWhere.y = R * Math.sin(a);
    cycWhere.h = a + Math.PI / 2;
    const inPlace = { ...f, joints: Object.fromEntries(Object.entries(f.joints).map(([n, p]) => [n, [p[0] - own, p[1], p[2]]])) } as typeof f;
    return rigPose(inPlace, body, cycWhere.x, cycWhere.y, cycWhere.h, (cycFrame % c.frames) / c.frames);
  }

  const mat = new Matrix4();
  function apply() {
    const placed = placeParts(doc, cyc ? cyclePose() : pose(walker));
    for (const p of pieces) {
      const at = placed.get(p.part);
      if (!at) {
        p.group.visible = false;
        continue;
      }
      const pl: Placement = Array.isArray(at) ? at[p.slice ?? 0] : at;
      mat.set(
        pl.x[0], pl.y[0], pl.z[0], pl.origin[0],
        pl.x[1], pl.y[1], pl.z[1], pl.origin[1],
        pl.x[2], pl.y[2], pl.z[2], pl.origin[2],
        0, 0, 0, 1,
      );
      p.group.matrix.copy(mat);
      p.group.visible = true;
      p.group.matrixWorldNeedsUpdate = true;
    }
  }
  apply();

  let acc = 0;
  const where = new Vector3();
  return {
    tick(dt: number) {
      if (cyc) {
        cycAcc = Math.min(cycAcc + dt, 4 / cyc.fps);
        let moved = false;
        while (cycAcc >= 1 / cyc.fps) {
          cycAcc -= 1 / cyc.fps;
          cycFrame += 1;
          moved = true;
        }
        if (moved) apply();
        return;
      }
      acc = Math.min(acc + dt, 4 / fps);
      let moved = false;
      while (acc >= 1 / fps) {
        acc -= 1 / fps;
        frame += 1;
        const p = lapPoint(frame, walker.c, strides, lapFrames);
        step(walker, p.x, p.y, p.h, 1 / fps);
        moved = true;
      }
      if (moved) apply();
    },
    where() {
      // rig (x, y) -> three (x, -y) on the ground
      if (cyc) return where.set(cycWhere.x * pitch, 0, -cycWhere.y * pitch);
      return where.set(walker.x * pitch, 0, -walker.y * pitch);
    },
    height: doc.height * pitch,
    radius: start.radius * pitch,
    setMaterial(m: Material) {
      material = m;
      for (const p of pieces) if (p.mesh) p.mesh.material = m;
    },
    recolor,
    dispose() {
      root.removeFromParent();
      for (const p of pieces) p.mesh?.dispose();
      for (const f of flats) {
        f.mesh.geometry.dispose();
        f.mat.dispose();
        f.tex.dispose();
      }
      geo.dispose();
    },
  };
}
