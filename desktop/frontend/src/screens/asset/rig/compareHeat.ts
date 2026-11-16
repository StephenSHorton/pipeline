// Compare's left side in the Weights colours: the earlier version's own weights, read only, painted with the same
// heat and territory colours as the Weights panel (./weights.ts paintColours). Bones are matched by name
// (screens/viewer/compare.ts boneMap): the weights are copied into the right skeleton's numbering, so the bone picked
// in the list lights up the same bone on both sides and every bone keeps its colour. Nothing here edits the mesh's
// skin; the colours come off (and the material goes back) on release.
import { BufferAttribute, DoubleSide, MeshLambertMaterial, type Material, type Object3D, type SkinnedMesh } from "three";
import { boneMap } from "@/screens/viewer/compare";
import { paintColours, type ShowMode, type WeightReader } from "./weights";

/** One mesh's four bones and weights per point, in the right skeleton's numbering. */
export class SkinWeights implements WeightReader {
  readonly n: number;
  constructor(
    readonly idx: Int32Array,
    readonly w: Float32Array,
  ) {
    this.n = w.length / 4;
  }

  weightOf(v: number, j: number): number {
    let s = 0;
    for (let k = 0; k < 4; k++) if (this.idx[v * 4 + k] === j) s += this.w[v * 4 + k];
    return s;
  }

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

  /** A mesh's skin attributes, its bone indices put through `map` (left index to right index). */
  static read(mesh: SkinnedMesh, map: Int32Array): SkinWeights {
    const geo = mesh.geometry;
    const si = geo.getAttribute("skinIndex");
    const sw = geo.getAttribute("skinWeight");
    const n = geo.attributes.position.count;
    const idx = new Int32Array(n * 4);
    const w = new Float32Array(n * 4);
    for (let v = 0; v < n; v++)
      for (let k = 0; k < 4; k++) {
        const j = si.getComponent(v, k);
        idx[v * 4 + k] = map[j] ?? j;
        w[v * 4 + k] = sw.getComponent(v, k);
      }
    return new SkinWeights(idx, w);
  }
}

/** The bone names of the first skinned mesh under `root`, or [] (not rigged). */
export function skeletonNames(root: Object3D): string[] {
  let names: string[] = [];
  root.traverse((o) => {
    if (!names.length && (o as SkinnedMesh).isSkinnedMesh) names = (o as SkinnedMesh).skeleton.bones.map((b) => b.name);
  });
  return names;
}

interface Part {
  mesh: SkinnedMesh;
  skin: SkinWeights;
  saved: { material: Material | Material[]; color: BufferAttribute | null };
  colors: BufferAttribute;
}

export class CompareHeat {
  private parts: Part[];
  private colMat = new MeshLambertMaterial({ vertexColors: true, side: DoubleSide, flatShading: true });
  private shown: { mode: ShowMode; bone: number } | null = null;

  private constructor(meshes: SkinnedMesh[], right: string[]) {
    this.parts = meshes.map((mesh) => {
      const geo = mesh.geometry;
      return {
        mesh,
        skin: SkinWeights.read(mesh, boneMap(mesh.skeleton.bones.map((b) => b.name), right)),
        saved: { material: mesh.material, color: (geo.getAttribute("color") as BufferAttribute) || null },
        colors: new BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3),
      };
    });
  }

  /** The skinned meshes under `root`, matched to the right side's bone names; null when there are none. */
  static bind(root: Object3D, right: string[]): CompareHeat | null {
    const meshes: SkinnedMesh[] = [];
    root.traverse((o) => {
      if ((o as SkinnedMesh).isSkinnedMesh) meshes.push(o as SkinnedMesh);
    });
    return meshes.length ? new CompareHeat(meshes, right) : null;
  }

  /** Heat for a bone (the right's index), or every bone's territory. Repaints only when it changes. */
  show(mode: ShowMode, bone: number): void {
    if (this.shown && this.shown.mode === mode && this.shown.bone === bone) return;
    const first = !this.shown;
    this.shown = { mode, bone };
    for (const p of this.parts) {
      const geo = p.mesh.geometry;
      if (first) p.saved = { material: p.mesh.material, color: (geo.getAttribute("color") as BufferAttribute) || null };
      paintColours(p.skin, mode, bone, p.colors.array as Float32Array);
      p.colors.needsUpdate = true;
    }
    this.keep();
  }

  /** Once a frame while shown: the colours stay on through the left model's own material swaps (setClay(false)). */
  keep(): void {
    if (!this.shown) return;
    for (const p of this.parts) {
      if (p.mesh.material === p.saved.material) p.mesh.material = this.colMat;
      if (p.mesh.geometry.getAttribute("color") !== p.colors) p.mesh.geometry.setAttribute("color", p.colors);
    }
  }

  /** Colours off, the material back, and the colour material freed. */
  release(): void {
    for (const p of this.parts) {
      const geo = p.mesh.geometry;
      if (p.mesh.material === this.colMat) p.mesh.material = p.saved.material;
      if (p.saved.color) geo.setAttribute("color", p.saved.color);
      else if (geo.getAttribute("color") === p.colors) geo.deleteAttribute("color");
    }
    this.shown = null;
    this.colMat.dispose();
  }
}
