// Turn a parsed GLB into a diorama piece: Lambert materials (Mokei's flat, no-tone-mapping lights turn PBR
// dark), shadows on, scaled to yard size, centred on x/z with its feet on y=0. The clay look is a second
// material set swapped by reference, so the AnimationMixer (bound to this scene) keeps working.
import {
  Box3,
  Group,
  Mesh,
  MeshLambertMaterial,
  SkinnedMesh,
  NearestFilter,
  Vector3,
  type Material,
  type MeshBasicMaterial,
  type Object3D,
  type Texture,
} from "three";
import { outlineMaterial, trackTargetSize } from "./frame";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";

/** Model size on the pad, in world units. Mokei's shadow frustum (±42), PCF radius and AO radius (2.2) assume this scale. */
export const MODEL_SIZE = 8;

export interface PreparedModel {
  /** Wrapper group: scaled and placed. Add it to the scene. */
  root: Group;
  /** The glTF scene inside root (the mixer binds to it). */
  scene: Object3D;
  height: number;
  /** Bounding-sphere radius after scaling (for framing). */
  radius: number;
  /** Footprint half-diagonal after scaling (for the pad). */
  footprint: number;
  /** Up to ~4000 of its vertices (x, y, z), in the diorama's frame: centred on x/z, y from mid-height (./frame.ts). */
  points: Float32Array;
  hasMap: boolean;
  skinned: boolean;
  vertices: number;
  faces: number;
  setClay(on: boolean): void;
  /** Wireframe on every material (legacy Wire). */
  setWire(on: boolean): void;
  dispose(): void;
}

type SrcMat = Material & {
  map?: Texture | null;
  color?: { clone(): unknown };
  vertexColors?: boolean;
  alphaMap?: Texture | null;
  alphaTest?: number;
  transparent?: boolean;
  opacity?: number;
  side?: number;
};

export function prepareModel(gltf: GLTF, opts: { clayColor: string; pixel: boolean }): PreparedModel {
  const scene = gltf.scene;
  const textured: Array<[Mesh, Material | Material[]]> = [];
  const clay: Array<[Mesh, Material | Material[]]> = [];
  const made = new Set<Material>();
  const clayMat = new MeshLambertMaterial({ color: opts.clayColor });
  made.add(clayMat);
  let hasMap = false;
  let skinned = false;
  let vertices = 0;
  let faces = 0;

  const lambert = (src: SrcMat): Material => {
    const m = new MeshLambertMaterial({
      map: src.map ?? null,
      alphaMap: src.alphaMap ?? null,
      vertexColors: !!src.vertexColors,
      transparent: !!src.transparent,
      opacity: src.opacity ?? 1,
      alphaTest: src.alphaTest ?? 0,
      side: src.side ?? 0,
    });
    if (src.color) m.color.copy(src.color as never);
    if (src.map) {
      hasMap = true;
      if (opts.pixel) {
        src.map.magFilter = NearestFilter;
        src.map.minFilter = NearestFilter;
        src.map.generateMipmaps = false;
        src.map.needsUpdate = true;
        m.flatShading = true;
      }
    }
    m.name = src.name;
    made.add(m);
    return m;
  };

  const meshes: Mesh[] = [];
  scene.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh) return;
    meshes.push(mesh);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if ((mesh as SkinnedMesh).isSkinnedMesh) {
      skinned = true;
      mesh.frustumCulled = false; // a playing clip moves it outside its bind-pose bounds
    }
    const geo = mesh.geometry;
    if (!geo.attributes.normal) geo.computeVertexNormals();
    const pos = geo.attributes.position?.count ?? 0;
    vertices += pos;
    faces += Math.floor((geo.index ? geo.index.count : pos) / 3);
    const src = mesh.material as SrcMat | SrcMat[];
    const tex = Array.isArray(src) ? src.map(lambert) : lambert(src);
    textured.push([mesh, tex]);
    clay.push([mesh, Array.isArray(src) ? src.map(() => clayMat) : clayMat]);
    mesh.material = tex;
    for (const s of Array.isArray(src) ? src : [src]) if (!made.has(s)) s.dispose();
  });

  // Fit: the longest of height and footprint becomes MODEL_SIZE; feet on y=0; centred on x/z.
  const root = new Group();
  root.name = "model";
  root.add(scene);
  scene.updateMatrixWorld(true);
  const box = new Box3().setFromObject(scene);
  const size = box.getSize(new Vector3());
  const longest = Math.max(size.y, size.x, size.z, 1e-6);
  const s = MODEL_SIZE / longest;
  root.scale.setScalar(s);
  root.updateMatrixWorld(true);
  box.setFromObject(root);
  const c = box.getCenter(new Vector3());
  root.position.set(-c.x, -box.min.y, -c.z);
  root.updateMatrixWorld(true);
  box.setFromObject(root);
  const fitted = box.getSize(new Vector3());
  const points = samplePoints(meshes, fitted.y / 2);

  // The outline (#356): near-white parts vanish on Mokei's pale ground and pad, so every mesh gets a back-face hull.
  const hullMats: MeshBasicMaterial[] = [];
  const hulls: Mesh[] = [];
  for (const mesh of meshes) {
    const sk = (mesh as SkinnedMesh).isSkinnedMesh;
    const mat = outlineMaterial(sk);
    let hull: Mesh;
    if (sk) {
      const src = mesh as SkinnedMesh;
      const skin = new SkinnedMesh(mesh.geometry, mat);
      skin.bind(src.skeleton, src.bindMatrix);
      skin.frustumCulled = false;
      hull = skin;
    } else hull = new Mesh(mesh.geometry, mat);
    hull.castShadow = false;
    hull.receiveShadow = false;
    hull.raycast = () => {}; // tools pick the model, never its outline
    hull.onBeforeRender = (renderer) => trackTargetSize(mat, renderer);
    mesh.add(hull);
    hullMats.push(mat);
    hulls.push(hull);
  }

  return {
    root,
    scene,
    height: fitted.y,
    radius: fitted.length() / 2,
    footprint: Math.hypot(fitted.x, fitted.z) / 2,
    points,
    hasMap,
    skinned,
    vertices,
    faces,
    setClay(on) {
      for (const [mesh, m] of on ? clay : textured) mesh.material = m;
    },
    setWire(on) {
      // Every material this model owns (textured and clay), so Clay keeps the wire, as legacy Texture does.
      for (const m of made) (m as Material & { wireframe?: boolean }).wireframe = on;
      for (const h of hulls) h.visible = !on;
    },
    dispose() {
      for (const m of made) {
        const mm = m as SrcMat;
        mm.map?.dispose();
        mm.alphaMap?.dispose();
        m.dispose();
      }
      for (const m of hullMats) m.dispose();
      scene.traverse((o) => {
        const mesh = o as Mesh;
        if (mesh.isMesh) mesh.geometry.dispose();
      });
    },
  };
}

/** Every k-th vertex of the meshes in root space, y shifted down by `mid` (the diorama lowers the model so). */
function samplePoints(meshes: Mesh[], mid: number, max = 4000): Float32Array {
  const total = meshes.reduce((n, m) => n + (m.geometry.attributes.position?.count ?? 0), 0);
  const step = Math.max(1, Math.ceil(total / max));
  const out: number[] = [];
  const v = new Vector3();
  for (const mesh of meshes) {
    const pos = mesh.geometry.attributes.position;
    if (!pos) continue;
    for (let i = 0; i < pos.count; i += step) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      out.push(v.x, v.y - mid, v.z);
    }
  }
  return new Float32Array(out);
}
