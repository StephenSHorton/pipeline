// Turn a parsed GLB into a diorama piece: Lambert materials (Mokei's flat, no-tone-mapping lights turn PBR
// dark), shadows on, scaled to yard size, centred on x/z with its feet on y=0. The clay look is a second
// material set swapped by reference, so the AnimationMixer (bound to this scene) keeps working.
import {
  Box3,
  Group,
  MeshLambertMaterial,
  NearestFilter,
  Vector3,
  type Material,
  type Mesh,
  type Object3D,
  type SkinnedMesh,
  type Texture,
} from "three";
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

  scene.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh) return;
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

  return {
    root,
    scene,
    height: fitted.y,
    radius: fitted.length() / 2,
    footprint: Math.hypot(fitted.x, fitted.z) / 2,
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
    },
    dispose() {
      for (const m of made) {
        const mm = m as SrcMat;
        mm.map?.dispose();
        mm.alphaMap?.dispose();
        m.dispose();
      }
      scene.traverse((o) => {
        const mesh = o as Mesh;
        if (mesh.isMesh) mesh.geometry.dispose();
      });
    },
  };
}
