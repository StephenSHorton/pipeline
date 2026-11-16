// The sprite stack's canvas: public/stack-lab.js createStackLab (4-1822) minus the DOM it wrote into. Scene, lights,
// ground and framing (139-376), the slab displays fillMesh / fillPlane / fillContour (913-1163), the chunked rebuild
// (1204-1227), the pixel pass and render (54-137, 1747-1772), the pointer side of the stack paint (index.html
// 5564-5590). New here: it mounts into a container and has dispose() (legacy never freed its WebGL context), the
// dock pushes state in with set(patch) instead of the lab reading DOM sliders, and it runs its own frame loop.
import {
  ACESFilmicToneMapping,
  type BufferGeometry,
  BoxGeometry,
  CanvasTexture,
  CircleGeometry,
  Color,
  DepthFormat,
  DepthTexture,
  DirectionalLight,
  DoubleSide,
  ExtrudeGeometry,
  type ExtrudeGeometryOptions,
  Group,
  HemisphereLight,
  InstancedBufferAttribute,
  InstancedMesh,
  LinearFilter,
  type Material,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  type MeshStandardMaterialParameters,
  MOUSE,
  type Object3D,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Raycaster,
  RGBAFormat,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Timer,
  UnsignedByteType,
  UnsignedIntType,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  applyPaletteEntry,
  beginStroke,
  buildMergedVoxelSurface,
  cellKept,
  type Core,
  createCore,
  crunchSize,
  layerShade,
  loopsToShapes,
  marchingSquares,
  nativeFromPixels,
  occupancyGrid,
  paintSphere,
  pbrOf,
  quantize,
  rebuildIslandMask,
  restBox,
  sampleVoxel,
  sheetPixels,
  type Slice,
  undoStroke,
} from "./core";
import { createGameLook, type GameLookState } from "./gameLook";
import { injectGrade, injectPixelBlit, injectVoxelPbr } from "./shaders";
import { hexToRgb, ORBIT, type PaletteEntry, type StackMeta, type StackState, worldPaintRadius } from "./stack";

export interface PaintMode {
  on: boolean;
  drop: boolean;
  erase: boolean;
  /** "#rrggbb" */
  color: string;
  /** Brush, as a fraction of the stack's diagonal. */
  radius: number;
}

export interface StackViewEvents {
  /** Drop sampled this colour (the dock sets the brush colour and turns Drop off). */
  onSample?: (rgb: [number, number, number]) => void;
  /** A stack is on screen (its layer count and resolution), or none is. */
  onChange?: (info: { ready: boolean; layers: number; resolution: number }) => void;
}

export interface StackView {
  /** Draw this stack (stack.json + sheet.png) with the dock's state. */
  load(meta: StackMeta, sheet: CanvasImageSource, state: StackState): void;
  /** Empty the canvas (Slicing…, Generating…, or no stack). */
  clear(): void;
  set(patch: Partial<StackState>): void;
  setPaint(p: PaintMode): void;
  setGame(patch: Partial<GameLookState>): void;
  quantize(k: number): PaletteEntry[];
  setPaletteEntry(i: number, patch: Partial<PaletteEntry>): void;
  selectPalette(i: number): void;
  undoStroke(): void;
  getSheetBlob(): Promise<Blob | null>;
  dispose(): void;
}

/** The sheet as an image; rejects on a miss. */
export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("no sheet"));
    im.src = url;
  });
}

interface Rec extends Slice {
  group: Group;
  mesh: InstancedMesh | null;
  plane: Mesh | null;
  contour: Group | null;
  sliceCanvas?: HTMLCanvasElement;
  sliceCtx?: CanvasRenderingContext2D;
  sliceTex?: CanvasTexture;
}

export function createStackView(container: HTMLElement, initial: StackState, game: Partial<GameLookState>, events: StackViewEvents = {}): StackView {
  const state: StackState = { ...initial };

  const canvas = document.createElement("canvas");
  // Inline, not a class: setSize(w, h, false) leaves the display size to CSS, and a class on a node built here
  // only exists if Tailwind's scan picks it up.
  canvas.style.cssText = "display:block;width:100%;height:100%;touch-action:none;outline:none";
  container.appendChild(canvas);

  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = SRGBColorSpace;

  const pixelCam = new OrthographicCamera(-1, 1, 1, -1, -1, 1);
  const pixelScene = new Scene();
  const pixelUniforms = {
    tDepth: { value: null as DepthTexture | null },
    resolution: { value: new Vector2(320, 180) },
    outline: { value: 1 },
  };
  const pixelMat = new MeshBasicMaterial({
    toneMapped: false,
    depthTest: false,
    depthWrite: false,
  });
  pixelMat.onBeforeCompile = (shader) => injectPixelBlit(shader, pixelUniforms);
  const pixelQuad = new Mesh(new PlaneGeometry(2, 2), pixelMat);
  pixelScene.add(pixelQuad);
  let pixelRT: WebGLRenderTarget | null = null;

  function ensurePixelTarget() {
    const canvasSize = renderer.getSize(new Vector2());
    const pr = renderer.getPixelRatio();
    const dw = Math.max(1, Math.round(canvasSize.x * pr));
    const dh = Math.max(1, Math.round(canvasSize.y * pr));
    const scale = Math.max(1, state.pixelSize);
    const lw = Math.max(32, Math.floor(dw / scale));
    const lh = Math.max(32, Math.floor(dh / scale));
    if (pixelRT && pixelRT.width === lw && pixelRT.height === lh) {
      pixelUniforms.outline.value = state.outline ? 1 : 0;
      return;
    }
    if (pixelRT) {
      pixelRT.depthTexture?.dispose();
      pixelRT.dispose();
    }
    const depth = new DepthTexture(lw, lh);
    depth.format = DepthFormat;
    depth.type = UnsignedIntType;
    pixelRT = new WebGLRenderTarget(lw, lh, {
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      format: RGBAFormat,
      type: UnsignedByteType,
      depthBuffer: true,
      depthTexture: depth,
    });
    pixelRT.texture.colorSpace = SRGBColorSpace;
    pixelMat.map = pixelRT.texture;
    pixelMat.needsUpdate = true;
    pixelUniforms.tDepth.value = depth;
    pixelUniforms.resolution.value.set(lw, lh);
    pixelUniforms.outline.value = state.outline ? 1 : 0;
  }

  const scene = new Scene();
  scene.background = new Color(0x0c0d0b);
  const camera = new PerspectiveCamera(35, 1, 0.05, 200);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = ORBIT.enableDamping;
  controls.dampingFactor = ORBIT.dampingFactor;
  controls.enablePan = ORBIT.enablePan;
  controls.minDistance = ORBIT.minDistance;
  controls.maxDistance = ORBIT.maxDistance;
  controls.minPolarAngle = ORBIT.minPolarAngle;
  controls.maxPolarAngle = ORBIT.maxPolarAngle;
  // Sync: legacy followed the mesh pane's camera, which its Orbit button turned at 0.7. There is no mesh pane on
  // this stage, so the stack turns itself at that speed; OrbitControls pauses the turn while you drag.
  controls.autoRotateSpeed = ORBIT.autoRotateSpeed;
  controls.autoRotate = state.auto;
  const ORBIT_BUTTONS = { ...controls.mouseButtons };

  scene.add(new HemisphereLight(0xe8e2d4, 0x1a1c16, 0.7));
  const key = new DirectionalLight(0xfff3d4, 1.15);
  key.position.set(2.4, 3.2, 1.6);
  scene.add(key);
  const fill = new DirectionalLight(0xe8e2d4, 0.55);
  fill.position.set(-1.4, 1.2, 3.2);
  scene.add(fill);
  const rim = new DirectionalLight(0x9bb08a, 0.45);
  rim.position.set(-2.2, 1.4, -1.8);
  scene.add(rim);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  const ground = new Mesh(new CircleGeometry(4.5, 64), new MeshBasicMaterial({ color: 0x141510 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const blob = new Mesh(new CircleGeometry(0.55, 48), new MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.32 }));
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = 0.004;
  scene.add(blob);
  const stackGroup = new Group();
  scene.add(stackGroup);
  // The same Game look as the mesh viewer, from the shared pipeline.pixelLook record.
  const gameLook = createGameLook({ renderer, scene, camera, controls, subject: stackGroup, key, ground, blob, defaults: game });
  const brushBall = new Mesh(
    new SphereGeometry(1, 20, 12),
    new MeshBasicMaterial({ color: 0x2ecc71, transparent: true, opacity: 0.28, depthWrite: false }),
  );
  brushBall.visible = false;
  scene.add(brushBall);

  const voxelGeo = new BoxGeometry(1, 1, 1);
  const bake = document.createElement("canvas");
  const bakeCtx = bake.getContext("2d", { willReadFrequently: true })!;
  let seenBin = new Int32Array(1);
  let voxelJob = 0;
  let texelsTimer: ReturnType<typeof setTimeout> | undefined;
  let roundTimer: ReturnType<typeof setTimeout> | undefined;
  let isoMesh: Mesh | null = null;
  let core: Core | null = null;
  let meta: StackMeta | null = null;
  let sheet: CanvasImageSource | null = null;
  let stackSlices: Rec[] = [];
  let ready = false;
  let paint: PaintMode = { on: false, drop: false, erase: false, color: "#2ecc71", radius: 0.04 };

  const grade = {
    brightness: { value: state.brightness },
    contrast: { value: state.contrast },
    saturation: { value: state.saturation },
  };

  function attachGrade(mat: Material, cacheKey: string) {
    mat.customProgramCacheKey = () => cacheKey;
    mat.onBeforeCompile = (shader) => injectGrade(shader, grade);
  }

  const isoMat = new MeshStandardMaterial({
    vertexColors: true,
    metalness: 0,
    roughness: 0.7,
    envMapIntensity: 0.35,
  });
  attachGrade(isoMat, "stack-iso-v1");

  const voxelMat = new MeshStandardMaterial({ metalness: 0, roughness: 0.7, envMapIntensity: 0.35 });
  voxelMat.customProgramCacheKey = () => "voxel-pbr-v2";
  voxelMat.onBeforeCompile = (shader) => injectVoxelPbr(shader, grade);

  function applyGrade() {
    grade.brightness.value = state.brightness;
    grade.contrast.value = state.contrast;
    grade.saturation.value = state.saturation;
  }

  function setEmpty(on: boolean) {
    ready = !on;
    events.onChange?.({ ready, layers: meta?.layers ?? 0, resolution: meta?.resolution ?? 0 });
  }

  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function applyStackState() {
    if (!meta || !stackSlices.length) return;
    const gap = meta.pitch * Math.max(0.55, state.spacing);
    const thick = meta.pitch * Math.max(0.05, state.height);
    const used = Math.max(1, Math.min(state.layers, meta.layers));
    for (const rec of stackSlices) {
      rec.group.position.y = rec.index * gap;
      rec.group.scale.set(1, state.look === "plane" ? 1 : thick, 1);
      rec.group.visible = rec.index < used;
    }
  }

  function frameStack() {
    if (!meta) return;
    applyStackState();
    const box = restBox(meta);
    const min = new Vector3(...box.min);
    const max = new Vector3(...box.max);
    const size = max.clone().sub(min);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const longest = Math.max(size.x, size.y, size.z) || 1;
    const fov = (camera.fov * Math.PI) / 180;
    const aspect = Math.max(camera.aspect, 0.08);
    const pad = 1.32;
    const halfV = Math.tan(fov / 2);
    const fitH = (size.y * pad) / (2 * halfV);
    const fitW = (Math.max(size.x, size.z) * pad) / (2 * halfV * aspect);
    const dist = Math.max(fitH, fitW, 0.8);
    const dir = new Vector3(0.72, 0.38, 1).normalize();
    camera.position.copy(center).addScaledVector(dir, dist);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.minDistance = Math.max(0.25, dist * 0.2);
    controls.maxDistance = Math.max(48, dist * 6);
    controls.update();
    blob.scale.setScalar(longest * 0.45);
  }

  /** Legacy cacheNative: cut slab rec.index out of the sheet and keep its opaque texels. */
  function cacheNative(rec: Rec) {
    const res = meta!.resolution;
    if (bake.width !== res) {
      bake.width = res;
      bake.height = res;
    }
    bakeCtx.imageSmoothingEnabled = false;
    bakeCtx.clearRect(0, 0, res, res);
    bakeCtx.drawImage(sheet!, rec.index * res, 0, res, res, 0, 0, res, res);
    rec.native = nativeFromPixels(bakeCtx.getImageData(0, 0, res, res).data, res);
  }

  function disposeIso() {
    if (!isoMesh) return;
    isoMesh.removeFromParent();
    isoMesh.geometry.dispose();
    isoMesh = null;
  }

  function setCubeVisible(on: boolean) {
    for (const rec of stackSlices) {
      if (rec.mesh) rec.mesh.visible = on;
    }
  }

  function rebuildSmoothSurface() {
    disposeIso();
    if (state.look !== "voxel" || !meta || !core || state.round <= 0) {
      setCubeVisible(true);
      applyStackState();
      return;
    }
    for (const rec of stackSlices) {
      if (!rec.native) cacheNative(rec);
    }
    const geo = buildMergedVoxelSurface(core, crunchSize(core), state.round);
    if (!geo) {
      setCubeVisible(true);
      applyStackState();
      return;
    }
    setCubeVisible(false);
    isoMesh = new Mesh(geo, isoMat);
    isoMesh.frustumCulled = false;
    stackGroup.add(isoMesh);
  }

  function scheduleSmooth() {
    clearTimeout(roundTimer);
    roundTimer = setTimeout(() => rebuildSmoothSurface(), 40);
  }

  function ensureSliceTexture(rec: Rec) {
    const res = meta!.resolution;
    if (!rec.sliceCanvas) {
      rec.sliceCanvas = document.createElement("canvas");
      rec.sliceCanvas.width = res;
      rec.sliceCanvas.height = res;
      rec.sliceCtx = rec.sliceCanvas.getContext("2d")!;
      rec.sliceTex = new CanvasTexture(rec.sliceCanvas);
      rec.sliceTex.flipY = false;
      rec.sliceTex.colorSpace = SRGBColorSpace;
      rec.sliceTex.magFilter = LinearFilter;
      rec.sliceTex.minFilter = LinearFilter;
    }
    const ctx = rec.sliceCtx!;
    const img = ctx.createImageData(res, res);
    img.data.fill(0);
    const native = rec.native;
    if (native) {
      for (let i = 0; i < native.n; i++) {
        const crunch = crunchSize(core!);
        const gx = crunch >= res ? native.ix[i] : Math.min(crunch - 1, ((native.ix[i] * crunch) / res) | 0);
        const gz = crunch >= res ? native.iz[i] : Math.min(crunch - 1, ((native.iz[i] * crunch) / res) | 0);
        if (!cellKept(core!, rec.index, gx, gz, crunch)) continue;
        const p = (native.iz[i] * res + native.ix[i]) * 4;
        img.data[p] = native.rgb[i * 3];
        img.data[p + 1] = native.rgb[i * 3 + 1];
        img.data[p + 2] = native.rgb[i * 3 + 2];
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    rec.sliceTex!.needsUpdate = true;
    return rec.sliceTex!;
  }

  function makeSliceMaterial(rec: Rec, extra: MeshStandardMaterialParameters & { cacheKey?: string } = {}) {
    const shade = layerShade(core!, rec.index);
    const { cacheKey, ...params } = extra;
    const mat = new MeshStandardMaterial({
      map: ensureSliceTexture(rec),
      color: new Color(shade, shade, shade),
      metalness: 0,
      roughness: 0.7,
      envMapIntensity: 0.35,
      alphaTest: 0.12,
      depthWrite: true,
      ...params,
    });
    attachGrade(mat, cacheKey || "stack-slice-v1");
    return mat;
  }

  function capUvgen(world: number): ExtrudeGeometryOptions["UVGenerator"] {
    return {
      generateTopUV(_geometry, vertices, a, b, c) {
        const to = (i: number) => new Vector2(vertices[i * 3] / world + 0.5, -vertices[i * 3 + 1] / world + 0.5);
        return [to(a), to(b), to(c)];
      },
      generateSideWallUV(_geometry, vertices, a, b, c, d) {
        const to = (i: number) => new Vector2(vertices[i * 3] / world + 0.5, -vertices[i * 3 + 1] / world + 0.5);
        return [to(a), to(b), to(c), to(d)];
      },
    };
  }

  function disposeObject(obj: Object3D | null) {
    if (!obj) return;
    obj.removeFromParent();
    const seen = new Set<Material>();
    obj.traverse((o) => {
      const m = o as Mesh;
      if (m.geometry && m.geometry !== voxelGeo) m.geometry.dispose();
      const mats = m.material ? (Array.isArray(m.material) ? m.material : [m.material]) : [];
      for (const mt of mats) {
        if (!mt || mt === voxelMat || seen.has(mt)) continue;
        seen.add(mt);
        mt.dispose();
      }
    });
  }

  function clearDisplay(rec: Rec) {
    disposeObject(rec.mesh);
    disposeObject(rec.plane);
    disposeObject(rec.contour);
    rec.mesh = null;
    rec.plane = null;
    rec.contour = null;
  }

  function fillMesh(rec: Rec, crunch: number) {
    const native = rec.native;
    if (!native || !native.n || !meta || !core) return;
    const res = meta.resolution;
    const world = res * meta.pitch;
    const texel = world / crunch;
    const shade = layerShade(core, rec.index);
    if (!rec.mesh) {
      const geo = voxelGeo.clone();
      geo.setAttribute("aPBR", new InstancedBufferAttribute(new Float32Array(native.n * 3), 3));
      const mesh = new InstancedMesh(geo, voxelMat, native.n);
      mesh.frustumCulled = false;
      mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(native.n * 3), 3);
      rec.group.add(mesh);
      rec.mesh = mesh;
    }
    const mesh = rec.mesh;
    const me = mesh.instanceMatrix.array;
    const ca = mesh.instanceColor!.array;
    const aPBR = mesh.geometry.getAttribute("aPBR");
    let count = 0;
    const write = (gx: number, gz: number, r: number, g: number, b: number, metal: number, rough: number, emit: number) => {
      const o = count * 16;
      const s = texel * 1.02;
      const wx = ((gx + 0.5) / crunch - 0.5) * world;
      const wz = ((gz + 0.5) / crunch - 0.5) * world;
      me[o] = s; me[o + 1] = 0; me[o + 2] = 0; me[o + 3] = 0;
      me[o + 4] = 0; me[o + 5] = 1; me[o + 6] = 0; me[o + 7] = 0;
      me[o + 8] = 0; me[o + 9] = 0; me[o + 10] = s; me[o + 11] = 0;
      me[o + 12] = wx; me[o + 13] = 0; me[o + 14] = wz; me[o + 15] = 1;
      const c = count * 3;
      ca[c] = (r * shade) / 255;
      ca[c + 1] = (g * shade) / 255;
      ca[c + 2] = (b * shade) / 255;
      if (aPBR) aPBR.setXYZ(count, metal, rough, emit);
      count++;
    };
    if (crunch >= res) {
      for (let i = 0; i < native.n; i++) {
        if (!cellKept(core, rec.index, native.ix[i], native.iz[i], crunch)) continue;
        const [m, rh, em] = pbrOf(core, native, i);
        write(native.ix[i], native.iz[i], native.rgb[i * 3], native.rgb[i * 3 + 1], native.rgb[i * 3 + 2], m, rh, em);
      }
    } else {
      const bins = crunch * crunch;
      if (seenBin.length < bins) seenBin = new Int32Array(bins);
      seenBin.fill(-1, 0, bins);
      const scale = crunch / res;
      for (let i = 0; i < native.n; i++) {
        const gx = Math.min(crunch - 1, (native.ix[i] * scale) | 0);
        const gz = Math.min(crunch - 1, (native.iz[i] * scale) | 0);
        if (!cellKept(core, rec.index, gx, gz, crunch)) continue;
        const k = gz * crunch + gx;
        const idx = seenBin[k];
        const r = native.rgb[i * 3];
        const g = native.rgb[i * 3 + 1];
        const b = native.rgb[i * 3 + 2];
        const [m, rh, em] = pbrOf(core, native, i);
        if (idx < 0) {
          seenBin[k] = count;
          write(gx, gz, r, g, b, m, rh, em);
        } else {
          const c = idx * 3;
          ca[c] = (r * shade) / 255;
          ca[c + 1] = (g * shade) / 255;
          ca[c + 2] = (b * shade) / 255;
          if (aPBR) aPBR.setXYZ(idx, m, rh, em);
        }
      }
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor!.needsUpdate = true;
    if (aPBR) aPBR.needsUpdate = true;
    mesh.visible = state.round <= 0;
  }

  function fillPlane(rec: Rec) {
    const world = meta!.resolution * meta!.pitch;
    const geo = new PlaneGeometry(world, world);
    geo.rotateX(-Math.PI / 2);
    const uv = geo.attributes.uv;
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      uv.setXY(i, pos.getX(i) / world + 0.5, pos.getZ(i) / world + 0.5);
    }
    uv.needsUpdate = true;
    const mat = makeSliceMaterial(rec, {
      side: DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -rec.index,
      polygonOffsetUnits: 1,
      cacheKey: "stack-plane-v1",
    });
    const mesh = new Mesh(geo, mat);
    mesh.frustumCulled = false;
    rec.group.add(mesh);
    rec.plane = mesh;
  }

  function fillContour(rec: Rec, crunch: number) {
    const world = meta!.resolution * meta!.pitch;
    const occ = occupancyGrid(core!, rec, crunch);
    const loops = marchingSquares(occ, crunch);
    const shapes = loopsToShapes(loops, crunch, world);
    const group = new Group();
    const mat = makeSliceMaterial(rec, {
      side: DoubleSide,
      alphaTest: 0.04,
      cacheKey: "stack-contour-v1",
    });
    const uvgen = capUvgen(world);
    for (const shape of shapes) {
      let geo: BufferGeometry;
      try {
        geo = new ExtrudeGeometry(shape, {
          depth: 1,
          steps: 1,
          bevelEnabled: false,
          curveSegments: 1,
          UVGenerator: uvgen,
        });
      } catch {
        continue;
      }
      geo.rotateX(-Math.PI / 2);
      geo.translate(0, -0.5, 0);
      geo.computeVertexNormals();
      const mesh = new Mesh(geo, mat);
      mesh.frustumCulled = false;
      group.add(mesh);
    }
    rec.group.add(group);
    rec.contour = group;
  }

  function rebuildRec(rec: Rec, crunch: number) {
    if (!rec.native) cacheNative(rec);
    if (state.look === "voxel") {
      if (!rec.mesh || rec.plane || rec.contour) {
        clearDisplay(rec);
      }
      fillMesh(rec, crunch);
      return;
    }
    clearDisplay(rec);
    if (state.look === "plane") fillPlane(rec);
    else fillContour(rec, crunch);
  }

  /** Repaint the slabs whose colours changed (a stroke, an undo, a swatch). */
  function refill(indices: number[]) {
    if (!core) return;
    const crunch = crunchSize(core);
    for (const i of indices) {
      const rec = stackSlices[i];
      if (!rec || !rec.native) continue;
      if (state.look === "voxel") fillMesh(rec, crunch);
      else ensureSliceTexture(rec);
    }
    // Round shows one welded surface instead of the cubes, so it has to be rebuilt to show new colours. Legacy
    // only did this after Reduce, so a stroke on the default look (Round 0.45) stayed invisible.
    if (state.round > 0 && state.look === "voxel") scheduleSmooth();
    else applyStackState();
  }

  function rebuildLookAsync(): Promise<void> {
    const job = ++voxelJob;
    clearTimeout(texelsTimer);
    if (!meta || !core) return Promise.resolve();
    const c = core;
    const crunch = crunchSize(c);
    const chunk = state.look === "contour" ? 4 : 10;
    return (async () => {
      for (const rec of stackSlices) {
        if (!rec.native) cacheNative(rec);
      }
      rebuildIslandMask(c, crunch);
      for (let i = 0; i < stackSlices.length; i++) {
        if (job !== voxelJob) return;
        rebuildRec(stackSlices[i], crunch);
        if (i % chunk === chunk - 1) {
          applyStackState();
          await new Promise((r) => requestAnimationFrame(r));
        }
      }
      if (job === voxelJob) {
        applyStackState();
        rebuildSmoothSurface();
      }
    })();
  }

  function clearStack() {
    voxelJob += 1;
    clearTimeout(roundTimer);
    clearTimeout(texelsTimer);
    disposeIso();
    for (const rec of stackSlices) {
      clearDisplay(rec);
      rec.sliceTex?.dispose();
      rec.group.removeFromParent();
    }
    stackSlices = [];
    sheet = null;
    meta = null;
    core = null;
    brushBall.visible = false;
  }

  /* ---------------------------------------------------------------- paint (index.html 5372-5590) */

  const paintRay = new Raycaster();
  const paintNdc = new Vector2();

  function pick(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    paintNdc.set(((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1, -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1);
    paintRay.setFromCamera(paintNdc, gameLook.viewCamera());
    const hits = paintRay.intersectObject(stackGroup, true);
    return hits[0] || null;
  }

  const brushRadius = () => worldPaintRadius(meta ? restBox(meta).diagonal : 0, paint.radius);

  function setBrush(point: Vector3 | null) {
    if (!point) {
      brushBall.visible = false;
      return;
    }
    brushBall.visible = true;
    brushBall.position.copy(point);
    brushBall.scale.setScalar(brushRadius() || 0.05);
  }

  function applyPaintAt(point: Vector3) {
    if (!core) return;
    const r = paintSphere(core, point, brushRadius(), hexToRgb(paint.color), paint.erase);
    if (r.dirty.length) refill(r.dirty);
  }

  function onPointerMove(e: PointerEvent) {
    if (!paint.on || !ready) return;
    const hit = pick(e.clientX, e.clientY);
    setBrush(hit ? hit.point : null);
    if (hit && e.buttons & 1 && !paint.drop) applyPaintAt(hit.point);
  }

  function onPointerDown(e: PointerEvent) {
    if (!paint.on || e.button !== 0 || !ready || !core) return;
    const hit = pick(e.clientX, e.clientY);
    if (!hit) return;
    e.preventDefault();
    if (paint.drop) {
      const rgb = sampleVoxel(core, hit.point);
      if (rgb) events.onSample?.(rgb);
      return;
    }
    beginStroke(core);
    applyPaintAt(hit.point);
  }

  const onPointerLeave = () => setBrush(null);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointerleave", onPointerLeave);

  /* ---------------------------------------------------------------- frame loop + size */

  const timer = new Timer();
  let raf = 0;
  function frame(ts: number) {
    raf = requestAnimationFrame(frame);
    timer.update(ts);
    controls.autoRotate = state.auto;
    controls.update(timer.getDelta());
    if (!ready) return;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.outputColorSpace = SRGBColorSpace;
    if (gameLook.state.on) {
      gameLook.render();
    } else if (state.pixel) {
      ensurePixelTarget();
      renderer.setRenderTarget(pixelRT);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      renderer.render(pixelScene, pixelCam);
    } else {
      renderer.render(scene, camera);
    }
  }
  raf = requestAnimationFrame(frame);

  const ro = new ResizeObserver(() => resize());
  ro.observe(container);
  resize();

  /* ---------------------------------------------------------------- the API */

  function set(patch: Partial<StackState>) {
    const prev = { ...state };
    Object.assign(state, patch);
    const changed = (k: keyof StackState) => prev[k] !== state[k];
    if (changed("outline")) pixelUniforms.outline.value = state.outline ? 1 : 0;
    if (changed("brightness") || changed("contrast") || changed("saturation")) applyGrade();
    if (!meta) return;
    const rebuild =
      changed("look") || changed("shade") || changed("main") || (changed("layers") && (state.round > 0 || state.main));
    if (rebuild) {
      applyStackState();
      void rebuildLookAsync();
      return;
    }
    if (changed("texels")) {
      // A Texels drag rebuilds 40 ms after it settles, as legacy.
      clearTimeout(texelsTimer);
      texelsTimer = setTimeout(() => void rebuildLookAsync(), 40);
    }
    if (changed("spacing") || changed("height") || changed("layers")) applyStackState();
    if (changed("round") || (changed("spacing") && state.round > 0)) scheduleSmooth();
  }

  return {
    load(m, img, s) {
      clearStack();
      Object.assign(state, s);
      meta = m;
      sheet = img;
      core = createCore(m, state);
      applyGrade();
      pixelUniforms.outline.value = state.outline ? 1 : 0;
      for (let i = 0; i < m.layers; i++) {
        const rec: Rec = { index: i, group: new Group(), mesh: null, plane: null, contour: null, native: null };
        stackGroup.add(rec.group);
        stackSlices.push(rec);
      }
      // The core's slices are these records, so its paint and palette edits land on what is drawn.
      core.slices = stackSlices;
      resize();
      frameStack();
      setEmpty(false);
      void rebuildLookAsync();
    },
    clear() {
      clearStack();
      setEmpty(true);
    },
    set,
    setPaint(p) {
      const was = paint.on;
      paint = { ...p };
      if (p.on !== was) {
        brushBall.visible = false;
        // Painting takes the left button; right-drag orbits (the Paint title says so; legacy's right-drag panned).
        controls.mouseButtons = p.on ? { LEFT: null, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE } : { ...ORBIT_BUTTONS };
      }
    },
    setGame(patch) {
      gameLook.set(patch);
    },
    quantize(k) {
      if (!core) return [];
      const pal = quantize(core, k);
      refill(stackSlices.map((r) => r.index));
      return pal;
    },
    setPaletteEntry(i, patch) {
      if (!core) return;
      const dirty = applyPaletteEntry(core, i, patch);
      if (dirty.length) refill(dirty);
    },
    selectPalette(i) {
      if (core) core.palSelected = i;
    },
    undoStroke() {
      if (core && undoStroke(core)) refill(stackSlices.map((r) => r.index));
    },
    getSheetBlob() {
      if (!core) return Promise.resolve(null);
      const px = sheetPixels(core);
      const c = document.createElement("canvas");
      c.width = px.width;
      c.height = px.height;
      c.getContext("2d")!.putImageData(new ImageData(px.data, px.width, px.height), 0, 0);
      return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/png"));
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      clearStack();
      controls.dispose();
      gameLook.dispose();
      pixelRT?.depthTexture?.dispose();
      pixelRT?.dispose();
      for (const m of [pixelMat, isoMat, voxelMat, ground.material, blob.material, brushBall.material]) m.dispose();
      for (const g of [pixelQuad.geometry, voxelGeo, ground.geometry, blob.geometry, brushBall.geometry]) g.dispose();
      renderer.dispose();
      // Without this the browser keeps the context until GC, and a few remounts hit its context limit.
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
