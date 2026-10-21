/** Sprite-stack pane for the factory inspector. Mesh stays in the main viewer. */
import { createPixelLook } from "./pixel-look.js";

export function createStackLab({ THREE, OrbitControls, pane, statsEl, emptyEl, sliders }) {
  const LOOKS = ["voxel", "plane", "contour"];
  const urlParams = new URLSearchParams(location.search);
  const urlLook = urlParams.get("look");
  const DEFAULTS = {
    spacing: 1,
    height: 1,
    texels: 120,
    brightness: 1,
    contrast: 1.2,
    saturation: 1.2,
    shade: true,
    auto: true,
    look: LOOKS.includes(urlLook) ? urlLook : "voxel",
    pixel: urlParams.get("pixel") !== "0",
    pixelSize: Math.max(1, Math.min(8, Number(urlParams.get("px")) || 3)),
    outline: urlParams.get("outline") !== "0",
    round: (() => {
      const raw = urlParams.get("round");
      if (raw == null || raw === "") return 0.45;
      return Math.max(0, Math.min(1, Number(raw) || 0));
    })(),
    main: urlParams.get("main") !== "0",
  };

  const state = {
    spacing: DEFAULTS.spacing,
    height: DEFAULTS.height,
    texels: DEFAULTS.texels,
    layers: 8,
    brightness: DEFAULTS.brightness,
    contrast: DEFAULTS.contrast,
    saturation: DEFAULTS.saturation,
    shade: DEFAULTS.shade,
    auto: DEFAULTS.auto,
    look: DEFAULTS.look,
    pixel: DEFAULTS.pixel,
    pixelSize: DEFAULTS.pixelSize,
    outline: DEFAULTS.outline,
    round: DEFAULTS.round,
    main: DEFAULTS.main,
  };

  const canvas = document.createElement("canvas");
  pane.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const pixelCam = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);
  const pixelScene = new THREE.Scene();
  const pixelUniforms = {
    tDepth: { value: null },
    resolution: { value: new THREE.Vector2(320, 180) },
    outline: { value: 1 },
  };
  const pixelMat = new THREE.MeshBasicMaterial({
    toneMapped: false,
    depthTest: false,
    depthWrite: false,
  });
  pixelMat.onBeforeCompile = (shader) => {
    shader.uniforms.tDepth = pixelUniforms.tDepth;
    shader.uniforms.resolution = pixelUniforms.resolution;
    shader.uniforms.outline = pixelUniforms.outline;
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform sampler2D tDepth;
uniform vec2 resolution;
uniform float outline;`
      )
      .replace(
        "#include <map_fragment>",
        `{
  vec2 filterSize = clamp(fwidth(vMapUv) * resolution, vec2(1e-5), vec2(1.0));
  vec2 texel = vMapUv * resolution - 0.5 * filterSize;
  vec2 texelOffset = smoothstep(vec2(1.0) - filterSize, vec2(1.0), fract(texel));
  vec2 filtered = (floor(texel) + 0.5 + texelOffset) / resolution;
  vec4 sampledDiffuseColor = texture2D(map, filtered);
  diffuseColor *= sampledDiffuseColor;
  if (outline > 0.5) {
    vec2 px = 1.0 / resolution;
    float d = texture2D(tDepth, filtered).r;
    float e = 0.0;
    e += abs(d - texture2D(tDepth, filtered + vec2(px.x, 0.0)).r);
    e += abs(d - texture2D(tDepth, filtered - vec2(px.x, 0.0)).r);
    e += abs(d - texture2D(tDepth, filtered + vec2(0.0, px.y)).r);
    e += abs(d - texture2D(tDepth, filtered - vec2(0.0, px.y)).r);
    float line = smoothstep(0.035, 0.12, e);
    diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * 0.72, line);
  }
}`
      );
  };
  pixelScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), pixelMat));
  let pixelRT = null;

  function ensurePixelTarget() {
    const canvasSize = renderer.getSize(new THREE.Vector2());
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
    const depth = new THREE.DepthTexture(lw, lh);
    depth.format = THREE.DepthFormat;
    depth.type = THREE.UnsignedIntType;
    pixelRT = new THREE.WebGLRenderTarget(lw, lh, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      depthTexture: depth,
    });
    pixelRT.texture.colorSpace = THREE.SRGBColorSpace;
    pixelMat.map = pixelRT.texture;
    pixelMat.needsUpdate = true;
    pixelUniforms.tDepth.value = depth;
    pixelUniforms.resolution.value.set(lw, lh);
    pixelUniforms.outline.value = state.outline ? 1 : 0;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0d0b);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 200);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enablePan = true;
  controls.minDistance = 0.3;
  controls.maxDistance = 48;
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = Math.PI;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.7;

  scene.add(new THREE.HemisphereLight(0xe8e2d4, 0x1a1c16, 0.7));
  const key = new THREE.DirectionalLight(0xfff3d4, 1.15);
  key.position.set(2.4, 3.2, 1.6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xe8e2d4, 0.55);
  fill.position.set(-1.4, 1.2, 3.2);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0x9bb08a, 0.45);
  rim.position.set(-2.2, 1.4, -1.8);
  scene.add(rim);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(4.5, 64),
    new THREE.MeshBasicMaterial({ color: 0x141510 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const blob = new THREE.Mesh(
    new THREE.CircleGeometry(0.55, 48),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.32 })
  );
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = 0.004;
  scene.add(blob);
  const stackGroup = new THREE.Group();
  scene.add(stackGroup);
  // Same game look as the mesh pane; the mesh pane's Game look controls drive both.
  const gameLook = createPixelLook({ THREE, renderer, scene, camera, controls, subject: stackGroup, key, ground, blob });
  const brushBall = new THREE.Mesh(
    new THREE.SphereGeometry(1, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0x2ecc71, transparent: true, opacity: 0.28, depthWrite: false })
  );
  brushBall.visible = false;
  scene.add(brushBall);

  const voxelGeo = new THREE.BoxGeometry(1, 1, 1);
  const bake = document.createElement("canvas");
  const bakeCtx = bake.getContext("2d", { willReadFrequently: true });
  let seenBin = new Int32Array(1);
  let voxelJob = 0;
  let texelsTimer = 0;
  let roundTimer = 0;
  let isoMesh = null;
  let islandMask = null;
  let islandCrunch = 0;
  let meta = null;
  let sheet = null;
  let stackSlices = [];
  let ready = false;
  let viewRadius = 4;
  const _sph = new THREE.Spherical();
  const _offset = new THREE.Vector3();

  const grade = {
    brightness: { value: DEFAULTS.brightness },
    contrast: { value: DEFAULTS.contrast },
    saturation: { value: DEFAULTS.saturation },
  };
  let palette = null;
  let palSelected = 0;

  function injectGrade(shader) {
    shader.uniforms.uBrightness = grade.brightness;
    shader.uniforms.uContrast = grade.contrast;
    shader.uniforms.uSaturation = grade.saturation;
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;`
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  vec3 g = diffuseColor.rgb * uBrightness;
  g = (g - 0.5) * uContrast + 0.5;
  float luma = dot(g, vec3(0.2126, 0.7152, 0.0722));
  g = mix(vec3(luma), g, uSaturation);
  diffuseColor.rgb = clamp(g, 0.0, 1.0);
}`
      );
  }

  function attachGrade(mat, key) {
    mat.customProgramCacheKey = () => key;
    mat.onBeforeCompile = (shader) => injectGrade(shader);
  }

  const isoMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0,
    roughness: 0.7,
    envMapIntensity: 0.35,
  });
  attachGrade(isoMat, "stack-iso-v1");

  const voxelMat = new THREE.MeshStandardMaterial({ metalness: 0, roughness: 0.7, envMapIntensity: 0.35 });
  voxelMat.customProgramCacheKey = () => "voxel-pbr-v2";
  voxelMat.onBeforeCompile = (shader) => {
    shader.uniforms.uBrightness = grade.brightness;
    shader.uniforms.uContrast = grade.contrast;
    shader.uniforms.uSaturation = grade.saturation;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
attribute vec3 aPBR;
varying vec3 vPBR;`
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
vPBR = aPBR;`
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vPBR;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;`
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  vec3 g = diffuseColor.rgb * uBrightness;
  g = (g - 0.5) * uContrast + 0.5;
  float luma = dot(g, vec3(0.2126, 0.7152, 0.0722));
  g = mix(vec3(luma), g, uSaturation);
  diffuseColor.rgb = clamp(g, 0.0, 1.0);
}`
      )
      .replace(
        "#include <metalnessmap_fragment>",
        `#include <metalnessmap_fragment>
metalnessFactor = vPBR.x;`
      )
      .replace(
        "#include <roughnessmap_fragment>",
        `#include <roughnessmap_fragment>
roughnessFactor = vPBR.y;`
      )
      .replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
totalEmissiveRadiance += diffuseColor.rgb * vPBR.z * 8.0;`
      );
  };

  function applyGrade() {
    grade.brightness.value = state.brightness;
    grade.contrast.value = state.contrast;
    grade.saturation.value = state.saturation;
  }

  function setEmpty(on, msg) {
    ready = !on;
    pane.classList.toggle("is-empty", on);
    if (emptyEl) {
      emptyEl.hidden = !on;
      if (msg) emptyEl.textContent = msg;
    }
    if (sliders?.root) sliders.root.hidden = on;
  }

  function resize() {
    const w = Math.max(1, pane.clientWidth);
    const h = Math.max(1, pane.clientHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function layerShade(index) {
    return state.shade ? 0.52 + 0.48 * (index / Math.max(1, meta.layers - 1)) : 1;
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
    const worldX = meta.extents[0];
    const worldZ = meta.extents[2];
    const restH = meta.layers * meta.pitch;
    const min = new THREE.Vector3(-worldX * 0.5, 0, -worldZ * 0.5);
    const max = new THREE.Vector3(worldX * 0.5, restH, worldZ * 0.5);
    const size = max.clone().sub(min);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const longest = Math.max(size.x, size.y, size.z) || 1;
    const fov = camera.fov * Math.PI / 180;
    const aspect = Math.max(camera.aspect, 0.08);
    const pad = 1.32;
    const halfV = Math.tan(fov / 2);
    const fitH = (size.y * pad) / (2 * halfV);
    const fitW = (Math.max(size.x, size.z) * pad) / (2 * halfV * aspect);
    const dist = Math.max(fitH, fitW, 0.8);
    viewRadius = dist;
    const dir = new THREE.Vector3(0.72, 0.38, 1).normalize();
    camera.position.copy(center).addScaledVector(dir, dist);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.minDistance = Math.max(0.25, dist * 0.2);
    controls.maxDistance = Math.max(48, dist * 6);
    controls.update();
    blob.scale.setScalar(longest * 0.45);
  }

  function cacheNative(rec) {
    const res = meta.resolution;
    if (bake.width !== res) {
      bake.width = res;
      bake.height = res;
    }
    bakeCtx.imageSmoothingEnabled = false;
    bakeCtx.clearRect(0, 0, res, res);
    bakeCtx.drawImage(sheet, rec.index * res, 0, res, res, 0, 0, res, res);
    const data = bakeCtx.getImageData(0, 0, res, res).data;
    const ix = [];
    const iz = [];
    const rgb = [];
    for (let z = 0; z < res; z++) {
      for (let x = 0; x < res; x++) {
        const p = (z * res + x) * 4;
        if (data[p + 3] < 24) continue;
        ix.push(x);
        iz.push(z);
        rgb.push(data[p], data[p + 1], data[p + 2]);
      }
    }
    rec.native = {
      n: ix.length,
      ix: new Uint16Array(ix),
      iz: new Uint16Array(iz),
      rgb: new Uint8Array(rgb),
      rgb0: new Uint8Array(rgb),
    };
  }

  function crunchSize() {
    return Math.max(8, Math.min(state.texels, meta.resolution));
  }

  function occupancyGrid(rec, crunch) {
    const native = rec.native;
    const res = meta.resolution;
    const occ = new Uint8Array(crunch * crunch);
    if (!native || !native.n) return occ;
    const scale = crunch / res;
    for (let i = 0; i < native.n; i++) {
      const gx = crunch >= res ? native.ix[i] : Math.min(crunch - 1, (native.ix[i] * scale) | 0);
      const gz = crunch >= res ? native.iz[i] : Math.min(crunch - 1, (native.iz[i] * scale) | 0);
      if (!cellKept(rec.index, gx, gz, crunch)) continue;
      occ[gz * crunch + gx] = 1;
    }
    return occ;
  }

  function cellKept(y, gx, gz, n) {
    if (!state.main || !islandMask || n !== islandCrunch) return true;
    const ny = Math.max(1, Math.min(state.layers, meta.layers));
    if (y < 0 || y >= ny || gx < 0 || gz < 0 || gx >= n || gz >= n) return false;
    return islandMask[(y * n + gz) * n + gx] === 1;
  }

  function rebuildIslandMask(crunch) {
    islandCrunch = crunch;
    const ny = Math.max(1, Math.min(state.layers, meta.layers));
    const n = crunch;
    const N = ny * n * n;
    const occ = new Uint8Array(N);
    const res = meta.resolution;
    const scale = n / res;
    for (let y = 0; y < ny; y++) {
      const native = stackSlices[y] && stackSlices[y].native;
      if (!native) continue;
      for (let i = 0; i < native.n; i++) {
        const gx = n >= res ? native.ix[i] : Math.min(n - 1, (native.ix[i] * scale) | 0);
        const gz = n >= res ? native.iz[i] : Math.min(n - 1, (native.iz[i] * scale) | 0);
        occ[(y * n + gz) * n + gx] = 1;
      }
    }
    if (!state.main) {
      islandMask = occ;
      return;
    }
    const parent = new Int32Array(N);
    for (let i = 0; i < N; i++) parent[i] = occ[i] ? i : -1;
    const find = (a) => {
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
    const union = (a, b) => {
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent[rb] = ra;
    };
    const at = (x, y, z) => (y * n + z) * n + x;
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
    const sizes = new Map();
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
      if (sizes.get(find(i)) >= maxSize) mask[i] = 1;
    }
    islandMask = mask;
  }

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

  function laplacianSmooth(pos, adj, lambda) {
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

  function disposeIso() {
    if (!isoMesh) return;
    isoMesh.removeFromParent();
    isoMesh.geometry.dispose();
    isoMesh = null;
  }

  function setCubeVisible(on) {
    for (const rec of stackSlices) {
      if (rec.mesh) rec.mesh.visible = on;
    }
  }

  function buildMergedVoxelSurface(crunch, amount) {
    const ny = Math.max(1, Math.min(state.layers, meta.layers));
    const n = crunch;
    const occ = new Uint8Array(ny * n * n);
    const col = new Uint8Array(ny * n * n * 3);
    const res = meta.resolution;
    const scale = n / res;
    for (let y = 0; y < ny; y++) {
      const native = stackSlices[y] && stackSlices[y].native;
      if (!native) continue;
      const shade = layerShade(y);
      for (let i = 0; i < native.n; i++) {
        const gx = n >= res ? native.ix[i] : Math.min(n - 1, (native.ix[i] * scale) | 0);
        const gz = n >= res ? native.iz[i] : Math.min(n - 1, (native.iz[i] * scale) | 0);
        const idx = (y * n + gz) * n + gx;
        if (!cellKept(y, gx, gz, n)) continue;
        occ[idx] = 1;
        col[idx * 3] = Math.round(native.rgb[i * 3] * shade);
        col[idx * 3 + 1] = Math.round(native.rgb[i * 3 + 1] * shade);
        col[idx * 3 + 2] = Math.round(native.rgb[i * 3 + 2] * shade);
      }
    }
    const filled = (x, y, z) => {
      if (x < 0 || y < 0 || z < 0 || x >= n || y >= ny || z >= n) return 0;
      return occ[(y * n + z) * n + x];
    };
    const world = meta.resolution * meta.pitch;
    const cellH = meta.pitch * Math.max(0.55, state.spacing);
    const vertId = new Map();
    const positions = [];
    const colors = [];
    const indices = [];
    const triFaceN = [];
    const vert = (x, y, z, r, g, b) => {
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
    const adj = Array.from({ length: vcount }, () => []);
    const seenE = new Set();
    const addE = (i, j) => {
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
    const welded = new THREE.BufferGeometry();
    welded.setAttribute("position", new THREE.BufferAttribute(pos, 3));
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
      let mx = fx + (nx - fx) * nMix;
      let my = fy + (ny - fy) * nMix;
      let mz = fz + (nz - fz) * nMix;
      const len = Math.hypot(mx, my, mz) || 1;
      outN[o] = mx / len;
      outN[o + 1] = my / len;
      outN[o + 2] = mz / len;
      outC[o] = colors[vi * 3];
      outC[o + 1] = colors[vi * 3 + 1];
      outC[o + 2] = colors[vi * 3 + 2];
    }
    welded.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(outP, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(outN, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(outC, 3));
    return geo;
  }

  function rebuildSmoothSurface() {
    disposeIso();
    if (state.look !== "voxel" || !meta || state.round <= 0) {
      setCubeVisible(true);
      applyStackState();
      return;
    }
    for (const rec of stackSlices) {
      if (!rec.native) cacheNative(rec);
    }
    const geo = buildMergedVoxelSurface(crunchSize(), state.round);
    if (!geo) {
      setCubeVisible(true);
      applyStackState();
      return;
    }
    setCubeVisible(false);
    isoMesh = new THREE.Mesh(geo, isoMat);
    isoMesh.frustumCulled = false;
    stackGroup.add(isoMesh);
  }

  function scheduleSmooth() {
    clearTimeout(roundTimer);
    roundTimer = setTimeout(() => rebuildSmoothSurface(), 40);
  }

  function ptKey(p) {
    return `${p[0].toFixed(4)},${p[1].toFixed(4)}`;
  }

  function stitchLoops(segs) {
    const adj = new Map();
    const coord = new Map();
    const addEdge = (a, b) => {
      const ka = ptKey(a);
      const kb = ptKey(b);
      if (ka === kb) return;
      coord.set(ka, a);
      coord.set(kb, b);
      if (!adj.has(ka)) adj.set(ka, []);
      if (!adj.has(kb)) adj.set(kb, []);
      adj.get(ka).push(kb);
      adj.get(kb).push(ka);
    };
    for (const [a, b] of segs) addEdge(a, b);
    const edgeId = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
    const used = new Set();
    const loops = [];
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
          let next = null;
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
          loops.push(loop.map((k) => coord.get(k)));
        }
      }
    }
    return loops;
  }

  function marchingSquares(occ, n) {
    const sample = (x, z) => {
      if (x < 0 || z < 0 || x >= n || z >= n) return 0;
      return occ[z * n + x] ? 1 : 0;
    };
    const segs = [];
    const add = (a, b) => segs.push([a, b]);
    for (let z = -1; z < n; z++) {
      for (let x = -1; x < n; x++) {
        const v0 = sample(x, z);
        const v1 = sample(x + 1, z);
        const v2 = sample(x + 1, z + 1);
        const v3 = sample(x, z + 1);
        const idx = v0 | (v1 << 1) | (v2 << 2) | (v3 << 3);
        if (idx === 0 || idx === 15) continue;
        const t = [x + 0.5, z];
        const r = [x + 1, z + 0.5];
        const b = [x + 0.5, z + 1];
        const l = [x, z + 0.5];
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

  function cleanLoop(pts) {
    const out = [];
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

  function signedArea(pts) {
    let area = 0;
    for (let i = 0, n = pts.length; i < n; i++) {
      const a = pts[i];
      const b = pts[(i + 1) % n];
      area += a[0] * b[1] - b[0] * a[1];
    }
    return area * 0.5;
  }

  function pointInLoop(x, z, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i][0];
      const zi = pts[i][1];
      const xj = pts[j][0];
      const zj = pts[j][1];
      const hit = ((zi > z) !== (zj > z)) && (x < ((xj - xi) * (z - zi)) / ((zj - zi) || 1e-15) + xi);
      if (hit) inside = !inside;
    }
    return inside;
  }

  function pixToWorld(px, pz, n, world) {
    const wx = ((px + 0.5) / n - 0.5) * world;
    const wz = ((pz + 0.5) / n - 0.5) * world;
    // Shape Y is -Z so rotateX(-PI/2) maps extrusion to +Y without mirroring XZ.
    return [wx, -wz];
  }

  function loopsToShapes(loops, n, world) {
    const metas = [];
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
    const isOuter = new Array(metas.length).fill(true);
    const parentOf = new Array(metas.length).fill(-1);
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
    const shapes = [];
    for (let i = 0; i < metas.length; i++) {
      if (!isOuter[i]) continue;
      const m = metas[i];
      let pts = m.pts;
      if (m.area < 0) pts = pts.slice().reverse();
      const shape = new THREE.Shape();
      shape.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < pts.length; k++) shape.lineTo(pts[k][0], pts[k][1]);
      shape.closePath();
      for (const h of m.holes) {
        let hp = h.pts;
        if (h.area > 0) hp = hp.slice().reverse();
        if (hp.length < 3) continue;
        const path = new THREE.Path();
        path.moveTo(hp[0][0], hp[0][1]);
        for (let k = 1; k < hp.length; k++) path.lineTo(hp[k][0], hp[k][1]);
        path.closePath();
        shape.holes.push(path);
      }
      shapes.push(shape);
    }
    return shapes;
  }

  function ensureSliceTexture(rec) {
    const res = meta.resolution;
    if (!rec.sliceCanvas) {
      rec.sliceCanvas = document.createElement("canvas");
      rec.sliceCanvas.width = res;
      rec.sliceCanvas.height = res;
      rec.sliceCtx = rec.sliceCanvas.getContext("2d");
      rec.sliceTex = new THREE.CanvasTexture(rec.sliceCanvas);
      rec.sliceTex.flipY = false;
      rec.sliceTex.colorSpace = THREE.SRGBColorSpace;
      rec.sliceTex.magFilter = THREE.LinearFilter;
      rec.sliceTex.minFilter = THREE.LinearFilter;
    }
    const ctx = rec.sliceCtx;
    const img = ctx.createImageData(res, res);
    img.data.fill(0);
    const native = rec.native;
    if (native) {
      for (let i = 0; i < native.n; i++) {
        const crunch = crunchSize();
        const gx = crunch >= res ? native.ix[i] : Math.min(crunch - 1, (native.ix[i] * crunch / res) | 0);
        const gz = crunch >= res ? native.iz[i] : Math.min(crunch - 1, (native.iz[i] * crunch / res) | 0);
        if (!cellKept(rec.index, gx, gz, crunch)) continue;
        const p = (native.iz[i] * res + native.ix[i]) * 4;
        img.data[p] = native.rgb[i * 3];
        img.data[p + 1] = native.rgb[i * 3 + 1];
        img.data[p + 2] = native.rgb[i * 3 + 2];
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    rec.sliceTex.needsUpdate = true;
    return rec.sliceTex;
  }

  function makeSliceMaterial(rec, extra = {}) {
    const shade = layerShade(rec.index);
    const { cacheKey, ...params } = extra;
    const mat = new THREE.MeshStandardMaterial({
      map: ensureSliceTexture(rec),
      color: new THREE.Color(shade, shade, shade),
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

  function capUvgen(world) {
    return {
      generateTopUV(_geometry, vertices, a, b, c) {
        const to = (i) => new THREE.Vector2(
          vertices[i * 3] / world + 0.5,
          -vertices[i * 3 + 1] / world + 0.5
        );
        return [to(a), to(b), to(c)];
      },
      generateSideWallUV(_geometry, vertices, a, b, c, d) {
        const to = (i) => new THREE.Vector2(
          vertices[i * 3] / world + 0.5,
          -vertices[i * 3 + 1] / world + 0.5
        );
        return [to(a), to(b), to(c), to(d)];
      },
    };
  }

  function disposeObject(obj) {
    if (!obj) return;
    obj.removeFromParent();
    const seen = new Set();
    obj.traverse((o) => {
      if (o.geometry && o.geometry !== voxelGeo) o.geometry.dispose();
      const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
      for (const m of mats) {
        if (!m || m === voxelMat || seen.has(m)) continue;
        seen.add(m);
        m.dispose();
      }
    });
  }

  function clearDisplay(rec) {
    disposeObject(rec.mesh);
    disposeObject(rec.plane);
    disposeObject(rec.contour);
    rec.mesh = null;
    rec.plane = null;
    rec.contour = null;
  }

  function fillMesh(rec, crunch) {
    const native = rec.native;
    if (!native || !native.n) return;
    const res = meta.resolution;
    const world = res * meta.pitch;
    const texel = world / crunch;
    const shade = layerShade(rec.index);
    if (!rec.mesh) {
      const geo = voxelGeo.clone();
      geo.setAttribute("aPBR", new THREE.InstancedBufferAttribute(new Float32Array(native.n * 3), 3));
      const mesh = new THREE.InstancedMesh(geo, voxelMat, native.n);
      mesh.frustumCulled = false;
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(native.n * 3), 3);
      rec.group.add(mesh);
      rec.mesh = mesh;
    }
    const mesh = rec.mesh;
    const me = mesh.instanceMatrix.array;
    const ca = mesh.instanceColor.array;
    const aPBR = mesh.geometry.getAttribute("aPBR");
    let count = 0;
    const pbrOf = (i) => {
      if (palette && native.pal && native.pal[i] < palette.length) {
        const e = palette[native.pal[i]];
        return [e.metal, e.rough, e.emit];
      }
      return [0, 0.7, 0];
    };
    const write = (gx, gz, r, g, b, metal, rough, emit) => {
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
        if (!cellKept(rec.index, native.ix[i], native.iz[i], crunch)) continue;
        const [m, rh, em] = pbrOf(i);
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
        if (!cellKept(rec.index, gx, gz, crunch)) continue;
        const k = gz * crunch + gx;
        const idx = seenBin[k];
        const r = native.rgb[i * 3];
        const g = native.rgb[i * 3 + 1];
        const b = native.rgb[i * 3 + 2];
        const [m, rh, em] = pbrOf(i);
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
    mesh.instanceColor.needsUpdate = true;
    if (aPBR) aPBR.needsUpdate = true;
    mesh.visible = state.round <= 0;
  }

  function fillPlane(rec) {
    const world = meta.resolution * meta.pitch;
    const geo = new THREE.PlaneGeometry(world, world);
    geo.rotateX(-Math.PI / 2);
    const uv = geo.attributes.uv;
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      uv.setXY(i, pos.getX(i) / world + 0.5, pos.getZ(i) / world + 0.5);
    }
    uv.needsUpdate = true;
    const mat = makeSliceMaterial(rec, {
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -rec.index,
      polygonOffsetUnits: 1,
      cacheKey: "stack-plane-v1",
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    rec.group.add(mesh);
    rec.plane = mesh;
  }

  function fillContour(rec, crunch) {
    const world = meta.resolution * meta.pitch;
    const occ = occupancyGrid(rec, crunch);
    const loops = marchingSquares(occ, crunch);
    const shapes = loopsToShapes(loops, crunch, world);
    const group = new THREE.Group();
    const mat = makeSliceMaterial(rec, {
      side: THREE.DoubleSide,
      alphaTest: 0.04,
      cacheKey: "stack-contour-v1",
    });
    const uvgen = capUvgen(world);
    for (const shape of shapes) {
      let geo;
      try {
        geo = new THREE.ExtrudeGeometry(shape, {
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
      const mesh = new THREE.Mesh(geo, mat);
      mesh.frustumCulled = false;
      group.add(mesh);
    }
    rec.group.add(group);
    rec.contour = group;
  }

  function rebuildRec(rec, crunch) {
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

  function lookLabel() {
    let name = "Voxel";
    if (state.look === "plane") name = "Plane";
    if (state.look === "contour") name = "Contour";
    if (state.pixel) name += " · Pixel";
    if (state.look === "voxel" && state.round > 0.001) name += ` · Round ${state.round.toFixed(2)}`;
    if (state.main) name += " · Main";
    return name;
  }

  function updateLookButtons() {
    const map = [
      [sliders?.lookVoxel, "voxel"],
      [sliders?.lookPlane, "plane"],
      [sliders?.lookContour, "contour"],
    ];
    for (const [el, look] of map) {
      if (el) el.setAttribute("aria-pressed", String(state.look === look));
    }
    sliders?.pixel?.setAttribute("aria-pressed", String(state.pixel));
    sliders?.outline?.setAttribute("aria-pressed", String(state.outline));
    sliders?.main?.setAttribute("aria-pressed", String(state.main));
    if (statsEl && meta) {
      const px = state.pixel ? ` · ${state.pixelSize}px` : "";
      statsEl.innerHTML = `<b>${lookLabel()}</b>${px} · <b>${meta.layers}</b> slabs · up to <b>${meta.resolution}²</b>`;
    }
  }

  function setLook(look) {
    if (!LOOKS.includes(look) || look === state.look) {
      updateLookButtons();
      return;
    }
    state.look = look;
    updateLookButtons();
    applyStackState();
    rebuildLookAsync();
  }

  function rebuildLookAsync() {
    const job = ++voxelJob;
    if (!meta) return Promise.resolve();
    const crunch = crunchSize();
    const chunk = state.look === "contour" ? 4 : 10;
    return (async () => {
      for (const rec of stackSlices) {
        if (!rec.native) cacheNative(rec);
      }
      rebuildIslandMask(crunch);
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

  function srgbToOkLab(r, g, b) {
    const lin = (c) => {
      c /= 255;
      return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    const R = lin(r), G = lin(g), B = lin(b);
    const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
    const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
    const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
    return [
      0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ];
  }

  function labDist2(a, b) {
    const dL = a[0] - b[0], da = a[1] - b[1], db = a[2] - b[2];
    return dL * dL + da * da + db * db;
  }

  function quantize(k) {
    k = Math.max(2, Math.min(32, k | 0));
    const pts = [];
    for (const rec of stackSlices) {
      const n = rec.native;
      if (!n) continue;
      const src = n.rgb0 || n.rgb;
      for (let i = 0; i < n.n; i++) pts.push(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
    }
    const N = pts.length / 3;
    if (!N) return [];
    k = Math.min(k, N);

    const B = 24;
    const bins = new Map();
    for (let i = 0; i < N; i++) {
      const r = pts[i * 3], g = pts[i * 3 + 1], b = pts[i * 3 + 2];
      const key = ((r * B / 256) | 0) << 16 | ((g * B / 256) | 0) << 8 | ((b * B / 256) | 0);
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
    const cands = [];
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

    const picked = [];
    const farEnough = (c, minD2) => {
      for (const p of picked) {
        if (labDist2(c.lab, p.lab) < minD2) return false;
      }
      return true;
    };
    const take = (list, limit, minD2) => {
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
        if (d < bestD) { bestD = d; best = c; }
      }
      assign[i] = best;
    }

    palette = picked.map((c) => ({
      r: Math.round(c.r),
      g: Math.round(c.g),
      b: Math.round(c.b),
      metal: 0,
      rough: 0.7,
      emit: c.chroma > 0.12 && c.lab[0] > 0.55 ? 0.35 : 0,
    }));
    palSelected = 0;
    let i = 0;
    const crunch = crunchSize();
    for (const rec of stackSlices) {
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
      if (state.look === "voxel") fillMesh(rec, crunch);
      else ensureSliceTexture(rec);
    }
    if (state.round > 0 && state.look === "voxel") scheduleSmooth();
    else applyStackState();
    return palette;
  }

  function applyPaletteEntry(index, patch) {
    if (!palette || !palette[index]) return;
    Object.assign(palette[index], patch);
    const e = palette[index];
    const crunch = crunchSize();
    for (const rec of stackSlices) {
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
      if (dirty) {
        if (state.look === "voxel") fillMesh(rec, crunch);
        else ensureSliceTexture(rec);
      }
    }
  }

  const paintRay = new THREE.Raycaster();
  const paintNdc = new THREE.Vector2();
  let strokeUndo = null;

  function beginStroke() {
    strokeUndo = stackSlices.map((rec) => (rec.native ? new Uint8Array(rec.native.rgb) : null));
  }

  function undoStroke() {
    if (!strokeUndo) return;
    const crunch = crunchSize();
    stackSlices.forEach((rec, i) => {
      if (!rec.native || !strokeUndo[i]) return;
      rec.native.rgb.set(strokeUndo[i]);
      if (state.look === "voxel") fillMesh(rec, crunch);
      else ensureSliceTexture(rec);
    });
    strokeUndo = null;
    applyStackState();
  }

  function paintSphere(center, radius, color, erase) {
    if (!meta || !stackSlices.length) return 0;
    const gap = meta.pitch * Math.max(0.55, state.spacing);
    const world = meta.resolution * meta.pitch;
    const r2 = radius * radius;
    const crunch = crunchSize();
    let painted = 0;
    for (const rec of stackSlices) {
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
          if (palette && palette[palSelected]) {
            const e = palette[palSelected];
            rgb = [e.r, e.g, e.b];
            if (native.pal) native.pal[i] = palSelected;
          }
          native.rgb[i * 3] = rgb[0];
          native.rgb[i * 3 + 1] = rgb[1];
          native.rgb[i * 3 + 2] = rgb[2];
        }
        dirty = true;
        painted++;
      }
      if (dirty) {
        if (state.look === "voxel") fillMesh(rec, crunch);
        else ensureSliceTexture(rec);
      }
    }
    return painted;
  }

  function sampleVoxel(center) {
    if (!meta) return null;
    const gap = meta.pitch * Math.max(0.55, state.spacing);
    const world = meta.resolution * meta.pitch;
    let best = null;
    let bestD = Infinity;
    for (const rec of stackSlices) {
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

  function pick(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    paintNdc.set(
      ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1,
      -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1
    );
    paintRay.setFromCamera(paintNdc, gameLook.viewCamera());
    const hits = paintRay.intersectObject(stackGroup, true);
    return hits[0] || null;
  }

  function getSheetBlob() {
    if (!meta) return Promise.resolve(null);
    const res = meta.resolution;
    const c = document.createElement("canvas");
    c.width = res * meta.layers;
    c.height = res;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(res, res);
    for (const rec of stackSlices) {
      img.data.fill(0);
      const native = rec.native;
      if (native) {
        for (let i = 0; i < native.n; i++) {
          const p = (native.iz[i] * res + native.ix[i]) * 4;
          img.data[p] = native.rgb[i * 3];
          img.data[p + 1] = native.rgb[i * 3 + 1];
          img.data[p + 2] = native.rgb[i * 3 + 2];
          img.data[p + 3] = 255;
        }
      }
      ctx.putImageData(img, rec.index * res, 0);
    }
    return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/png"));
  }

  function clearStack() {
    voxelJob += 1;
    clearTimeout(roundTimer);
    disposeIso();
    islandMask = null;
    islandCrunch = 0;
    while (stackGroup.children.length) stackGroup.remove(stackGroup.children[0]);
    stackSlices = [];
    sheet = null;
    meta = null;
  }

  function bindSliders() {
    if (!sliders) return;
    const setVal = (el, label, value, digits) => {
      if (!el) return;
      el.value = String(value);
      if (label) label.textContent = digits == null ? String(value) : Number(value).toFixed(digits);
    };
    const on = (el, fn) => el && el.addEventListener("input", fn);

    on(sliders.spacing, () => {
      state.spacing = Number(sliders.spacing.value);
      if (sliders.spacingVal) sliders.spacingVal.textContent = Number(state.spacing).toFixed(1);
      applyStackState();
      if (state.round > 0) scheduleSmooth();
    });
    on(sliders.height, () => {
      state.height = Number(sliders.height.value);
      if (sliders.heightVal) sliders.heightVal.textContent = Number(state.height).toFixed(2);
      applyStackState();
    });
    on(sliders.layers, () => {
      state.layers = Number(sliders.layers.value);
      if (sliders.layersVal) sliders.layersVal.textContent = String(state.layers);
      applyStackState();
      if (state.round > 0 || state.main) rebuildLookAsync();
    });
    on(sliders.texels, () => {
      state.texels = Number(sliders.texels.value);
      if (sliders.texelsVal) sliders.texelsVal.textContent = String(state.texels);
      clearTimeout(texelsTimer);
      texelsTimer = setTimeout(() => rebuildLookAsync(), 40);
    });
    on(sliders.round, () => {
      state.round = Math.max(0, Math.min(1, Number(sliders.round.value) || 0));
      if (sliders.roundVal) sliders.roundVal.textContent = state.round.toFixed(2);
      updateLookButtons();
      scheduleSmooth();
    });
    sliders.main?.addEventListener("click", () => {
      state.main = !state.main;
      updateLookButtons();
      rebuildLookAsync();
    });
    on(sliders.bright, () => {
      state.brightness = Number(sliders.bright.value);
      if (sliders.brightVal) sliders.brightVal.textContent = Number(state.brightness).toFixed(2);
      applyGrade();
    });
    on(sliders.contrast, () => {
      state.contrast = Number(sliders.contrast.value);
      if (sliders.contrastVal) sliders.contrastVal.textContent = Number(state.contrast).toFixed(2);
      applyGrade();
    });
    on(sliders.saturate, () => {
      state.saturation = Number(sliders.saturate.value);
      if (sliders.saturateVal) sliders.saturateVal.textContent = Number(state.saturation).toFixed(2);
      applyGrade();
    });
    sliders.shade?.addEventListener("click", () => {
      state.shade = !state.shade;
      sliders.shade.setAttribute("aria-pressed", String(state.shade));
      rebuildLookAsync();
    });
    sliders.spin?.addEventListener("click", () => {
      state.auto = !state.auto;
      controls.enabled = true;
      controls.autoRotate = false;
      sliders.spin.setAttribute("aria-pressed", String(state.auto));
    });
    sliders.lookVoxel?.addEventListener("click", () => setLook("voxel"));
    sliders.lookPlane?.addEventListener("click", () => setLook("plane"));
    sliders.lookContour?.addEventListener("click", () => setLook("contour"));
    sliders.pixel?.addEventListener("click", () => {
      state.pixel = !state.pixel;
      updateLookButtons();
    });
    sliders.outline?.addEventListener("click", () => {
      state.outline = !state.outline;
      pixelUniforms.outline.value = state.outline ? 1 : 0;
      updateLookButtons();
    });
    on(sliders.pixelSize, () => {
      state.pixelSize = Math.max(1, Math.min(8, Number(sliders.pixelSize.value) || 3));
      if (sliders.pixelSizeVal) sliders.pixelSizeVal.textContent = String(state.pixelSize);
      updateLookButtons();
    });
    sliders.reset?.addEventListener("click", () => {
      if (!meta) return;
      const texels = Math.min(DEFAULTS.texels, meta.resolution);
      const rebuild = state.texels !== texels || state.shade !== DEFAULTS.shade || state.round !== DEFAULTS.round || state.main !== DEFAULTS.main;
      state.spacing = DEFAULTS.spacing;
      state.height = DEFAULTS.height;
      state.layers = meta.layers;
      state.texels = texels;
      state.brightness = DEFAULTS.brightness;
      state.contrast = DEFAULTS.contrast;
      state.saturation = DEFAULTS.saturation;
      state.shade = DEFAULTS.shade;
      state.auto = DEFAULTS.auto;
      state.pixel = DEFAULTS.pixel;
      state.pixelSize = DEFAULTS.pixelSize;
      state.outline = DEFAULTS.outline;
      state.round = DEFAULTS.round;
      state.main = DEFAULTS.main;
      disposeIso();
      setCubeVisible(state.round <= 0);
      controls.autoRotate = false;
      setVal(sliders.spacing, sliders.spacingVal, state.spacing, 1);
      setVal(sliders.height, sliders.heightVal, state.height, 2);
      setVal(sliders.layers, sliders.layersVal, state.layers);
      setVal(sliders.texels, sliders.texelsVal, state.texels);
      setVal(sliders.bright, sliders.brightVal, state.brightness, 2);
      setVal(sliders.contrast, sliders.contrastVal, state.contrast, 2);
      setVal(sliders.saturate, sliders.saturateVal, state.saturation, 2);
      setVal(sliders.pixelSize, sliders.pixelSizeVal, state.pixelSize);
      setVal(sliders.round, sliders.roundVal, state.round, 2);
      sliders.shade?.setAttribute("aria-pressed", "true");
      sliders.spin?.setAttribute("aria-pressed", "true");
      updateLookButtons();
      applyGrade();
      if (rebuild) rebuildLookAsync();
      else applyStackState();
    });
  }

  function applyMetaToSliders() {
    if (!meta || !sliders) return;
    if (meta.defaults && typeof meta.defaults === "object") {
      const keepLook = DEFAULTS.look;
      const keepRound = DEFAULTS.round;
      const keepMain = DEFAULTS.main;
      Object.assign(DEFAULTS, meta.defaults);
      DEFAULTS.look = keepLook;
      DEFAULTS.round = keepRound;
      DEFAULTS.main = keepMain;
    }
    state.layers = meta.layers;
    state.texels = Math.min(DEFAULTS.texels, meta.resolution);
    state.spacing = DEFAULTS.spacing;
    state.height = DEFAULTS.height;
    state.brightness = DEFAULTS.brightness;
    state.contrast = DEFAULTS.contrast;
    state.saturation = DEFAULTS.saturation;
    if (sliders.layers) {
      sliders.layers.max = String(meta.layers);
      sliders.layers.value = String(state.layers);
    }
    if (sliders.texels) {
      sliders.texels.max = String(meta.resolution);
      sliders.texels.value = String(state.texels);
    }
    if (sliders.layersVal) sliders.layersVal.textContent = String(state.layers);
    if (sliders.texelsVal) sliders.texelsVal.textContent = String(state.texels);
    if (sliders.spacing) sliders.spacing.value = String(state.spacing);
    if (sliders.height) sliders.height.value = String(state.height);
    if (sliders.bright) sliders.bright.value = String(state.brightness);
    if (sliders.contrast) sliders.contrast.value = String(state.contrast);
    if (sliders.saturate) sliders.saturate.value = String(state.saturation);
    if (sliders.spacingVal) sliders.spacingVal.textContent = Number(state.spacing).toFixed(1);
    if (sliders.heightVal) sliders.heightVal.textContent = Number(state.height).toFixed(2);
    if (sliders.brightVal) sliders.brightVal.textContent = Number(state.brightness).toFixed(2);
    if (sliders.contrastVal) sliders.contrastVal.textContent = Number(state.contrast).toFixed(2);
    if (sliders.saturateVal) sliders.saturateVal.textContent = Number(state.saturation).toFixed(2);
    if (sliders.pixelSize) sliders.pixelSize.value = String(state.pixelSize);
    if (sliders.pixelSizeVal) sliders.pixelSizeVal.textContent = String(state.pixelSize);
    if (sliders.round) sliders.round.value = String(state.round);
    if (sliders.roundVal) sliders.roundVal.textContent = Number(state.round).toFixed(2);
    applyGrade();
    updateLookButtons();
  }

  async function load(asset) {
    clearStack();
    setEmpty(true, "Slicing…");
    resize();
    try {
      const json = await fetch(`/out/${asset}/stack/stack.json`).then((r) => {
        if (!r.ok) throw new Error("no stack");
        return r.json();
      });
      const img = await new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = `/out/${asset}/stack/sheet.png`;
      });
      meta = json;
      sheet = img;
      applyMetaToSliders();
      updateLookButtons();
      for (let i = 0; i < meta.layers; i++) {
        const rec = { index: i, group: new THREE.Group(), mesh: null, native: null };
        stackGroup.add(rec.group);
        stackSlices.push(rec);
      }
      setEmpty(false);
      resize();
      frameStack();
      rebuildLookAsync();
      return true;
    } catch {
      if (statsEl) statsEl.textContent = "No sprite stack";
      setEmpty(true, "No sprite stack yet. Generate writes out/<stem>/stack/.");
      return false;
    }
  }

  function getView() {
    _sph.setFromVector3(_offset.copy(camera.position).sub(controls.target));
    return { theta: _sph.theta, phi: _sph.phi };
  }

  function tick(view) {
    controls.enabled = true;
    controls.autoRotate = false;
    controls.update();
    if (view) {
      const radius = camera.position.distanceTo(controls.target);
      _sph.set(radius, view.phi, view.theta);
      _sph.makeSafe();
      camera.position.copy(controls.target).add(_offset.setFromSpherical(_sph));
      camera.lookAt(controls.target);
    }
    if (!ready) return;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
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

  bindSliders();
  resize();
  updateLookButtons();

  return {
    load,
    resize,
    tick,
    pixelLook: gameLook,
    clear() {
      clearStack();
      setEmpty(true, "Generating…");
    },
    unload(msg) {
      clearStack();
      setEmpty(true, msg || "No sprite stack yet. Generate writes out/<stem>/stack/.");
    },
    paintSphere,
    sampleVoxel,
    pick,
    beginStroke,
    undoStroke,
    getSheetBlob,
    quantize,
    getPalette: () => palette,
    setPaletteEntry: applyPaletteEntry,
    selectPalette(i) { palSelected = i; },
    selectedPalette: () => palSelected,
    setLook,
    getLook: () => state.look,
    setBrush(point, radius) {
      if (!point) {
        brushBall.visible = false;
        return;
      }
      brushBall.visible = true;
      brushBall.position.copy(point);
      brushBall.scale.setScalar(radius || 0.05);
    },
    setRotateEnabled(on) {
      controls.enableRotate = on;
    },
    getView,
    isSync: () => state.auto,
    onDragStart(fn) { controls.addEventListener("start", fn); },
    onDragEnd(fn) { controls.addEventListener("end", fn); },
    canvas,
  };
}
