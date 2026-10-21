/**
 * Game look for the mesh pane: render the scene small, then blow it up with
 * crisp pixels. Low-res target, depth/normal outlines on the pixel grid,
 * an isometric orthographic camera that snaps to whole texels (with the
 * sub-texel remainder shifted back on the upscale so motion stays smooth),
 * real cast shadows, and an optional posterize + ordered dither.
 *
 * The perspective camera and OrbitControls still drive everything; this only
 * changes how a frame is drawn. docs/pixel-look.md has the engine port.
 */
export const ISO_POLAR = Math.PI / 2 - Math.atan(1 / Math.SQRT2); // 35.26° above the ground

const BLIT_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const BLIT_FRAG = /* glsl */ `
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform vec2 lowRes;      // visible texels
uniform vec2 rtRes;       // lowRes + 2 (one texel margin per side)
uniform vec2 frac;        // sub-texel camera remainder, in texels
uniform float outline;
uniform float isOrtho;
uniform float camNear;
uniform float camFar;
uniform float texelWorld; // world size of one texel at the focus (ortho) or per unit depth (persp)
uniform float levels;     // posterize steps per channel, 0 = off
uniform float dither;
varying vec2 vUv;

float viewZ(float d) {
  if (isOrtho > 0.5) return camNear + d * (camFar - camNear);
  return (camNear * camFar) / (camFar - d * (camFar - camNear));
}

float bayer4(vec2 p) {
  vec2 q = mod(p, 4.0);
  int i = int(q.x) + int(q.y) * 4;
  float m[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
  return m[i] / 16.0 - 0.5;
}

void main() {
  vec2 t = vUv * lowRes + 1.0 + frac;
  vec2 fs = clamp(fwidth(t), vec2(1e-5), vec2(1.0));
  vec2 tt = t - 0.5 * fs;
  vec2 smoothUv = (floor(tt) + 0.5 + smoothstep(vec2(1.0) - fs, vec2(1.0), fract(tt))) / rtRes;
  vec2 cell = floor(t);
  vec2 c = (cell + 0.5) / rtRes;
  vec4 color = texture2D(tColor, smoothUv);

  if (outline > 0.5) {
    vec2 px = 1.0 / rtRes;
    float dc = texture2D(tDepth, c).r;
    float zc = viewZ(dc);
    vec3 nc = normalize(texture2D(tNormal, c).rgb * 2.0 - 1.0);
    // Surfaces seen edge-on step a lot in depth per texel; widen the threshold there.
    float facing = clamp(abs(nc.z), 0.04, 1.0);
    float thr = 3.0 * (isOrtho > 0.5 ? texelWorld : texelWorld * zc) / facing;
    float depthEdge = 0.0;
    float normalEdge = 0.0;
    vec2 offs[4] = vec2[4](vec2(px.x, 0.0), vec2(-px.x, 0.0), vec2(0.0, px.y), vec2(0.0, -px.y));
    for (int i = 0; i < 4; i++) {
      float zn = viewZ(texture2D(tDepth, c + offs[i]).r);
      vec3 nn = normalize(texture2D(tNormal, c + offs[i]).rgb * 2.0 - 1.0);
      float dz = zn - zc; // > 0: neighbor is farther, so this texel is the silhouette
      // A depth jump between texels facing the same way is one surface seen edge-on (far ground), not an edge.
      depthEdge += (dz > thr && dot(nc, nn) < 0.995) ? 1.0 : 0.0;
      float bias = smoothstep(-0.01, 0.01, dot(nc - nn, vec3(1.0)));
      float sameSurface = dz > -thr ? 1.0 : 0.0;
      normalEdge += (1.0 - dot(nc, nn)) * bias * sameSurface;
    }
    if (depthEdge > 0.0) color.rgb *= 0.55;
    else if (normalEdge > 0.25 && dc < 0.9999) color.rgb *= 1.3;
  }

  gl_FragColor = vec4(color.rgb, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  if (levels > 1.5) {
    float steps = levels - 1.0;
    vec3 v = gl_FragColor.rgb * steps + (dither > 0.5 ? 0.75 * bayer4(cell) : 0.0);
    gl_FragColor.rgb = clamp(floor(v + 0.5) / steps, 0.0, 1.0);
  }
}`;

export function createPixelLook({ THREE, renderer, scene, camera, controls, subject, key, ground, blob, defaults = {} }) {
  const state = {
    on: false,
    pixelSize: 4,
    outline: true,
    iso: true,
    shadows: true,
    levels: 0,
    dither: false,
    lines: 0, // > 0: a fixed internal height (240 for the PS1 look) instead of screen size / pixelSize
    ...defaults,
  };

  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0005;
  ground.receiveShadow = true;
  if (!ground.geometry.boundingSphere) ground.geometry.computeBoundingSphere();
  const GROUND_GROW = 4; // the game look has no horizon, so the ground runs past the frame
  const groundReach = ground.geometry.boundingSphere.radius * GROUND_GROW + 1;
  // An unlit ground (MeshBasicMaterial) can't show a shadow, so lay a shadow-only copy on it.
  let catcher = null;
  if (ground.material.isMeshBasicMaterial) {
    catcher = new THREE.Mesh(ground.geometry, new THREE.ShadowMaterial({ opacity: 0.45 }));
    catcher.position.copy(ground.position);
    catcher.position.y += 0.001;
    catcher.quaternion.copy(ground.quaternion);
    catcher.receiveShadow = true;
    catcher.visible = false;
    ground.parent.add(catcher);
  }
  const groundColor = ground.material.color.clone();
  const bgColor = scene.background && scene.background.isColor ? scene.background.clone() : null;
  const GAME_GROUND = new THREE.Color(0x5d6152);
  const GAME_BG = new THREE.Color(0x23251f);
  const savedPolar = { min: controls.minPolarAngle, max: controls.maxPolarAngle };
  const envIntensity = scene.environmentIntensity ?? 1;

  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
  const persp = new THREE.PerspectiveCamera();
  const normalMat = new THREE.MeshNormalMaterial();

  const uniforms = {
    tColor: { value: null },
    tDepth: { value: null },
    tNormal: { value: null },
    lowRes: { value: new THREE.Vector2(1, 1) },
    rtRes: { value: new THREE.Vector2(3, 3) },
    frac: { value: new THREE.Vector2() },
    outline: { value: 1 },
    isOrtho: { value: 1 },
    camNear: { value: 0.01 },
    camFar: { value: 100 },
    texelWorld: { value: 0.01 },
    levels: { value: 0 },
    dither: { value: 0 },
  };
  const blitMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: BLIT_VERT,
    fragmentShader: BLIT_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const blitScene = new THREE.Scene();
  blitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), blitMat));
  const blitCam = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);

  let colorRT = null;
  let normalRT = null;
  let fitKey = "";
  let radius = 1;
  const _size = new THREE.Vector2();
  const _box = new THREE.Box3();
  const _sphere = new THREE.Sphere();
  const _right = new THREE.Vector3();
  const _up = new THREE.Vector3();
  const _keyDir = key.position.clone().normalize();
  const _clear = new THREE.Color();

  function ensureTargets(lw, lh) {
    const w = lw + 2;
    const h = lh + 2;
    if (colorRT && colorRT.width === w && colorRT.height === h) return;
    colorRT?.depthTexture?.dispose();
    colorRT?.dispose();
    normalRT?.dispose();
    const depth = new THREE.DepthTexture(w, h);
    depth.format = THREE.DepthFormat;
    depth.type = THREE.UnsignedIntType;
    colorRT = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      depthTexture: depth,
    });
    colorRT.texture.colorSpace = THREE.SRGBColorSpace;
    normalRT = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
    });
    uniforms.tColor.value = colorRT.texture;
    uniforms.tDepth.value = depth;
    uniforms.tNormal.value = normalRT.texture;
    uniforms.rtRes.value.set(w, h);
  }

  // Size the shadow camera to whatever is loaded, again whenever its meshes are swapped.
  function fitShadow() {
    let id = "";
    subject.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      id += o.uuid;
    });
    if (id === fitKey) return;
    fitKey = id;
    _box.setFromObject(subject);
    radius = _box.isEmpty() ? 1 : Math.max(0.25, _box.getBoundingSphere(_sphere).radius);
    const sc = key.shadow.camera;
    const r = radius * 1.6;
    sc.left = -r;
    sc.right = r;
    sc.top = r;
    sc.bottom = -r;
    sc.near = 0.05;
    sc.far = radius * 8;
    sc.updateProjectionMatrix();
    key.position.copy(key.target.position).addScaledVector(_keyDir, radius * 4);
    key.shadow.normalBias = radius * 0.01;
  }

  function apply() {
    const on = state.on;
    key.castShadow = on && state.shadows;
    ground.material.color.copy(on ? GAME_GROUND : groundColor);
    if (bgColor) scene.background.copy(on ? GAME_BG : bgColor);
    if (blob) blob.visible = !(on && state.shadows);
    if (catcher) catcher.visible = on && state.shadows;
    ground.scale.setScalar(on ? GROUND_GROW : 1);
    if (catcher) catcher.scale.copy(ground.scale);
    scene.environmentIntensity = on ? envIntensity * 0.35 : envIntensity; // matte, like the game
    if (on && state.iso) {
      controls.minPolarAngle = controls.maxPolarAngle = ISO_POLAR;
    } else {
      controls.minPolarAngle = savedPolar.min;
      controls.maxPolarAngle = savedPolar.max;
    }
    controls.update();
    uniforms.outline.value = state.outline ? 1 : 0;
    uniforms.levels.value = state.levels;
    uniforms.dither.value = state.dither ? 1 : 0;
  }

  function render() {
    if (!state.on) {
      renderer.render(scene, camera);
      return;
    }
    fitShadow();
    renderer.getDrawingBufferSize(_size);
    const px = Math.max(1, state.pixelSize);
    const lh = state.lines > 0 ? state.lines : Math.max(32, Math.floor(_size.y / px));
    const lw = state.lines > 0 ? Math.max(32, Math.round((_size.x / Math.max(1, _size.y)) * lh)) : Math.max(32, Math.floor(_size.x / px));
    ensureTargets(lw, lh);
    uniforms.lowRes.value.set(lw, lh);

    const dist = camera.position.distanceTo(controls.target);
    const reach = radius * 2 + groundReach;
    const near = Math.max(dist * 0.1, dist - reach); // a tight near keeps far ground depth precise enough for outlines
    const far = dist + reach;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    let view;
    if (state.iso) {
      const texel = (2 * dist * tanHalf) / camera.zoom / lh;
      const halfW = (texel * (lw + 2)) / 2;
      const halfH = (texel * (lh + 2)) / 2;
      ortho.left = -halfW;
      ortho.right = halfW;
      ortho.top = halfH;
      ortho.bottom = -halfH;
      ortho.near = dist - reach;
      ortho.far = far;
      ortho.updateProjectionMatrix();
      ortho.quaternion.copy(camera.quaternion);
      _right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      _up.set(0, 1, 0).applyQuaternion(camera.quaternion);
      const r = camera.position.dot(_right) / texel;
      const u = camera.position.dot(_up) / texel;
      const fr = r - Math.round(r);
      const fu = u - Math.round(u);
      ortho.position.copy(camera.position).addScaledVector(_right, -fr * texel).addScaledVector(_up, -fu * texel);
      ortho.updateMatrixWorld();
      uniforms.frac.value.set(fr, fu);
      uniforms.isOrtho.value = 1;
      uniforms.camNear.value = ortho.near;
      uniforms.camFar.value = ortho.far;
      uniforms.texelWorld.value = texel;
      view = ortho;
    } else {
      persp.fov = camera.fov;
      persp.zoom = camera.zoom;
      persp.aspect = lw / lh;
      persp.near = near;
      persp.far = far;
      persp.setViewOffset(lw, lh, -1, -1, lw + 2, lh + 2);
      persp.position.copy(camera.position);
      persp.quaternion.copy(camera.quaternion);
      persp.updateMatrixWorld();
      uniforms.frac.value.set(0, 0);
      uniforms.isOrtho.value = 0;
      uniforms.camNear.value = near;
      uniforms.camFar.value = far;
      uniforms.texelWorld.value = (2 * tanHalf) / camera.zoom / lh;
      view = persp;
    }

    renderer.setRenderTarget(colorRT);
    renderer.render(scene, view);
    if (state.outline) {
      const bg = scene.background;
      const shadowAuto = renderer.shadowMap.autoUpdate;
      scene.background = null;
      scene.overrideMaterial = normalMat;
      const clearAlpha = renderer.getClearAlpha();
      renderer.getClearColor(_clear);
      renderer.shadowMap.autoUpdate = false;
      renderer.setRenderTarget(normalRT);
      renderer.setClearColor(0x000000, 0);
      renderer.render(scene, view);
      renderer.setClearColor(_clear, clearAlpha);
      renderer.shadowMap.autoUpdate = shadowAuto;
      scene.overrideMaterial = null;
      scene.background = bg;
    }
    renderer.setRenderTarget(null);
    renderer.render(blitScene, blitCam);
  }

  apply();

  return {
    state,
    render,
    set(patch) {
      Object.assign(state, patch);
      apply();
    },
    // The camera the screen actually shows, for raycasts.
    viewCamera() {
      return state.on && state.iso ? ortho : camera;
    },
  };
}

/** Wire the Game look buttons and sliders (ids game-*) and remember the choice. */
export function bindPixelLookUi(looks, doc = document) {
  const list = Array.isArray(looks) ? looks : [looks];
  const look = {
    state: list[0].state,
    set(patch) {
      for (const l of list) l.set(patch);
    },
  };
  const KEY = "pipeline.pixelLook";
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (saved && typeof saved === "object") look.set(saved);
  } catch {}
  const q = new URLSearchParams(location.search).get("game");
  if (q != null) look.set({ on: q !== "0" });

  const $ = (id) => doc.getElementById(id);
  const toggles = { "game-look": "on", "game-look-bar": "on", "game-iso": "iso", "game-outline": "outline", "game-shadow": "shadows", "game-dither": "dither" };
  const px = $("game-px");
  const pxVal = $("game-px-val");
  const levels = $("game-levels");
  const levelsVal = $("game-levels-val");

  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(look.state));
    } catch {}
  }
  function show() {
    for (const [id, k] of Object.entries(toggles)) $(id)?.setAttribute("aria-pressed", String(!!look.state[k]));
    if (px) px.value = String(look.state.pixelSize);
    if (pxVal) pxVal.textContent = String(look.state.pixelSize);
    if (levels) levels.value = String(Math.max(1, look.state.levels));
    if (levelsVal) levelsVal.textContent = look.state.levels > 1 ? String(look.state.levels) : "off";
    $("game-240")?.setAttribute("aria-pressed", String(!!look.state.on && look.state.lines === 240));
    if (pxVal && look.state.lines > 0) pxVal.textContent = `${look.state.lines}p`;
  }
  for (const [id, k] of Object.entries(toggles)) {
    $(id)?.addEventListener("click", () => {
      look.set({ [k]: !look.state[k] });
      show();
      save();
    });
  }
  // PS1 240p: what the game's PSX shader targets. A perspective camera, 240 lines whatever the window,
  // no outlines or soft shadows. Pressing it again goes back to the size-based look.
  const ps1 = $("game-240");
  const PS1 = { on: true, lines: 240, iso: false, outline: false, shadows: false };
  ps1?.addEventListener("click", () => {
    look.set(look.state.on && look.state.lines === 240 ? { lines: 0, iso: true, outline: true, shadows: true } : PS1);
    show();
    save();
  });
  px?.addEventListener("input", () => {
    look.set({ pixelSize: Math.max(1, Math.min(8, Number(px.value) || 4)), lines: 0 });
    show();
    save();
  });
  levels?.addEventListener("input", () => {
    const n = Math.round(Number(levels.value) || 1);
    look.set({ levels: n > 1 ? n : 0 });
    show();
    save();
  });
  show();
}
