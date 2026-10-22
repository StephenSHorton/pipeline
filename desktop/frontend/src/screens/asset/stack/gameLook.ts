// The shared Game look (public/pixel-look.js createPixelLook, 1-386), ported for the stack's own canvas: render
// small, outline depth/normal edges on the pixel grid, an iso orthographic camera that snaps to whole texels, cast
// shadows, optional posterize + dither. The viewer's PixelPass is an R3F pass with none of that yet, and the stack
// is plain three, so this is a TypeScript port rather than a reuse. Same "pipeline.pixelLook" record as the viewer.
import {
  Box3,
  type BufferGeometry,
  type Camera,
  Color,
  DepthFormat,
  DepthTexture,
  type DirectionalLight,
  LinearFilter,
  MathUtils,
  Mesh,
  type MeshBasicMaterial,
  MeshNormalMaterial,
  NearestFilter,
  type Object3D,
  OrthographicCamera,
  PCFSoftShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  ShadowMaterial,
  Sphere,
  SRGBColorSpace,
  type Texture,
  UnsignedIntType,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  type WebGLRenderer,
} from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export const ISO_POLAR = Math.PI / 2 - Math.atan(1 / Math.SQRT2); // 35.26° above the ground

export const BLIT_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

export const BLIT_FRAG = /* glsl */ `
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

export interface GameLookState {
  on: boolean;
  pixelSize: number;
  outline: boolean;
  iso: boolean;
  shadows: boolean;
  levels: number;
  dither: boolean;
  /** > 0: a fixed internal height (240 for the PS1 look) instead of screen size / pixelSize. */
  lines: number;
}

export const GAME_DEFAULT: GameLookState = { on: false, pixelSize: 4, outline: true, iso: true, shadows: true, levels: 0, dither: false, lines: 0 };

/**
 * The shared look as pixel-look.js bindPixelLookUi restores it: localStorage "pipeline.pixelLook" over the
 * defaults, then ?game=. src/viewer/pixelLook.ts reads only on / pixelSize / lines of the same record; the stack
 * needs iso, outline, shadows, levels and dither too, so it reads the whole record here.
 */
export function readGameLook(raw: string | null, search = ""): GameLookState {
  let saved: Record<string, unknown> = {};
  try {
    const v = JSON.parse(raw || "null");
    if (v && typeof v === "object") saved = v as Record<string, unknown>;
  } catch {
    /* a broken record is the defaults */
  }
  const out: GameLookState = { ...GAME_DEFAULT };
  const put = out as unknown as Record<string, unknown>;
  for (const k of Object.keys(GAME_DEFAULT) as (keyof GameLookState)[]) {
    if (saved[k] !== undefined && typeof saved[k] === typeof GAME_DEFAULT[k]) put[k] = saved[k];
  }
  const q = new URLSearchParams(search).get("game");
  if (q != null) out.on = q !== "0";
  return out;
}

export interface GameLookParts {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  controls: OrbitControls;
  subject: Object3D;
  key: DirectionalLight;
  ground: Mesh<BufferGeometry, MeshBasicMaterial>;
  blob: Mesh | null;
  defaults?: Partial<GameLookState>;
}

export interface GameLook {
  state: GameLookState;
  render(): void;
  set(patch: Partial<GameLookState>): void;
  /** The camera the screen actually shows, for raycasts. */
  viewCamera(): Camera;
  dispose(): void;
}

export function createGameLook({ renderer, scene, camera, controls, subject, key, ground, blob, defaults = {} }: GameLookParts): GameLook {
  const state: GameLookState = { ...GAME_DEFAULT, ...defaults };

  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0005;
  ground.receiveShadow = true;
  if (!ground.geometry.boundingSphere) ground.geometry.computeBoundingSphere();
  const GROUND_GROW = 4; // the game look has no horizon, so the ground runs past the frame
  const groundReach = ground.geometry.boundingSphere!.radius * GROUND_GROW + 1;
  // An unlit ground (MeshBasicMaterial) can't show a shadow, so lay a shadow-only copy on it.
  let catcher: Mesh | null = null;
  if (ground.material.isMeshBasicMaterial) {
    catcher = new Mesh(ground.geometry, new ShadowMaterial({ opacity: 0.45 }));
    catcher.position.copy(ground.position);
    catcher.position.y += 0.001;
    catcher.quaternion.copy(ground.quaternion);
    catcher.receiveShadow = true;
    catcher.visible = false;
    ground.parent!.add(catcher);
  }
  const groundColor = ground.material.color.clone();
  const bgColor = scene.background instanceof Color ? scene.background.clone() : null;
  const GAME_GROUND = new Color(0x5d6152);
  const GAME_BG = new Color(0x23251f);
  const savedPolar = { min: controls.minPolarAngle, max: controls.maxPolarAngle };
  const envIntensity = scene.environmentIntensity ?? 1;

  const ortho = new OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
  const persp = new PerspectiveCamera();
  const normalMat = new MeshNormalMaterial();

  const uniforms = {
    tColor: { value: null as Texture | null },
    tDepth: { value: null as Texture | null },
    tNormal: { value: null as Texture | null },
    lowRes: { value: new Vector2(1, 1) },
    rtRes: { value: new Vector2(3, 3) },
    frac: { value: new Vector2() },
    outline: { value: 1 },
    isOrtho: { value: 1 },
    camNear: { value: 0.01 },
    camFar: { value: 100 },
    texelWorld: { value: 0.01 },
    levels: { value: 0 },
    dither: { value: 0 },
  };
  const blitMat = new ShaderMaterial({
    uniforms,
    vertexShader: BLIT_VERT,
    fragmentShader: BLIT_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const blitScene = new Scene();
  const blitQuad = new Mesh(new PlaneGeometry(2, 2), blitMat);
  blitScene.add(blitQuad);
  const blitCam = new OrthographicCamera(-1, 1, 1, -1, -1, 1);

  let colorRT: WebGLRenderTarget | null = null;
  let normalRT: WebGLRenderTarget | null = null;
  let fitKey = "";
  let radius = 1;
  const _size = new Vector2();
  const _box = new Box3();
  const _sphere = new Sphere();
  const _right = new Vector3();
  const _up = new Vector3();
  const _keyDir = key.position.clone().normalize();
  const _clear = new Color();

  function ensureTargets(lw: number, lh: number) {
    const w = lw + 2;
    const h = lh + 2;
    if (colorRT && colorRT.width === w && colorRT.height === h) return;
    colorRT?.depthTexture?.dispose();
    colorRT?.dispose();
    normalRT?.dispose();
    const depth = new DepthTexture(w, h);
    depth.format = DepthFormat;
    depth.type = UnsignedIntType;
    colorRT = new WebGLRenderTarget(w, h, {
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      depthBuffer: true,
      depthTexture: depth,
    });
    colorRT.texture.colorSpace = SRGBColorSpace;
    normalRT = new WebGLRenderTarget(w, h, {
      minFilter: NearestFilter,
      magFilter: NearestFilter,
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
      if (!(o as Mesh).isMesh) return;
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
    if (bgColor && scene.background instanceof Color) scene.background.copy(on ? GAME_BG : bgColor);
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
    const tanHalf = Math.tan(MathUtils.degToRad(camera.fov) / 2);
    let view: Camera;
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
      const bgNow = scene.background;
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
      scene.background = bgNow;
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
    viewCamera() {
      return state.on && state.iso ? ortho : camera;
    },
    // Not in pixel-look.js (the page never tore it down): the stage remounts, so free the GPU side.
    dispose() {
      colorRT?.depthTexture?.dispose();
      colorRT?.dispose();
      normalRT?.dispose();
      blitMat.dispose();
      blitQuad.geometry.dispose();
      normalMat.dispose();
      if (catcher) {
        (catcher.material as ShadowMaterial).dispose();
        catcher.removeFromParent();
      }
    },
  };
}
