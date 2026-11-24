// The sprite-stack look's render (pipeline#6). Good sprite stacks are small flat pixel slices stacked one pixel
// apart, kept pixel-sharp when enlarged, with a dark outline. Here the stack's slices are flat pictures in the 3D
// scene (so the view turns as ever), and this pass draws them small: the subject alone goes into a low-resolution
// target where one cell is about one pixel, which is then enlarged by a whole number with no smoothing, and every
// empty pixel touching the subject becomes the outline. The ground, grid and shadow are drawn at full size behind.
import {
  Color,
  HalfFloatType,
  Mesh,
  NearestFilter,
  type Object3D,
  OrthographicCamera,
  type PerspectiveCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  type WebGLRenderer,
  WebGLRenderTarget,
} from "three";

/** The outline's colour: near black, a little cool. */
export const OUTLINE_INK = 0x120f18;

/**
 * How many screen pixels one cell takes, as a whole number (at least 1): the cell's world size over the world size
 * of one screen pixel at the distance looked at. So the subject is drawn with one cell to one low-resolution pixel.
 */
export function cellScale(cellWorld: number, distance: number, fovDeg: number, screenHeightPx: number): number {
  const worldPerPx = (2 * distance * Math.tan((fovDeg * Math.PI) / 360)) / Math.max(1, screenHeightPx);
  return Math.max(1, Math.round(cellWorld / Math.max(1e-9, worldPerPx)));
}

/** The low-resolution target's size for a screen of w x h pixels at `scale`: it covers the whole screen. */
export const lowSize = (w: number, h: number, scale: number) => ({
  w: Math.max(1, Math.ceil(w / scale)),
  h: Math.max(1, Math.ceil(h / scale)),
});

const VERT = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

// Each screen pixel reads its low-resolution pixel (whole-number enlargement, no filtering). An empty pixel with a
// filled one beside it (not diagonally) is the outline; any other empty pixel shows what is behind.
const FRAG = /* glsl */ `
uniform sampler2D tMap;
uniform float scale;
uniform vec3 ink;
void main() {
  ivec2 size = textureSize(tMap, 0);
  ivec2 p = ivec2(floor(gl_FragCoord.xy / scale));
  vec4 c = texelFetch(tMap, p, 0);
  if (c.a > 0.5) {
    gl_FragColor = vec4(c.rgb, 1.0);
  } else {
    float n = 0.0;
    n += texelFetch(tMap, clamp(p + ivec2(1, 0), ivec2(0), size - 1), 0).a;
    n += texelFetch(tMap, clamp(p - ivec2(1, 0), ivec2(0), size - 1), 0).a;
    n += texelFetch(tMap, clamp(p + ivec2(0, 1), ivec2(0), size - 1), 0).a;
    n += texelFetch(tMap, clamp(p - ivec2(0, 1), ivec2(0), size - 1), 0).a;
    if (n < 0.5) discard;
    gl_FragColor = vec4(ink, 1.0);
  }
  #include <colorspace_fragment>
}`;

export interface SpritePass {
  /** Draw the scene with `subject` in the sprite-stack look. cellWorld: one cell's world size. */
  render(subject: Object3D, cellWorld: number, distance: number): void;
  dispose(): void;
}

export function createSpritePass(renderer: WebGLRenderer, scene: Scene, camera: PerspectiveCamera): SpritePass {
  let target: WebGLRenderTarget | null = null;
  const uniforms = {
    tMap: { value: null as WebGLRenderTarget["texture"] | null },
    scale: { value: 1 },
    ink: { value: new Color(OUTLINE_INK) },
  };
  const mat = new ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const quad = new Mesh(new PlaneGeometry(2, 2), mat);
  quad.frustumCulled = false;
  const blit = new Scene();
  blit.add(quad);
  const blitCam = new OrthographicCamera(-1, 1, 1, -1, -1, 1);
  const clear = new Color();

  function ensure(w: number, h: number) {
    if (target && target.width === w && target.height === h) return target;
    target?.dispose();
    // Half float: the slices' colours are stored linear, and 8 bits linear would band the dark ones.
    target = new WebGLRenderTarget(w, h, {
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      format: RGBAFormat,
      type: HalfFloatType,
      depthBuffer: true,
      generateMipmaps: false,
    });
    uniforms.tMap.value = target.texture;
    return target;
  }

  return {
    render(subject, cellWorld, distance) {
      const dw = renderer.domElement.width;
      const dh = renderer.domElement.height;
      const scale = cellScale(cellWorld, distance, camera.fov, dh);
      const low = lowSize(dw, dh, scale);
      const rt = ensure(low.w, low.h);
      uniforms.scale.value = scale;

      // 1. Everything but the subject, full size.
      const wasVisible = subject.visible;
      subject.visible = false;
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);

      // 2. The subject alone, small, on a clear ground. The target covers low.w*scale x low.h*scale screen pixels
      // from the bottom left, so its pixels line up with the screen's in whole blocks.
      const shown = scene.children.map((c) => c.visible);
      for (const c of scene.children) c.visible = false;
      subject.visible = true;
      let p: Object3D | null = subject;
      const hidden: Object3D[] = [];
      while (p && p !== scene) {
        if (!p.visible) {
          hidden.push(p);
          p.visible = true;
        }
        p = p.parent;
      }
      const bg = scene.background;
      scene.background = null;
      renderer.getClearColor(clear);
      const clearAlpha = renderer.getClearAlpha();
      renderer.setClearColor(0x000000, 0);
      camera.setViewOffset(dw, dh, 0, dh - low.h * scale, low.w * scale, low.h * scale);
      renderer.setRenderTarget(rt);
      renderer.clear();
      renderer.render(scene, camera);
      camera.clearViewOffset();
      renderer.setRenderTarget(null);
      renderer.setClearColor(clear, clearAlpha);
      scene.background = bg;
      scene.children.forEach((c, i) => (c.visible = shown[i]));
      for (const h of hidden) h.visible = false;
      subject.visible = wasVisible;

      // 3. Enlarged onto the screen, with the outline, over step 1.
      const autoClear = renderer.autoClear;
      renderer.autoClear = false;
      renderer.render(blit, blitCam);
      renderer.autoClear = autoClear;
    },
    dispose() {
      target?.dispose();
      quad.geometry.dispose();
      mat.dispose();
    },
  };
}
