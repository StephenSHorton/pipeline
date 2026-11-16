// Viewer defaults held to numbers (#356, teardown's checklist in #352; the same numbers are in
// scripts/workbench/playbook.md "Viewer and thumbnail defaults"):
// - framing: at the orbit home the model fills 60-70% of the frame on its limiting side, not its bounding sphere
//   (FILL is fit to the farthest point from the centre; a lopsided model, a pig's head low and forward, lands ~62%);
// - near-white parts on Mokei's pale ground (#e9eef8, the pad #f7f9fd, Clay #f7f9fd) get an OUTLINE_PX outline.
// No bloom and no gloss to check here: the diorama is Lambert + N8AO, NoToneMapping (prepareModel drops emissive).
import { BackSide, Color, MeshBasicMaterial, Vector2, type WebGLRenderer } from "three";

/** Twice the model's farthest point from the view centre, as a share of the frame's matching side, at the orbit home. */
export const FILL = 0.68;
/** Outline width in CSS pixels (in game pixels under the PS1 pass, whose small target is the frame). */
export const OUTLINE_PX = 1.5;
/** Mokei's ink (aoColor #25335a), a little lighter so it reads as an edge, not a stroke. */
export const OUTLINE_COLOR = "#3a4568";

/** Half extents of the points seen by Mokei's ClayCameraRig at (azimuth, elevation) degrees, orthographic.
 * The camera sits at (sin az cos el, sin el, cos az cos el) and looks at the origin (CameraRig.tsx). */
export function viewExtents(points: ArrayLike<number>, azimuth: number, elevation: number): { x: number; y: number } {
  const az = (azimuth * Math.PI) / 180;
  const el = (elevation * Math.PI) / 180;
  // three's lookAt basis for that camera: right = (cos az, 0, -sin az), up = (-sin el sin az, cos el, -sin el cos az).
  const rx = Math.cos(az), rz = -Math.sin(az);
  const ux = -Math.sin(el) * Math.sin(az), uy = Math.cos(el), uz = -Math.sin(el) * Math.cos(az);
  let x = 0, y = 0;
  for (let i = 0; i + 2 < points.length; i += 3) {
    const px = points[i], py = points[i + 1], pz = points[i + 2];
    x = Math.max(x, Math.abs(px * rx + pz * rz));
    y = Math.max(y, Math.abs(px * ux + py * uy + pz * uz));
  }
  return { x, y };
}

/** Ortho zoom (px per world unit) that fits the extents to FILL of a w x h frame. */
export function fitZoom(ext: { x: number; y: number }, w: number, h: number, fill = FILL): number {
  const zx = ext.x > 1e-6 ? (w * fill) / (2 * ext.x) : Infinity;
  const zy = ext.y > 1e-6 ? (h * fill) / (2 * ext.y) : Infinity;
  const z = Math.min(zx, zy);
  return Number.isFinite(z) ? z : 1;
}

/** The outline's hull material: back faces pushed out along the normal by a fixed number of screen pixels. */
export function outlineMaterial(skinned: boolean): MeshBasicMaterial {
  const res = new Vector2(1, 1);
  const m = new MeshBasicMaterial({ color: new Color(OUTLINE_COLOR), side: BackSide });
  m.name = "outline";
  m.customProgramCacheKey = () => `viewer-outline-v1-${skinned ? "s" : "m"}`;
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uOutlinePx = { value: OUTLINE_PX };
    shader.uniforms.uRes = { value: res };
    shader.vertexShader = shader.vertexShader
      .replace("void main() {", "uniform float uOutlinePx;\nuniform vec2 uRes;\nvoid main() {")
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
#ifdef USE_SKINNING
  vec3 oN = objectNormal;
#else
  vec3 oN = normal;
#endif
  vec2 cN = (projectionMatrix * modelViewMatrix * vec4(oN, 0.0)).xy;
  float cL = length(cN);
  if (cL > 1e-5) gl_Position.xy += cN / cL * uOutlinePx * 2.0 / uRes * gl_Position.w;`,
      );
  };
  m.userData.res = res;
  return m;
}

/** Keep the hull's pixel width right for whatever it is drawn into (the canvas, or the PS1 pass's target). */
export function trackTargetSize(m: MeshBasicMaterial, renderer: WebGLRenderer): void {
  const res = m.userData.res as Vector2;
  const full = renderer.getDrawingBufferSize(res);
  const fw = full.x;
  const rt = renderer.getRenderTarget();
  // PostFX draws into a full-size buffer (CSS px = that / pixel ratio); the PS1 pass into a small one (game px).
  if (rt && rt.width < fw * 0.9) res.set(rt.width, rt.height);
  else res.divideScalar(renderer.getPixelRatio());
}
