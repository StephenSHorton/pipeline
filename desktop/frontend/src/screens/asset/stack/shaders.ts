// The stack's shader patches, word for word from public/stack-lab.js: the t3ssel8r pixel-art AA blit with its
// depth outline (66-100), the Bright / Contrast / Saturate grade (215-238) and the per-voxel PBR of the palette
// (253-306). Plain string edits on three's chunks, so the tests can check every chunk they hook still exists.

export interface ShaderLike {
  uniforms: Record<string, { value: unknown }>;
  vertexShader: string;
  fragmentShader: string;
}

export interface GradeUniforms {
  brightness: { value: number };
  contrast: { value: number };
  saturation: { value: number };
}

const GRADE_UNIFORMS = `
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;`;

const GRADE = `#include <color_fragment>
{
  vec3 g = diffuseColor.rgb * uBrightness;
  g = (g - 0.5) * uContrast + 0.5;
  float luma = dot(g, vec3(0.2126, 0.7152, 0.0722));
  g = mix(vec3(luma), g, uSaturation);
  diffuseColor.rgb = clamp(g, 0.0, 1.0);
}`;

function gradeUniforms(shader: ShaderLike, grade: GradeUniforms): void {
  shader.uniforms.uBrightness = grade.brightness;
  shader.uniforms.uContrast = grade.contrast;
  shader.uniforms.uSaturation = grade.saturation;
}

/** Slices, contour slabs and the Round surface (legacy injectGrade). */
export function injectGrade(shader: ShaderLike, grade: GradeUniforms): void {
  gradeUniforms(shader, grade);
  shader.fragmentShader = shader.fragmentShader
    .replace("#include <common>", `#include <common>${GRADE_UNIFORMS}`)
    .replace("#include <color_fragment>", GRADE);
}

/** The voxel cubes: the grade plus metal / rough / glow per instance from the aPBR attribute (glow × 8). */
export function injectVoxelPbr(shader: ShaderLike, grade: GradeUniforms): void {
  gradeUniforms(shader, grade);
  shader.vertexShader = shader.vertexShader
    .replace(
      "#include <common>",
      `#include <common>
attribute vec3 aPBR;
varying vec3 vPBR;`,
    )
    .replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
vPBR = aPBR;`,
    );
  shader.fragmentShader = shader.fragmentShader
    .replace(
      "#include <common>",
      `#include <common>
varying vec3 vPBR;${GRADE_UNIFORMS}`,
    )
    .replace("#include <color_fragment>", GRADE)
    .replace(
      "#include <metalnessmap_fragment>",
      `#include <metalnessmap_fragment>
metalnessFactor = vPBR.x;`,
    )
    .replace(
      "#include <roughnessmap_fragment>",
      `#include <roughnessmap_fragment>
roughnessFactor = vPBR.y;`,
    )
    .replace(
      "#include <emissivemap_fragment>",
      `#include <emissivemap_fragment>
totalEmissiveRadiance += diffuseColor.rgb * vPBR.z * 8.0;`,
    );
}

export interface PixelUniforms {
  tDepth: { value: unknown };
  resolution: { value: unknown };
  outline: { value: number };
}

/** The stack's own Pixel look: fwidth AA on the upscale, and Outline darkens depth edges to 0.72. */
export function injectPixelBlit(shader: ShaderLike, u: PixelUniforms): void {
  shader.uniforms.tDepth = u.tDepth;
  shader.uniforms.resolution = u.resolution;
  shader.uniforms.outline = u.outline;
  shader.fragmentShader = shader.fragmentShader
    .replace(
      "#include <common>",
      `#include <common>
uniform sampler2D tDepth;
uniform vec2 resolution;
uniform float outline;`,
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
}`,
    );
}
