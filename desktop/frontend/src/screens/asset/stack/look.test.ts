// The stack's looks: the shader patches hook chunks three still has, the Game look is pixel-look.js's (same
// shaders, same shared record), and the view is wired the way tests/test_pixel_look.py and test_stack_looks.py pin
// the legacy stack-lab.js (subject: stackGroup, gameLook.render(), the paint ray through gameLook.viewCamera(),
// the Pixel pass's MeshBasicMaterial with toneMapped false and the 0.72 outline).
import { readFileSync } from "node:fs";
import path from "node:path";
import { ShaderChunk } from "three";
import { describe, expect, it } from "vitest";
import { BLIT_FRAG, BLIT_VERT, GAME_DEFAULT, readGameLook } from "./gameLook";
import { injectGrade, injectPixelBlit, injectVoxelPbr, type ShaderLike } from "./shaders";

const here = path.dirname(new URL(import.meta.url).pathname);
const src = (rel: string) => readFileSync(path.join(here, rel), "utf8");

const shader = (): ShaderLike => ({
  uniforms: {},
  vertexShader: "#include <common>\nvoid main() {\n#include <begin_vertex>\n}",
  fragmentShader: [
    "#include <common>",
    "#include <map_fragment>",
    "#include <color_fragment>",
    "#include <metalnessmap_fragment>",
    "#include <roughnessmap_fragment>",
    "#include <emissivemap_fragment>",
  ].join("\n"),
});
const grade = { brightness: { value: 1 }, contrast: { value: 1.2 }, saturation: { value: 1.2 } };

describe("shader patches", () => {
  it("every chunk they hook is still in three", () => {
    for (const c of ["common", "map_fragment", "color_fragment", "metalnessmap_fragment", "roughnessmap_fragment", "emissivemap_fragment", "begin_vertex"])
      expect(ShaderChunk[c as keyof typeof ShaderChunk], c).toBeTypeOf("string");
    expect(ShaderChunk.map_fragment).toContain("vMapUv");
  });

  it("the grade adds its uniforms and the brightness / contrast / saturation block", () => {
    const s = shader();
    injectGrade(s, grade);
    expect(Object.keys(s.uniforms)).toEqual(["uBrightness", "uContrast", "uSaturation"]);
    expect(s.uniforms.uContrast).toBe(grade.contrast);
    expect(s.fragmentShader).toContain("#include <common>\nuniform float uBrightness;\nuniform float uContrast;\nuniform float uSaturation;");
    expect(s.fragmentShader).toContain("g = mix(vec3(luma), g, uSaturation);");
  });

  it("the voxel PBR reads aPBR per instance: metal, rough, glow × 8", () => {
    const s = shader();
    injectVoxelPbr(s, grade);
    expect(s.vertexShader).toContain("attribute vec3 aPBR;\nvarying vec3 vPBR;");
    expect(s.vertexShader).toContain("#include <begin_vertex>\nvPBR = aPBR;");
    expect(s.fragmentShader).toContain("metalnessFactor = vPBR.x;");
    expect(s.fragmentShader).toContain("roughnessFactor = vPBR.y;");
    expect(s.fragmentShader).toContain("totalEmissiveRadiance += diffuseColor.rgb * vPBR.z * 8.0;");
  });

  it("the Pixel pass replaces map_fragment with the fwidth AA and darkens depth edges to 0.72", () => {
    const s = shader();
    injectPixelBlit(s, { tDepth: { value: null }, resolution: { value: null }, outline: { value: 1 } });
    expect(s.fragmentShader).not.toContain("#include <map_fragment>");
    expect(s.fragmentShader).toContain("vec2 filterSize = clamp(fwidth(vMapUv) * resolution, vec2(1e-5), vec2(1.0));");
    expect(s.fragmentShader).toContain("diffuseColor.rgb * 0.72");
  });
});

describe("the Game look", () => {
  it("draws with pixel-look.js's own shaders", () => {
    // The standalone viewer's module is the reference (the desktop copy went with the classic page).
    const legacy = readFileSync(path.join(here, "../../../../../../viewer/pixel-look.js"), "utf8");
    expect(legacy).toContain(BLIT_VERT);
    expect(legacy).toContain(BLIT_FRAG);
    expect(BLIT_FRAG).toContain("vUv * lowRes + 1.0 + frac");
    expect(BLIT_FRAG).toContain("bayer4");
    expect(BLIT_FRAG).toContain("#include <colorspace_fragment>");
  });

  it("reads the shared pipeline.pixelLook record over the defaults, then ?game=", () => {
    expect(readGameLook(null)).toEqual(GAME_DEFAULT);
    expect(readGameLook("{broken")).toEqual(GAME_DEFAULT);
    const saved = JSON.stringify({ on: true, pixelSize: 6, iso: false, levels: 5, dither: true, lines: 240, junk: 1, outline: "yes" });
    expect(readGameLook(saved)).toEqual({ on: true, pixelSize: 6, outline: true, iso: false, shadows: true, levels: 5, dither: true, lines: 240 });
    expect(readGameLook(saved, "?game=0").on).toBe(false);
    expect(readGameLook(null, "?game=1").on).toBe(true);
  });
});

describe("the view's wiring (as legacy stack-lab.js)", () => {
  const view = src("view.ts");
  const gameLook = src("gameLook.ts");

  it("shares the Game look with the stack group as its subject, and paints through its camera", () => {
    expect(view).toContain("createGameLook({ renderer, scene, camera, controls, subject: stackGroup, key, ground, blob, defaults: game })");
    expect(view).toContain("gameLook.render();");
    expect(view).toContain("paintRay.setFromCamera(paintNdc, gameLook.viewCamera());");
    for (const s of ["ShadowMaterial", "isMeshBasicMaterial", "OrthographicCamera", "Math.round(r)", "MeshNormalMaterial", "DepthTexture", "PCFSoftShadowMap"])
      expect(gameLook, s).toContain(s);
  });

  it("Pixel is a MeshBasicMaterial blit that skips tone mapping; Main rebuilds the island mask; orbit is full", () => {
    expect(view).toMatch(/new MeshBasicMaterial\(\{\s*toneMapped: false,/);
    expect(view).toContain("rebuildIslandMask(c, crunch);");
    expect(view).toContain("controls.maxPolarAngle = ORBIT.maxPolarAngle;");
  });

  it("frees its WebGL context on dispose", () => {
    expect(view).toContain("renderer.dispose();");
    expect(view).toContain("renderer.forceContextLoss();");
    expect(view).toContain("cancelAnimationFrame(raf);");
  });
});
