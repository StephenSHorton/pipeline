// The PS1 look, simple but real: render the scene into a small target and blit it to the canvas with nearest
// filtering. Mounted instead of PostFX (both take over rendering with a priority useFrame). Full
// pixel-look.js parity (iso snap, outline, posterize, dither) comes later.
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import {
  Mesh,
  MeshBasicMaterial,
  NearestFilter,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  WebGLRenderTarget,
} from "three";
import { pixelTarget, type PixelLook } from "./pixelLook";

export function PixelPass({ look }: { look: PixelLook }) {
  const size = useThree((s) => s.size);
  const { w, h } = pixelTarget(look, size.width, size.height);

  const blit = useMemo(() => {
    const rt = new WebGLRenderTarget(1, 1, { minFilter: NearestFilter, magFilter: NearestFilter, depthBuffer: true });
    const mat = new MeshBasicMaterial({ map: rt.texture, depthTest: false, depthWrite: false, toneMapped: false });
    const quad = new Mesh(new PlaneGeometry(2, 2), mat);
    const scene = new Scene();
    scene.add(quad);
    const cam = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    return { rt, mat, quad, scene, cam };
  }, []);

  useEffect(() => {
    blit.rt.setSize(w, h);
  }, [blit, w, h]);

  useEffect(
    () => () => {
      blit.rt.dispose();
      blit.mat.dispose();
      blit.quad.geometry.dispose();
    },
    [blit],
  );

  useFrame(({ gl, scene, camera }) => {
    gl.setRenderTarget(blit.rt);
    gl.clear();
    gl.render(scene, camera);
    gl.setRenderTarget(null);
    gl.render(blit.scene, blit.cam);
  }, 1);

  return null;
}
