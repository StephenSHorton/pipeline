// The legacy loadFirstGlb (ix:4946-4967): try each URL in order; skip a miss, a JSON answer, a body under
// 12 bytes or one without the glTF magic; parse the first that works. All fail: "no glb loaded".
// No Draco decoder is wired: the pipeline writes plain GLBs, and the hosted decoder would not load offline.
// A Draco/KTX2 file fails its parse with that message and the next URL is tried.
import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";

export interface LoadedGlb {
  url: string;
  gltf: GLTF;
  bytes: number;
  ms: number;
}

let loader: GLTFLoader | null = null;
function gltfLoader(): GLTFLoader {
  if (!loader) {
    loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder); // bundled, works offline
  }
  return loader;
}

export async function loadFirstGlb(urls: string[], signal?: AbortSignal): Promise<LoadedGlb> {
  for (const url of urls) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const t0 = performance.now();
    try {
      const res = await fetch(url, { signal });
      if (!res.ok) continue;
      const type = res.headers.get("content-type") || "";
      if (type.includes("application/json") && !type.includes("gltf")) continue;
      const buf = await res.arrayBuffer();
      if (buf.byteLength < 12) continue;
      const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
      if (magic !== "glTF") continue;
      const gltf = await gltfLoader().parseAsync(buf, "");
      return { url, gltf, bytes: buf.byteLength, ms: Math.round(performance.now() - t0) };
    } catch (err) {
      if ((err as { name?: string })?.name === "AbortError") throw err;
      console.warn("skip", url, err);
    }
  }
  throw new Error("no glb loaded");
}
