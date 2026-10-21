// The asset viewer: the GLB as a clay diorama (src/viewer/Diorama.tsx) under a glass HUD (./Hud.tsx).
// Everything not ported yet (rig check panel, export, 2D, stack, paint, cleanup, the turnaround figures from
// views.json, articulated pivots) is one link away in the
// classic view (/legacy.html?asset=<stem>).
import { useLook } from "mokei/clay";
import { useMaterialColor } from "mokei/theme";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { AnimationClip } from "three";
import { navigate } from "@/app/router";
import { onJobFinished } from "@/lib/jobs";
import { RIGGED, fileOf, fileTag, isPixelFile, modelCandidates, preferredClip, viewableLods } from "@/viewer/candidates";
import { Diorama, resetLook } from "@/viewer/Diorama";
import { loadFirstGlb } from "@/viewer/loadGlb";
import { readPixelLook, toggle240, writePixelLook, type PixelLook } from "@/viewer/pixelLook";
import { prepareModel, type PreparedModel } from "@/viewer/prepareModel";
import { useAssetDocs } from "@/viewer/useAssetDocs";
import { Hud, type ModelInfo } from "./Hud";

type LoadState = { state: "loading" } | { state: "ready"; info: ModelInfo } | { state: "error"; message: string };

export function ViewerScreen({ stem, lod, flowVersion }: { stem: string; lod?: string; flowVersion?: number }) {
  const [version, setVersion] = useState(0);
  const docs = useAssetDocs(stem, version);
  const clayColor = useMaterialColor("base");

  const [load, setLoad] = useState<LoadState>({ state: "loading" });
  const [model, setModel] = useState<PreparedModel | null>(null);
  const [clips, setClips] = useState<AnimationClip[]>([]);
  const [clip, setClip] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [clay, setClay] = useState(false);
  const [wire, setWire] = useState(false);
  const [spin, setSpin] = useState(true);
  const [pixel, setPixelState] = useState<PixelLook>(() => readPixelLook());

  const setPixel = useCallback((p: PixelLook) => {
    setPixelState(p);
    writePixelLook(p);
  }, []);

  // A finished job on this asset (a rig, a remake) re-reads its files and reloads the model.
  useEffect(() => onJobFinished((j) => j.stem === stem && setVersion((v) => v + 1)), [stem]);
  // A pick in the versions strip re-reads the flow: reload the model (the camera stays). Not on the first run.
  const seenFlow = useRef(flowVersion);
  useEffect(() => {
    if (seenFlow.current === flowVersion) return;
    seenFlow.current = flowVersion;
    setVersion((v) => v + 1);
  }, [flowVersion]);
  // The look store is global: give it back when the viewer goes away.
  useEffect(() => () => resetLook(), []);

  // Load the model: the legacy candidate order, first GLB that parses wins.
  useEffect(() => {
    if (!docs) return;
    const ctl = new AbortController();
    const cand = modelCandidates({ stem, lod, lods: docs.lods?.lods ?? [], hasRig: docs.hasRig, rigFiles: docs.rigFiles });
    setLoad({ state: "loading" });
    loadFirstGlb(cand.urls, ctl.signal)
      .then(({ url, gltf, ms }) => {
        if (ctl.signal.aborted) return;
        const file = fileOf(url);
        const shownLod = /rigged/i.test(file) ? RIGGED : cand.lod;
        const made = prepareModel(gltf, { clayColor, pixel: isPixelFile(file, shownLod) });
        const names = gltf.animations.map((a) => a.name);
        setModel(made);
        setClips(gltf.animations);
        setClip(preferredClip(names));
        setPlaying(true);
        setLoad({
          state: "ready",
          info: { url, file, lod: shownLod, tag: fileTag(file, shownLod), ms, vertices: made.vertices, faces: made.faces, hasMap: made.hasMap },
        });
      })
      .catch((err: unknown) => {
        if (ctl.signal.aborted) return;
        setModel(null);
        setClips([]);
        setClip(null);
        setLoad({ state: "error", message: String((err as Error)?.message || err) });
      });
    return () => ctl.abort();
  }, [docs, stem, lod, clayColor]);

  // GPU memory: a model is freed once the next one replaces it, and when the viewer closes.
  useEffect(() => () => model?.dispose(), [model]);

  const lods = useMemo(() => viewableLods(docs?.lods?.lods), [docs]);
  const info = load.state === "ready" ? load.info : null;
  const isPs1 = !!info && (/^psx/i.test(info.file) || docs?.intent?.output === "ps1");

  const pickLod = useCallback((id: string) => navigate({ name: "asset", stem, lod: id }, { replace: true }), [stem]);

  return (
    // The clay runs on under the asset screen's versions strip (--strip-h, set by TakesStrip), so the stage and the
    // strip share one ground instead of meeting at a seam; the HUD keeps to the stage above it.
    <div className="absolute inset-x-0 top-0 bottom-[calc(-1*var(--strip-h,0px))] overflow-hidden">
      <OrbitSurface>
        <Diorama model={model} clips={clips} clip={clip} playing={playing} clay={clay} wire={wire} spin={spin} pixel={pixel} />
      </OrbitSurface>
      <div className="pointer-events-none absolute inset-x-0 top-0 bottom-[var(--strip-h,0px)]">
        <Hud
          stem={stem}
          docs={docs}
          load={load}
          lods={lods}
          hasRig={!!docs?.hasRig}
          onPickLod={pickLod}
          clips={clips.map((c) => c.name)}
          clip={clip}
          playing={playing}
          onClip={(name) => {
            setClip(name);
            setPlaying(true);
          }}
          onPlaying={setPlaying}
          clay={clay}
          canClay={!!info?.hasMap}
          onClay={setClay}
          wire={wire}
          onWire={setWire}
          spin={spin}
          onSpin={setSpin}
          pixel={pixel}
          isPs1={isPs1}
          onPixel={(on) => setPixel({ ...pixel, on })}
          on240={() => setPixel(toggle240(pixel))}
          onRenamed={(next) => navigate({ name: "asset", stem: next, ...(lod ? { lod } : {}) }, { replace: true })}
        />
      </div>
    </div>
  );
}

/**
 * Orbit from the DOM: left-drag turns (azimuth) and tilts (elevation); right-drag or shift-drag pans with
 * ClayGround's formula; the wheel zooms through ClayCameraRig. The HUD sits above with pointer-events off.
 */
function OrbitSurface({ children }: { children: React.ReactNode }) {
  const drag = useRef<{ x: number; y: number; pan: boolean; id: number } | null>(null);
  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.button !== 2) return;
    drag.current = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, id: e.pointerId };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    d.x = e.clientX;
    d.y = e.clientY;
    const look = useLook.getState();
    if (d.pan) {
      const az = (look.cameraAzimuth * Math.PI) / 180;
      // Screen px to world units: 1/zoom across; down the screen the ground is foreshortened by sin(elevation).
      const scale = 1 / Math.max(1e-3, look.cameraZoom);
      const tilt = 1 / Math.max(0.2, Math.sin((look.cameraElevation * Math.PI) / 180));
      look.setLook({
        panX: look.panX - Math.cos(az) * dx * scale - Math.sin(az) * dy * scale * tilt,
        panZ: look.panZ + Math.sin(az) * dx * scale - Math.cos(az) * dy * scale * tilt,
      });
    } else {
      look.setLook({
        cameraAzimuth: look.cameraAzimuth - dx * 0.4,
        cameraElevation: Math.max(4, Math.min(82, look.cameraElevation + dy * 0.3)),
      });
    }
  };
  const up = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (drag.current?.id === e.pointerId) drag.current = null;
  };
  return (
    <div
      className="absolute inset-0 cursor-grab touch-none select-none active:cursor-grabbing"
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onContextMenu={(e) => e.preventDefault()}
    >
      {children}
    </div>
  );
}
