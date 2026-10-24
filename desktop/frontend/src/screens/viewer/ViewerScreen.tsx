// The asset viewer for the 3D and Rig stages: the GLB as a clay diorama (src/viewer/Diorama.tsx) under a glass
// HUD (./Hud.tsx), with the Cleanup, Paint and Detail levels rail and the Rig check dock (./panels/). Panels reach
// the model through the tool API (src/viewer/tools.ts), which this screen feeds: the model, the file on screen,
// and loadFile / reload. The old non-Mokei page (and its link from the HUD) is retired at Stephen's ask; the
// articulated pivots (articulated.json) it showed have no view here yet. Compare (./compare.ts) splits the picture in
// two: an earlier version on the left (./CompareSide.tsx), this one on the right, one camera and clip for both.
import { useLook } from "mokei/clay";
import { useMaterialColor } from "mokei/theme";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode, type RefObject } from "react";
import type { AnimationClip } from "three";
import { navigate } from "@/app/router";
import { outUrl } from "@/lib/api";
import { flowEntry } from "@/lib/flow";
import { onJobFinished } from "@/lib/jobs";
import type { StageProps } from "@/screens/asset/types";
import { RIGGED, fileOf, fileTag, isPixelFile, modelCandidates, preferredClip, viewableLods } from "@/viewer/candidates";
import { Diorama, resetLook } from "@/viewer/Diorama";
import { loadFirstGlb } from "@/viewer/loadGlb";
import { readPixelLook, toggle240, writePixelLook, type PixelLook } from "@/viewer/pixelLook";
import { prepareModel, type PreparedModel } from "@/viewer/prepareModel";
import { outRelFile, resetViewerTools, setViewerLoader, setViewerModel, useViewerTools, viewerTools, type ViewerTool } from "@/viewer/tools";
import { useAssetDocs } from "@/viewer/useAssetDocs";
import { useNarrow } from "@/lib/useNarrow";
import { cn } from "@/lib/utils";
import { compareTakes, comparing, leftTake, paneClears, paneLayout, roomShift, sideLabels, useCompare, type Insets, type PaneClears } from "./compare";
import { CompareSide, SideLabel } from "./CompareSide";
import { Hud } from "./Hud";
import type { ViewerStage } from "./panels/panels";
import { usePanels } from "./panels/store";
import type { LoadState, ViewerPanelBase, ViewerView } from "./types";

/** A file loadFile asked for (cache-busted by t). */
interface Override {
  file: string;
  t: number;
}
type Waiter = { resolve: () => void; reject: (err: Error) => void };

export function ViewerScreen({ lod, versions, ...stage }: StageProps & { lod?: string; versions?: ReactNode }) {
  const { stem, flowVersion } = stage;
  const [version, setVersion] = useState(0);
  // Re-reads the docs without dropping a loadFile override (reload, loadFile); `version` drops it.
  const [docsBump, setDocsBump] = useState(0);
  const docs = useAssetDocs(stem, `${version}.${docsBump}`);
  const clayColor = useMaterialColor("base");

  const [load, setLoad] = useState<LoadState>({ state: "loading" });
  const loadRef = useRef(load);
  loadRef.current = load;
  const [model, setModel] = useState<PreparedModel | null>(null);
  const [clips, setClips] = useState<AnimationClip[]>([]);
  const [clip, setClip] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [clay, setClay] = useState(false);
  const [wire, setWire] = useState(false);
  // Off until asked for (Stephen: "Turn off spin by default"): the model holds still while you look it over.
  const [spin, setSpin] = useState(false);
  const [pixel, setPixelState] = useState<PixelLook>(() => readPixelLook());

  const setPixel = useCallback((p: PixelLook) => {
    setPixelState(p);
    writePixelLook(p);
  }, []);

  // loadFile's file: `want` is what to load next, `shown` what is on screen. Legacy load() went back to the usual
  // candidates after a job, a version pick or another LOD; so does this (want = null).
  const want = useRef<Override | null>(null);
  const shown = useRef<Override | null>(null);
  const waiters = useRef<Waiter[]>([]);
  const settle = useCallback((err: Error | null) => {
    const ws = waiters.current;
    waiters.current = [];
    for (const w of ws) (err ? w.reject(err) : w.resolve());
  }, []);

  // A finished job on this asset (a rig, a remake) re-reads its files and reloads the model.
  useEffect(
    () =>
      onJobFinished((j) => {
        // A sheet render writes rig/preview/ only: the model and the Rig check on it stay as they are.
        if (j.stem !== stem || j.kind === "render") return;
        want.current = null;
        setVersion((v) => v + 1);
      }),
    [stem],
  );
  // A pick in the versions panel re-reads the flow: reload the model (the camera stays). Not on the first run.
  const seenFlow = useRef(flowVersion);
  useEffect(() => {
    if (seenFlow.current === flowVersion) return;
    seenFlow.current = flowVersion;
    want.current = null;
    setVersion((v) => v + 1);
  }, [flowVersion]);
  // Another LOD from the picker or a stage change: back to the candidates (the load effect below runs on lod).
  const seenLod = useRef(lod);
  if (seenLod.current !== lod) {
    seenLod.current = lod;
    want.current = null;
  }
  // The look store is global: give it back when the viewer goes away. The tool API and the open panel too
  // (resetViewerTools disarms the tool, legacy left Carve and Paint armed).
  useEffect(
    () => () => {
      resetLook();
      resetViewerTools();
      usePanels.getState().close();
      useCompare.getState().stop();
    },
    [],
  );
  useEffect(() => () => settle(new Error("The viewer closed.")), [settle]);
  useEffect(() => {
    setViewerLoader({
      loadFile(file) {
        want.current = { file: file.replace(/^\/+/, ""), t: Date.now() };
        return new Promise<void>((resolve, reject) => {
          waiters.current.push({ resolve, reject });
          setDocsBump((b) => b + 1);
        });
      },
      reload() {
        want.current = shown.current ? { ...shown.current, t: Date.now() } : want.current;
        return new Promise<void>((resolve, reject) => {
          waiters.current.push({ resolve, reject });
          setDocsBump((b) => b + 1);
        });
      },
    });
    return () => setViewerLoader(null);
  }, []);

  // Load the model: the legacy candidate order, first GLB that parses wins.
  useEffect(() => {
    if (!docs) return;
    const ctl = new AbortController();
    const forced = want.current;
    const cand = modelCandidates({ stem, lod, lods: docs.lods?.lods ?? [], hasRig: docs.hasRig, rigFiles: docs.rigFiles });
    // A loadFile file is the only URL tried (legacy loadFrom([url])): a miss keeps the model on screen.
    const urls = forced ? [outUrl(stem, forced.file, forced.t)] : cand.urls;
    const before = loadRef.current;
    setLoad({ state: "loading" });
    loadFirstGlb(urls, ctl.signal)
      .then(({ url, gltf, ms }) => {
        if (ctl.signal.aborted) return;
        shown.current = forced;
        const file = fileOf(url);
        // A loadFile file names its LOD when lods.json lists it (Cleanup's "Clean" once the docs are re-read).
        const shownLod = forced ? (viewableLods(docs.lods?.lods).find((l) => l.file === forced.file)?.id ?? "") : /rigged/i.test(file) ? RIGGED : cand.lod;
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
        settle(null);
      })
      .catch((err: unknown) => {
        if (ctl.signal.aborted) return;
        const message = String((err as Error)?.message || err);
        if (forced) {
          want.current = shown.current;
          if (before.state !== "loading") setLoad(before);
          settle(new Error(message));
          return;
        }
        settle(new Error(message));
        setModel(null);
        setClips([]);
        setClip(null);
        setLoad({ state: "error", message });
      });
    return () => ctl.abort();
  }, [docs, stem, lod, clayColor, settle]);

  // GPU memory: a model is freed once the next one replaces it, and when the viewer closes.
  useEffect(() => () => model?.dispose(), [model]);

  const lods = useMemo(() => viewableLods(docs?.lods?.lods), [docs]);
  const info = load.state === "ready" ? load.info : null;
  const isPs1 = !!info && (/^psx/i.test(info.file) || docs?.intent?.output === "ps1");

  // Picking the LOD the URL already names (a loadFile file such as Cleanup's is on screen) reloads the candidates:
  // navigating to the same URL would change nothing.
  const pickLod = useCallback(
    (id: string) => {
      if (id !== lod) return navigate({ name: "asset", stem, lod: id }, { replace: true });
      want.current = null;
      setDocsBump((b) => b + 1);
    },
    [stem, lod],
  );

  // The tool API follows the model on screen; a new model disarms the tool. Not while loading: currentFile
  // stays the file on screen until the next one has loaded (legacy loadFrom set it after the parse).
  useEffect(() => {
    if (load.state === "loading") return;
    setViewerModel(model, load.state === "ready" ? outRelFile(load.info.url, stem) : "");
  }, [model, load, stem]);
  const tool = useViewerTools((s) => s.tool);

  const onClip = useCallback((name: string) => {
    setClip(name);
    setPlaying(true);
  }, []);
  // Compare: the left take while it is on and there is one to show (the chosen one, else the version before).
  const viewerStage: ViewerStage = stage.stage === "rig" ? "rig" : "mesh";
  const entry = flowEntry(stage.flow, viewerStage);
  const canCompare = compareTakes(entry).length > 0;
  const compareOn = useCompare((s) => comparing(s, stem, viewerStage)) && canCompare;
  const chosenLeft = useCompare((s) => s.left);
  const leftId = compareOn ? leftTake(entry, chosenLeft) : null;
  const left = leftId ? (entry?.takes.find((t) => t.id === leftId) ?? null) : null;
  const labels = sideLabels(entry, leftId);
  // Another stage (3D and Rig share this screen) starts single again.
  useEffect(() => {
    const c = useCompare.getState();
    if (c.stem && (c.stem !== stem || c.stage !== viewerStage)) c.stop();
  }, [stem, viewerStage]);
  // The panes sit in the room the HUD leaves (./compare.ts paneLayout), so neither model is centred under a card.
  const root = useRef<HTMLDivElement>(null);
  const panes = usePaneLayout(root, !!left);
  const onCompare = useCallback(
    (on: boolean) => (on ? useCompare.getState().start(stem, viewerStage) : useCompare.getState().stop()),
    [stem, viewerStage],
  );

  const view: ViewerView = { clips: clips.map((c) => c.name), clip, onClip, playing, onPlaying: setPlaying, spin, onSpin: setSpin, clay, onClay: setClay, wire, onWire: setWire };
  const panel: ViewerPanelBase = {
    ...stage,
    stage: viewerStage,
    docs,
    load,
    lods,
    onPickLod: pickLod,
    reloadDocs: () => setDocsBump((b) => b + 1),
    view,
    tools: viewerTools,
  };

  return (
    // The clay fills the stage; the versions panel rides in the HUD's left column, so nothing sits under the ground.
    // Comparing, the two dioramas split the stage: left and right, stacked on phones and when the room is thin. The HUD
    // floats over both, and each model is centred in its pane's room the HUD leaves (Diorama's room). The right one
    // keeps its place in the tree, so turning Compare on or off never remounts it (the model stays; the camera keeps
    // its orbit, the zoom re-frames).
    <div ref={root} className="absolute inset-0 overflow-hidden">
      <div className={cn("absolute inset-0 flex", left && panes?.stacked && "flex-col")}>
        {left ? (
          // Its own surface without the panel tool: a drag turns both sides, a brush only paints the right.
          <div className={cn("relative min-h-0 min-w-0 flex-1 border-hair", panes?.stacked ? "border-b" : "border-r")}>
            <OrbitSurface tool={null}>
              <CompareSide
                stem={stem}
                stage={viewerStage}
                take={left}
                label={labels.left}
                shown={info ? outRelFile(info.url, stem) : ""}
                clayColor={clayColor}
                clip={clip}
                playing={playing}
                clay={clay}
                wire={wire}
                spin={spin}
                pixel={pixel}
                stacked={!!panes?.stacked}
                room={panes ? { shift: roomShift(panes.first), frame: panes.frame } : undefined}
                clear={panes?.first ?? NO_INSETS}
              />
            </OrbitSurface>
          </div>
        ) : null}
        <div className="relative min-h-0 min-w-0 flex-1">
          <OrbitSurface tool={tool}>
            <Diorama
              model={model}
              clips={clips}
              clip={clip}
              playing={playing}
              clay={clay}
              wire={wire}
              spin={spin}
              pixel={pixel}
              room={left && panes ? { shift: roomShift(panes.second), frame: panes.frame } : undefined}
            />
          </OrbitSurface>
          {left ? <SideLabel side="right" text={labels.right} stacked={!!panes?.stacked} clear={panes?.second ?? NO_INSETS} /> : null}
        </div>
      </div>
      <div className="pointer-events-none absolute inset-0">
        <Hud
          versions={versions}
          stem={stem}
          docs={docs}
          load={load}
          lods={lods}
          onPickLod={pickLod}
          clips={view.clips}
          clip={clip}
          playing={playing}
          onClip={onClip}
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
          compare={canCompare ? { on: compareOn, onToggle: onCompare } : null}
          onRenamed={(next) => navigate({ name: "asset", stem: next, ...(lod ? { lod } : {}) }, { replace: true })}
          panel={panel}
        />
      </div>
    </div>
  );
}

const NO_INSETS: Insets = { top: 0, right: 0, bottom: 0, left: 0 };

/**
 * Each pane's room while comparing, measured from the HUD's columns (data-hud-col in ./Hud.tsx): their sizes are
 * enough, since each is anchored to its edge. Re-measured when one changes (a panel opens, the clips arrive).
 */
function usePaneLayout(root: RefObject<HTMLDivElement | null>, on: boolean): (PaneClears & { stacked: boolean }) | null {
  const narrow = useNarrow();
  const [layout, setLayout] = useState<(PaneClears & { stacked: boolean }) | null>(null);
  useLayoutEffect(() => {
    const el = root.current;
    if (!on || !el) return setLayout(null);
    const col = (name: string) => el.querySelector<HTMLElement>(`[data-hud-col=${name}]`);
    const size = (e: Element | null) => (e ? e.getBoundingClientRect() : { width: 0, height: 0 });
    const measure = () => {
      const right = col("right");
      // The right column under the view card: the camera stack and the dock (the view card is the top inset).
      const under = right ? [...right.children].filter((c) => !c.matches("[data-hud-col=top]")) : [];
      const room = paneLayout({
        width: el.clientWidth,
        height: el.clientHeight,
        narrow,
        left: size(col("left")),
        right: Math.max(0, ...under.map((c) => size(c).width)),
        top: size(col("top")).height,
        bottom: size(col("bottom")).height,
      });
      const next = { ...paneClears(el.clientWidth, el.clientHeight, room), stacked: room.stacked };
      setLayout((was) => (JSON.stringify(was) === JSON.stringify(next) ? was : next));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    for (const name of ["left", "right", "top", "bottom"]) {
      const c = col(name);
      if (c) ro.observe(c);
    }
    return () => ro.disconnect();
  }, [root, on, narrow]);
  return layout;
}

/**
 * Orbit from the DOM: left-drag turns (azimuth) and tilts (elevation); right-drag or shift-drag pans with
 * ClayGround's formula; the wheel zooms through ClayCameraRig. The HUD sits above with pointer-events off.
 * With a panel tool armed (src/viewer/tools.ts), the left button and every move go to the tool, and right-drag
 * turns instead of panning (legacy Carve/Paint: "Orbit with right-drag while this is on").
 */
function OrbitSurface({ tool, children }: { tool: ViewerTool | null; children?: React.ReactNode }) {
  const drag = useRef<{ x: number; y: number; pan: boolean; id: number } | null>(null);
  const painting = useRef<number | null>(null);
  const pointer = (e: ReactPointerEvent<HTMLDivElement>, button: number) => {
    let hit: ReturnType<typeof viewerTools.raycastAt> | undefined;
    const { clientX, clientY } = e;
    return {
      clientX,
      clientY,
      button,
      buttons: e.buttons,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
      pointerId: e.pointerId,
      hit: () => (hit === undefined ? (hit = viewerTools.raycastAt(clientX, clientY)) : hit),
    };
  };
  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.button !== 2) return;
    if (tool && e.button === 0) {
      painting.current = e.pointerId;
      e.currentTarget.setPointerCapture(e.pointerId);
      tool.onDown?.(pointer(e, 0));
      return;
    }
    drag.current = { x: e.clientX, y: e.clientY, pan: !tool && (e.button === 2 || e.shiftKey), id: e.pointerId };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (tool && !(d && d.id === e.pointerId)) {
      tool.onMove?.(pointer(e, -1));
      return;
    }
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
    if (painting.current === e.pointerId) {
      painting.current = null;
      tool?.onUp?.(pointer(e, 0));
    }
  };
  // A tool disarmed mid-stroke must not get the up; the next tool starts clean.
  useEffect(() => {
    painting.current = null;
  }, [tool]);
  return (
    <div
      className={cn("absolute inset-0 touch-none select-none", tool ? "" : "cursor-grab active:cursor-grabbing")}
      style={tool ? { cursor: tool.cursor ?? "crosshair" } : undefined}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onPointerLeave={() => tool?.onLeave?.()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {children}
    </div>
  );
}
