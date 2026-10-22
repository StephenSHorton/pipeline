// The viewer's glass HUD (Mokei playground look): panels at the edges, the model in the clear middle.
// Panels drop the backdrop blur on purpose: at 0.9 white the frost barely shows, and in the macOS webview
// (WebKit applies -webkit-backdrop-filter) it re-blurs the WebGL canvas behind every panel on every frame.
import { House, Minus, Pause, Play, Plus, RotateCcw, RotateCw } from "lucide-react";
import { useLook } from "mokei/clay";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { legacyUrl, renameAsset } from "@/lib/api";
import { degradedLine, displayStem } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { KIND_NAMES, RIGGED, slugStem } from "@/viewer/candidates";
import type { PixelLook } from "@/viewer/pixelLook";
import type { AssetDocs } from "@/viewer/useAssetDocs";
import { PanelRail } from "./panels/PanelRails";
import { PANEL, TOGGLE } from "./classes";
import type { LoadState, ModelInfo, ViewerPanelBase } from "./types";

export type { LoadState, ModelInfo } from "./types";

export { PANEL, TOGGLE } from "./classes";

export interface HudProps {
  stem: string;
  docs: AssetDocs | null;
  load: LoadState;
  clips: string[];
  clip: string | null;
  playing: boolean;
  onClip: (name: string) => void;
  onPlaying: (on: boolean) => void;
  clay: boolean;
  canClay: boolean;
  onClay: (on: boolean) => void;
  wire: boolean;
  onWire: (on: boolean) => void;
  spin: boolean;
  onSpin: (on: boolean) => void;
  pixel: PixelLook;
  isPs1: boolean;
  onPixel: (on: boolean) => void;
  on240: () => void;
  onRenamed: (stem: string) => void;
  /** What the rail and dock panels get (ViewerPanelProps minus id, open, onClose). */
  panel: ViewerPanelBase;
}

export function Hud(p: HudProps) {
  const info = p.load.state === "ready" ? p.load.info : null;
  return (
    <div data-hud className="pointer-events-none absolute inset-0 z-[5] text-ink tracking-[var(--track)]">
      {/* The asset card, then the panel rail under it (legacy left cubby rail). The column stops above the bottom
          pickers on phones and scrolls the open panel instead of running off the screen. */}
      <div className="absolute top-3 left-3 flex max-h-[calc(100%-24px)] w-[min(340px,calc(100%-24px))] flex-col gap-2 max-sm:max-h-[calc(100%-160px)]">
        <AssetCard {...p} info={info} />
        <PanelRail side="left" base={p.panel} />
      </div>
      {/* Wide: view toggles and the camera stack top right, the Rig check dock under them. Phones: one column at
          the bottom, in normal flow, so nothing overlaps: the toggles with Reset view, then the model and clip
          pickers (the dock's panels join the left rail there). */}
      <div className="absolute top-3 right-3 flex max-h-[calc(100%-24px)] flex-col items-end gap-2 max-sm:hidden">
        <ViewCard {...p} />
        <CameraStack />
        <PanelRail side="right" base={p.panel} className="w-[min(342px,calc(100vw-24px))]" />
      </div>
      <Status load={p.load} />
      <div className="absolute inset-x-3 bottom-3 flex flex-col items-center gap-2">
        <div className="flex max-w-full items-center gap-2 sm:hidden">
          <ViewCard {...p} />
          <Card size="hud" className={cn(PANEL, "p-1")}>
            <ResetView />
          </Card>
        </div>
        <div className="flex max-w-full flex-wrap items-end justify-center gap-2">
          {p.clips.length ? <ClipPicker {...p} /> : null}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ asset card (top left) */

function AssetCard({ stem, docs, info, onRenamed }: HudProps & { info: ModelInfo | null }) {
  const kind = docs?.intent?.output ? KIND_NAMES[docs.intent.output] : null;
  const insp = docs?.inspect;
  // inspect.json describes one file; when another file is on screen its own counts are shown instead.
  const inspFile = typeof insp?.path === "string" ? insp.path.split("/").pop() : "";
  const useInsp = !!insp && typeof insp.faces === "number" && (!info || !inspFile || inspFile === info.file);
  const verts = useInsp ? insp!.vertices : info?.vertices;
  const faces = useInsp ? insp!.faces : info?.faces;
  const degraded = degradedLine(docs?.degraded);
  const summary = docs?.summary;
  return (
    <Card size="hud" className={cn(PANEL, "w-full shrink-0 gap-1.5 px-4 pt-3 pb-3")}>
      <TitleInput stem={stem} onRenamed={onRenamed} />
      <div className="flex flex-wrap items-center gap-1.5">
        {kind ? <Badge variant="info" size="hud-sm">{kind}</Badge> : null}
        {/* The rigged model is already named by the selected Rigged tab. */}
        {info && info.lod !== RIGGED && info.tag.toLowerCase() !== kind?.toLowerCase() ? <Badge variant="slate" size="hud-sm">{info.tag}</Badge> : null}
      </div>
      {typeof faces === "number" || info ? (
        <p className="m-0 text-[length:calc(13.5px*var(--fs))] text-ink-3 tabular-nums" title={info ? `Loaded in ${info.ms} ms` : undefined}>
          {typeof verts === "number" ? <><b className="font-semibold text-ink-2">{verts.toLocaleString()}</b> verts · </> : null}
          {typeof faces === "number" ? <><b className="font-semibold text-ink-2">{faces.toLocaleString()}</b> faces</> : null}
          {info ? <> · {info.file}</> : null}
        </p>
      ) : null}
      {degraded ? (
        <p role="status" className="m-0 line-clamp-2 text-[length:calc(13px*var(--fs))] text-ink-3" title={degraded}>
          {degraded}
        </p>
      ) : null}
      {summary?.level ? <RigLine summary={summary} /> : null}
      <a
        href={legacyUrl(info?.lod === RIGGED ? { asset: stem, lod: RIGGED } : { asset: stem })}
        className="hit-44 mt-0.5 w-fit rounded-sm text-[length:calc(13.5px*var(--fs))] font-medium text-primary no-underline outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring"
        title="The articulated pivots live in the classic view for now."
      >
        Open in classic view
      </a>
    </Card>
  );
}

// warn is accent2 (signal yellow, dark ink on it); fail is a dark red on its tint: no orange, no #ef4444 text.
const LEVEL: Record<string, { variant: "success" | "signal" | "danger"; word: string }> = {
  pass: { variant: "success", word: "Rig looks right" },
  warn: { variant: "signal", word: "Rig check: warn" },
  fail: { variant: "danger", word: "Rig check: fail" },
};

function RigLine({ summary }: { summary: NonNullable<AssetDocs["summary"]> }) {
  const lv = LEVEL[String(summary.level)] ?? { variant: "signal" as const, word: `Rig check: ${summary.level}` };
  const problems = (summary.problems ?? []).map((q) => q.label).filter(Boolean) as string[];
  const tip = problems.length ? problems.join("\n") : "Every check passed.";
  // A <details>, so the problem list opens by keyboard and touch too, not only on hover.
  return (
    <details className="group/rig min-w-0">
      <summary
        className="flex min-w-0 cursor-pointer list-none items-start gap-1.5 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden"
        title={problems.length ? "Show the problems" : undefined}
      >
        <Badge variant={lv.variant} size="hud-sm" className="shrink-0">
          {lv.word}
        </Badge>
        {summary.line ? <span className="line-clamp-2 pt-0.5 text-[length:calc(13px*var(--fs))] text-ink-3">{summary.line}</span> : null}
      </summary>
      <p className="m-0 mt-1 text-[length:calc(13px*var(--fs))] whitespace-pre-line text-ink-2">{tip}</p>
    </details>
  );
}

/** Rename on Enter or blur (legacy commitRename): PATCH /assets/<stem> {name}; Escape or an empty/unchanged slug reverts. */
function TitleInput({ stem, onRenamed }: { stem: string; onRenamed: (stem: string) => void }) {
  const [value, setValue] = useState(displayStem(stem));
  const busy = useRef(false);
  useEffect(() => setValue(displayStem(stem)), [stem]);
  const commit = async () => {
    if (busy.current) return;
    const next = slugStem(value);
    if (!next || next === stem) {
      setValue(displayStem(stem));
      return;
    }
    busy.current = true;
    try {
      const res = await renameAsset(stem, value);
      onRenamed(res.stem || next);
    } catch (err) {
      setValue(displayStem(stem));
      toast((err as Error)?.message || String(err));
    } finally {
      busy.current = false;
    }
  };
  return (
    <input
      aria-label="Asset name"
      title={`Rename this asset. Saved as out/${stem}/`}
      value={value}
      spellCheck={false}
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => void commit()}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") {
          setValue(displayStem(stem));
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur());
        }
      }}
      className="-mx-1.5 w-[calc(100%+12px)] rounded-[8px] border border-transparent bg-transparent px-1.5 py-0.5 text-[1.05rem] font-semibold text-ink outline-none hover:border-input focus:border-ring focus:bg-white"
    />
  );
}

/* ------------------------------------------------------------------ view toggles + camera (top right) */

function ViewCard(p: HudProps) {
  return (
    <Card size="hud" className={cn(PANEL, "flex-row flex-wrap items-center justify-end gap-0.5 p-1")}>
      <Toggle on={p.spin} onClick={() => p.onSpin(!p.spin)} title="Slowly turn the model on its pad.">
        Spin
      </Toggle>
      <Toggle on={p.clay} disabled={!p.canClay} onClick={() => p.onClay(!p.clay)} title="Show the model in plain white clay instead of its texture.">
        Clay
      </Toggle>
      <Toggle on={p.wire} onClick={() => p.onWire(!p.wire)} title="Show the mesh as wireframe.">
        Wire
      </Toggle>
      <Separator orientation="vertical" className="mx-0.5 h-5 w-px bg-hair" />
      <Toggle on={p.pixel.on} onClick={() => p.onPixel(!p.pixel.on)} title="Render at a low internal resolution and upscale with hard pixel edges, like the game will.">
        Pixel
      </Toggle>
      {p.isPs1 ? (
        <Toggle
          on={p.pixel.on && p.pixel.lines === 240}
          onClick={p.on240}
          title="Preview at 240 lines, the way the game's PSX shader draws it."
        >
          PS1 240p
        </Toggle>
      ) : null}
    </Card>
  );
}

function Toggle({ on, onClick, title, disabled, children }: { on: boolean; onClick: () => void; title: string; disabled?: boolean; children: ReactNode }) {
  return (
    <Button type="button" variant="hud-ghost" aria-pressed={on} disabled={disabled} title={title} onClick={onClick} className={TOGGLE}>
      {children}
    </Button>
  );
}

function CameraStack() {
  const zoomBy = useLook((s) => s.zoomBy);
  const rotateBy = useLook((s) => s.rotateBy);
  const step = () => Math.max(1, useLook.getState().cameraZoom * 0.12);
  return (
    <Card size="hud" className={cn(PANEL, "flex w-[47px] flex-col items-center py-1.5 max-sm:hidden")}>
      <CamBtn label="Zoom in" onClick={() => zoomBy(step())}>
        <Plus size={21} strokeWidth={2} />
      </CamBtn>
      <CamBtn label="Zoom out" onClick={() => zoomBy(-step())}>
        <Minus size={21} strokeWidth={2} />
      </CamBtn>
      <Separator className="my-[3px] h-px w-6 bg-hair" />
      <CamBtn label="Rotate left" onClick={() => rotateBy(-15)}>
        <RotateCcw size={19} strokeWidth={2.1} />
      </CamBtn>
      <CamBtn label="Rotate right" onClick={() => rotateBy(15)}>
        <RotateCw size={19} strokeWidth={2.1} />
      </CamBtn>
      <Separator className="my-[3px] h-px w-6 bg-hair" />
      <ResetView />
    </Card>
  );
}

function ResetView() {
  const resetView = useLook((s) => s.resetView);
  return (
    <CamBtn label="Reset view" onClick={resetView}>
      <House size={19} strokeWidth={2.1} />
    </CamBtn>
  );
}

function CamBtn({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button type="button" variant="hud-cam" size="hud-cam" aria-label={label} onClick={onClick} className="hit-44" />}>
        {children}
      </TooltipTrigger>
      <TooltipContent side="left">{label}</TooltipContent>
    </Tooltip>
  );
}

/* ------------------------------------------------------------------ clip picker (bottom; the LODs are in Detail levels) */

function ClipPicker({ clips, clip, playing, onClip, onPlaying }: HudProps) {
  return (
    <Card size="hud" className={cn(PANEL, "max-w-full flex-row items-center gap-1 p-1")}>
      <Tooltip>
        <TooltipTrigger
          render={<Button type="button" variant="hud-cam" size="hud-cam" aria-label={playing ? "Pause" : "Play"} onClick={() => onPlaying(!playing)} className="hit-44 h-[29px] w-[32px]" />}
        >
          {playing ? <Pause size={17} strokeWidth={2.2} /> : <Play size={17} strokeWidth={2.2} />}
        </TooltipTrigger>
        <TooltipContent>{playing ? "Pause" : "Play"}</TooltipContent>
      </Tooltip>
      <Tabs value={clip ?? ""} onValueChange={(v) => typeof v === "string" && onClip(v)}>
        <TabsList variant="hud" aria-label="Clips" className="max-w-full flex-wrap justify-start bg-transparent p-0">
          {clips.map((name) => (
            <TabsTrigger key={name} variant="hud" value={name} title={`Play the “${name}” animation on the mesh.`} className="hit-44">
              {name.replace(/_/g, " ")}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
    </Card>
  );
}

/* ------------------------------------------------------------------ status (centre) */

function Status({ load }: { load: LoadState }) {
  if (load.state === "ready") return null;
  const error = load.state === "error";
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <Card size="hud" className={cn(PANEL, "max-w-[420px] px-4 py-2.5 text-center")} role={error ? "alert" : "status"}>
        {error ? `Could not load the model: ${load.message}` : "Loading mesh…"}
      </Card>
    </div>
  );
}
