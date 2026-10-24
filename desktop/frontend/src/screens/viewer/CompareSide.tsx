// Compare's left side (./compare.ts): an earlier version of the stage as a follower diorama (src/viewer/Diorama.tsx),
// loaded read-only from out/<stem>/takes/<stage>/<id>/. One camera, clip, Play/Pause and set of view toggles drive
// both sides; in the Rig stage, with Weights open, the left shows the picked bone's heat from its own weights
// (../asset/rig/compareHeat.ts). Brushes, the Rig check and every save act on the right side only.
import { useEffect, useMemo, useRef, useState } from "react";
import type { AnimationClip } from "three";
import { Card } from "@/components/ui/card";
import { outUrl } from "@/lib/api";
import type { FlowTake } from "@/lib/flow";
import { cn } from "@/lib/utils";
import { CompareHeat, skeletonNames } from "@/screens/asset/rig/compareHeat";
import { useWeights, weightsSession } from "@/screens/asset/rig/weightsApi";
import { fileOf, isPixelFile } from "@/viewer/candidates";
import { Diorama, type Room } from "@/viewer/Diorama";
import { loadFirstGlb } from "@/viewer/loadGlb";
import type { PixelLook } from "@/viewer/pixelLook";
import { prepareModel, type PreparedModel } from "@/viewer/prepareModel";
import { PANEL } from "./classes";
import { missingBone, takeModelFiles, type Insets } from "./compare";
import { usePanels } from "./panels/store";
import type { ViewerStage } from "./panels/panels";

export interface CompareSideProps {
  stem: string;
  stage: ViewerStage;
  take: FlowTake;
  /** "Version 2". */
  label: string;
  /** The right side's file under out/<stem>/ ("" until it has loaded): the left shows the same one when it can. */
  shown: string;
  clayColor: string;
  clip: string | null;
  playing: boolean;
  clay: boolean;
  wire: boolean;
  spin: boolean;
  pixel: PixelLook;
  /** The panes are one above the other (phones, thin room): the label goes by the line across the middle. */
  stacked: boolean;
  /** Where the model goes in this pane, and the pane's room the HUD leaves (compare.ts paneClears). */
  room?: Room;
  clear: Insets;
}

type Left = { state: "loading" } | { state: "ready"; model: PreparedModel; clips: AnimationClip[] } | { state: "error"; message: string };

export function CompareSide(p: CompareSideProps) {
  const { stem, stage, take, shown, clayColor } = p;
  const files = useMemo(() => (shown ? takeModelFiles(stage, take.files, shown) : []), [stage, take, shown]);
  const [left, setLeft] = useState<Left>({ state: "loading" });
  const model = left.state === "ready" ? left.model : null;

  // Takes never change: no cache bust (takeSrc does the same for the thumbnails).
  const key = files.join("|");
  useEffect(() => {
    if (!shown) return;
    if (!files.length) {
      setLeft({ state: "error", message: "This version has no model to show." });
      return;
    }
    const ctl = new AbortController();
    setLeft({ state: "loading" });
    loadFirstGlb(
      files.map((f) => outUrl(stem, `takes/${stage}/${take.id}/${f}`, false)),
      ctl.signal,
    )
      .then(({ url, gltf }) => {
        if (ctl.signal.aborted) return;
        setLeft({ state: "ready", model: prepareModel(gltf, { clayColor, pixel: isPixelFile(fileOf(url), "") }), clips: gltf.animations });
      })
      .catch((err: unknown) => {
        if (!ctl.signal.aborted) setLeft({ state: "error", message: String((err as Error)?.message || err) });
      });
    return () => ctl.abort();
    // The files by value (key): a flow re-read hands over a new take with the same files, and the model stays.
  }, [stem, stage, take.id, key, shown, clayColor]);

  // GPU memory: the left model goes when another version replaces it and when compare closes.
  useEffect(() => () => model?.dispose(), [model]);

  const heatNote = useLeftHeat(stem, stage, model, p.label);
  const clips = left.state === "ready" ? left.clips : [];
  const clipNote = model && p.clip && !clips.some((c) => c.name === p.clip) ? `No “${p.clip.replace(/_/g, " ")}” clip in ${p.label}.` : "";
  const status = left.state === "loading" ? `Loading ${p.label}…` : left.state === "error" ? `Could not load ${p.label}: ${left.message}` : "";

  return (
    <>
      <Diorama follower model={model} clips={clips} clip={p.clip} playing={p.playing} clay={p.clay} wire={p.wire} spin={p.spin} pixel={p.pixel} room={p.room} />
      <SideLabel side="left" text={p.label} notes={[status, heatNote, clipNote]} stacked={p.stacked} clear={p.clear} />
    </>
  );
}

/**
 * With Weights open in the Rig stage, the left model in the same colours as the right: the bone picked in the list
 * (matched by name), or every bone's territory. Returns the note for a bone this version lacks.
 */
function useLeftHeat(stem: string, stage: ViewerStage, model: PreparedModel | null, label: string): string {
  const open = usePanels((s) => s.open === "weights") && stage === "rig";
  const bone = useWeights((s) => s.bone);
  const show = useWeights((s) => s.show);
  const want = useRef({ bone, show });
  want.current = { bone, show };
  // The Weights panel binds its session after it opens (and again on a reload): follow it once a frame.
  const [right, setRight] = useState<string[] | null>(null);
  useEffect(() => {
    if (!open || !model) return;
    let heat: CompareHeat | null = null;
    let names: string[] | null = null;
    let raf = requestAnimationFrame(function tick() {
      const s = weightsSession();
      const now = s && s.stem === stem ? s.names : null;
      if (now !== names) {
        names = now;
        heat?.release();
        heat = now ? CompareHeat.bind(model.root, now) : null;
        setRight(now);
      }
      heat?.show(want.current.show, want.current.bone);
      heat?.keep();
      raf = requestAnimationFrame(tick);
    });
    return () => {
      cancelAnimationFrame(raf);
      heat?.release();
      setRight(null);
    };
  }, [open, model, stem]);
  const leftNames = useMemo(() => (model ? skeletonNames(model.root) : []), [model]);
  if (!open || !model || !right) return "";
  if (!leftNames.length) return `${label} has no skin weights.`;
  return show === "heat" && missingBone(leftNames, right, bone) ? `No “${right[bone]}” bone in ${label}.` : "";
}

/** A side's name, at the divider: the left's top right, the right's top left (stacked: either side of the line across
 * the middle), inside the pane's room clear of the HUD. Notes under it: loading, a missing bone or clip. Never takes
 * the pointer. */
export function SideLabel({ side, text, notes = [], stacked, clear }: { side: "left" | "right"; text: string; notes?: string[]; stacked: boolean; clear: Insets }) {
  const lines = notes.filter(Boolean);
  const at = (k: keyof Insets) => clear[k] + 12;
  const place =
    side === "left"
      ? stacked
        ? { right: at("right"), bottom: at("bottom") }
        : { top: at("top"), right: at("right") }
      : stacked
        ? { top: at("top"), right: at("right") }
        : { top: at("top"), left: at("left") };
  return (
    <div
      style={place}
      className={cn("pointer-events-none absolute z-[1] flex max-w-[calc(100%-24px)] flex-col gap-1", side === "right" && !stacked ? "items-start" : "items-end")}
    >
      <Card size="hud" role="note" className={cn(PANEL, "pointer-events-none px-2.5 py-1 font-semibold text-ink tabular-nums")}>
        {text}
      </Card>
      {lines.map((l) => (
        <Card key={l} size="hud" role="status" className={cn(PANEL, "pointer-events-none max-w-[280px] px-2.5 py-1 text-[length:calc(12.5px*var(--fs))] leading-snug text-ink-2")}>
          {l}
        </Card>
      ))}
    </div>
  );
}
