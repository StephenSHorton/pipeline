// The animation principles side by side (pipeline#8): the same walk and punch without and with them, for both the
// skeleton a video model follows and the sprite-stack walker hung on it. No drawing of the walker here, only the
// columns, their framing in cells and the skeleton's 2D drawing (OpenPose colours, as LTX-2.3 is given it).
import { OPENPOSE, OPENPOSE_COLORS, OPENPOSE_LIMBS, seenJoints } from "./openpose";
import type { V3 } from "./rig";
import { makeCycle, placeFrame, spotFrame, type Body, type CycleName, type JointName, type SkeletonCycle } from "./skeleton";

export interface CompareColumn {
  cycle: CycleName;
  principles: boolean;
  label: string;
}

/** Walk then punch, each without and then with: the pairs to judge sit next to each other. */
export const COMPARE_COLUMNS: CompareColumn[] = [
  { cycle: "walk", principles: false, label: "Walk, without" },
  { cycle: "walk", principles: true, label: "Walk, with" },
  { cycle: "attack", principles: false, label: "Punch, without" },
  { cycle: "attack", principles: true, label: "Punch, with" },
];

export const COMPARE_COPY = {
  open: "Compare principles",
  openTitle: "The same walk and punch without and with the animation principles, side by side: the skeleton the video model follows, and this walker on it.",
  title: "Animation principles: without and with",
  blurb:
    "With: a deeper wind-up, held a frame, before a two-frame strike led by the hips; the body carrying on past the hit (both versions freeze three frames on it), the fist back on a low arc and the head settling last; eased poses; arms and head trailing the body; bob, twist and swing pushed about 25%. Each pair has the same feet, frame count and moment of impact.",
  skeleton: "Skeleton the video model follows",
  walker: "Sprite-stack walker on the same frames",
  close: "Close the comparison",
} as const;

/** ?compare=principles opens the Stack stage on the comparison (a walking character only). */
export const compareFromSearch = (search = "") => new URLSearchParams(search).get("compare") === "principles";

/** A column's box in cells, for a figure `height` cells tall: wide enough for the punch's reach, tall enough for
 * the head and a strip of ground. The punch stands left of the middle, as it reaches right. */
export function columnFrame(height: number, cycle: CycleName) {
  return { width: 0.9 * height, height: 1.15 * height, ground: 0.08 * height, x: cycle === "attack" ? -0.2 * height : 0 };
}

export const compareCycles = (body: Body): SkeletonCycle[] =>
  COMPARE_COLUMNS.map((c) => makeCycle(c.cycle, body, { principles: c.principles }));

/** The skeleton of frame k on the spot, facing right, as the camera sees it (the far eye and ear left out). */
export const skeletonAt = (c: SkeletonCycle, k: number): Partial<Record<JointName, V3>> =>
  seenJoints(placeFrame(spotFrame(c, ((k % c.frames) + c.frames) % c.frames), 0, 0, 0));

/** Where a joint lands on a canvas w x h showing one column (its frame from columnFrame). */
export function toCanvas(p: V3, w: number, h: number, f: ReturnType<typeof columnFrame>): [number, number] {
  const k = w / f.width;
  return [w / 2 + (p[0] + f.x) * k, h - f.ground * k - p[2] * k];
}

/** One skeleton frame on black, as the skeleton video draws it: limbs, then joints. */
export function drawSkeleton(ctx: CanvasRenderingContext2D, joints: Partial<Record<JointName, V3>>, f: ReturnType<typeof columnFrame>) {
  const { width: w, height: h } = ctx.canvas;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  const pts = OPENPOSE.map((n) => (joints[n] ? toCanvas(joints[n]!, w, h, f) : null));
  const r = Math.max(2, w / 110);
  ctx.lineCap = "round";
  ctx.lineWidth = r;
  OPENPOSE_LIMBS.forEach(([a, b], i) => {
    const pa = pts[a];
    const pb = pts[b];
    if (!pa || !pb) return;
    ctx.strokeStyle = `rgb(${OPENPOSE_COLORS[i % 18].join(",")})`;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  });
  pts.forEach((p, i) => {
    if (!p) return;
    ctx.fillStyle = `rgb(${OPENPOSE_COLORS[i].join(",")})`;
    ctx.beginPath();
    ctx.arc(p[0], p[1], r * 1.1, 0, Math.PI * 2);
    ctx.fill();
  });
  // the ground, dim, so a drop or a lunge reads against it
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.fillRect(0, Math.round(h - f.ground * (w / f.width)), w, 1);
}
