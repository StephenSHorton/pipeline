// Skeleton cycles made in code (pipeline#14): walk, run and attack as one skeleton format that drives both an LTX
// video (as an OpenPose skeleton video, see scripts/skeleton-cycle.ts) and the sprite-stack walker (as the rig's
// Pose, see rigPose). No drawing here, only numbers.
//
// A cycle is made for a Body (bone lengths, the same names as the rig's RigConfig), so the same numbers fit a sprite
// stack exactly (its rig is its body) and, projected, a drawing of that character. Joints use Apple Vision's names
// (runners/bodypose), which render as OpenPose-18.
//
// Frame: the character's own, as the rig's: +x forward, +y its left, +z up, the ground at z = 0. A cycle starts with
// the body over the origin facing +x; one loop moves it `travel` along +x (0 for the attack). Feet are solved like
// the rig's: a planted foot is a fixed point on the ground, knees and elbows come from two-bone IK, so no bone
// stretches and a planted foot never slides.
//
// The animation principles (pipeline#8) are built into the cycles, and on by default; `principles: false` gives the
// plain cycles pipeline#14 made, kept to compare against. With them:
//  - slow in / slow out: every key pose is eased into and out of, never linear;
//  - timing: the punch holds its guard and its wind-up, strikes in two frames, and stops dead for three frames on
//    impact (hit-stop; the plain punch already held its hit that long);
//  - anticipation: the punch coils back, down and away before it strikes;
//  - follow-through and overlap: hips lead, then shoulders, then the fist; the arms swing behind the shoulders and
//    the forearms behind the upper arms; the head trails the body and settles after it;
//  - arcs: hands swing on arcs, the fist comes back on a curve, not the line it went out on;
//  - exaggeration: the bob, twist, lean and arm swing are pushed about 25% past the plain cycles.
// The feet, the stride and the frame counts are the same with and without, so the two can be laid side by side and
// still loop, keep bones their length and never slide a planted foot.
import { solveTwoBone, type ArmPose, type LegPose, type Pose, type RigConfig, type V3 } from "./rig";

export const SKELETON_FORMAT = "pipeline-skeleton-cycle";

/** Apple Vision joint names, as runners/bodypose writes them; skeleton videos draw them as OpenPose-18. */
export const JOINTS = [
  "head_joint", "neck_1_joint", "root",
  "right_shoulder_1_joint", "right_forearm_joint", "right_hand_joint",
  "left_shoulder_1_joint", "left_forearm_joint", "left_hand_joint",
  "right_upLeg_joint", "right_leg_joint", "right_foot_joint",
  "left_upLeg_joint", "left_leg_joint", "left_foot_joint",
  "right_eye_joint", "left_eye_joint", "right_ear_joint", "left_ear_joint",
] as const;
export type JointName = (typeof JOINTS)[number];

export type CycleName = "walk" | "run" | "attack";
export const CYCLES: CycleName[] = ["walk", "run", "attack"];

/** Bone lengths and where the hips and shoulders sit, in any one unit (cells for a sprite stack). */
export interface Body {
  thigh: number;
  shin: number;
  /** The ankle's height over the sole. */
  ankle: number;
  hipHalf: number;
  hipFwd: number;
  kneeHint: number;
  standHip: number;
  /** Pelvis up to the shoulders' height (Vision's neck_1_joint). */
  spine: number;
  /** Pelvis up to where the head turns. */
  neck: number;
  shoulderHalf: number;
  shoulderFwd: number;
  upperArm: number;
  foreArm: number;
  /** Where the head turns up to the top of the head. */
  head: number;
}

/** A body from a sprite-stack character's rig and its height in cells. */
export const bodyFromRig = (c: RigConfig, height: number): Body => ({
  thigh: c.thigh, shin: c.shin, ankle: c.ankle, hipHalf: c.hipHalf, hipFwd: c.hipFwd, kneeHint: c.kneeHint,
  standHip: c.standHip, spine: c.spine, neck: c.neck, shoulderHalf: c.shoulderHalf, shoulderFwd: c.shoulderFwd,
  upperArm: c.upperArm, foreArm: c.foreArm, head: Math.max(1, height - c.standHip - c.neck),
});

export interface SkeletonFrame {
  joints: Record<JointName, V3>;
  /** Right, left: the foot is planted. */
  contact: [boolean, boolean];
  /** Each foot's turn from the facing (radians; right, left). */
  footYaw: [number, number];
  /** The hips' and the shoulders' turn from the facing (radians). */
  hips: number;
  shoulders: number;
  /** The upper body's tilt forward (radians). */
  lean: number;
  /** The head's own turn from the shoulders and its tilt forward past the lean (radians), when it trails the body
   * (the principles); absent, it turns with the shoulders and tilts with the lean. */
  head?: { turn: number; nod: number };
}

export interface SkeletonCycle {
  format: typeof SKELETON_FORMAT;
  version: 1;
  name: CycleName;
  /** Made with the animation principles (pipeline#8), or plain (pipeline#14's). */
  principles: boolean;
  fps: number;
  /** Frames in one loop: frame `frames` is frame 0 moved `travel` along +x. */
  frames: number;
  travel: number;
  body: Body;
  poses: SkeletonFrame[];
}

const TAU = Math.PI * 2;
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
/** A point in a yawed frame: forward f, left l, up u. */
const yawed = (a: number, f: number, l: number, u: number): V3 => [Math.cos(a) * f - Math.sin(a) * l, Math.sin(a) * f + Math.cos(a) * l, u];
/** Smooth periodic bump: 0 outside [a, d], rising over [a, b], 1 over [b, c], falling over [c, d]. */
function bump(t: number, a: number, b: number, c: number, d: number) {
  if (t <= a || t >= d) return 0;
  if (t < b) return ease((t - a) / (b - a));
  if (t <= c) return 1;
  return 1 - ease((t - c) / (d - c));
}

interface Arm {
  hand: V3;
  /** The way the elbow bends (IK hint), or the elbow itself when the arm is posed bone by bone. */
  hint?: V3;
  elbow?: V3;
}

interface Build {
  pelvis: V3;
  hips: number;
  shoulders: number;
  lean: number;
  ankles: [V3, V3];
  contact: [boolean, boolean];
  footYaw?: [number, number];
  arms: [Arm, Arm];
  head?: { turn: number; nod: number };
}

const SIDES = ["right", "left"] as const;
/** Side 0 (right) is -1 along the left axis, side 1 (left) +1, as in the rig. */
const sideSign = (i: number) => (i ? 1 : -1);

/** The upper body's axis: up, tilted forward by lean along +x. */
const upAxis = (lean: number): V3 => [Math.sin(lean), 0, Math.cos(lean)];

/** Shoulder positions for a pelvis, a lean and the shoulders' turn. */
function shoulderAt(b: Body, pelvis: V3, lean: number, shoulders: number, i: number): V3 {
  const top = add(pelvis, scale(upAxis(lean), b.spine));
  return add(top, yawed(shoulders, b.shoulderFwd * Math.cos(lean), sideSign(i) * b.shoulderHalf, -b.shoulderFwd * Math.sin(lean)));
}

function hipAt(b: Body, pelvis: V3, hips: number, i: number): V3 {
  return add(pelvis, yawed(hips, b.hipFwd, sideSign(i) * b.hipHalf, 0));
}

/** Every joint from a pelvis, turns, ankles and hands: knees and elbows by two-bone IK. */
function build(b: Body, s: Build): SkeletonFrame {
  const j = {} as Record<JointName, V3>;
  const up = upAxis(s.lean);
  j.root = s.pelvis;
  j.neck_1_joint = add(s.pelvis, scale(up, b.spine));
  const yaw = s.footYaw ?? [0, 0];
  for (let i = 0; i < 2; i++) {
    const n = SIDES[i];
    const hip = hipAt(b, s.pelvis, s.hips, i);
    // a foot never out of the leg's reach (a part stack cannot stretch): pulled in along the line to the hip
    const ankle: V3 = [...s.ankles[i]];
    const d = dist(ankle, hip);
    const most = b.thigh + b.shin - 0.02;
    if (d > most) for (let k = 0; k < 3; k++) ankle[k] = hip[k] + ((ankle[k] - hip[k]) * most) / d;
    const kh: V3 = [Math.cos(yaw[i]) * b.kneeHint, Math.sin(yaw[i]) * b.kneeHint, 0];
    j[`${n}_upLeg_joint`] = hip;
    j[`${n}_leg_joint`] = solveTwoBone(hip, ankle, b.thigh, b.shin, kh);
    j[`${n}_foot_joint`] = ankle;
    const sh = shoulderAt(b, s.pelvis, s.lean, s.shoulders, i);
    const arm = s.arms[i];
    j[`${n}_shoulder_1_joint`] = sh;
    j[`${n}_forearm_joint`] = arm.elbow ?? solveTwoBone(sh, arm.hand, b.upperArm, b.foreArm, arm.hint ?? [-1, 0, 0]);
    j[`${n}_hand_joint`] = arm.hand;
  }
  // the head, in the shoulders' turn, tilted with the lean (and, trailing, by its own turn and nod)
  const pivot = add(s.pelvis, scale(up, b.neck));
  const h = b.head;
  const hy = s.head ? s.shoulders + s.head.turn : s.shoulders;
  const hl = s.head ? s.lean + s.head.nod : s.lean;
  const headPt = (f: number, l: number, u: number): V3 => {
    const p = yawed(hy, f * Math.cos(hl) + u * Math.sin(hl), l, u * Math.cos(hl) - f * Math.sin(hl));
    return add(pivot, p);
  };
  j.head_joint = headPt(0.34 * h, 0, 0.42 * h);
  for (let i = 0; i < 2; i++) {
    j[`${SIDES[i]}_eye_joint`] = headPt(0.3 * h, sideSign(i) * 0.16 * h, 0.5 * h);
    j[`${SIDES[i]}_ear_joint`] = headPt(0, sideSign(i) * 0.42 * h, 0.45 * h);
  }
  const out: SkeletonFrame = { joints: j, contact: s.contact, footYaw: yaw, hips: s.hips, shoulders: s.shoulders, lean: s.lean };
  if (s.head) out.head = { ...s.head };
  return out;
}

/** Standing still facing +x, arms hanging: where every cycle's turn starts. */
export function standFrame(b: Body): SkeletonFrame {
  const pelvis: V3 = [0, 0, b.standHip];
  const arms = [0, 1].map((i): Arm => {
    const sh = shoulderAt(b, pelvis, 0, 0, i);
    const L = (b.upperArm + b.foreArm) * 0.97;
    return { hand: add(sh, [0.04 * L, sideSign(i) * 0.06 * L, -Math.sqrt(1 - 0.0052) * L]), hint: [-1, 0, 0] };
  }) as [Arm, Arm];
  return build(b, {
    pelvis, hips: 0, shoulders: 0, lean: 0, contact: [true, true],
    ankles: [0, 1].map((i): V3 => [b.hipFwd, sideSign(i) * b.hipHalf, b.ankle]) as [V3, V3], arms,
  });
}

/** A gait (walk or run): the rig's step clock written out over one loop. */
interface Gait {
  stride: number;
  duty: number;
  lift: number;
  bob: number;
  twist: number;
  lean: number;
  armSwing: number;
  /** Forearm bend forward from the upper arm (radians); 0 lets the elbow hang back, by IK. */
  elbowBend: number;
  /** >1 lifts the swinging heel early, behind the body (a run); 1 is an even arc. */
  heelKick: number;
  /** The principles' overlap and follow-through; absent in the plain gaits. */
  overlap?: Overlap;
}

/** Follow-through and overlap in a gait, as parts of a loop (0..1) each part runs behind the one that drives it. */
interface Overlap {
  /** Where in the loop the hips are lowest: a walk's down pose, just after the foot lands. */
  bobLow: number;
  /** The upper arm behind the shoulders' swing, and the forearm behind the upper arm. */
  armLag: number;
  foreLag: number;
  /** The head nods forward after the body drops (radians), this far behind the low. */
  headNod: number;
  headLag: number;
  /** How much of the shoulders' turn the head takes back, to keep looking ahead. */
  headHold: number;
  /** The upper body pitches with the bob (radians), this far behind the low. */
  leanBob: number;
  leanLag: number;
}

const legLen = (b: Body) => b.thigh + b.shin;

function gaitOf(name: "walk" | "run", b: Body, principles = true): Gait {
  const L = legLen(b);
  // a walk's stride and lift as the sprite stack's own rig has them (28.6 and 3.06 cells for 18 of leg)
  if (!principles)
    return name === "walk"
      ? { stride: 1.55 * L, duty: 0.6, lift: 0.17 * L, bob: 0.03 * L, twist: 0.14, lean: 0.03, armSwing: 0.42, elbowBend: 0, heelKick: 1 }
      : { stride: 2.5 * L, duty: 0.36, lift: 0.42 * L, bob: 0.06 * L, twist: 0.22, lean: 0.2, armSwing: 0.75, elbowBend: 1.45, heelKick: 1.8 };
  // The same feet (stride, duty, lift, heel kick), the body above them pushed about 25% further (exaggeration), and
  // the arms and head running behind it (overlap). A walk drops into its down pose just after each foot lands and
  // rises over the passing pose; a run is lowest mid-stance, as a spring.
  return name === "walk"
    ? {
        stride: 1.55 * L, duty: 0.6, lift: 0.17 * L, bob: 0.045 * L, twist: 0.18, lean: 0.05, armSwing: 0.55, elbowBend: 0.3, heelKick: 1,
        overlap: { bobLow: 0.07, armLag: 0.04, foreLag: 0.05, headNod: 0.05, headLag: 0.06, headHold: 0.6, leanBob: 0.015, leanLag: 0.04 },
      }
    : {
        stride: 2.5 * L, duty: 0.36, lift: 0.42 * L, bob: 0.075 * L, twist: 0.27, lean: 0.25, armSwing: 0.9, elbowBend: 1.45, heelKick: 1.8,
        overlap: { bobLow: 0.18, armLag: 0.04, foreLag: 0.05, headNod: 0.07, headLag: 0.06, headHold: 0.6, leanBob: 0.03, leanLag: 0.05 },
      };
}

/** Where in the loop a gait's hips are lowest. */
const bobLowOf = (g: Gait) => g.overlap?.bobLow ?? g.duty / 2;

/** The pelvis height that keeps every planted foot in reach through the loop (as the rig lowers its hips). */
function gaitHip(b: Body, g: Gait, samples = 240) {
  const reach = legLen(b) - 0.3;
  const stance = g.duty * g.stride;
  let base = b.standHip;
  for (let k = 0; k < samples; k++) {
    const p = k / samples;
    const pi = p % 1;
    if (pi >= g.duty) continue;
    // the right foot (side 0) is planted; its ground distance from the hip; the hips' twist moves the hip too
    const far = Math.abs(stance / 2 - pi * g.stride) + b.hipHalf * Math.sin(Math.abs(g.twist));
    const most = b.ankle + Math.sqrt(Math.max(1, reach * reach - far * far));
    base = Math.min(base, most + g.bob * Math.cos(TAU * 2 * (p - bobLowOf(g))));
  }
  return base;
}

function gaitFrame(b: Body, g: Gait, hipBase: number, p: number): SkeletonFrame {
  if (g.overlap) return gaitFrameOverlap(b, g, g.overlap, hipBase, p);
  const stance = g.duty * g.stride;
  const x = p * g.stride;
  // lowest mid-stance, highest mid-swing (for a run, in the flight)
  const pelvis: V3 = [x, 0, hipBase - g.bob * Math.cos(TAU * 2 * (p - g.duty / 2))];
  const tw = g.twist * Math.cos(TAU * p);
  const hips = tw;
  const shoulders = -1.2 * tw;
  const ankles: V3[] = [];
  const contact: boolean[] = [];
  for (let i = 0; i < 2; i++) {
    const pi = (((p + i * 0.5) % 1) + 1) % 1;
    const tdX = (p - pi) * g.stride + b.hipFwd + stance / 2; // where this foot last touched down
    const y = sideSign(i) * b.hipHalf;
    if (pi < g.duty) {
      ankles.push([tdX, y, b.ankle]);
      contact.push(true);
    } else {
      const u = (pi - g.duty) / (1 - g.duty);
      const e = ease(u);
      const lift = g.lift * Math.sin(Math.PI * Math.pow(u, 1 / g.heelKick));
      ankles.push([tdX + g.stride * e, y, b.ankle + lift]);
      contact.push(false);
    }
  }
  const arms = [0, 1].map((i): Arm => {
    const s = sideSign(i);
    const sh = shoulderAt(b, pelvis, g.lean, shoulders, i);
    const a = s * g.armSwing * Math.cos(TAU * p) - g.lean * 0.5;
    if (g.elbowBend > 0) {
      // bone by bone: the upper arm swings, the forearm bent forward from it
      const elbow = add(sh, [Math.sin(a) * b.upperArm, 0, -Math.cos(a) * b.upperArm]);
      const fa = a + g.elbowBend;
      return { elbow, hand: add(elbow, [Math.sin(fa) * b.foreArm, 0, -Math.cos(fa) * b.foreArm]) };
    }
    const L = (b.upperArm + b.foreArm) * 0.93;
    return { hand: add(sh, [Math.sin(a) * L, 0, -Math.cos(a) * L]), hint: [-1, 0, 0] };
  }) as [Arm, Arm];
  return build(b, {
    pelvis, hips, shoulders, lean: g.lean, ankles: ankles as [V3, V3], contact: contact as [boolean, boolean], arms,
  });
}

/** The feet of a gait at loop position p, as gaitFrame places them: planted ones fixed on the ground. */
function gaitFeet(b: Body, g: Gait, p: number) {
  const stance = g.duty * g.stride;
  const ankles: V3[] = [];
  const contact: boolean[] = [];
  for (let i = 0; i < 2; i++) {
    const pi = (((p + i * 0.5) % 1) + 1) % 1;
    const tdX = (p - pi) * g.stride + b.hipFwd + stance / 2;
    const y = sideSign(i) * b.hipHalf;
    if (pi < g.duty) {
      ankles.push([tdX, y, b.ankle]);
      contact.push(true);
    } else {
      const u = (pi - g.duty) / (1 - g.duty);
      const lift = g.lift * Math.sin(Math.PI * Math.pow(u, 1 / g.heelKick));
      ankles.push([tdX + g.stride * ease(u), y, b.ankle + lift]);
      contact.push(false);
    }
  }
  return { ankles: ankles as [V3, V3], contact: contact as [boolean, boolean] };
}

/** A gait with the principles: the same feet as the plain one; the hips' bob eased round the down pose, the
 * upper body pitching after it, the arms swinging behind the shoulders bone by bone (upper arm, then forearm: a
 * drag, never a stiff pendulum) and the head nodding after the body and holding its look ahead. */
function gaitFrameOverlap(b: Body, g: Gait, o: Overlap, hipBase: number, p: number): SkeletonFrame {
  const low = o.bobLow;
  const pelvis: V3 = [p * g.stride, 0, hipBase - g.bob * Math.cos(TAU * 2 * (p - low))];
  const lean = g.lean + o.leanBob * Math.cos(TAU * 2 * (p - low - o.leanLag));
  const tw = g.twist * Math.cos(TAU * p);
  const hips = tw;
  const shoulders = -1.2 * tw;
  const { ankles, contact } = gaitFeet(b, g, p);
  const arms = [0, 1].map((i): Arm => {
    const s = sideSign(i);
    const sh = shoulderAt(b, pelvis, lean, shoulders, i);
    const upper = s * g.armSwing * Math.cos(TAU * (p - o.armLag)) - lean * 0.5;
    const fore = s * g.armSwing * Math.cos(TAU * (p - o.armLag - o.foreLag)) - lean * 0.5 + g.elbowBend;
    const elbow = add(sh, [Math.sin(upper) * b.upperArm, 0, -Math.cos(upper) * b.upperArm]);
    return { elbow, hand: add(elbow, [Math.sin(fore) * b.foreArm, 0, -Math.cos(fore) * b.foreArm]) };
  }) as [Arm, Arm];
  const head = { turn: -o.headHold * shoulders, nod: o.headNod * Math.cos(TAU * 2 * (p - low - o.headLag)) };
  return build(b, { pelvis, hips, shoulders, lean, ankles, contact, arms, head });
}

/** A right straight punch from a guard, and back to the guard: feet planted all through, left foot forward. */
function attackFrame(b: Body, p: number): SkeletonFrame {
  const L = legLen(b);
  const arm = b.upperArm + b.foreArm;
  const t = ((p % 1) + 1) % 1;
  const wind = bump(t, 0.18, 0.32, 0.36, 0.44); // drawing back
  const hit = bump(t, 0.38, 0.47, 0.6, 0.86); // out and back
  const lean = 0.06 + 0.12 * hit - 0.04 * wind;
  const hips = -0.42 + 0.5 * hit - 0.08 * wind;
  const shoulders = -0.55 + 0.95 * hit - 0.15 * wind;
  const crouch = 0.1 * L;
  const pelvis: V3 = [0.1 * L * hit - 0.04 * L * wind, 0, b.standHip - crouch - 0.02 * L * hit];
  const ankles: [V3, V3] = [
    [-0.36 * L + b.hipFwd, -1.25 * b.hipHalf, b.ankle], // right foot back
    [0.3 * L + b.hipFwd, 1.1 * b.hipHalf, b.ankle], // left foot forward
  ];
  const chin = b.neck + 0.2 * b.head;
  // guard: both fists before the chin; the right one drives straight out at shoulder height
  const left: Arm = {
    hand: add(pelvis, [0.34 * arm - 0.05 * arm * hit, 0.12 * b.shoulderHalf, chin - 0.05 * b.head * hit]),
    hint: [-0.2, 0.6, -1],
  };
  const shR = shoulderAt(b, pelvis, lean, shoulders, 0);
  const guardR: V3 = add(pelvis, [0.24 * arm - 0.06 * arm * wind, -0.35 * b.shoulderHalf, chin - 0.1 * b.head]);
  const strikeR: V3 = add(shR, [0.97 * arm * Math.cos(0.12), 0.45 * b.shoulderHalf, -0.03 * arm]);
  const right: Arm = {
    hand: [lerp(guardR[0], strikeR[0], hit), lerp(guardR[1], strikeR[1], hit), lerp(guardR[2], strikeR[2], hit)],
    hint: [-0.3, -0.6 * (1 - hit), -1],
  };
  return build(b, {
    pelvis, hips, shoulders, lean, ankles, contact: [true, true], footYaw: [-0.7, -0.25], arms: [right, left],
  });
}

/** Ease in only: a strike speeds up into the hit. */
const easeIn = (t: number) => t * t;

/** A key: [frame of the 24-frame punch, value, the ease into it (slow in / slow out unless named)]. */
type Key = [number, number, ((t: number) => number)?];
/** The value at loop position t (0..1) on keys from frame 0 to frame 24, eased between each pair. Equal keys either
 * side of a span hold it still. */
function track(t: number, keys: Key[]) {
  const f = t * 24;
  for (let i = 1; i < keys.length; i++) {
    const [f1, v1, e] = keys[i];
    if (f <= f1) {
      const [f0, v0] = keys[i - 1];
      return f1 === f0 ? v1 : lerp(v0, v1, (e ?? ease)((f - f0) / (f1 - f0)));
    }
  }
  return keys[keys.length - 1][1];
}

// The punch with the principles, as key poses on its 24 frames: guard held (0-3), the wind-up (3-9: anticipation,
// coiled back and down, the fist drawn to the cheek) held a frame, the strike in two frames (10-12, speeding in;
// hips first, then shoulders, then the fist), impact held dead for three frames (12-14, hit-stop), the shoulders
// and lean carrying on past it (16, follow-through), and the slow way back to the guard (16-22), the fist on a
// low arc and the head settling last. The feet are the plain punch's.
const P_HIPS: Key[] = [[0, -0.42], [3, -0.42], [9, -0.57], [10, -0.57], [11, 0.04, easeIn], [12, 0.08], [14, 0.08], [15, 0.12], [21, -0.42], [24, -0.42]];
const P_SHOULDERS: Key[] = [[0, -0.55], [3, -0.55], [9, -0.87], [10, -0.87], [11, -0.35, easeIn], [12, 0.4], [14, 0.4], [16, 0.52], [22, -0.55], [24, -0.55]];
const P_LEAN: Key[] = [[0, 0.06], [3, 0.06], [9, -0.06], [10, -0.06], [12, 0.18, easeIn], [14, 0.18], [16, 0.24], [22, 0.06], [24, 0.06]];
/** The pelvis forward and its drop, in leg lengths. */
const P_FWD: Key[] = [[0, 0], [3, 0], [9, -0.09], [10, -0.09], [11, 0.06, easeIn], [12, 0.1], [14, 0.1], [16, 0.13], [22, 0], [24, 0]];
const P_CROUCH: Key[] = [[0, 0.1], [3, 0.1], [9, 0.14], [10, 0.14], [12, 0.125], [14, 0.125], [16, 0.135], [22, 0.1], [24, 0.1]];
/** The right fist: drawn back (0..1), out to the hit (0..1), and the dip of its way back (0..1). */
const P_DRAW: Key[] = [[0, 0], [3, 0], [9, 1], [10, 1], [12, 0], [24, 0]];
const P_OUT: Key[] = [[0, 0], [10, 0], [12, 1, easeIn], [16, 1], [22, 0], [24, 0]];
const P_DIP: Key[] = [[0, 0], [16, 0], [19, 1], [22, 0], [24, 0]];
/** The left fist pulled back to the chin as the right goes out (0..1), back last. */
const P_PULL: Key[] = [[0, 0], [10, 0], [12, 1], [16, 1], [23, 0], [24, 0]];
/** The head: chin down in the wind-up, left behind by the lunge, thrown on after it, settling. */
const P_NOD: Key[] = [[0, 0], [3, 0], [9, 0.07], [10, 0.07], [12, -0.06], [14, -0.06], [17, 0.1], [20, -0.02], [23, 0], [24, 0]];

function attackFramePrincipled(b: Body, p: number): SkeletonFrame {
  const L = legLen(b);
  const arm = b.upperArm + b.foreArm;
  const t = ((p % 1) + 1) % 1;
  const lean = track(t, P_LEAN);
  const hips = track(t, P_HIPS);
  const shoulders = track(t, P_SHOULDERS);
  const pelvis: V3 = [track(t, P_FWD) * L, 0, b.standHip - track(t, P_CROUCH) * L];
  const ankles: [V3, V3] = [
    [-0.36 * L + b.hipFwd, -1.25 * b.hipHalf, b.ankle],
    [0.3 * L + b.hipFwd, 1.1 * b.hipHalf, b.ankle],
  ];
  // the lunge and the wind-up move the hips further over the feet than the plain punch: on a body standing with
  // near-straight legs, sink the hips as far as it takes to keep both planted feet in reach (they must not slide)
  const most = L - 0.03;
  for (let i = 0; i < 2; i++) {
    const hip = hipAt(b, pelvis, hips, i);
    const far = Math.hypot(hip[0] - ankles[i][0], hip[1] - ankles[i][1]);
    pelvis[2] = Math.min(pelvis[2], ankles[i][2] + Math.sqrt(Math.max(1, most * most - far * far)));
  }
  const chin = b.neck + 0.2 * b.head;
  const draw = track(t, P_DRAW);
  const out = track(t, P_OUT);
  const pull = track(t, P_PULL);
  const left: Arm = {
    hand: add(pelvis, [0.34 * arm - 0.1 * arm * pull, 0.12 * b.shoulderHalf, chin + 0.04 * b.head * pull]),
    hint: [-0.2, 0.6, -1],
  };
  const shR = shoulderAt(b, pelvis, lean, shoulders, 0);
  const guardR: V3 = add(pelvis, [0.24 * arm, -0.35 * b.shoulderHalf, chin - 0.1 * b.head]);
  // drawn back to the cheek, on a curve out to the side (an arc, not a line)
  const back: V3 = [
    guardR[0] - 0.16 * arm * draw,
    guardR[1] - 0.25 * b.shoulderHalf * Math.sin(Math.PI * Math.min(1, draw)) * (1 - out),
    guardR[2] + 0.08 * b.head * draw,
  ];
  const strikeR: V3 = add(shR, [0.97 * arm * Math.cos(0.12), 0.45 * b.shoulderHalf, -0.03 * arm]);
  const dip = track(t, P_DIP);
  const right: Arm = {
    hand: [lerp(back[0], strikeR[0], out), lerp(back[1], strikeR[1], out), lerp(back[2], strikeR[2], out) - 0.12 * arm * dip],
    hint: [-0.3 - 0.3 * draw, -0.6 * (1 - out), -1],
  };
  // the head keeps its eyes on the target: it takes back part of the shoulders' turn from the guard
  const head = { turn: -0.6 * (shoulders - P_SHOULDERS[0][1]), nod: track(t, P_NOD) };
  // a fist never past the arm's reach (a body with short arms and wide shoulders): pulled in towards its shoulder
  const shL = shoulderAt(b, pelvis, lean, shoulders, 1);
  for (const [sh, a] of [[shR, right], [shL, left]] as [V3, Arm][]) {
    const d = dist(a.hand, sh);
    if (d > arm - 0.02) a.hand = add(sh, scale([a.hand[0] - sh[0], a.hand[1] - sh[1], a.hand[2] - sh[2]], (arm - 0.02) / d));
  }
  return build(b, {
    pelvis, hips, shoulders, lean, ankles, contact: [true, true], footYaw: [-0.7, -0.25], arms: [right, left], head,
  });
}

export interface CycleOptions {
  fps?: number;
  /** Frames in one loop. */
  frames?: number;
  /** The animation principles (pipeline#8); on unless false, which gives pipeline#14's plain cycle. */
  principles?: boolean;
}

/** Frames a loop has by default at 24 fps (LTX's rate): a walk's two steps in 28, a run's in 18, a punch in 24. */
export const CYCLE_FRAMES: Record<CycleName, number> = { walk: 28, run: 18, attack: 24 };

/** One cycle's frame at loop position p (0..1, any real: p = 1 is frame 0 moved one travel on). */
export function cycleFrameAt(name: CycleName, b: Body, p: number, principles = true): SkeletonFrame {
  if (name === "attack") return principles ? attackFramePrincipled(b, p) : attackFrame(b, p);
  const g = gaitOf(name, b, principles);
  const fr = gaitFrame(b, g, gaitHip(b, g), p - Math.floor(p));
  const shift = Math.floor(p) * g.stride;
  return shift ? moveFrame(fr, shift) : fr;
}

export function cycleTravel(name: CycleName, b: Body) {
  return name === "attack" ? 0 : gaitOf(name, b).stride;
}

export function makeCycle(name: CycleName, b: Body, o: CycleOptions = {}): SkeletonCycle {
  const frames = o.frames ?? CYCLE_FRAMES[name];
  const principles = o.principles ?? true;
  const poses: SkeletonFrame[] = [];
  for (let k = 0; k < frames; k++) poses.push(cycleFrameAt(name, b, k / frames, principles));
  return {
    format: SKELETON_FORMAT, version: 1, name, principles, fps: o.fps ?? 24, frames, travel: cycleTravel(name, b), body: b, poses,
  };
}

/** Frame k of a cycle, any k: whole loops move it on by `travel`. */
export function cycleFrame(c: SkeletonCycle, k: number): SkeletonFrame {
  const loops = Math.floor(k / c.frames);
  const f = c.poses[k - loops * c.frames];
  return loops ? moveFrame(f, loops * c.travel) : f;
}

/** Frame k of a cycle on the spot: its travel so far taken back (not the pelvis, which sways: a punch leans in on
 * planted feet), so the ground moves under it as on a treadmill. What a video model's skeleton and the side-by-side
 * compare play. */
export function spotFrame(c: SkeletonCycle, k: number): SkeletonFrame {
  return moveFrame(cycleFrame(c, k), -(k / c.frames) * c.travel);
}

function moveFrame(f: SkeletonFrame, dx: number): SkeletonFrame {
  const joints = {} as Record<JointName, V3>;
  for (const n of JOINTS) joints[n] = [f.joints[n][0] + dx, f.joints[n][1], f.joints[n][2]];
  return { ...f, joints };
}

/** A frame placed in the world: turned to face heading h about its own origin, then moved to (x, y). Facing left
 * on a side view is h = π. */
export function placeFrame(f: SkeletonFrame, x: number, y: number, h: number): Record<JointName, V3> {
  const c = Math.cos(h);
  const s = Math.sin(h);
  const out = {} as Record<JointName, V3>;
  for (const n of JOINTS) {
    const p = f.joints[n];
    out[n] = [x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]];
  }
  return out;
}

/** Between two frames in the body's own frame (t from a to b): the pelvis, ankles and hands move straight, knees
 * and elbows are solved again, so bones keep their lengths. Used for the turn into a cycle. */
export function blendFrames(bd: Body, a: SkeletonFrame, b: SkeletonFrame, t: number): SkeletonFrame {
  const L = (p: V3, q: V3): V3 => [lerp(p[0], q[0], t), lerp(p[1], q[1], t), lerp(p[2], q[2], t)];
  const J = (n: JointName) => L(a.joints[n], b.joints[n]);
  const pelvis = J("root");
  const hips = lerp(a.hips, b.hips, t);
  const shoulders = lerp(a.shoulders, b.shoulders, t);
  const lean = lerp(a.lean, b.lean, t);
  const arms = [0, 1].map((i): Arm => {
    // keep each hand at its own distance from its shoulder, so the arm never stretches
    const n = SIDES[i];
    const sh = shoulderAt(bd, pelvis, lean, shoulders, i);
    const ra = dist(a.joints[`${n}_hand_joint`], a.joints[`${n}_shoulder_1_joint`]);
    const rb = dist(b.joints[`${n}_hand_joint`], b.joints[`${n}_shoulder_1_joint`]);
    const off = L(
      [0, 1, 2].map((k) => a.joints[`${n}_hand_joint`][k] - a.joints[`${n}_shoulder_1_joint`][k]) as V3,
      [0, 1, 2].map((k) => b.joints[`${n}_hand_joint`][k] - b.joints[`${n}_shoulder_1_joint`][k]) as V3,
    );
    const ol = Math.hypot(...off) || 1;
    const want = lerp(ra, rb, t);
    const elbowA = a.joints[`${n}_forearm_joint`];
    const elbowB = b.joints[`${n}_forearm_joint`];
    const hint = L(
      [elbowA[0] - a.joints[`${n}_hand_joint`][0], elbowA[1] - a.joints[`${n}_hand_joint`][1], elbowA[2] - a.joints[`${n}_hand_joint`][2]],
      [elbowB[0] - b.joints[`${n}_hand_joint`][0], elbowB[1] - b.joints[`${n}_hand_joint`][1], elbowB[2] - b.joints[`${n}_hand_joint`][2]],
    );
    return { hand: add(sh, scale(off, want / ol)), hint };
  }) as [Arm, Arm];
  const ha = a.head ?? { turn: 0, nod: 0 };
  const hb = b.head ?? { turn: 0, nod: 0 };
  const head = a.head || b.head ? { turn: lerp(ha.turn, hb.turn, t), nod: lerp(ha.nod, hb.nod, t) } : undefined;
  return build(bd, {
    pelvis, hips, shoulders, lean, arms, head,
    ankles: [J("right_foot_joint"), J("left_foot_joint")],
    contact: [a.contact[0] && b.contact[0], a.contact[1] && b.contact[1]],
    footYaw: [lerp(a.footYaw[0], b.footYaw[0], t), lerp(a.footYaw[1], b.footYaw[1], t)],
  });
}

export interface Shot {
  /** Each frame: the skeleton in the world, its facing, and the frame in the body's own frame. */
  frames: { joints: Record<JointName, V3>; heading: number; frame: SkeletonFrame; x: number }[];
  /** The first frame of the loop (after the turn). */
  loopStart: number;
}

/** A clip for a picture of the character standing facing the camera: it turns to face left or right over `turn`
 * frames while easing into the cycle, then plays the cycle for the rest. `inPlace` keeps the body over the origin
 * (a treadmill, as a video model wants it); otherwise it travels. The camera looks along +y, so facing the camera
 * is heading -π/2. */
export function shot(c: SkeletonCycle, facing: "left" | "right", total: number, turn = 16, inPlace = true): Shot {
  const end = facing === "right" ? 0 : -Math.PI;
  const start = -Math.PI / 2;
  const stand = standFrame(c.body);
  const frames: Shot["frames"] = [];
  for (let k = 0; k < total; k++) {
    let f: SkeletonFrame;
    let h: number;
    let x = 0;
    if (k < turn) {
      const t = ease(k / turn);
      const first = cycleFrame(c, 0);
      f = blendFrames(c.body, stand, first, t);
      h = lerp(start, end, t);
    } else {
      f = cycleFrame(c, k - turn);
      h = end;
      // the cycle's travel so far (not the pelvis, which sways: a punch leans in on planted feet), moved back to
      // the origin when in place
      x = ((k - turn) / c.frames) * c.travel;
      if (inPlace) f = moveFrame(f, -x);
    }
    frames.push({ joints: placeFrame(f, 0, 0, h), heading: h, frame: f, x: inPlace ? 0 : x });
  }
  return { frames, loopStart: turn };
}

/** The rig's Pose for a skeleton frame placed at (x, y) facing h: what placeParts needs to hang a sprite stack's
 * parts on it. The frame must be made for this character's body (bodyFromRig of its rig). */
export function rigPose(f: SkeletonFrame, body: Body, x: number, y: number, h: number, phase = 0): Pose {
  const j = placeFrame(f, x, y, h);
  const legs: LegPose[] = [];
  const arms: ArmPose[] = [];
  for (let i = 0; i < 2; i++) {
    const n = SIDES[i];
    const ankle = j[`${n}_foot_joint`];
    legs.push({
      hip: j[`${n}_upLeg_joint`], knee: j[`${n}_leg_joint`], ankle, sole: [ankle[0], ankle[1], ankle[2] - body.ankle],
      rot: h + f.footYaw[i], planted: f.contact[i],
    });
    arms.push({ shoulder: j[`${n}_shoulder_1_joint`], elbow: j[`${n}_forearm_joint`], hand: j[`${n}_hand_joint`] });
  }
  const pelvis = j.root;
  const top = j.neck_1_joint;
  // where the head turns: on the line from the pelvis up through the shoulders' centre, at the neck's height
  const k = body.neck / body.spine;
  const neck: V3 = [pelvis[0] + (top[0] - pelvis[0]) * k, pelvis[1] + (top[1] - pelvis[1]) * k, pelvis[2] + (top[2] - pelvis[2]) * k];
  const P: Pose = {
    x, y, heading: h, phase, pelvis, top, neck, hipsAngle: h + f.hips, shouldersAngle: h + f.shoulders, legs, arms,
    lean: f.lean,
  };
  if (f.head) {
    P.headAngle = h + f.shoulders + f.head.turn;
    P.headLean = f.lean + f.head.nod;
  }
  return P;
}
