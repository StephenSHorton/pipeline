// The walk for a character made of part stacks (pipeline#2). No drawing here, only numbers.
//
// Ported from loktar00/sprite-stack-animation js/rig.js (MIT, Copyright (c) 2026 loktar00; see NOTICE).
// Two ideas carry it:
//  1. The step clock runs on distance, not time: each step() adds (ground covered / stride) to one phase. A leg is
//     planted for the first `duty` of its cycle and swings for the rest; the legs are half a cycle apart. Because
//     the clock only moves when the body does, feet cannot slide.
//  2. A planted foot is a fixed point in the world. The body travels over it, and the knee is solved every frame
//     from where the hip and the foot are: analytic two-bone IK (the law of cosines). Elbows use the same solver.
//
// Changed from the kit: hips and shoulders may sit forward or behind the body's middle (hipFwd, shoulderFwd); the
// neck has its own height; the hip limit allows for bob and twist, and a swinging foot is kept within reach, so the
// legs never stretch (a part stack cannot stretch).
//
// World axes: x and y on the ground, z up; a character facing heading h looks along (cos h, sin h). All lengths
// are in cells (one voxel of the part stacks). Side index 0 is the character's right, 1 its left.

export type V3 = [number, number, number];

export interface RigConfig {
  thigh: number;
  shin: number;
  /** The ankle's height over the sole. */
  ankle: number;
  /** Half the distance between the hips. */
  hipHalf: number;
  /** How far forward of the body's middle the hips are (negative: behind). */
  hipFwd: number;
  /** 1: knees bend forward; -1: backward, like a bird. */
  kneeHint: number;
  /** Hip height when standing (under thigh + shin + ankle, so the knees stay a little bent). */
  standHip: number;
  /** Ground covered by one full cycle (two steps). */
  stride: number;
  /** The part of a cycle a foot is planted: over 0.5 is a walk, under it a run. */
  duty: number;
  /** How high a swinging foot rises. */
  lift: number;
  /** Hips rise and fall, twice a cycle. */
  bob: number;
  /** Hips turn with the leading leg (radians); the shoulders turn against it. */
  twist: number;
  /** From the hips up to the shoulders. */
  spine: number;
  /** From the hips up to where the head turns. */
  neck: number;
  arms: boolean;
  shoulderHalf: number;
  /** How far forward of the body's middle the shoulders are. */
  shoulderFwd: number;
  upperArm: number;
  foreArm: number;
  armSwing: number;
}

export const RIG_DEFAULTS: RigConfig = {
  thigh: 7, shin: 7, ankle: 1.5, hipHalf: 2.4, hipFwd: 0, kneeHint: 1, standHip: 13.2, stride: 18, duty: 0.6, lift: 2.4,
  bob: 0.35, twist: 0.16, spine: 9.5, neck: 10.5, arms: true, shoulderHalf: 7.5, shoulderFwd: 0, upperArm: 4.6,
  foreArm: 4.6, armSwing: 0.55,
};

interface Foot {
  x: number;
  y: number;
  z: number;
  rot: number;
  planted: boolean;
  fx: number;
  fy: number;
  frot: number;
}

export interface Walker {
  c: RigConfig;
  x: number;
  y: number;
  h: number;
  dx: number;
  dy: number;
  phase: number;
  hipZ: number;
  /** Touchdowns so far. */
  steps: number;
  feet: [Foot, Foot];
}

export interface LegPose {
  hip: V3;
  knee: V3;
  ankle: V3;
  sole: V3;
  rot: number;
  planted: boolean;
}

export interface ArmPose {
  shoulder: V3;
  elbow: V3;
  hand: V3;
}

export interface Pose {
  x: number;
  y: number;
  heading: number;
  phase: number;
  pelvis: V3;
  top: V3;
  /** Where the head turns (on the spine, above the pelvis). */
  neck: V3;
  hipsAngle: number;
  shouldersAngle: number;
  legs: LegPose[];
  arms: ArmPose[];
  /** The upper body's tilt forward along the heading (radians); the walk leaves it out (upright). */
  lean?: number;
  /** The head's own turn and forward tilt, when it trails the body (pipeline#8); else it turns with the shoulders
   * and tilts with the lean. */
  headAngle?: number;
  headLean?: number;
}

const TAU = Math.PI * 2;
const ease = (t: number) => t * t * (3 - 2 * t);
function turn(a: number, b: number, t: number) {
  let d = (b - a) % TAU;
  if (d > Math.PI) d -= TAU;
  if (d < -Math.PI) d += TAU;
  return a + d * t;
}

/** Two-bone IK: the middle joint for a chain root -> target with bone lengths a and b, bending towards hint. Out
 * of reach, the limb is straight. */
export function solveTwoBone(root: V3, target: V3, a: number, b: number, hint: V3): V3 {
  let dx = target[0] - root[0];
  let dy = target[1] - root[1];
  let dz = target[2] - root[2];
  const len = Math.hypot(dx, dy, dz) || 1e-6;
  dx /= len;
  dy /= len;
  dz /= len;
  const d = Math.min(Math.max(len, Math.abs(a - b) + 1e-3), a + b - 1e-3);
  const cosA = (a * a + d * d - b * b) / (2 * a * d);
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const along = hint[0] * dx + hint[1] * dy + hint[2] * dz;
  let px = hint[0] - along * dx;
  let py = hint[1] - along * dy;
  let pz = hint[2] - along * dz;
  const pl = Math.hypot(px, py, pz);
  if (pl < 1e-6) {
    px = 0;
    py = 0;
    pz = 1;
  } else {
    px /= pl;
    py /= pl;
    pz /= pl;
  }
  return [root[0] + dx * a * cosA + px * a * sinA, root[1] + dy * a * cosA + py * a * sinA, root[2] + dz * a * cosA + pz * a * sinA];
}

/** Side 0 (right) is -1, side 1 (left) is +1: a heading of 0 faces +x, so the left is +y. */
const sideSign = (i: number) => (i ? 1 : -1);

function hipGround(w: Walker, i: number, h: number): [number, number] {
  const s = sideSign(i);
  const c = w.c;
  return [w.x + Math.cos(h) * c.hipFwd - Math.sin(h) * s * c.hipHalf, w.y + Math.sin(h) * c.hipFwd + Math.cos(h) * s * c.hipHalf];
}

export function createWalker(cfg: Partial<RigConfig> = {}): Walker {
  const c = { ...RIG_DEFAULTS, ...cfg };
  const foot = (): Foot => ({ x: 0, y: 0, z: 0, rot: 0, planted: true, fx: 0, fy: 0, frot: 0 });
  const w: Walker = { c, x: 0, y: 0, h: 0, dx: 1, dy: 0, phase: 0, hipZ: c.standHip, steps: 0, feet: [foot(), foot()] };
  place(w, 0, 0, 0);
  return w;
}

/** Put the walker somewhere without walking there: both feet under the hips, or, with `walking`, where the walk
 * has them at phase 0 (one foot just down half a stance ahead, the other behind), so the first step is like every
 * other. */
export function place(w: Walker, x: number, y: number, h: number, walking = false) {
  w.x = x;
  w.y = y;
  w.h = h;
  w.dx = Math.cos(h);
  w.dy = Math.sin(h);
  w.hipZ = w.c.standHip;
  w.phase = 0;
  for (let i = 0; i < 2; i++) {
    const g = hipGround(w, i, h);
    const f = w.feet[i];
    const p = (i * 0.5) % 1;
    // past the stance (a run's short duty) the foot starts in the air, from where it would have lifted
    const ahead = !walking ? 0 : p < w.c.duty ? (w.c.duty * w.c.stride) / 2 - p * w.c.stride : -(w.c.duty * w.c.stride) / 2;
    f.x = g[0] + w.dx * ahead;
    f.y = g[1] + w.dy * ahead;
    f.z = 0;
    f.rot = h;
    f.planted = !walking || p < w.c.duty;
    f.fx = f.x;
    f.fy = f.y;
    f.frot = h;
  }
}

/** Move the body to (x, y) facing h. Call once per fixed time step; dt only smooths the hip height. */
export function step(w: Walker, x: number, y: number, h: number, dt = 1 / 60) {
  const c = w.c;
  const mx = x - w.x;
  const my = y - w.y;
  const dist = Math.hypot(mx, my);
  w.x = x;
  w.y = y;
  w.h = h;
  if (dist > 1e-6) {
    w.dx = mx / dist;
    w.dy = my / dist;
  }
  w.phase = (w.phase + dist / c.stride) % 1;
  const stance = c.duty * c.stride;
  let far = 0;
  for (let i = 0; i < 2; i++) {
    const f = w.feet[i];
    const g = hipGround(w, i, h);
    const p = (w.phase + i * 0.5) % 1;
    if (p < c.duty) {
      if (!f.planted) {
        // touchdown half a stance ahead of the hip, so the hip passes over it mid-stance
        const ahead = stance / 2 - p * c.stride;
        f.planted = true;
        f.x = g[0] + w.dx * ahead;
        f.y = g[1] + w.dy * ahead;
        f.rot = h;
        w.steps++;
      }
      f.z = 0; // planted: x and y are not touched until the foot lifts
      far = Math.max(far, Math.hypot(f.x - g[0], f.y - g[1]));
    } else {
      if (f.planted) {
        f.planted = false;
        f.fx = f.x;
        f.fy = f.y;
        f.frot = f.rot;
      }
      const u = (p - c.duty) / (1 - c.duty);
      const e = ease(u);
      const ahead = (1 - p) * c.stride + stance / 2;
      f.x = f.fx + (g[0] + w.dx * ahead - f.fx) * e;
      f.y = f.fy + (g[1] + w.dy * ahead - f.fy) * e;
      f.z = c.lift * Math.sin(Math.PI * u);
      f.rot = turn(f.frot, h, e);
    }
  }
  // hip height: as high as standing allows, never so high that a planted foot is out of reach, even at the top of
  // the bob (pose() adds it) and with the hips turned by the twist
  const reach = c.thigh + c.shin - 0.3;
  far += c.hipHalf * Math.sin(Math.abs(c.twist));
  const most = c.ankle + Math.sqrt(Math.max(1, reach * reach - far * far)) - c.bob;
  const want = Math.min(c.standHip, most);
  w.hipZ += (want - w.hipZ) * Math.min(1, dt * 14);
  if (w.hipZ > most) w.hipZ = most;
}

/** Every joint for drawing, as plain data. */
export function pose(w: Walker): Pose {
  const c = w.c;
  const ph = w.phase * TAU;
  const fwd: V3 = [Math.cos(w.h), Math.sin(w.h), 0];
  const twist = c.twist * Math.cos(ph);
  const hipsAngle = w.h + twist;
  const shouldersAngle = w.h - twist * 1.2;
  const pz = w.hipZ - c.bob * Math.cos(2 * ph);
  const P: Pose = {
    x: w.x, y: w.y, heading: w.h, phase: w.phase, pelvis: [w.x, w.y, pz], top: [w.x, w.y, pz + c.spine],
    neck: [w.x, w.y, pz + c.neck],
    hipsAngle, shouldersAngle, legs: [], arms: [],
  };
  for (let i = 0; i < 2; i++) {
    const s = sideSign(i);
    const f = w.feet[i];
    const hip: V3 = [
      w.x + Math.cos(hipsAngle) * c.hipFwd - Math.sin(hipsAngle) * s * c.hipHalf,
      w.y + Math.sin(hipsAngle) * c.hipFwd + Math.cos(hipsAngle) * s * c.hipHalf,
      pz,
    ];
    const ankle: V3 = [f.x, f.y, f.z + c.ankle];
    if (!f.planted) {
      // a foot in the air never trails out of the leg's reach (the kit's lagging swing stretched the shin):
      // pulled in along the line to the hip. A planted foot is never moved.
      const d = Math.hypot(ankle[0] - hip[0], ankle[1] - hip[1], ankle[2] - hip[2]);
      const most = c.thigh + c.shin - 0.05;
      if (d > most) for (let k = 0; k < 3; k++) ankle[k] = hip[k] + ((ankle[k] - hip[k]) * most) / d;
    }
    P.legs.push({
      hip, knee: solveTwoBone(hip, ankle, c.thigh, c.shin, [fwd[0] * c.kneeHint, fwd[1] * c.kneeHint, 0]),
      ankle, sole: [ankle[0], ankle[1], ankle[2] - c.ankle], rot: f.rot, planted: f.planted,
    });
    if (!c.arms) continue;
    // the arm swings against the leg on its side; the elbow bends backwards
    const sh: V3 = [
      w.x + Math.cos(shouldersAngle) * c.shoulderFwd - Math.sin(shouldersAngle) * s * c.shoulderHalf,
      w.y + Math.sin(shouldersAngle) * c.shoulderFwd + Math.cos(shouldersAngle) * s * c.shoulderHalf,
      pz + c.spine,
    ];
    const a = s * c.armSwing * Math.cos(ph);
    const L = (c.upperArm + c.foreArm) * 0.93;
    const hand: V3 = [sh[0] + fwd[0] * Math.sin(a) * L, sh[1] + fwd[1] * Math.sin(a) * L, sh[2] - Math.cos(a) * L];
    P.arms.push({ shoulder: sh, elbow: solveTwoBone(sh, hand, c.upperArm, c.foreArm, [-fwd[0], -fwd[1], 0]), hand });
  }
  return P;
}

/** The demo loop: a circle of exactly `strides` strides, walked in `lapFrames` steps of 1/60 s, so the last frame
 * leads into the first. Returns the body's place and heading for a frame (counter-clockwise, facing along it). */
export function lapPoint(frame: number, cfg: Pick<RigConfig, "stride">, strides: number, lapFrames: number) {
  const radius = (strides * cfg.stride) / TAU;
  const t = ((frame % lapFrames) + lapFrames) % lapFrames;
  const a = (t / lapFrames) * TAU;
  return { x: radius * Math.cos(a), y: radius * Math.sin(a), h: a + Math.PI / 2, radius };
}
