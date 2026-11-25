// A walking character's data (pipeline#2): stack/character.json and its part strips, as scripts/stack_character.py
// writes them (`walk`). No DOM and no renderer here: the cells each part draws, and where each part hangs on the
// rig every frame, as plain numbers.
import type { Pose, RigConfig, V3 } from "./rig";

export const CHARACTER_FORMAT = "pipeline-sprite-character";

export type PartKind = "torso" | "head" | "thigh" | "shin" | "foot" | "upperarm" | "forearm";

export interface CharacterPart {
  name: string;
  /** The strip, next to character.json: n slices side by side, bottom slice first, each cw x ch. */
  file: string;
  cw: number;
  ch: number;
  n: number;
  /** The body layer of the bottom slice. */
  z0: number;
  /** The body layer of the joint the part hangs from (its z = 0). */
  joint_layer: number;
  side: "R" | "L" | null;
}

export interface CharacterDoc {
  format: string;
  version: number;
  height: number;
  rig: RigConfig;
  walk: { stride: number; duty: number; speed: number; fps: number; lap: { strides: number; frames: number } };
  parts: CharacterPart[];
}

export const isCharacterDoc = (x: unknown): x is CharacterDoc =>
  !!x && typeof x === "object" && (x as CharacterDoc).format === CHARACTER_FORMAT && Array.isArray((x as CharacterDoc).parts);

export const partKind = (name: string) => name.split("_")[0] as PartKind;

/** One part's cubes: centres in the part's own frame (cells; x forward, y left, z up from its joint), colours, and
 * each cube's body layer. Only cubes with an open face: the inside of a part never shows, however it turns. */
export interface PartCells {
  count: number;
  pos: Float32Array;
  rgb: Uint8Array;
  layer: Int16Array;
  /** Each cube's slice (0 = bottom), for parts drawn a slice at a time (the torso's twist). */
  slice: Int16Array;
}

/** The cubes of one part from its strip's RGBA (width cw * n, height ch). */
export function partCells(part: CharacterPart, rgba: ArrayLike<number>): PartCells {
  const { cw, ch, n } = part;
  const width = cw * n;
  const on = (s: number, x: number, y: number) =>
    s >= 0 && s < n && x >= 0 && x < cw && y >= 0 && y < ch && rgba[(y * width + s * cw + x) * 4 + 3] >= 128;
  const pos: number[] = [];
  const rgb: number[] = [];
  const layer: number[] = [];
  const slice: number[] = [];
  const cx = Math.floor(cw / 2);
  const cy = Math.floor(ch / 2);
  for (let s = 0; s < n; s++)
    for (let y = 0; y < ch; y++)
      for (let x = 0; x < cw; x++) {
        if (!on(s, x, y)) continue;
        if (on(s - 1, x, y) && on(s + 1, x, y) && on(s, x - 1, y) && on(s, x + 1, y) && on(s, x, y - 1) && on(s, x, y + 1)) continue;
        const i = (y * width + s * cw + x) * 4;
        pos.push(x - cx, y - cy, part.z0 + s + 0.5 - part.joint_layer);
        rgb.push(rgba[i], rgba[i + 1], rgba[i + 2]);
        layer.push(part.z0 + s);
        slice.push(s);
      }
  return {
    count: layer.length,
    pos: Float32Array.from(pos),
    rgb: Uint8Array.from(rgb),
    layer: Int16Array.from(layer),
    slice: Int16Array.from(slice),
  };
}

/** Where a part hangs this frame: origin (its joint, rig cells) and its three axes (columns: forward, left, up), the
 * up axis scaled when a limb's two joints are further apart or nearer than its drawn length. */
export interface Placement {
  origin: V3;
  x: V3;
  y: V3;
  z: V3;
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const turned = (origin: V3, a: number): Placement => ({
  origin,
  x: [Math.cos(a), Math.sin(a), 0],
  y: [-Math.sin(a), Math.cos(a), 0],
  z: [0, 0, 1],
});

/** Turned to a, and tilted forward by lean along heading h (the upper body of a run or a punch). */
function leaning(origin: V3, a: number, h: number, lean: number): Placement {
  if (!lean) return turned(origin, a);
  // Rz(h) . Ry(lean) . Rz(a - h): turn within the body, tilt the body forward, face it along h
  const r = a - h;
  const cl = Math.cos(lean);
  const sl = Math.sin(lean);
  const ch = Math.cos(h);
  const sh = Math.sin(h);
  const tilt = (v: V3): V3 => [v[0] * cl + v[2] * sl, v[1], -v[0] * sl + v[2] * cl];
  const face = (v: V3): V3 => [ch * v[0] - sh * v[1], sh * v[0] + ch * v[1], v[2]];
  return {
    origin,
    x: face(tilt([Math.cos(r), Math.sin(r), 0])),
    y: face(tilt([-Math.sin(r), Math.cos(r), 0])),
    z: face(tilt([0, 0, 1])),
  };
}

/** A limb from joint `from` down to joint `to`, drawn `bone` cells long: its up axis points from `to` back to
 * `from`, its forward axis as near `facing` as that allows. */
export function limb(from: V3, to: V3, bone: number, facing: number): Placement {
  const d = sub(from, to);
  const l = len(d) || 1e-6;
  const z: V3 = [d[0] / l, d[1] / l, d[2] / l];
  const f: V3 = [Math.cos(facing), Math.sin(facing), 0];
  const along = f[0] * z[0] + f[1] * z[1] + f[2] * z[2];
  let x: V3 = [f[0] - along * z[0], f[1] - along * z[1], f[2] - along * z[2]];
  const xl = len(x);
  x = xl < 1e-6 ? [1, 0, 0] : [x[0] / xl, x[1] / xl, x[2] / xl];
  const y = cross(z, x);
  const k = l / Math.max(1e-6, bone);
  return { origin: from, x, y, z: [z[0] * k, z[1] * k, z[2] * k] };
}

/** Where every part (and, for the torso, every slice) hangs in a pose. The torso twists slice by slice from the
 * hips' turn at the bottom to the shoulders' at the top. */
export function placeParts(doc: CharacterDoc, P: Pose): Map<string, Placement | Placement[]> {
  const c = doc.rig;
  const out = new Map<string, Placement | Placement[]>();
  for (const part of doc.parts) {
    const kind = partKind(part.name);
    const s = part.side === "L" ? 1 : 0;
    const leg = P.legs[s];
    const arm = P.arms[s];
    switch (kind) {
      case "torso": {
        const slices: Placement[] = [];
        for (let i = 0; i < part.n; i++) {
          const t = Math.max(0, Math.min(1, (part.z0 + i + 0.5 - part.joint_layer) / Math.max(1, c.spine)));
          slices.push(leaning(P.pelvis, P.hipsAngle + (P.shouldersAngle - P.hipsAngle) * t, P.heading, P.lean ?? 0));
        }
        out.set(part.name, slices);
        break;
      }
      case "head":
        out.set(part.name, leaning(P.neck, P.shouldersAngle, P.heading, P.lean ?? 0));
        break;
      case "thigh":
        out.set(part.name, limb(leg.hip, leg.knee, c.thigh, P.hipsAngle));
        break;
      case "shin":
        out.set(part.name, limb(leg.knee, leg.ankle, c.shin, P.hipsAngle));
        break;
      case "foot":
        out.set(part.name, turned(leg.ankle, leg.rot));
        break;
      case "upperarm":
        if (arm) out.set(part.name, limb(arm.shoulder, arm.elbow, c.upperArm, P.shouldersAngle));
        break;
      case "forearm":
        if (arm) out.set(part.name, limb(arm.elbow, arm.hand, c.foreArm, P.shouldersAngle));
        break;
    }
  }
  return out;
}

/** Apply a placement to a point in the part's frame. */
export function placePoint(p: Placement, local: V3): V3 {
  return [
    p.origin[0] + p.x[0] * local[0] + p.y[0] * local[1] + p.z[0] * local[2],
    p.origin[1] + p.x[1] * local[0] + p.y[1] * local[1] + p.z[1] * local[2],
    p.origin[2] + p.x[2] * local[0] + p.y[2] * local[1] + p.z[2] * local[2],
  ];
}
