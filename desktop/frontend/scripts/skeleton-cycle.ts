// Skeleton cycles (pipeline#14) out to files: for each cycle and facing, the cycle itself (pipeline-skeleton-cycle
// JSON), its side view as Apple Vision 2D joints (what runners/pose_error.py compares), an OpenPose-18 skeleton video
// on black (the control video LTX-2.3's Union-Control IC-LoRA takes) and the walking character's sprite stack posed by
// the same frames, drawn from the same side at the same scale. Also the start picture for LTX: the character's drawn
// front view on white, placed where the skeleton's first frame stands.
//
//   bun scripts/skeleton-cycle.ts --character ../../out/walker/stack --front ../../out/walker/character/views/front.png \
//     --out /tmp/cycles [--cycle walk|run|attack|all] [--facing right|left|both] [--frames 73] [--turn 16] [--size 704]
//     [--pose-only]
//
// Each pose file also says where the standing figure is in the picture ("figure": its top, its ground line and its
// middle, 0..1 of the picture): the Animate stage (scripts/animate.py) fits a character's start picture to that box.
// --pose-only writes only the pose files (what Animate reads), no videos; it is what Animate runs for a character.
//
// Every clip turns from facing the camera (the start picture) to the side over --turn frames, then plays the loop
// on the spot. Frame k of the skeleton video and frame k of the stack video are the same pose.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { isCharacterDoc, partCells, placeParts, placePoint, type CharacterDoc } from "../src/screens/asset/stack/character/parts";
import type { V3 } from "../src/screens/asset/stack/character/rig";
import { bodyFromRig, CYCLES, makeCycle, rigPose, shot, type CycleName, type JointName } from "../src/screens/asset/stack/character/skeleton";

const arg = (k: string, d?: string) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? process.argv[i + 1] : d;
};
const charDir = path.resolve(arg("character") ?? "");
const front = arg("front");
const out = path.resolve(arg("out") ?? "cycles");
const which = arg("cycle", "all")!;
const facings = (arg("facing", "both") === "both" ? ["right", "left"] : [arg("facing")!]) as ("right" | "left")[];
const total = Number(arg("frames", "73"));
const turn = Number(arg("turn", "16"));
const W = Number(arg("size", "704"));
const H = W;
const poseOnly = process.argv.includes("--pose-only");
mkdirSync(out, { recursive: true });

function readRGBA(file: string): { w: number; h: number; px: Uint8Array } {
  const probe = spawnSync("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", file]);
  const [w, h] = probe.stdout.toString().trim().split(",").map(Number);
  const r = spawnSync("ffmpeg", ["-v", "error", "-i", file, "-f", "rawvideo", "-pix_fmt", "rgba", "-"], { maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(`cannot read ${file}`);
  return { w, h, px: new Uint8Array(r.stdout) };
}

function writeVideo(file: string, frames: Uint8Array[], fps: number) {
  const r = spawnSync(
    "ffmpeg",
    ["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", `${W}x${H}`, "-framerate", String(fps), "-i", "-",
      "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", file],
    { input: Buffer.concat(frames.map((f) => Buffer.from(f.buffer, f.byteOffset, f.byteLength))), maxBuffer: 1 << 30 },
  );
  if (r.status !== 0) throw new Error(`ffmpeg: ${r.stderr}`);
}

function writePNG(file: string, px: Uint8Array) {
  const r = spawnSync("ffmpeg", ["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", `${W}x${H}`, "-i", "-", file], { input: Buffer.from(px) });
  if (r.status !== 0) throw new Error(`ffmpeg: ${r.stderr}`);
}

// --- the character
const docPath = path.join(charDir, "character.json");
const doc = JSON.parse(readFileSync(docPath, "utf8")) as CharacterDoc;
if (!isCharacterDoc(doc)) throw new Error(`${docPath} is not a pipeline-sprite-character`);
const body = bodyFromRig(doc.rig, doc.height);

// Screen: the camera looks along +y (the character's right side is nearest when it faces right), x to the right,
// z up; one cell is `px` pixels; the ground `ground` pixels down; the body over the middle.
const pxPerCell = Math.floor((H * 0.82) / doc.height);
const ground = Math.round(H - (H - doc.height * pxPerCell) / 2);
const toScreen = (p: V3): [number, number] => [W / 2 + p[0] * pxPerCell, ground - p[2] * pxPerCell];

// --- the start picture: the drawn front view, scaled so its height is the stack's height, its middle over x = 0
if (front) {
  const img = readRGBA(path.resolve(front));
  let x0 = img.w, x1 = -1, y0 = img.h, y1 = -1;
  for (let y = 0; y < img.h; y++)
    for (let x = 0; x < img.w; x++)
      if (img.px[(y * img.w + x) * 4 + 3] >= 128) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
  const k = (doc.height * pxPerCell) / (y1 - y0 + 1);
  const pic = new Uint8Array(W * H * 4).fill(255);
  const left = W / 2 - ((x0 + x1 + 1) / 2) * k;
  const top = ground - (y1 + 1) * k;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const sx = Math.floor((x + 0.5 - left) / k);
      const sy = Math.floor((y + 0.5 - top) / k);
      if (sx < 0 || sy < 0 || sx >= img.w || sy >= img.h) continue;
      const i = (sy * img.w + sx) * 4;
      const a = img.px[i + 3] / 255;
      const o = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) pic[o + c] = Math.round(img.px[i + c] * a + 255 * (1 - a));
    }
  writePNG(path.join(out, "start.png"), pic);
}

// --- drawing
function disc(px: Uint8Array, cx: number, cy: number, r: number, col: number[]) {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(H - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(W - 1, Math.ceil(cx + r)); x++)
      if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) px.set(col, (y * W + x) * 4);
}
function line(px: Uint8Array, a: [number, number], b: [number, number], r: number, col: number[]) {
  const [ax, ay] = a;
  const [bx, by] = b;
  const dx = bx - ax;
  const dy = by - ay;
  const l2 = dx * dx + dy * dy || 1e-9;
  for (let y = Math.max(0, Math.floor(Math.min(ay, by) - r)); y <= Math.min(H - 1, Math.ceil(Math.max(ay, by) + r)); y++)
    for (let x = Math.max(0, Math.floor(Math.min(ax, bx) - r)); x <= Math.min(W - 1, Math.ceil(Math.max(ax, bx) + r)); x++) {
      const t = Math.max(0, Math.min(1, ((x + 0.5 - ax) * dx + (y + 0.5 - ay) * dy) / l2));
      if ((x + 0.5 - ax - t * dx) ** 2 + (y + 0.5 - ay - t * dy) ** 2 <= r * r) px.set(col, (y * W + x) * 4);
    }
}

// OpenPose-18, as runners/skeleton_video.py draws it: the Vision joint for each index, limbs and colours
const OP: JointName[] = ["head_joint", "neck_1_joint", "right_shoulder_1_joint", "right_forearm_joint", "right_hand_joint",
  "left_shoulder_1_joint", "left_forearm_joint", "left_hand_joint", "right_upLeg_joint", "right_leg_joint",
  "right_foot_joint", "left_upLeg_joint", "left_leg_joint", "left_foot_joint", "right_eye_joint",
  "left_eye_joint", "right_ear_joint", "left_ear_joint"];
const LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
const COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];

/** Joints the camera sees: an eye or ear on the far side of the head is left out, as a pose detector would. */
function seen(j: Record<JointName, V3>): Partial<Record<JointName, V3>> {
  const s: Partial<Record<JointName, V3>> = { ...j };
  const mid = (j.left_ear_joint[1] + j.right_ear_joint[1]) / 2;
  const reach = Math.abs(j.left_ear_joint[1] - j.right_ear_joint[1]) / 2;
  for (const n of ["left_eye_joint", "right_eye_joint", "left_ear_joint", "right_ear_joint"] as JointName[])
    if (j[n][1] - mid > 0.35 * reach) delete s[n];
  return s;
}

function skeletonFrame(j: Partial<Record<JointName, V3>>): Uint8Array {
  const px = new Uint8Array(W * H * 4);
  for (let i = 3; i < px.length; i += 4) px[i] = 255;
  const pts = OP.map((n) => (j[n] ? toScreen(j[n]!) : null));
  LIMBS.forEach(([a, b], li) => {
    if (pts[a] && pts[b]) line(px, pts[a]!, pts[b]!, Math.max(4, W / 120) / 2, [...COLORS[li % 18], 255]);
  });
  pts.forEach((p, k) => p && disc(px, p[0], p[1], Math.max(4, W / 160), [...COLORS[k], 255]));
  return px;
}

// The stack: each part's surface cells, drawn as squares from a camera a little above, far ones first. walkView
// (and three.js with it) is loaded only when the stack is drawn.
const sliceShade = poseOnly ? () => 1 : (await import("../src/screens/asset/stack/character/walkView")).sliceShade;
const cellsByPart = poseOnly ? new Map() :
  new Map(doc.parts.map((p) => [p.name, partCells(p, readRGBA(path.join(charDir, p.file)).px)]));
const EL = (Number(arg("elevation", "20")) * Math.PI) / 180;
function stackFrame(pose: ReturnType<typeof rigPose>): Uint8Array {
  const placed = placeParts(doc, pose);
  const splats: { x: number; y: number; d: number; c: number[] }[] = [];
  const top = Math.max(1, doc.height - 1);
  for (const part of doc.parts) {
    const at = placed.get(part.name);
    const cells = cellsByPart.get(part.name);
    if (!at || !cells) continue;
    for (let i = 0; i < cells.count; i++) {
      const pl = Array.isArray(at) ? at[cells.slice[i]] : at;
      const w = placePoint(pl, [cells.pos[i * 3], cells.pos[i * 3 + 1], cells.pos[i * 3 + 2]]);
      const shade = sliceShade(cells.layer[i], top);
      splats.push({
        x: W / 2 + w[0] * pxPerCell,
        y: ground - (w[2] * Math.cos(EL) + w[1] * Math.sin(EL)) * pxPerCell,
        d: w[1] * Math.cos(EL) - w[2] * Math.sin(EL),
        c: [cells.rgb[i * 3] * shade, cells.rgb[i * 3 + 1] * shade, cells.rgb[i * 3 + 2] * shade].map(Math.round),
      });
    }
  }
  splats.sort((a, b) => b.d - a.d);
  const px = new Uint8Array(W * H * 4).fill(255);
  const on = new Uint8Array(W * H);
  const half = (pxPerCell * 1.1) / 2;
  for (const s of splats)
    for (let y = Math.max(0, Math.round(s.y - half)); y < Math.min(H, Math.round(s.y + half)); y++)
      for (let x = Math.max(0, Math.round(s.x - half)); x < Math.min(W, Math.round(s.x + half)); x++) {
        px.set([s.c[0], s.c[1], s.c[2], 255], (y * W + x) * 4);
        on[y * W + x] = 1;
      }
  // the sprite look's dark outline round the figure
  const edge = Math.max(1, Math.round(pxPerCell / 4));
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      if (on[y * W + x]) continue;
      let near = false;
      for (let dy = -edge; dy <= edge && !near; dy++)
        for (let dx = -edge; dx <= edge && !near; dx++) {
          const yy = y + dy;
          const xx = x + dx;
          near = yy >= 0 && xx >= 0 && yy < H && xx < W && on[yy * W + xx] === 1;
        }
      if (near) px.set([28, 24, 30, 255], (y * W + x) * 4);
    }
  // the ground line
  for (let x = 0; x < W; x++) if (!on[ground * W + x]) px.set([200, 200, 200, 255], (ground * W + x) * 4);
  return px;
}

const names = (which === "all" ? CYCLES : [which]) as CycleName[];
for (const name of names) {
  const cycle = makeCycle(name, body);
  for (const facing of facings) {
    const stem = `${name}_${facing}`;
    const s = shot(cycle, facing, total, turn);
    writeFileSync(path.join(out, `${stem}.cycle.json`), JSON.stringify({ ...cycle, shot: { facing, total, turn, loopStart: s.loopStart } }));
    const pose2d = s.frames.map((f) => {
      const j = seen(f.joints);
      const joints2d: Record<string, [number, number, number]> = {};
      for (const [n, p] of Object.entries(j)) {
        const [x, y] = toScreen(p!);
        joints2d[n] = [+(x / W).toFixed(4), +(y / H).toFixed(4), 1];
      }
      return { joints2d };
    });
    // the standing figure, as the start picture is placed: its top, its ground line and its middle
    const figure = { top: +((ground - doc.height * pxPerCell) / H).toFixed(4), ground: +(ground / H).toFixed(4), x: 0.5 };
    writeFileSync(path.join(out, `${stem}.pose.json`), JSON.stringify({ frames: pose2d, turn_frames: turn, cycle_frames: cycle.frames, figure }));
    if (poseOnly) {
      console.log(`${stem}: ${total} frames (pose only)`);
      continue;
    }
    writeVideo(path.join(out, `${stem}_skel.mp4`), s.frames.map((f) => skeletonFrame(seen(f.joints))), cycle.fps);
    writeVideo(path.join(out, `${stem}_stack.mp4`), s.frames.map((f) => stackFrame(rigPose(f.frame, body, 0, 0, f.heading))), cycle.fps);
    console.log(`${stem}: ${total} frames (turn ${turn}, loop of ${cycle.frames}, travel ${cycle.travel.toFixed(2)} cells)`);
  }
}
