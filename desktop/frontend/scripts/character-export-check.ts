// Loads an exported walking character the way a game would (pipeline#2): asset.json -> stack/character/
// character.json -> every part strip, checks each strip is n slices of cw x ch, then walks the rig two laps and
// checks a planted foot never moves and no limb stretches. Prints what it found; exits 1 on any miss.
//
//   bun scripts/character-export-check.ts ../../out/walker/export
import { readFileSync } from "node:fs";
import path from "node:path";
import { createWalker, lapPoint, place, pose, step } from "../src/screens/asset/stack/character/rig";
import { isCharacterDoc } from "../src/screens/asset/stack/character/parts";

const dir = path.resolve(process.argv[2] ?? "");
const fail = (why: string): never => {
  console.error(`FAIL: ${why}`);
  process.exit(1);
};

/** Width and height from a PNG's IHDR. */
function pngSize(file: string): [number, number] {
  const b = readFileSync(file);
  if (b.toString("latin1", 1, 4) !== "PNG") fail(`${file} is not a PNG`);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

const asset = JSON.parse(readFileSync(path.join(dir, "asset.json"), "utf8"));
const rel = asset?.stack?.character;
if (typeof rel !== "string") fail("asset.json has no stack.character");
const docPath = path.join(dir, rel);
const doc = JSON.parse(readFileSync(docPath, "utf8"));
if (!isCharacterDoc(doc)) fail("character.json is not a pipeline-sprite-character");
console.log(`${rel}: ${doc.parts.length} parts, ${doc.height} cells tall, stride ${doc.walk.stride}`);
for (const p of doc.parts) {
  const [w, h] = pngSize(path.join(path.dirname(docPath), p.file));
  if (w !== p.cw * p.n || h !== p.ch) fail(`${p.file} is ${w}x${h}, wants ${p.cw * p.n}x${p.ch}`);
  console.log(`  ${p.name.padEnd(11)} ${String(p.n).padStart(3)} slices of ${p.cw}x${p.ch}, joint at layer ${p.joint_layer}`);
}

const w = createWalker(doc.rig);
const { strides, frames } = doc.walk.lap;
const s0 = lapPoint(0, w.c, strides, frames);
place(w, s0.x, s0.y, s0.h, true);
let prev = pose(w);
let planted = 0;
let worstSlide = 0;
let worstStretch = 0;
const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
for (let f = 1; f <= frames * 2; f++) {
  const p = lapPoint(f, w.c, strides, frames);
  step(w, p.x, p.y, p.h, 1 / doc.walk.fps);
  const P = pose(w);
  P.legs.forEach((L, i) => {
    const A = prev.legs[i];
    if (A.planted && L.planted) {
      planted++;
      worstSlide = Math.max(worstSlide, d(A.sole, L.sole));
    }
    worstStretch = Math.max(worstStretch, Math.abs(d(L.knee, L.ankle) - w.c.shin), Math.abs(d(L.hip, L.knee) - w.c.thigh));
  });
  prev = P;
}
console.log(`walked 2 laps (${frames * 2} steps, ${w.steps} footfalls): planted-foot frames ${planted}, most a planted foot moved ${worstSlide.toFixed(6)} cells, most a leg bone stretched ${worstStretch.toFixed(6)} cells`);
if (worstSlide > 1e-9) fail("a planted foot slid");
if (worstStretch > 1e-3) fail("a leg stretched");
console.log("OK");
