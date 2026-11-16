import { AnimationClip, AnimationMixer, LoopOnce, LoopRepeat, NumberKeyframeTrack, Object3D } from "three";
import { describe, expect, it, vi } from "vitest";

vi.mock("mokei/clay", () => ({}));
vi.mock("mokei/theme", () => ({}));

describe("fadeOthers", () => {
  it("fades out a finished play-once clip, so the next clip is not mixed with its last pose", async () => {
    const { fadeOthers } = await import("./Diorama");
    const root = new Object3D();
    root.name = "root";
    // Death turns the model over (rotation.z 0 -> 1.5) and holds; Idle keeps it upright.
    const death = new AnimationClip("Death", 1, [new NumberKeyframeTrack("root.rotation[z]", [0, 1], [0, 1.5])]);
    const idle = new AnimationClip("Idle", 1, [new NumberKeyframeTrack("root.rotation[z]", [0, 1], [0, 0])]);
    const m = new AnimationMixer(root);
    const all = { Death: m.clipAction(death), Idle: m.clipAction(idle) };
    all.Death.setLoop(LoopOnce, 1);
    all.Death.clampWhenFinished = true;
    all.Idle.setLoop(LoopRepeat, Infinity);
    all.Death.play();
    m.update(1.5); // finished and held
    expect(all.Death.isRunning()).toBe(false);
    expect(root.rotation.z).toBeCloseTo(1.5);
    fadeOthers(all, "Idle");
    all.Idle.reset().fadeIn(0.2).play();
    m.update(0.5);
    expect(root.rotation.z).toBeCloseTo(0, 3);
  });
});
