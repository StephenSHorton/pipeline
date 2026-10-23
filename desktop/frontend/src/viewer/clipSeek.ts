// The Diorama's clip player, for a panel that has to hold a clip still at one moment (the Rig check's "worst
// moment", legacy pauseClipAt / resumeClip / restPose, index.html 5038-5060, 4782-4783). The Animator registers its
// mixer and actions here; the HUD's clip and Play/Pause state still come from ViewerScreen. Only the current side
// registers: Compare's left side follows its clip time (clipTime).
import type { AnimationAction, AnimationMixer } from "three";

interface Player {
  mixer: AnimationMixer;
  actions: Record<string, AnimationAction>;
}

let player: Player | null = null;

/** Diorama's Animator: its mixer and actions while mounted, null when it goes. */
export function setClipPlayer(p: Player | null): void {
  player = p;
}

/** Hold `name` still at `t` seconds; the other clips stop. False when there is no such clip. */
export function pauseClipAt(name: string, t: number): boolean {
  const a = player?.actions[name];
  if (!player || !a) return false;
  for (const [n, x] of Object.entries(player.actions)) if (n !== name) x.stop();
  a.reset().play();
  a.setEffectiveWeight(1);
  a.time = Math.min(t, a.getClip().duration);
  a.paused = true;
  player.mixer.update(0);
  return true;
}

/** Play `name` on from where it is held (or from the start once the rest pose stopped it). */
export function resumeClip(name: string | null): void {
  const a = name ? player?.actions[name] : undefined;
  if (!a) return;
  a.paused = false;
  a.setEffectiveWeight(1);
  if (!a.isRunning()) a.reset().play();
}

/** Where `name` is now in seconds, or null when it is neither playing nor held (Compare's left side follows it). */
export function clipTime(name: string): number | null {
  const a = player?.actions[name];
  return a && (a.isRunning() || a.paused) && a.getEffectiveWeight() > 0 ? a.time : null;
}

/** Every clip off and the mixer applied once: the model at its rest pose. */
export function restPose(): void {
  if (!player) return;
  player.mixer.stopAllAction();
  player.mixer.update(0);
}
