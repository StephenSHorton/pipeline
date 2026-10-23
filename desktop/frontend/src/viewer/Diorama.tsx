// The clay diorama: Mokei's SceneCanvas, lights, ground, a soft clay pad and the asset standing on it, an
// orthographic orbit (ClayCameraRig) framed on the model, contact shadow and AO from PostFX, or the PS1 pass.
//
// ClayCameraRig always looks at (pan.x, 0, pan.z): the whole diorama is lowered by half the model's height so
// the camera centres on the model, not its feet. The look store is global: every mount re-frames it.
//
// Compare draws a second diorama, the follower (an earlier version, screens/viewer/CompareSide.tsx). Its camera rig
// reads the same look store, so orbit, zoom and pan move both; it never frames the store, never registers with the
// tool API or the clip player, and takes its spin angle and clip time from the current side.
import { useFrame, useThree } from "@react-three/fiber";
import { ClayCameraRig, ClayGround, Lights, PostFX, RoundCyl, SceneCanvas, applyLook, lookDefaults, setViewHome, useLook } from "mokei/clay";
import { useEffect, useLayoutEffect, useRef } from "react";
import { AnimationMixer, LoopOnce, LoopRepeat, type AnimationAction, type AnimationClip, type Group } from "three";
import { clipTime, setClipPlayer } from "./clipSeek";
import { PixelPass } from "./PixelPass";
import type { PixelLook } from "./pixelLook";
import type { PreparedModel } from "./prepareModel";
import { ToolBridge, ToolOverlays } from "./ToolBridge";

export const PAD_H = 0.45;
/** The orbit home: a little lower than Yardline's 37 deg, to see more of the model's side. */
export const HOME = { azimuth: 36, elevation: 26 };
/** One-shot clips hold their last frame instead of looping. */
const ONCE = /^(death|die|dead)$/i;
/** The current side's spin angle, for a follower to copy (one Spin turns both). */
const turned = { y: 0 };

export interface DioramaProps {
  model: PreparedModel | null;
  clips: AnimationClip[];
  clip: string | null;
  playing: boolean;
  clay: boolean;
  wire: boolean;
  spin: boolean;
  pixel: PixelLook;
  /** Compare's left side: follows the current diorama (see the top of this file). */
  follower?: boolean;
}

export function Diorama(props: DioramaProps) {
  const ground = useLook((s) => s.ground);
  return (
    <SceneCanvas quality="high" className="!absolute inset-0" style={{ background: ground }}>
      <color attach="background" args={[ground]} />
      <NoPointerEvents />
      {props.follower ? null : <ToolBridge />}
      <ClayCameraRig home={{ x: 0, z: 0 }} />
      <Lights />
      <Stage {...props} />
      {props.follower ? null : <ToolOverlays />}
      {props.pixel.on ? <PixelPass look={props.pixel} /> : <PostFX />}
    </SceneCanvas>
  );
}

/**
 * The page drives the orbit from the DOM (left-drag turns, right-drag pans, wheel zooms through the rig), so
 * R3F pointer events are off: ClayGround's own left-drag pan would fight the orbit. An armed panel tool gets the
 * pointer from the same DOM surface and raycasts itself (./tools.ts raycastAt).
 */
function NoPointerEvents() {
  const setEvents = useThree((s) => s.setEvents);
  useEffect(() => setEvents({ enabled: false }), [setEvents]);
  return null;
}

function Stage({ model, clips, clip, playing, clay, wire, spin, follower }: DioramaProps) {
  const turn = useRef<Group>(null);
  const height = model?.height ?? 4;
  const padR = Math.max(4.2, (model?.footprint ?? 3) * 1.25 + 0.6);

  useFrame((_, dt) => {
    if (!turn.current) return;
    if (follower) turn.current.rotation.y = turned.y;
    else {
      if (spin) turn.current.rotation.y += dt * 0.21; // the legacy autoRotateSpeed 0.7
      turned.y = turn.current.rotation.y;
    }
  });

  useEffect(() => {
    model?.setClay(clay);
  }, [model, clay]);

  useEffect(() => {
    model?.setWire(wire);
  }, [model, wire]);

  return (
    <group position-y={-height / 2}>
      <group position-y={-PAD_H}>
        <ClayGround />
      </group>
      <RoundCyl radius={padR} height={PAD_H} fillet={0.18} material="base" position={[0, -PAD_H / 2, 0]} cast={false} segments={64} />
      <group ref={turn}>{model ? <primitive object={model.root} /> : null}</group>
      {model && !follower ? <Framer model={model} /> : null}
      {model && clips.length ? <Animator model={model} clips={clips} clip={clip} playing={playing} follower={follower} /> : null}
    </group>
  );
}

/** Frame the model: ortho zoom is px per world unit; zoomReferenceWidth = canvas width makes cameraZoom exactly that. */
function Framer({ model }: { model: PreparedModel }) {
  const size = useThree((s) => s.size);
  const first = useRef(true);
  useLayoutEffect(() => {
    if (!size.width || !size.height) return;
    const zoom = Math.min(size.width, size.height) / (2 * model.radius * 1.12);
    const home = { cameraZoom: zoom, cameraAzimuth: HOME.azimuth, cameraElevation: HOME.elevation };
    setViewHome(home);
    const camera = first.current ? { ...home, panX: 0, panZ: 0 } : { cameraZoom: zoom }; // a resize keeps the orbit
    first.current = false;
    applyLook({ ...camera, zoomReferenceWidth: size.width, zoomMin: zoom * 0.35, zoomMax: zoom * 5 });
  }, [model, size.width, size.height]);
  return null;
}

/** Fade out every clip but `keep` that still shows. Not only the running ones: a play-once clip (Death) that finished
 * holds its last pose with full weight but reports not running, and left in, it kept the model tipped over under the
 * next clip. */
export function fadeOthers(all: Record<string, AnimationAction>, keep: string): void {
  for (const [name, a] of Object.entries(all)) if (name !== keep && a.getEffectiveWeight() > 0) a.fadeOut(0.2);
}

function Animator({ model, clips, clip, playing, follower }: { model: PreparedModel; clips: AnimationClip[]; clip: string | null; playing: boolean; follower?: boolean }) {
  const mixer = useRef<AnimationMixer | null>(null);
  const actions = useRef<Record<string, AnimationAction>>({});
  const current = useRef<string | null>(null);

  useEffect(() => {
    const m = new AnimationMixer(model.scene);
    mixer.current = m;
    actions.current = {};
    for (const c of clips) {
      const a = m.clipAction(c);
      if (ONCE.test(c.name)) {
        a.setLoop(LoopOnce, 1);
        a.clampWhenFinished = true;
      } else a.setLoop(LoopRepeat, Infinity);
      actions.current[c.name] = a;
    }
    current.current = null;
    // The Rig check holds a clip still at its worst moment (./clipSeek.ts). Not a follower's: it copies the time.
    if (!follower) setClipPlayer({ mixer: m, actions: actions.current });
    return () => {
      if (!follower) setClipPlayer(null);
      m.stopAllAction();
      m.uncacheRoot(model.scene);
      mixer.current = null;
      current.current = null;
    };
  }, [model, clips, follower]);

  useEffect(() => {
    const all = actions.current;
    if (!clip) return;
    const next = all[clip];
    if (!next || current.current === clip) return;
    fadeOthers(all, clip);
    next.reset().fadeIn(current.current ? 0.2 : 0).play();
    current.current = clip;
  }, [clip, clips, model]);

  useEffect(() => {
    for (const a of Object.values(actions.current)) a.paused = !playing;
  }, [playing, clip, clips, model]);

  useFrame((_, dt) => {
    const m = mixer.current;
    if (!m) return;
    m.update(playing ? dt : 0);
    // A follower runs its fades on its own clock and holds its clip at the current side's time (held still there by
    // the Rig check too). A clip of another length loops (or stops, play-once) at its own end.
    const a = follower && clip ? actions.current[clip] : undefined;
    const t = a && clip ? clipTime(clip) : null;
    if (!a || t === null) return;
    const d = a.getClip().duration;
    a.time = ONCE.test(clip!) ? Math.min(t, d) : d > 0 ? t % d : 0;
    m.update(0);
  });
  return null;
}

/** On leaving the viewer, give the shared look store back its defaults (Home may draw clay too). */
export function resetLook() {
  applyLook({ ...lookDefaults });
  setViewHome({ cameraZoom: lookDefaults.cameraZoom, cameraAzimuth: lookDefaults.cameraAzimuth, cameraElevation: lookDefaults.cameraElevation });
}
