// The clay diorama: Mokei's SceneCanvas, lights, ground, a soft clay pad and the asset standing on it, an
// orthographic orbit (ClayCameraRig) framed on the model, contact shadow and AO from PostFX, or the PS1 pass.
//
// ClayCameraRig always looks at (pan.x, 0, pan.z): the whole diorama is lowered by half the model's height so
// the camera centres on the model, not its feet. The look store is global: every mount re-frames it.
import { useFrame, useThree } from "@react-three/fiber";
import { ClayCameraRig, ClayGround, Lights, PostFX, RoundCyl, SceneCanvas, applyLook, lookDefaults, setViewHome, useLook } from "mokei/clay";
import { useEffect, useLayoutEffect, useRef } from "react";
import { AnimationMixer, LoopOnce, LoopRepeat, type AnimationAction, type AnimationClip, type Group } from "three";
import { setClipPlayer } from "./clipSeek";
import { PixelPass } from "./PixelPass";
import type { PixelLook } from "./pixelLook";
import type { PreparedModel } from "./prepareModel";
import { ToolBridge, ToolOverlays } from "./ToolBridge";

export const PAD_H = 0.45;
/** The orbit home: a little lower than Yardline's 37 deg, to see more of the model's side. */
export const HOME = { azimuth: 36, elevation: 26 };
/** One-shot clips hold their last frame instead of looping. */
const ONCE = /^(death|die|dead)$/i;

export interface DioramaProps {
  model: PreparedModel | null;
  clips: AnimationClip[];
  clip: string | null;
  playing: boolean;
  clay: boolean;
  wire: boolean;
  spin: boolean;
  pixel: PixelLook;
}

export function Diorama(props: DioramaProps) {
  const ground = useLook((s) => s.ground);
  return (
    <SceneCanvas quality="high" className="!absolute inset-0" style={{ background: ground }}>
      <color attach="background" args={[ground]} />
      <NoPointerEvents />
      <ToolBridge />
      <ClayCameraRig home={{ x: 0, z: 0 }} />
      <Lights />
      <Stage {...props} />
      <ToolOverlays />
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

function Stage({ model, clips, clip, playing, clay, wire, spin }: DioramaProps) {
  const turn = useRef<Group>(null);
  const height = model?.height ?? 4;
  const padR = Math.max(4.2, (model?.footprint ?? 3) * 1.25 + 0.6);

  useFrame((_, dt) => {
    if (spin && turn.current) turn.current.rotation.y += dt * 0.21; // the legacy autoRotateSpeed 0.7
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
      {model ? <Framer model={model} /> : null}
      {model && clips.length ? <Animator model={model} clips={clips} clip={clip} playing={playing} /> : null}
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

function Animator({ model, clips, clip, playing }: { model: PreparedModel; clips: AnimationClip[]; clip: string | null; playing: boolean }) {
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
    // The Rig check holds a clip still at its worst moment (./clipSeek.ts).
    setClipPlayer({ mixer: m, actions: actions.current });
    return () => {
      setClipPlayer(null);
      m.stopAllAction();
      m.uncacheRoot(model.scene);
      mixer.current = null;
      current.current = null;
    };
  }, [model, clips]);

  useEffect(() => {
    const all = actions.current;
    const next = clip ? all[clip] : undefined;
    if (!next || current.current === clip) return;
    for (const [name, a] of Object.entries(all)) if (name !== clip && a.isRunning()) a.fadeOut(0.2);
    next.reset().fadeIn(current.current ? 0.2 : 0).play();
    current.current = clip;
  }, [clip, clips, model]);

  useEffect(() => {
    for (const a of Object.values(actions.current)) a.paused = !playing;
  }, [playing, clip, clips, model]);

  useFrame((_, dt) => {
    mixer.current?.update(playing ? dt : 0);
  });
  return null;
}

/** On leaving the viewer, give the shared look store back its defaults (Home may draw clay too). */
export function resetLook() {
  applyLook({ ...lookDefaults });
  setViewHome({ cameraZoom: lookDefaults.cameraZoom, cameraAzimuth: lookDefaults.cameraAzimuth, cameraElevation: lookDefaults.cameraElevation });
}
