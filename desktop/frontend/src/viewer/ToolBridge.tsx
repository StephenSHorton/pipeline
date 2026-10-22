// The Canvas side of the viewer tool API (./tools.ts): <ToolBridge/> hands R3F's camera, gl and scene to the
// store, <ToolOverlays/> draws the panels' mesh overlays, and useViewerOverlay is how a panel adds one. Both
// components mount at the top of the Canvas, outside the lowered stage group, so overlays sit in world space.
import { useThree } from "@react-three/fiber";
import { Fragment, useEffect, type ReactNode } from "react";
import { setOverlay, setViewerThree, useViewerTools } from "./tools";

export function ToolBridge() {
  const get = useThree((s) => s.get);
  useEffect(() => {
    // A getter, not a snapshot: R3F swaps the default camera when ClayCameraRig mounts its own.
    setViewerThree(() => {
      const s = get();
      return { camera: s.camera, gl: s.gl, scene: s.scene };
    });
    return () => setViewerThree(null);
  }, [get]);
  return null;
}

export function ToolOverlays() {
  const overlays = useViewerTools((s) => s.overlays);
  return (
    <>
      {Object.entries(overlays).map(([key, node]) => (
        <Fragment key={key}>{node}</Fragment>
      ))}
    </>
  );
}

/**
 * Draw `node` inside the viewer's Canvas while the calling component is mounted (null draws nothing). `key` must
 * be unique per panel, e.g. "carve-ball". The node is R3F JSX (<mesh>, a component using useFrame), in world space.
 */
export function useViewerOverlay(key: string, node: ReactNode | null): void {
  useEffect(() => {
    setOverlay(key, node);
  }, [key, node]);
  useEffect(() => () => setOverlay(key, null), [key]);
}
