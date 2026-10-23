// The Weights panel's store and its one factory call. Module-level, so the panel's choices (bone, brush, size) and
// the edit itself (the WeightsSession on the mesh) outlive a close and a reload of the same mesh.
import { create } from "zustand";
import { factoryJson } from "@/lib/api";
import { toast } from "@/lib/ui";
import { recheckRig } from "./actions";
import { BRUSH_SIZE, BRUSH_STRENGTH, savedToast, weightsPath, type BrushMode, type ShowMode } from "./weights";
import type { WeightsSession } from "./weightsView";

export interface WeightsSaved {
  ok: boolean;
  stem: string;
  take: string | null;
  files: string[];
  changed: number;
  fbx: string | null;
  fbx_note: string;
  blender: boolean;
  skipped?: string[];
}

interface WeightsStore {
  /** The chosen bone (skeleton index), -1 for none. */
  bone: number;
  show: ShowMode;
  /** The armed brush, or null (a click on the model picks a bone). */
  brush: BrushMode | null;
  /** Brush radius as a share of the model's size. */
  size: number;
  strength: number;
  saving: boolean;
  /** Bumped on every edit, so the panel's numbers follow. */
  rev: number;
}

export const useWeights = create<WeightsStore>(() => ({
  bone: -1,
  show: "heat",
  brush: null,
  size: BRUSH_SIZE.value,
  strength: BRUSH_STRENGTH.value,
  saving: false,
  rev: 0,
}));

let session: WeightsSession | null = null;

/** The edit on the model (one per asset; a new asset's starts over). */
export function weightsSession(): WeightsSession | null {
  return session;
}

export function setWeightsSession(s: WeightsSession | null): void {
  session = s;
}

/** A new asset: its own bone and no armed brush. */
export function weightsFor(stem: string): void {
  if (session && session.stem !== stem) {
    session = null;
    useWeights.setState({ bone: -1, brush: null });
  }
}

/** POST /rig/<stem>/weights (scripts/rig_weights.py). */
export function postWeights(stem: string, body: unknown, post: typeof factoryJson = factoryJson): Promise<WeightsSaved> {
  return post<WeightsSaved>(weightsPath(stem), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

/**
 * Save: every point's four bones and weights into the rig, then the checks again (inspect.json reads the new weights;
 * the flow shows the new rig version). A busy asset (a job writing it) answers 409 and the edit stays on screen.
 */
export async function saveWeights(
  s: WeightsSession,
  deps: { post?: typeof factoryJson; recheck?: (stem: string) => Promise<void>; say?: (text: string) => void } = {},
): Promise<WeightsSaved | null> {
  const say = deps.say ?? toast;
  if (useWeights.getState().saving) return null;
  useWeights.setState({ saving: true });
  try {
    const out = await postWeights(s.stem, s.body(), deps.post);
    s.markSaved();
    say(savedToast(out));
    await (deps.recheck ?? recheckRig)(s.stem);
    return out;
  } catch (err) {
    say(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    useWeights.setState((st) => ({ saving: false, rev: st.rev + 1 }));
  }
}
