// The strip's writes: Build from the concept (legacy buildFromConcept, index.html 3542-3577) and Stack from picture
// (stackFromPicture, 3266-3285). Each takes its side effects as `deps` so api.test.ts runs them against fakes.
import { create } from "zustand";
import { navigate } from "@/app/router";
import { enc, factoryJson, readOutJson, saveIntentQuiet, type AssetOutput } from "@/lib/api";
import { useFlow, type Stage } from "@/lib/flow";
import type { JobOpts } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { startHomeJob } from "@/screens/home/chain";
import { intentBody, type NestedIntent } from "@/screens/home/make";
import { reloadIntent, useStage } from "../stage";
import { buildIntent, buildSpec, buildStage, buildTitle, withKeyFacing, type FlatKey } from "./logic";

export interface BuildDeps {
  readKey: (stem: string) => Promise<FlatKey | null>;
  saveIntent: (stem: string, intent: NestedIntent) => Promise<void>;
  /** Starts the job. Home's starter: its finish toast ("<name> ready: …", "Export →") and Try again come with it. */
  start: (spec: Record<string, unknown>, opts: JobOpts) => Promise<void>;
  /** Switches `stem`'s stage, but only while it is still the open asset (useStage holds one stem). */
  enter: (stem: string, stage: Stage) => void;
  /** The job card lives on Home (the legacy overlay): go there to watch the build. */
  showJob: () => void;
}

export const defaultBuildEnter = (stem: string, stage: Stage) => {
  if (useStage.getState().stem === stem) useStage.getState().enter(stage);
};

const buildDeps: BuildDeps = {
  readKey: (stem) => readOutJson<FlatKey>(stem, "flat/key.json"),
  async saveIntent(stem, intent) {
    await saveIntentQuiet(stem, intentBody(intent));
    await reloadIntent(stem);
  },
  start: startHomeJob,
  enter: defaultBuildEnter,
  showJob: () => navigate({ name: "home" }),
};

/** Build `output` from the concept picture `still` (out/<stem>/...). A sprite opens the 2D sheet instead. */
export async function buildFromConcept(
  stem: string,
  still: string,
  output: AssetOutput,
  saved: NestedIntent | null,
  deps: BuildDeps = buildDeps,
): Promise<void> {
  try {
    let intent = buildIntent(saved, output);
    // The concept knows which way its head points (flat/key.json: a canonical draw is side-on, head right).
    if (intent.output === "ps1" && still.endsWith("flat/source.png")) intent = withKeyFacing(intent, still, await deps.readKey(stem));
    if (intent.output === "sprite") return deps.enter(stem, "flat");
    if (!saved || saved.output !== intent.output) await deps.saveIntent(stem, intent);
    // Back from the job card, the asset opens on what was built.
    deps.enter(stem, buildStage(intent.output));
    deps.showJob();
    await deps.start(buildSpec(stem, still, intent), {
      title: buildTitle(stem, intent.output),
    });
  } catch (err) {
    toast(`Could not start: ${(err as Error)?.message || String(err)}`);
  }
}

export interface StackDeps {
  post: (stem: string) => Promise<unknown>;
  reload: (stem: string) => Promise<unknown>;
}

export const defaultStackPost = (stem: string) =>
  factoryJson(`/stack/${enc(stem)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });

/** Re-reads the flow only while `stem` is still the open asset: load() would switch the store back to it. */
export const defaultStackReload = (stem: string) => (useFlow.getState().stem === stem ? useFlow.getState().load(stem) : Promise.resolve(null));

const stackDeps: StackDeps = {
  post: defaultStackPost,
  reload: defaultStackReload,
};

/** The stem whose Stack from picture is in flight ("" for none). Module level, like legacy stackingPicture, so a
 * strip that remounts (Ideas, Home and back) still holds the button, and another asset's strip never shows it. */
export const useStacking = create<{ stem: string }>(() => ({ stem: "" }));

/** POST /stack/<s> {}: the sprite stack straight from the concept picture. The flow is re-read either way (the
 * Stack stage re-reads its files on the flow version). Resolves to the error message, or "" when it worked. */
export async function stackFromPicture(stem: string, deps: StackDeps = stackDeps): Promise<string> {
  let error = "";
  useStacking.setState({ stem });
  try {
    await deps.post(stem);
  } catch (err) {
    error = (err as Error)?.message || String(err);
    toast(error);
  } finally {
    if (useStacking.getState().stem === stem) useStacking.setState({ stem: "" });
  }
  await deps.reload(stem);
  return error;
}
