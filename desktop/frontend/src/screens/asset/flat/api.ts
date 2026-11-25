// The 2D sheet's factory calls. POSTs to /2d/* queue a flow refresh in factoryJson, so the rail and the versions
// strip follow a revise without a call here.
import { arrayBufferToBase64, enc, factoryJson } from "@/lib/api";
import { NO_OFFER, type AnimateOffer, type ClipFields, type FlatFlags, type Sheet, type StepsAnswer, type StepsBody } from "./flat";

/** GET /2d/<stem>: parts, mechanics, note, center. A miss is null (a stem with no sheet yet). */
export const getSheet = (stem: string) => factoryJson<Sheet>(`/2d/${enc(stem)}`).catch(() => null);

/** GET /2d/status: voice, sfx_mode, times. A miss is {} (legacy refreshAudioFlags). */
export const getFlatFlags = () => factoryJson<FlatFlags>("/2d/status").catch(() => ({}) as FlatFlags);

/** POST /2d/steps: runs one step at once (no job) and answers center, parts, hear, notes. */
export const postSteps = (body: StepsBody) =>
  factoryJson<StepsAnswer>("/2d/steps", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** The first video goes as {video}; otherwise every file is a frame (legacy clipFields). */
export async function clipFields(files: (Blob & { name: string })[]): Promise<ClipFields> {
  const video = files.find((f) => /^video\//.test(f.type) || /\.(mp4|webm)$/i.test(f.name));
  if (video) return { video: arrayBufferToBase64(await video.arrayBuffer()) };
  const frames: string[] = [];
  for (const f of files) frames.push(arrayBufferToBase64(await f.arrayBuffer()));
  return { frames };
}

/** GET /2d/animate/<stem>: what Animate offers this asset (the code-made cycles, each facing, with ready prompts;
 * the other skeletons), for its picker. A miss offers nothing. */
export const getAnimateOffer = (stem: string) =>
  factoryJson<Partial<AnimateOffer>>(`/2d/animate/${enc(stem)}`)
    .then((d): AnimateOffer => ({
      cycles: Array.isArray(d.cycles) ? d.cycles : [],
      skeletons: Array.isArray(d.skeletons) ? d.skeletons.filter((k) => k && typeof k === "object") : [],
    }))
    .catch(() => NO_OFFER);
