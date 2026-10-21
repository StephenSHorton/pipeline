// The Ideas grid's factory calls (legacy loadIdeas / keepIdea / useIdea / the Clear unkept handler). Every POST
// goes to /2d/*, so factoryJson already queues the 300 ms flow refresh. More ideas is a job: startJob, not here.
import { enc, factoryJson } from "@/lib/api";

export interface IdeaItem {
  file: string;
  seed?: number;
  prompt?: string;
  keep?: boolean;
  used?: boolean;
}

/** GET /2d/<stem>/ideas. `still` names the local image model ("" when none is installed). */
export interface IdeasDoc {
  ok?: boolean;
  stem?: string;
  prompt?: string;
  items: IdeaItem[];
  still?: string;
}

const post = <T>(path: string, body: unknown) =>
  factoryJson<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const getIdeas = (stem: string) => factoryJson<IdeasDoc>(`/2d/${enc(stem)}/ideas`);

/** The response is the whole ideas doc: it replaces the grid's. */
export const keepIdea = (stem: string, file: string, keep: boolean) => post<IdeasDoc>("/2d/keep", { stem, file, keep });

/** Make this idea the concept picture (flat/source.png). The response is not read. */
export const makeConcept = (stem: string, file: string) => post<IdeasDoc & { source?: string; used?: string }>("/2d/use", { stem, file });

/** Delete every idea that is neither kept nor in use. The response is the ideas doc. */
export const clearIdeas = (stem: string) => post<IdeasDoc>("/2d/clear", { stem });
