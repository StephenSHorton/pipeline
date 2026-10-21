import { afterEach, describe, expect, it, vi } from "vitest";
import type { Job } from "@/lib/jobs";
import { clearIdeas, getIdeas, keepIdea, makeConcept } from "./api";
import {
  clearConfirmText,
  draftFor,
  endedStatus,
  finishAction,
  ideasJobHere,
  ideasPending,
  IDEAS_COPY,
  ideaUrl,
  liveSlotText,
  loadStatus,
  setIdeasDraft,
  shouldFollow,
  unkeptCount,
} from "./ideas";

const STEM = "a-stenciled-wooden-supply-crate";
const job = (p: Partial<Job> = {}): Job => ({ id: "j1", stem: STEM, kind: "ideas", state: "running", progress: null, log: "", ...p });
const stages = (n: number) => [{ key: "load", label: "Load" }, ...Array.from({ length: n }, (_, i) => ({ key: `idea${i + 1}`, label: `Picture ${i + 1}` }))];

describe("copy", () => {
  it("matches the legacy page word for word", () => {
    expect(IDEAS_COPY.noModel).toBe("No local image model. Install mflux to draw ideas.");
    expect(IDEAS_COPY.empty).toBe("No ideas yet. Describe the thing below.");
    expect(IDEAS_COPY.useTitle).toBe("Make this the concept. 2D, 3D and the stack start from it.");
    expect(IDEAS_COPY.useWait).toBe("Wait for the ideas to finish.");
    expect(IDEAS_COPY.placeholder).toBe("Describe one object…");
    expect(IDEAS_COPY.promptTitle).toBe("One object per picture. Make each piece of a set as its own asset.");
    expect(IDEAS_COPY.clearTitle).toBe("Delete the ideas you did not keep.");
    expect(IDEAS_COPY.moreTitle).toBe("Draw four more from this prompt.");
    expect(IDEAS_COPY.busy).toBe("The factory is busy with another job.");
    expect(IDEAS_COPY.allKept).toBe("Every idea here is kept or in use.");
  });

  it("asks before clearing, singular and plural", () => {
    expect(clearConfirmText(1)).toBe("Delete the 1 idea you did not keep?");
    expect(clearConfirmText(3)).toBe("Delete the 3 ideas you did not keep?");
  });
});

describe("ideaUrl", () => {
  it("is /out/<stem>/ideas/<file>?s=<seed>", () => {
    expect(ideaUrl(STEM, { file: "idea-01.png", seed: 1003865282 })).toBe(`/out/${STEM}/ideas/idea-01.png?s=1003865282`);
  });
});

describe("the job here", () => {
  it("is a running ideas job on this stem only", () => {
    expect(ideasJobHere(job(), STEM)).toBe(true);
    expect(ideasJobHere(job({ stem: "other" }), STEM)).toBe(false);
    expect(ideasJobHere(job({ kind: "flat2d" }), STEM)).toBe(false);
    expect(ideasJobHere(job({ state: "ok" }), STEM)).toBe(false);
    expect(ideasJobHere(null, STEM)).toBe(false);
  });

  it("counts the pictures still to draw: count, else the idea stages, else 1", () => {
    expect(ideasPending(job({ count: 4, result: { items: [{ file: "idea-05.png" }] } }), STEM)).toBe(3);
    expect(ideasPending(job({ progress: { stages: stages(2), stage: "load", stage_index: 0, stage_count: 3 } }), STEM)).toBe(2);
    expect(ideasPending(job(), STEM)).toBe(1);
    expect(ideasPending(job({ stem: "other", count: 4 }), STEM)).toBe(0);
  });
});

describe("liveSlotText", () => {
  it("says which picture and the step", () => {
    const j = job({ count: 4, result: { items: [{ file: "a" }] }, log: "x\nStep 2 of 8\nStep 3 of 8 [ 37%]" });
    expect(liveSlotText(j, 3)).toBe("Picture 2 of 4 · step 3/8");
  });

  it("says why a picture is drawn again", () => {
    expect(liveSlotText(job({ count: 4, log: "Again, ground was cloudy · Step 3 of 8" }), 4)).toBe("Picture 1 of 4 · again (cloudy) · step 3/8");
    expect(liveSlotText(job({ count: 4, log: "There is a Shadow on the ground. Drawing again" }), 4)).toBe("Picture 1 of 4 · again (shadow)");
  });

  it("loading and stopping win", () => {
    const loading = job({ count: 4, progress: { stages: stages(4), stage: "load", stage_index: 0, stage_count: 5 } });
    expect(liveSlotText(loading, 4)).toBe("Loading the model…");
    expect(liveSlotText({ ...loading, stopping: true }, 4)).toBe("Stopping…");
  });
});

describe("status lines", () => {
  it("no model, no job → the install hint", () => {
    expect(loadStatus({ items: [], still: "" }, false)).toBe(IDEAS_COPY.noModel);
    expect(loadStatus({ items: [], still: "" }, true)).toBe(null);
    expect(loadStatus({ items: [], still: "qwen-2512" }, false)).toBe(null);
    expect(loadStatus(null, false)).toBe(null);
  });

  it("an ended job: Stopped. or its error", () => {
    expect(endedStatus({ state: "cancelled" })).toBe("Stopped.");
    expect(endedStatus({ state: "error", error: "No local image model. Install mflux." })).toBe("No local image model. Install mflux.");
    expect(endedStatus({ state: "error" })).toBe("");
  });

  it("counts the unkept, not the kept or used", () => {
    expect(unkeptCount([{ file: "a" }, { file: "b", keep: true }, { file: "c", used: true }, { file: "d", keep: false }])).toBe(2);
    expect(unkeptCount(undefined)).toBe(0);
  });
});

describe("api", () => {
  const g = globalThis as unknown as { fetch: typeof fetch };
  const realFetch = g.fetch;
  afterEach(() => {
    g.fetch = realFetch;
  });

  it("reads and writes the ideas doc with the legacy bodies", async () => {
    const calls: Array<[string, RequestInit | undefined]> = [];
    g.fetch = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(u), init]);
      return new Response(JSON.stringify({ ok: true, prompt: "crate", items: [] }), { status: 200 });
    }) as typeof fetch;
    await getIdeas(STEM);
    await keepIdea(STEM, "idea-01.png", true);
    await makeConcept(STEM, "idea-02.png");
    await clearIdeas(STEM);
    expect(calls.map(([u, i]) => [u, i?.method ?? "GET", i?.body ? JSON.parse(String(i.body)) : null])).toEqual([
      [`/2d/${STEM}/ideas`, "GET", null],
      ["/2d/keep", "POST", { stem: STEM, file: "idea-01.png", keep: true }],
      ["/2d/use", "POST", { stem: STEM, file: "idea-02.png" }],
      ["/2d/clear", "POST", { stem: STEM }],
    ]);
  });
});

describe("draft", () => {
  it("is kept for its asset only", () => {
    setIdeasDraft(STEM, "a crate");
    expect(draftFor(STEM)).toBe("a crate");
    expect(draftFor("other")).toBe("");
  });
});

describe("shouldFollow (legacy followIdeas)", () => {
  const one = { items: [{ file: "a.png" }] };
  it("hands over at the first picture of a watched job, once, unless that asset is open", () => {
    expect(shouldFollow(job({ result: one }), true, null, new Set())).toBe(true);
    expect(shouldFollow(job({ result: one }), true, "other", new Set())).toBe(true);
    expect(shouldFollow(job({ result: one }), true, STEM, new Set())).toBe(false);
    expect(shouldFollow(job({ result: one }), true, null, new Set(["j1"]))).toBe(false);
  });
  it("waits for a picture, a watched running ideas job", () => {
    expect(shouldFollow(job(), true, null, new Set())).toBe(false);
    expect(shouldFollow(job({ result: one }), false, null, new Set())).toBe(false);
    expect(shouldFollow(job({ result: one, state: "ok" }), true, null, new Set())).toBe(false);
    expect(shouldFollow(job({ result: one, kind: "flat2d" }), true, null, new Set())).toBe(false);
    expect(shouldFollow(null, true, null, new Set())).toBe(false);
  });
});

describe("finishAction (legacy finishIdeasJob)", () => {
  it("drops quietly on that asset, opens when watching, else toasts", () => {
    expect(finishAction(true, STEM, STEM)).toBe("drop");
    expect(finishAction(false, STEM, STEM)).toBe("drop");
    expect(finishAction(true, null, STEM)).toBe("open");
    expect(finishAction(true, "other", STEM)).toBe("open");
    expect(finishAction(false, null, STEM)).toBe("toast");
    expect(finishAction(false, "other", STEM)).toBe("toast");
  });
});
