// Build and Stack from picture against fakes: nothing here reaches the factory.
import { afterEach, describe, expect, it, vi } from "vitest";
import { useFlow } from "@/lib/flow";
import { useUi } from "@/lib/ui";
import { useStage } from "../stage";
import { buildFromConcept, defaultBuildEnter, defaultStackReload, stackFromPicture, useStacking, type BuildDeps } from "./api";

const fakes = (over: Partial<BuildDeps> = {}) => {
  const calls: string[] = [];
  const deps: BuildDeps = {
    readKey: vi.fn(async () => ({ facing: "right" as const, canonical: true })),
    saveIntent: vi.fn(async () => void calls.push("save")),
    start: vi.fn(async () => void calls.push("start")),
    enter: vi.fn((stem, s) => void calls.push(`enter ${stem} ${s}`)),
    showJob: vi.fn(() => void calls.push("home")),
    ...over,
  };
  return { deps, calls };
};

const flow0 = useFlow.getState();
const stage0 = useStage.getState();
afterEach(() => {
  useUi.setState({ toast: null });
  useFlow.setState(flow0, true);
  useStage.setState(stage0, true);
  useStacking.setState({ stem: "" });
});

describe("buildFromConcept", () => {
  it("PS1 from the drawing: reads key.json, saves the new output, lands on 3D, starts the job", async () => {
    const { deps, calls } = fakes();
    await buildFromConcept("cow", "out/cow/flat/source.png", "ps1", { output: "model", subject: "animal" }, deps);
    expect(deps.readKey).toHaveBeenCalledWith("cow");
    expect(deps.saveIntent).toHaveBeenCalledWith("cow", {
      output: "ps1",
      subject: "animal",
      psx: { height: 48, facing: "right" },
      facing_source: "drawn",
    });
    expect(calls).toEqual(["save", "enter cow mesh", "home", "start"]);
    expect(deps.start).toHaveBeenCalledWith(
      expect.objectContaining({
        image: "out/cow/flat/source.png",
        stem: "cow",
        quality: "psx",
        facing: "right",
        psx_height: 48,
      }),
      { title: "Building the PS1 model · cow" },
    );
  });

  it("same output: no intent save; a stack lands on Stack", async () => {
    const { deps, calls } = fakes();
    await buildFromConcept("s", "out/s/input.png", "stack", { output: "stack", poly: "high" }, deps);
    expect(deps.readKey).not.toHaveBeenCalled();
    expect(calls).toEqual(["enter s stack", "home", "start"]);
    expect(deps.start).toHaveBeenCalledWith(expect.objectContaining({ quality: "stack", poly: "high" }), {
      title: "Building the sprite stack · s",
    });
  });

  it("a sprite opens the 2D sheet and starts nothing", async () => {
    const { deps, calls } = fakes();
    await buildFromConcept("s", "out/s/flat/source.png", "sprite", null, deps);
    expect(calls).toEqual(["enter s flat"]);
  });

  it("another asset opened during the awaits: its stage is left alone", async () => {
    const enter = vi.fn();
    useStage.setState({ stem: "cow", enter });
    // The intent save is the await; the user opens "other" while it runs.
    const { deps } = fakes({
      saveIntent: vi.fn(async () => void useStage.setState({ stem: "other" })),
      enter: defaultBuildEnter,
    });
    await buildFromConcept("cow", "out/cow/input.png", "model", null, deps);
    expect(enter).not.toHaveBeenCalled();
    expect(deps.start).toHaveBeenCalled();
  });

  it("still the open asset: the default enter switches its stage", () => {
    const enter = vi.fn();
    useStage.setState({ stem: "cow", enter });
    defaultBuildEnter("cow", "mesh");
    expect(enter).toHaveBeenCalledWith("mesh");
  });

  it("a start that throws says Could not start", async () => {
    const { deps } = fakes({
      start: vi.fn(async () => Promise.reject(new Error("no slot"))),
    });
    await buildFromConcept("s", "out/s/flat/source.png", "model", { output: "model" }, deps);
    expect(useUi.getState().toast?.text).toBe("Could not start: no slot");
  });
});

describe("stackFromPicture", () => {
  it("posts, then re-reads the flow", async () => {
    const post = vi.fn(async () => ({ ok: true }));
    const reload = vi.fn(async () => null);
    expect(await stackFromPicture("s", { post, reload })).toBe("");
    expect(post).toHaveBeenCalledWith("s");
    expect(reload).toHaveBeenCalledWith("s");
  });

  it("an error toasts and still re-reads the flow", async () => {
    const reload = vi.fn(async () => null);
    expect(
      await stackFromPicture("s", {
        post: async () => Promise.reject(new Error("no picture")),
        reload,
      }),
    ).toBe("no picture");
    expect(useUi.getState().toast?.text).toBe("no picture");
    expect(reload).toHaveBeenCalled();
  });

  it("holds the stacking stem while the post runs, and clears it when the post rejects", async () => {
    let seen = "";
    const post = vi.fn(async () => {
      seen = useStacking.getState().stem;
      throw new Error("busy");
    });
    await stackFromPicture("spider", { post, reload: async () => null });
    expect(seen).toBe("spider");
    expect(useStacking.getState().stem).toBe("");
  });

  it("another asset opened before the post answers: the flow store is not moved back", async () => {
    const load = vi.fn(async () => null);
    useFlow.setState({ stem: "a", load });
    const post = vi.fn(async () => void useFlow.setState({ stem: "b" }));
    await stackFromPicture("a", { post, reload: defaultStackReload });
    expect(load).not.toHaveBeenCalled();
    // Still on it: the flow is re-read.
    await stackFromPicture("b", {
      post: async () => null,
      reload: defaultStackReload,
    });
    expect(load).toHaveBeenCalledWith("b");
  });

  it("the default post is POST /stack/<s> with {}", async () => {
    const fetchMock = vi.fn(async () => new Response('{"ok":true}', { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    try {
      await stackFromPicture("a-mech-spider", {
        post: (await import("./api")).defaultStackPost,
        reload: async () => null,
      });
      expect(fetchMock).toHaveBeenCalledWith("/stack/a-mech-spider", expect.objectContaining({ method: "POST", body: "{}" }));
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
