// The rig finish seam: it owns exactly the jobs Home's chain.ts skips ("the viewer owns rig jobs").
import { describe, expect, it } from "vitest";
import { isRigJob } from "./finish";

describe("isRigJob", () => {
  it("rig and render jobs, nothing else", () => {
    expect(isRigJob({ kind: "rig" })).toBe(true);
    expect(isRigJob({ kind: "render" })).toBe(true);
    for (const kind of ["generate", "flat2d", "ideas"] as const) expect(isRigJob({ kind } as never)).toBe(false);
  });
});
