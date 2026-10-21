import { beforeEach, describe, expect, it } from "vitest";
import { DIR_PENDING, DIR_UNAVAILABLE, openFolder, refreshDataDir, useDataDir } from "./dataDir";

beforeEach(() => useDataDir.setState({ text: DIR_PENDING, tone: "" }));

describe("factory folder line", () => {
  it("shows the path", async () => {
    await refreshDataDir(async () => "/Users/me/Library/Application Support/Pipeline");
    expect(useDataDir.getState()).toEqual({ text: "/Users/me/Library/Application Support/Pipeline", tone: "" });
  });
  it("an empty answer changes nothing", async () => {
    await refreshDataDir(async () => "");
    expect(useDataDir.getState().text).toBe(DIR_PENDING);
  });
  it("a failure says unavailable only over …", async () => {
    const fail = async (): Promise<string> => {
      throw new Error("down");
    };
    await refreshDataDir(fail);
    expect(useDataDir.getState().text).toBe("Factory folder unavailable");
    useDataDir.setState({ text: "/old/path" });
    await refreshDataDir(fail);
    expect(useDataDir.getState().text).toBe("/old/path");
    expect(DIR_UNAVAILABLE).toBe("Factory folder unavailable");
  });
  it("Open errors replace the line in red; a fresh path clears the red", async () => {
    await openFolder(async () => {
      throw new Error("Finder is not available");
    });
    expect(useDataDir.getState()).toEqual({ text: "Finder is not available", tone: "error" });
    await refreshDataDir(async () => "/p");
    expect(useDataDir.getState()).toEqual({ text: "/p", tone: "" });
  });
  it("a working Open leaves the line alone", async () => {
    useDataDir.setState({ text: "/p" });
    await openFolder(async () => {});
    expect(useDataDir.getState()).toEqual({ text: "/p", tone: "" });
  });
});
