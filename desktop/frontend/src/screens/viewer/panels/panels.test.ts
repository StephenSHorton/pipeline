// The viewer panels' copy and open rules. The titles and the Rig check's × are word for word from the legacy
// cubbies (index.html 2090-2142) and rig-inspect.js 896; "Close <title>" is new, after the Audio panel. The open
// rules are legacy openStep (5817-5829) minus its bugs: no hide/tray, nothing stays hidden after a job.
import { afterEach, describe, expect, it } from "vitest";
import { closeLabel, closeTitle, openOn, PANEL_TITLES, panelDomId, panelsFor, RIG_CLOSE_LABEL, RIG_CLOSE_TITLE, togglePanel } from "./panels";
import { openViewerPanel, usePanels } from "./store";

afterEach(() => usePanels.setState({ open: null }));

describe("copy", () => {
  it("cubby titles match the legacy page", () => {
    expect(PANEL_TITLES).toEqual({ cleanup: "Cleanup", paint: "Paint", lods: "Detail levels", rig: "Rig check" });
    expect(RIG_CLOSE_TITLE).toBe("Close the rig check");
    expect(RIG_CLOSE_LABEL).toBe("Close");
    expect(closeTitle("rig")).toBe("Close the rig check");
    expect(closeLabel("rig")).toBe("Close");
    expect(closeLabel("cleanup")).toBe("Close Cleanup");
    expect(closeTitle("lods")).toBe("Close Detail levels");
  });

  it("does not port the CSS-hidden Hide copy", () => {
    for (const id of ["cleanup", "paint", "lods", "rig"] as const) {
      expect(closeLabel(id)).not.toMatch(/^Hide/);
      expect(closeTitle(id)).not.toMatch(/^Hide|^Show/);
    }
  });
});

describe("panelsFor", () => {
  it("left rail in legacy order on 3D and Rig; the Rig check only on Rig", () => {
    expect(panelsFor("mesh", "left")).toEqual(["cleanup", "paint", "lods"]);
    expect(panelsFor("rig", "left")).toEqual(["cleanup", "paint", "lods"]);
    expect(panelsFor("mesh", "right")).toEqual([]);
    expect(panelsFor("rig", "right")).toEqual(["rig"]);
  });
});

describe("togglePanel (legacy openStep)", () => {
  it("opens one, switching closes the other, clicking the open one closes it", () => {
    expect(togglePanel(null, "cleanup")).toBe("cleanup");
    expect(togglePanel("cleanup", "paint")).toBe("paint");
    expect(togglePanel("paint", "rig")).toBe("rig");
    expect(togglePanel("rig", "rig")).toBeNull();
  });

  it("the store keeps one open across rail and dock, and can open one from outside React (Inspect)", () => {
    usePanels.getState().toggle("lods");
    usePanels.getState().toggle("rig");
    expect(usePanels.getState().open).toBe("rig");
    usePanels.getState().toggle("rig");
    expect(usePanels.getState().open).toBeNull();
    openViewerPanel("rig");
    openViewerPanel("rig");
    expect(usePanels.getState().open).toBe("rig");
  });
});

describe("openOn", () => {
  it("a panel the stage does not show is closed (the Rig check off Rig)", () => {
    expect(openOn("rig", "mesh")).toBeNull();
    expect(openOn("rig", "rig")).toBe("rig");
    expect(openOn("paint", "mesh")).toBe("paint");
    expect(openOn(null, "rig")).toBeNull();
  });
});

describe("panelDomId", () => {
  it("is stable per panel", () => {
    expect(panelDomId("cleanup")).toBe("viewer-panel-cleanup");
  });
});
