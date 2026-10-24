import { describe, expect, it } from "vitest";
import { parseRoute, routeHref, UNSORTED } from "./router";

describe("router", () => {
  it("reads the asset, settings and home forms", () => {
    expect(parseRoute("?asset=hen&lod=psx")).toEqual({ name: "asset", stem: "hen", lod: "psx" });
    expect(parseRoute("", "#/asset/hen")).toEqual({ name: "asset", stem: "hen" });
    expect(parseRoute("?view=settings&project=farm")).toEqual({ name: "settings" });
    expect(parseRoute("")).toEqual({ name: "home" });
    expect(parseRoute("?view=junk")).toEqual({ name: "home" });
  });

  it("keeps ?project= on both Home sections", () => {
    expect(parseRoute("?view=library&project=mlg-farming")).toEqual({ name: "home", section: "library", project: "mlg-farming" });
    expect(parseRoute("?view=make&project=mlg-farming")).toEqual({ name: "home", section: "make", project: "mlg-farming" });
    expect(parseRoute(`?view=library&project=${UNSORTED}`)).toEqual({ name: "home", section: "library", project: "_unsorted" });
    expect(parseRoute("?view=library&project=")).toEqual({ name: "home", section: "library" });
    // An asset route never carries a project.
    expect(parseRoute("?asset=hen&project=farm")).toEqual({ name: "asset", stem: "hen" });
  });

  it("writes what it reads", () => {
    for (const s of ["?view=library&project=mlg-farming", "?view=make&project=_unsorted", "?view=library", "?asset=hen&lod=psx", "?view=settings"]) {
      expect(routeHref(parseRoute(s))).toBe(`/${s}`);
    }
    expect(routeHref({ name: "home" })).toBe("/");
    expect(routeHref({ name: "home", section: "make", project: "" })).toBe("/?view=make");
  });
});
