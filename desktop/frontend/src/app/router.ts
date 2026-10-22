// A tiny query router. No path routes: the Go host owns every path prefix (/out, /assets, /2d, ...) and
// serves the app only at "/". ?asset=<stem> opens the asset screen;
// ?view=library|make focuses a Home section; ?view=settings is Settings. #/asset/<stem> is accepted too and rewritten to the query form.
import { useSyncExternalStore } from "react";

export type Route =
  | { name: "home"; section?: "make" | "library" }
  | { name: "asset"; stem: string; lod?: string }
  | { name: "settings" };

export function parseRoute(search: string, hash = ""): Route {
  const h = hash.match(/^#\/asset\/([^?#]+)/);
  if (h) return { name: "asset", stem: decodeURIComponent(h[1]) };
  const q = new URLSearchParams(search);
  const stem = q.get("asset");
  if (stem) return { name: "asset", stem, ...(q.get("lod") ? { lod: q.get("lod")! } : {}) };
  const view = q.get("view");
  if (view === "settings") return { name: "settings" };
  return view === "make" || view === "library" ? { name: "home", section: view } : { name: "home" };
}

export function routeHref(r: Route): string {
  const q = new URLSearchParams();
  if (r.name === "asset") {
    q.set("asset", r.stem);
    if (r.lod) q.set("lod", r.lod);
  } else if (r.name === "settings") q.set("view", "settings");
  else if (r.section) q.set("view", r.section);
  const s = q.toString();
  return `/${s ? `?${s}` : ""}`;
}

const listeners = new Set<() => void>();
let snapshot = "";
let route: Route = { name: "home" };

function read(): Route {
  const key = location.search + location.hash;
  if (key !== snapshot) {
    snapshot = key;
    route = parseRoute(location.search, location.hash);
  }
  return route;
}

function emit() {
  for (const fn of listeners) fn();
}

if (typeof window !== "undefined") {
  window.addEventListener("popstate", emit);
  window.addEventListener("hashchange", () => {
    const r = parseRoute(location.search, location.hash);
    history.replaceState({}, "", routeHref(r));
    emit();
  });
  if (/^#\/asset\//.test(location.hash)) history.replaceState({}, "", routeHref(parseRoute(location.search, location.hash)));
}

export function navigate(r: Route, opts: { replace?: boolean } = {}): void {
  const href = routeHref(r);
  if (href === location.pathname + location.search && !location.hash) return;
  if (opts.replace) history.replaceState({}, "", href);
  else history.pushState({}, "", href);
  emit();
}

export function useRoute(): Route {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    read,
    read,
  );
}
