// The library grid from GET /assets (legacy refreshAssets / filterLibrary / deleteAsset): thumbnail, name, kind,
// relative time, the per-card Degraded line (inspect.json), Rigged badge, kind filter + search, Delete. Grouped by
// project (?project=: All, one project, Unsorted), with a Move menu on each card.
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, Folder, FolderInput, Search, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MoveMenu, ProjectSwitcher } from "@/components/ProjectMenu";
import { deleteAsset, listAssets, flowFiles, outUrl, readInspect, type Asset, type Project } from "@/lib/api";
import { degradedLine, displayStem, onJobEnded } from "@/lib/jobs";
import { assetProject, membership, projectFilterName, refreshProjects, useProjectList } from "@/lib/projects";
import { askConfirm } from "@/lib/ui";
import { openAsset } from "@/app/open";
import { navigate, UNSORTED } from "@/app/router";
import { cn } from "@/lib/utils";
import { assetKind, DELETE_MESSAGE, KIND_LABEL, LIB_FILTERS, matchesFilter, relativeTime, type LibKind } from "./make";

/** Per-card extras, cached by id + mtime so a refresh does not ask again. */
const extras = new Map<string, Promise<{ degraded: string; rigged: boolean }>>();
function cardExtras(a: Asset) {
  const key = `${a.id}@${a.mtime}`;
  let p = extras.get(key);
  if (!p) {
    // The flow's file list first: inspect.json is read only when it is there (no 404 per card).
    p = a.has_mesh
      ? flowFiles(a.id).then(async (f) => {
          const doc = f?.has("inspect.json") ? await readInspect(a.id, a.mtime) : null;
          return { degraded: degradedLine(doc?.degraded), rigged: !!f?.has("rigged.glb") };
        })
      : Promise.resolve({ degraded: "", rigged: false });
    extras.set(key, p);
  }
  return p;
}

/** The kind switch / filter as hud tabs: the approved design's white pill on the glass strip, 44 px tall. */
export const KIND_TAB =
  "inline-flex h-auto min-h-11 items-center rounded-[10px] px-[0.9rem] text-[0.82rem] leading-none font-semibold text-ink-3 hover:bg-secondary data-active:bg-white data-active:text-blue-deep data-active:shadow-[0_1px_2px_rgba(30,41,90,0.08),0_2px_8px_rgba(30,41,90,0.06)]";

export function useAssets() {
  const [assets, setAssets] = useState<Asset[] | null>(null);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try {
      const r = await listAssets();
      setAssets(r.assets || []);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setAssets((a) => a || []);
    }
  }, []);
  useEffect(() => {
    void refresh();
    return onJobEnded(() => void refresh());
  }, [refresh]);
  return { assets, error, refresh, setAssets };
}

// The Library page only: Make no longer shows a "Recent" slice of it (HomeScreen.tsx).
export function Library({
  assets,
  error,
  refresh,
  onDeleted,
  project,
}: {
  assets: Asset[] | null;
  error: string;
  refresh: () => Promise<void>;
  onDeleted?: (id: string) => void;
  /** The route's ?project=: undefined All, UNSORTED, or a project id. */
  project?: string;
}) {
  const [kind, setKind] = useState<LibKind>("");
  const [query, setQuery] = useState("");
  const [gone, setGone] = useState<Set<string>>(new Set());
  const [delError, setDelError] = useState("");

  const projects = useProjectList();
  const list = useMemo(() => projects ?? [], [projects]);
  // Each row's project from the projects list once it has loaded (a move refreshes that list, not GET /assets).
  const all = useMemo(() => {
    const members = projects ? membership(projects) : null;
    return (assets || []).filter((a) => !gone.has(a.id)).map((a) => ({ ...a, project: assetProject(a, members) }));
  }, [assets, gone, projects]);
  const names = useMemo(() => new Map(list.map((p) => [p.id, p.name])), [list]);
  const shown = useMemo(() => all.filter((a) => matchesFilter(a, kind, query, project, (a.project && names.get(a.project)) || "")), [all, kind, query, project, names]);
  const counts = useMemo(() => {
    const n = new Map<string, number>();
    for (const a of all) if (a.project) n.set(a.project, (n.get(a.project) ?? 0) + 1);
    return { all: all.length, unsorted: all.length - [...n.values()].reduce((x, y) => x + y, 0), of: (id: string) => n.get(id) ?? 0 };
  }, [all]);
  // replace: flipping the filter is not a page to go Back through.
  const pick = (id: string | undefined) => navigate({ name: "home", section: "library", project: id }, { replace: true });
  const title = project ? projectFilterName(projects, project) : "Library";

  const remove = async (a: Asset) => {
    if (!(await askConfirm(DELETE_MESSAGE(a.id)))) return;
    setGone((g) => new Set(g).add(a.id));
    setDelError("");
    try {
      await deleteAsset(a.id);
      onDeleted?.(a.id);
    } catch (err) {
      setDelError(err instanceof Error ? err.message : String(err));
    }
    // The factory drops a deleted stem from its project too.
    await Promise.all([refresh(), refreshProjects()]);
    setGone(new Set());
  };

  return (
    <section aria-labelledby="mk-library" className="w-full max-w-[52rem]">
      <h2 id="mk-library" className="m-0 mb-3 truncate px-1 text-[1.25rem] font-semibold text-ink">
        {title}
      </h2>

      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 px-1">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <ProjectSwitcher
            value={project}
            projects={list}
            counts={counts}
            onPick={pick}
            trigger={
              <Button
                variant="outline"
                aria-label={`Project: ${projectFilterName(projects, project)}`}
                title="Show one project's assets"
                className="h-auto min-h-11 max-w-[16rem] gap-1.5 rounded-[14px] border-[var(--topbar-border)] bg-[var(--topbar-bg)] px-3.5 text-[0.82rem] font-semibold text-ink hover:bg-white aria-expanded:bg-white"
              >
                <Folder aria-hidden="true" className="size-4 text-ink-3" />
                <span className="truncate">{projectFilterName(projects, project)}</span>
                <ChevronDown aria-hidden="true" className="size-3.5 text-ink-3" />
              </Button>
            }
          />
          <Tabs value={kind || "all"} onValueChange={(v) => typeof v === "string" && setKind((v === "all" ? "" : v) as LibKind)}>
            <TabsList variant="hud" aria-label="Show" className="flex-wrap gap-1 rounded-[14px] border border-[var(--topbar-border)] bg-[var(--topbar-bg)] p-1">
              {LIB_FILTERS.map((f) => (
                <TabsTrigger key={f.id || "all"} variant="hud" value={f.id || "all"} className={KIND_TAB}>
                  {f.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </div>
        <label className="relative block w-full max-w-[16rem]">
          <span className="sr-only">Search assets</span>
          <Search aria-hidden="true" className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search assets" className="h-10 rounded-xl bg-white/90 pl-9" />
        </label>
      </div>

      {error ? <p className="mb-2 px-1 text-[0.8rem] text-danger-text">{error}</p> : null}
      {delError ? (
        <p role="alert" className="mb-2 px-1 text-[0.8rem] text-danger-text">
          {delError}
        </p>
      ) : null}

      {assets === null ? null : !all.length ? (
        <p className="px-1 text-[0.85rem] text-ink-3">No assets yet. Prompt one into existence.</p>
      ) : !shown.length ? (
        <p className="px-1 text-[0.85rem] text-ink-3">
          {project && !kind && !query.trim() ? (project === UNSORTED ? "Every asset is in a project." : `Nothing in ${title} yet. Make with it open, or move assets in from All projects.`) : "Nothing matches."}
        </p>
      ) : (
        <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] items-start gap-3 p-0">
          {shown.map((a) => (
            <AssetCard
              key={a.id}
              asset={a}
              projects={list}
              projectName={!project && a.project ? names.get(a.project) || "" : ""}
              onDelete={() => void remove(a)}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function AssetCard({ asset: a, projects, projectName, onDelete }: { asset: Asset; projects: readonly Project[]; projectName: string; onDelete: () => void }) {
  const [x, setX] = useState<{ degraded: string; rigged: boolean }>({ degraded: "", rigged: false });
  const [broken, setBroken] = useState(false);
  useEffect(() => {
    let live = true;
    void cardExtras(a).then((v) => live && setX(v));
    return () => {
      live = false;
    };
  }, [a]);
  const kind = assetKind(a);
  // Titles mix "hay_bale" and "hay bale"; on the card both read as words.
  const name = (a.title || displayStem(a.id)).replace(/_/g, " ");
  const shot = a.preview || a.input;
  // Every asset opens in the asset screen; one with no mesh (a sprite, Ideas, a drawing never built) lands on the
  // stage its flow says is current (openStem → defaultStage), not on 3D.
  const open = () => openAsset(a.id);

  return (
    <li className="group/tile relative flex flex-col gap-2 rounded-2xl border border-[var(--glass-border)] bg-[var(--glass-bg)] p-2 shadow-[var(--glass-shadow)] transition-[box-shadow,translate] hover:-translate-y-0.5 hover:shadow-[0_0_0_1px_color-mix(in_srgb,var(--primary)_30%,transparent),0_12px_28px_rgba(30,41,90,0.12)]">
      <button
        type="button"
        onClick={open}
        title="Open"
        className="flex cursor-pointer flex-col gap-2 rounded-xl border-0 bg-transparent p-0 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <div aria-hidden="true" className="relative flex aspect-[4/3] w-full items-end justify-center overflow-hidden rounded-[11px] bg-clay-sky">
          {shot && !broken ? (
            <img src={outUrl(a.id, shot, a.mtime)} alt="" loading="lazy" onError={() => setBroken(true)} className="size-full object-cover" />
          ) : (
            <Placeholder />
          )}
          {/* On the picture, so the meta line keeps its full width. */}
          {x.rigged ? (
            <Badge variant="success" className="absolute bottom-1.5 left-1.5 h-[22px] rounded-[7px] px-2 text-[0.7rem] font-semibold">
              Rigged
            </Badge>
          ) : null}
        </div>
        <div className="min-w-0 px-1 pb-0.5">
          <div className="line-clamp-2 text-[0.9rem] font-semibold break-words text-ink">{name}</div>
          <div className="mt-0.5 flex min-w-0 items-center gap-1.5 text-[0.75rem] text-ink-3">
            {/* The project only in All: inside one, every card would say the same. */}
            <span className="truncate" title={projectName ? `In ${projectName}` : undefined}>
              {KIND_LABEL[kind] || "Asset"} · {projectName ? `${projectName} · ` : ""}
              {relativeTime(a.mtime)}
            </span>
          </div>
        </div>
      </button>
      {x.degraded ? (
        <p className="m-0 line-clamp-2 px-1 text-[0.72rem] text-ink-3" title={x.degraded}>
          {x.degraded}
        </p>
      ) : null}
      <MoveMenu
        stem={a.id}
        current={a.project ?? null}
        projects={projects}
        align="end"
        trigger={
          <Button
            variant="hud-icon"
            size="icon-sm"
            aria-label={`Move ${name} to a project`}
            title="Move to project"
            className={cn(TILE_ACTION, "right-11")}
          >
            <FolderInput className="size-3.5 text-ink-3" />
          </Button>
        }
      />
      <Button
        variant="hud-icon"
        size="icon-sm"
        aria-label={`Delete ${name}`}
        title={`Delete out/${a.id}/ from this machine`}
        onClick={onDelete}
        className={cn(TILE_ACTION, "right-3")}
      >
        <Trash2 className="size-3.5 text-ink-3" />
      </Button>
    </li>
  );
}

/** The card's corner buttons: shown on hover or focus (always on touch), and while their menu is open. */
const TILE_ACTION =
  "hit-44 absolute! top-3 opacity-0 group-focus-within/tile:opacity-100 group-hover/tile:opacity-100 focus-visible:opacity-100 data-popup-open:opacity-100 [@media(hover:none)]:opacity-100";

function Placeholder() {
  return (
    <svg width="100%" height="100%" viewBox="0 0 120 90" fill="none">
      <ellipse cx="60" cy="74" rx="40" ry="10" className="fill-ground-ellipse" />
      <ellipse cx="60" cy="72" rx="22" ry="5" className="fill-clay-road" />
      <path d="M60 26 76 35v18L60 62 44 53V35z" className="fill-clay-wall stroke-[var(--mokei-detail-light)]" strokeWidth="1" />
      <path d="M44 35 60 44l16-9M60 44v18" className="stroke-[var(--mokei-detail-light)]" strokeWidth="1" />
      <path d="M60 26 76 35 60 44 44 35z" className="fill-primary" opacity="0.9" />
    </svg>
  );
}
