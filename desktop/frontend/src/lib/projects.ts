// Projects group assets like folders (Stephen: grouping only, no style or palette UI): an asset is in one project or
// none ("Unsorted"). One cached GET /projects for the Library switcher, Make's "Into" picker and the viewer's asset
// card, refreshed after every change here and when a job ends (a filed stem shows once its folder exists).
import { useEffect } from "react";
import { create } from "zustand";
import { UNSORTED } from "@/app/router";
import { createProject, deleteProject, listProjects, moveAsset, renameProject, type Asset, type Project } from "./api";
import { onJobEnded } from "./jobs";
import { askConfirm, askText, toast } from "./ui";

interface ProjectsStore {
  /** null until the first answer. */
  projects: Project[] | null;
  error: string;
  refresh: () => Promise<void>;
}

let seq = 0;

export const useProjects = create<ProjectsStore>((set) => ({
  projects: null,
  error: "",
  refresh: async () => {
    const mine = ++seq;
    try {
      const r = await listProjects();
      if (mine === seq) set({ projects: r.projects || [], error: "" });
    } catch (err) {
      // An older factory (or host) without /projects: no projects, and the Library still renders.
      if (mine === seq) set((s) => ({ projects: s.projects || [], error: err instanceof Error ? err.message : String(err) }));
    }
  },
}));

export const refreshProjects = () => useProjects.getState().refresh();

onJobEnded(() => void refreshProjects());

/** The cached list, asked again on mount (not at import: the tests import this module). */
export function useProjectList(): Project[] | null {
  useEffect(() => void refreshProjects(), []);
  return useProjects((s) => s.projects);
}

/* ---------------------------------------------------------------- pure helpers */

/** stem → project id, one pass over every project. */
export function membership(projects: readonly Project[]): Map<string, string> {
  const m = new Map<string, string>();
  for (const p of projects) for (const stem of p.assets) m.set(stem, p.id);
  return m;
}

/** The project an asset row is in: the projects list when it has loaded (it is what moves update), else the row's
 * own field. null is Unsorted. */
export function assetProject(a: Asset, members: Map<string, string> | null): string | null {
  return members ? (members.get(a.id) ?? null) : (a.project ?? null);
}

export function projectOf(projects: readonly Project[] | null, stem: string): Project | null {
  return projects?.find((p) => p.assets.includes(stem)) ?? null;
}

/** The route's ?project= as a project to file into: Unsorted (from the Library's filter) and a gone id mean none. */
export function intoProject(projects: readonly Project[] | null, id: string | undefined): Project | null {
  if (!id || id === UNSORTED) return null;
  return projects?.find((p) => p.id === id) ?? null;
}

export function projectFilterName(projects: readonly Project[] | null, id: string | undefined): string {
  if (!id) return "All projects";
  if (id === UNSORTED) return "Unsorted";
  return projects?.find((p) => p.id === id)?.name ?? id;
}

export const DELETE_PROJECT_MESSAGE = (name: string, n: number) =>
  `Delete the project "${name}"?\n\n${n === 0 ? "It has no assets." : `Its ${n === 1 ? "asset stays" : `${n} assets stay`} on disk and ${n === 1 ? "becomes" : "become"} Unsorted.`} No asset files are touched.`;

/* ---------------------------------------------------------------- actions (each refreshes the list) */

const say = (err: unknown) => toast(err instanceof Error ? err.message : String(err));

/** File a stem into `to` (null: no project). Toasts a failure; resolves whether it moved. */
export async function fileAsset(stem: string, to: string | null, from: string | null): Promise<boolean> {
  if (to === from) return true;
  try {
    await moveAsset(stem, to, from);
    return true;
  } catch (err) {
    say(err);
    return false;
  } finally {
    await refreshProjects();
  }
}

/** "New project…": ask the name, create it with `assets` moved in. null when cancelled or refused. */
export async function newProject(assets?: string[]): Promise<Project | null> {
  const name = await askText("New project", { placeholder: "MLG Farming", confirmLabel: "Create" });
  if (!name) return null;
  try {
    return (await createProject(name, assets)).project;
  } catch (err) {
    say(err);
    return null;
  } finally {
    await refreshProjects();
  }
}

/** Renames the display name only; the id (and so the URL) stays. */
export async function renameProjectAsk(p: Project): Promise<void> {
  const name = await askText("Rename project", { initial: p.name, confirmLabel: "Rename" });
  if (!name || name === p.name) return;
  try {
    await renameProject(p.id, name);
  } catch (err) {
    say(err);
  }
  await refreshProjects();
}

/** Resolves whether it was deleted. The assets stay, Unsorted. */
export async function deleteProjectAsk(p: Project): Promise<boolean> {
  if (!(await askConfirm(DELETE_PROJECT_MESSAGE(p.name, p.assets.length)))) return false;
  try {
    await deleteProject(p.id);
    return true;
  } catch (err) {
    say(err);
    return false;
  } finally {
    await refreshProjects();
  }
}
