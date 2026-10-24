// The project menus: the Library's switcher, Make's "Into" picker and the Move menu (a library card, the viewer's
// asset card). One base-ui menu each, on the popover white with the glass edge, so they scale to ~15 projects (the
// popup scrolls) where hud tabs would wrap.
import type { ReactElement } from "react";
import { FolderPlus, Pencil, Trash2 } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { Project } from "@/lib/api";
import { UNSORTED } from "@/app/router";
import { deleteProjectAsk, fileAsset, newProject, renameProjectAsk } from "@/lib/projects";

/** w-auto: the default is the trigger's width, too narrow under a small ▾ chip. */
const POPUP = "w-auto min-w-52 max-w-72 rounded-xl border border-[var(--glass-border)] p-1 shadow-[var(--glass-shadow)] ring-0";
const ITEM = "min-h-9 rounded-lg px-2 text-[0.85rem]";
// No project is "" in the radio group: a slug is never empty.
const NONE = "";

type Side = "bottom" | "top";

/** Move one asset: each project, No project, New project… (creates it with the asset in). */
export function MoveMenu({ stem, current, projects, trigger, side = "bottom", align = "start" }: {
  stem: string;
  current: string | null;
  projects: readonly Project[];
  trigger: ReactElement;
  side?: Side;
  align?: "start" | "end";
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={trigger} />
      <DropdownMenuContent side={side} align={align} className={POPUP}>
        <DropdownMenuRadioGroup value={current ?? NONE} onValueChange={(v) => void fileAsset(stem, (v as string) || null, current)}>
          <DropdownMenuLabel>Move to project</DropdownMenuLabel>
          {projects.map((p) => (
            <DropdownMenuRadioItem key={p.id} value={p.id} closeOnClick className={ITEM}>
              <span className="truncate">{p.name}</span>
            </DropdownMenuRadioItem>
          ))}
          <DropdownMenuRadioItem value={NONE} closeOnClick className={ITEM}>
            No project
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuItem className={ITEM} onClick={() => void newProject([stem])}>
          <FolderPlus className="text-ink-3" />
          New project…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The Library's switcher: All / each project / Unsorted with their counts, then New, Rename and Delete. */
export function ProjectSwitcher({ value, projects, counts, onPick, trigger }: {
  /** The route's ?project= (undefined is All). */
  value: string | undefined;
  projects: readonly Project[];
  counts: { all: number; unsorted: number; of: (id: string) => number };
  onPick: (id: string | undefined) => void;
  trigger: ReactElement;
}) {
  const open = projects.find((p) => p.id === value);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={trigger} />
      <DropdownMenuContent className={POPUP}>
        <DropdownMenuRadioGroup value={value ?? NONE} onValueChange={(v) => onPick((v as string) || undefined)}>
          <DropdownMenuRadioItem value={NONE} closeOnClick className={ITEM}>
            All projects
            <DropdownMenuShortcut className="tracking-normal tabular-nums">{counts.all}</DropdownMenuShortcut>
          </DropdownMenuRadioItem>
          {projects.length ? <DropdownMenuSeparator /> : null}
          {projects.map((p) => (
            <DropdownMenuRadioItem key={p.id} value={p.id} closeOnClick className={ITEM}>
              <span className="truncate">{p.name}</span>
              <DropdownMenuShortcut className="tracking-normal tabular-nums">{counts.of(p.id)}</DropdownMenuShortcut>
            </DropdownMenuRadioItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuRadioItem value={UNSORTED} closeOnClick className={ITEM}>
            Unsorted
            <DropdownMenuShortcut className="tracking-normal tabular-nums">{counts.unsorted}</DropdownMenuShortcut>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuItem className={ITEM} onClick={() => void newProject().then((p) => p && onPick(p.id))}>
            <FolderPlus className="text-ink-3" />
            New project…
          </DropdownMenuItem>
          {open ? (
            <>
              <DropdownMenuItem className={ITEM} onClick={() => void renameProjectAsk(open)}>
                <Pencil className="text-ink-3" />
                Rename project…
              </DropdownMenuItem>
              <DropdownMenuItem variant="destructive" className={ITEM} onClick={() => void deleteProjectAsk(open).then((gone) => gone && onPick(undefined))}>
                <Trash2 />
                Delete project…
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** Make's "Into": which project new assets are filed in (the route's ?project=); No project; New project…. */
export function IntoMenu({ value, projects, onPick, trigger }: {
  value: string | null;
  projects: readonly Project[];
  onPick: (id: string | undefined) => void;
  trigger: ReactElement;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={trigger} />
      <DropdownMenuContent align="center" className={POPUP}>
        <DropdownMenuRadioGroup value={value ?? NONE} onValueChange={(v) => onPick((v as string) || undefined)}>
          <DropdownMenuLabel>New assets go into</DropdownMenuLabel>
          {projects.map((p) => (
            <DropdownMenuRadioItem key={p.id} value={p.id} closeOnClick className={ITEM}>
              <span className="truncate">{p.name}</span>
            </DropdownMenuRadioItem>
          ))}
          <DropdownMenuRadioItem value={NONE} closeOnClick className={ITEM}>
            No project
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuItem className={ITEM} onClick={() => void newProject().then((p) => p && onPick(p.id))}>
          <FolderPlus className="text-ink-3" />
          New project…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
