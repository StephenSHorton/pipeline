// Make's building modes, under the Sprite stack kind: Building (three drawn walls, POST /2d/building, a factory job
// with its stages on the job card; it opens on Stack when done) and Join buildings (buildings already made, joined
// into a row or a corner, POST /2d/join; no model, so it answers at once and opens the joined one on Stack).
// Mokei parts as the Make panel uses them: the glass card, the segmented tabs, outline buttons.
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { ArrowDown, ArrowUp, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { factoryJson, listAssets, outUrl, type Asset } from "@/lib/api";
import { displayStem, onJobEnded, useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { openAsset } from "@/app/open";
import { startHomeJob } from "./chain";
import { freshStem } from "./make";
import {
  BACK_PLACEHOLDER,
  BAY_CHOICES,
  buildingBody,
  buildingChips,
  buildingName,
  buildingProblem,
  DEFAULT_BUILDING,
  FRONT_PLACEHOLDER,
  joinable,
  joinBody,
  joinedText,
  joinName,
  joinProblem,
  LAYOUTS,
  LOOKS,
  MAX_PARTS,
  RIDGES,
  ROOF_COLOURS,
  ROOFS,
  WALL_PLACEHOLDER,
  type BuildingChoice,
  type Layout,
} from "./building";

const CARD =
  "rounded-[20px] border border-[var(--glass-border)] bg-[var(--glass-bg)] p-3 shadow-[var(--glass-shadow)] [-webkit-backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))] [backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))]";
const FIELD =
  "block w-full resize-y rounded-xl border border-input bg-white px-3 py-2 text-[0.9rem] leading-snug text-ink outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";
const LABEL = "mb-1 block text-[0.78rem] font-semibold text-ink-2";
const SELECT =
  "hit-44 h-9 cursor-pointer rounded-[10px] border border-input bg-white px-2 text-[0.82rem] text-ink tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring";
const SEG_TAB = "min-h-9 px-3 text-[0.8rem]";

function Seg<T extends string>({ label, value, items, onPick }: { label: string; value: T; items: readonly { id: T; label: string; hint: string }[]; onPick: (v: T) => void }) {
  return (
    <div>
      <span className={LABEL}>{label}</span>
      <Tabs value={value} onValueChange={(v) => typeof v === "string" && onPick(v as T)}>
        <TabsList aria-label={label} className="h-auto">
          {items.map((it) => (
            <TabsTrigger key={it.id} value={it.id} title={it.hint} className={SEG_TAB}>
              {it.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
    </div>
  );
}

function Bays({ label, hint, value, onPick }: { label: string; hint: string; value: number | null; onPick: (v: number | null) => void }) {
  return (
    <label title={hint}>
      <span className={LABEL}>{label}</span>
      <select value={value ?? ""} onChange={(e) => onPick(e.target.value === "" ? null : Number(e.target.value))} className={SELECT}>
        <option value="">As drawn</option>
        {BAY_CHOICES.map((n) => (
          <option key={n} value={n}>
            {n} window bay{n === 1 ? "" : "s"}
          </option>
        ))}
      </select>
    </label>
  );
}

function Line({ line }: { line: { text: string; state: "" | "running" | "error" } | null }) {
  if (!line) return null;
  return (
    <p role={line.state === "error" ? "alert" : "status"} className={cn("m-0 text-center text-[0.82rem]", line.state === "error" ? "font-medium text-danger-text" : "text-ink-3")}>
      {line.text}
    </p>
  );
}

function Go({ ready, busy, children }: { ready: boolean; busy: boolean; children: ReactNode }) {
  return (
    <Button
      type="submit"
      aria-disabled={!ready}
      className={cn(
        "h-auto min-h-11 gap-2 rounded-xl px-[1.1rem] text-[0.9rem] font-semibold",
        ready ? "shadow-[0_1px_2px_rgba(37,99,235,0.25),0_6px_16px_rgba(37,99,235,0.2)] hover:bg-blue-deep" : cn("opacity-50 hover:bg-primary", busy ? "cursor-wait" : "cursor-not-allowed"),
      )}
    >
      {children}
      <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 12h14M13 6l6 6-6 6" />
      </svg>
    </Button>
  );
}

/* ---------------------------------------------------------------- Building */

/** project: the id new assets are filed into (null: none). */
export function BuildingForm({ project }: { project: string | null }) {
  const [c, setC] = useState<BuildingChoice>(DEFAULT_BUILDING);
  const [line, setLine] = useState<{ text: string; state: "" | "running" | "error" } | null>(null);
  const [starting, setStarting] = useState(false);
  const busy = useJobs((s) => s.busy) || starting;
  const set = (p: Partial<BuildingChoice>) => {
    setC((cur) => ({ ...cur, ...p }));
    setLine(null);
  };
  const problem = buildingProblem(c);
  const ready = !problem && !busy;
  const custom = !ROOF_COLOURS.some((r) => r.id === c.roofColour);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (problem) {
      setLine({ text: problem, state: "error" });
      return;
    }
    setStarting(true);
    try {
      let stem: string;
      try {
        stem = await freshStem(buildingName(c));
      } catch (err) {
        setLine({ text: `Couldn't check the names already taken: ${err instanceof Error ? err.message : err}`, state: "error" });
        return;
      }
      void startHomeJob(buildingBody(c, stem, project), { path: "/2d/building", kind: "building", title: `Making the building ${displayStem(stem)}` });
    } finally {
      setStarting(false);
    }
  };

  return (
    <form onSubmit={(e) => void submit(e)} aria-label="Make a building" className="flex flex-col gap-3">
      <div className={cn(CARD, "flex flex-col gap-3")}>
        <label>
          <span className={LABEL}>Plain wall</span>
          <textarea
            rows={2}
            value={c.wall}
            onChange={(e) => set({ wall: e.target.value })}
            placeholder={WALL_PLACEHOLDER}
            title="Drawn first, with no door or shop front: it is every side of the building."
            spellCheck={false}
            className={FIELD}
          />
        </label>
        <label>
          <span className={LABEL}>Front: what to add</span>
          <textarea rows={2} value={c.front} onChange={(e) => set({ front: e.target.value })} placeholder={FRONT_PLACEHOLDER} title="Added to a copy of the plain wall. Say awning to get one." spellCheck={false} className={FIELD} />
        </label>
        <label>
          <span className={LABEL}>
            Back: what to add <span className="font-normal text-ink-3">(optional)</span>
          </span>
          <textarea rows={2} value={c.back} onChange={(e) => set({ back: e.target.value })} placeholder={BACK_PLACEHOLDER} title="Added to another copy. Left empty, the back is the plain wall." spellCheck={false} className={FIELD} />
        </label>
        <div className="flex items-end justify-between gap-2">
          <label className="min-w-0 flex-1">
            <span className={LABEL}>Name</span>
            <Input value={c.name} onChange={(e) => set({ name: e.target.value })} placeholder={displayStem(buildingName(c))} spellCheck={false} className="h-9 rounded-[10px] bg-white text-[0.85rem]" />
          </label>
          <Go ready={ready} busy={busy}>
            Make building
          </Go>
        </div>
      </div>

      <div className={cn(CARD, "flex flex-wrap items-end gap-x-5 gap-y-3")}>
        <Seg label="Look" value={c.look} items={LOOKS} onPick={(look) => set({ look })} />
        <Bays label="Wide" hint="Window bays across the front. As drawn keeps the plain wall's." value={c.wide} onPick={(wide) => set({ wide })} />
        <Bays label="Deep" hint="Window bays along the side. As drawn keeps the plain wall's." value={c.deep} onPick={(deep) => set({ deep })} />
        <Seg label="Roof" value={c.roof} items={ROOFS} onPick={(roof) => set({ roof })} />
        {c.roof === "gable" ? (
          <>
            <Seg label="Ridge" value={c.ridge} items={RIDGES} onPick={(ridge) => set({ ridge })} />
            <div>
              <span className={LABEL}>Roof colour</span>
              <div role="radiogroup" aria-label="Roof colour" className="flex items-center gap-1.5">
                {ROOF_COLOURS.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    role="radio"
                    aria-checked={c.roofColour === r.id}
                    title={r.label}
                    onClick={() => set({ roofColour: r.id })}
                    className="hit-44 size-8 cursor-pointer rounded-[9px] border-2 border-white shadow-[0_0_0_1px_var(--color-input)] outline-none focus-visible:ring-2 focus-visible:ring-ring aria-checked:shadow-[0_0_0_2px_var(--color-primary)]"
                    style={{ background: r.hex }}
                  >
                    <span className="sr-only">{r.label}</span>
                  </button>
                ))}
                <label title="Any colour" className={cn("hit-44 relative size-8 cursor-pointer overflow-hidden rounded-[9px] border-2 border-white", custom ? "shadow-[0_0_0_2px_var(--color-primary)]" : "shadow-[0_0_0_1px_var(--color-input)]")} style={{ background: custom ? c.roofColour : "conic-gradient(#e66,#ec6,#6c6,#6ce,#66e,#e6e,#e66)" }}>
                  <span className="sr-only">Another colour</span>
                  <input type="color" value={custom ? c.roofColour : "#8a4a34"} onChange={(e) => set({ roofColour: e.target.value })} className="absolute inset-0 cursor-pointer opacity-0" />
                </label>
              </div>
            </div>
          </>
        ) : null}
      </div>
      <p className="m-0 text-center text-[0.82rem] text-ink-3">{buildingChips(c).join(" · ")}. The image model draws the plain wall, then the edit model adds the front and back: about 8 minutes with both models loading.</p>
      <Line line={line} />
    </form>
  );
}

/* ---------------------------------------------------------------- Join buildings */

export function JoinForm({ project }: { project: string | null }) {
  const [layout, setLayout] = useState<Layout>("row");
  const [parts, setParts] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [assets, setAssets] = useState<Asset[] | null>(null);
  const [line, setLine] = useState<{ text: string; state: "" | "running" | "error" } | null>(null);
  const [joining, setJoining] = useState(false);
  const busy = useJobs((s) => s.busy) || joining;

  useEffect(() => {
    let live = true;
    const load = () =>
      listAssets()
        .then((r) => live && setAssets(joinable(r.assets)))
        .catch((err) => live && (setAssets([]), setLine({ text: `Couldn't list the buildings: ${err instanceof Error ? err.message : err}`, state: "error" })));
    void load();
    const off = onJobEnded(() => void load());
    return () => {
      live = false;
      off();
    };
  }, []);

  const pickLayout = (l: Layout) => {
    setLayout(l);
    setLine(null);
    if (l === "corner") setParts((p) => p.slice(0, 2));
  };
  const add = (stem: string) => {
    setParts((p) => (p.length >= (layout === "corner" ? 2 : MAX_PARTS) ? p : [...p, stem]));
    setLine(null);
  };
  const move = (i: number, by: number) =>
    setParts((p) => {
      const j = i + by;
      if (j < 0 || j >= p.length) return p;
      const n = [...p];
      [n[i], n[j]] = [n[j], n[i]];
      return n;
    });
  const drop = (i: number) => setParts((p) => p.filter((_, k) => k !== i));

  const problem = joinProblem(layout, parts);
  const ready = !problem && !busy;
  const full = parts.length >= (layout === "corner" ? 2 : MAX_PARTS);
  const role = (i: number) => (layout === "corner" ? (i === 0 ? "Main building" : "Wing, turned to face right") : i === 0 ? "Left end" : i === parts.length - 1 ? "Right end" : `${i + 1}`);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (problem) {
      setLine({ text: problem, state: "error" });
      return;
    }
    setJoining(true);
    setLine({ text: "Joining…", state: "running" });
    try {
      const stem = await freshStem(joinName(name, layout, parts));
      const out = await factoryJson<{ size?: { width: number; length: number; height: number }; parts?: unknown[] }>("/2d/join", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(joinBody(stem, layout, parts, project)),
      });
      setLine(null);
      setParts([]);
      setName("");
      toast(joinedText(displayStem(stem), out), "Export →", () => openAsset(stem, "export"));
      openAsset(stem, "stack");
    } catch (err) {
      setLine({ text: err instanceof Error ? err.message : String(err), state: "error" });
    } finally {
      setJoining(false);
    }
  };

  const thumb = (stem: string) => {
    const a = assets?.find((x) => x.id === stem);
    const shot = a?.preview;
    return shot ? <img src={outUrl(stem, shot, a.mtime)} alt="" className="size-10 shrink-0 rounded-[9px] bg-input object-contain" /> : <span className="size-10 shrink-0 rounded-[9px] bg-input" />;
  };

  return (
    <form onSubmit={(e) => void submit(e)} aria-label="Join buildings" className="flex flex-col gap-3">
      <div className={cn(CARD, "flex flex-col gap-3")}>
        <Seg label="Layout" value={layout} items={LAYOUTS} onPick={pickLayout} />
        <div>
          <span className={LABEL}>{layout === "corner" ? "The main building, then its wing" : `Left to right (2 to ${MAX_PARTS})`}</span>
          {parts.length ? (
            <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
              {parts.map((stem, i) => (
                <li key={`${stem}-${i}`} className="flex items-center gap-2 rounded-xl bg-muted p-[0.3rem] pr-1">
                  {thumb(stem)}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[0.85rem] font-medium text-ink">{displayStem(stem)}</span>
                    <span className="block text-[0.72rem] text-ink-3">{role(i)}</span>
                  </span>
                  <Button type="button" variant="ghost" size="icon" aria-label={`Move ${displayStem(stem)} ${layout === "row" ? "left" : "up"}`} disabled={i === 0} onClick={() => move(i, -1)} className="hit-44 size-8 text-ink-3">
                    <ArrowUp aria-hidden="true" className="size-4" />
                  </Button>
                  <Button type="button" variant="ghost" size="icon" aria-label={`Move ${displayStem(stem)} ${layout === "row" ? "right" : "down"}`} disabled={i === parts.length - 1} onClick={() => move(i, 1)} className="hit-44 size-8 text-ink-3">
                    <ArrowDown aria-hidden="true" className="size-4" />
                  </Button>
                  <Button type="button" variant="ghost" size="icon" aria-label={`Take ${displayStem(stem)} out`} onClick={() => drop(i)} className="hit-44 size-8 text-ink-3">
                    <X aria-hidden="true" className="size-4" />
                  </Button>
                </li>
              ))}
            </ol>
          ) : (
            <p className="m-0 text-[0.82rem] text-ink-3">Pick buildings below.</p>
          )}
        </div>
        <div className="flex items-end justify-between gap-2">
          <label className="min-w-0 flex-1">
            <span className={LABEL}>Name</span>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={displayStem(joinName("", layout, parts))} spellCheck={false} className="h-9 rounded-[10px] bg-white text-[0.85rem]" />
          </label>
          <Go ready={ready} busy={busy}>
            Join
          </Go>
        </div>
      </div>

      <div className={CARD}>
        <span className={LABEL}>Your buildings</span>
        {assets === null ? (
          <p className="m-0 text-[0.82rem] text-ink-3">Loading…</p>
        ) : assets.length ? (
          <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(8.5rem,1fr))] gap-2 p-0">
            {assets.map((a) => (
              <li key={a.id}>
                <Button
                  type="button"
                  variant="outline"
                  disabled={full}
                  title={full ? (layout === "corner" ? "A corner is two buildings" : `At most ${MAX_PARTS}`) : `Add ${displayStem(a.id)}`}
                  onClick={() => add(a.id)}
                  className="h-auto w-full flex-col items-center gap-1 rounded-xl bg-white p-2 text-[0.78rem] font-medium text-ink-2"
                >
                  {a.preview ? <img src={outUrl(a.id, a.preview, a.mtime)} alt="" className="h-20 w-full rounded-lg object-contain" /> : <span className="h-20 w-full rounded-lg bg-input" />}
                  <span className="w-full truncate">{displayStem(a.id)}</span>
                </Button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="m-0 text-[0.82rem] text-ink-3">No buildings yet. Make one with Building first.</p>
        )}
      </div>
      <Line line={line} />
    </form>
  );
}
