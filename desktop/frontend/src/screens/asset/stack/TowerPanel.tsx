// The Stack stage's Tower panel (pipeline#20): a building drawn from walls stepped up into a tower with setbacks,
// from two numbers, how many tiers and floors in each. Each tier is one window bay narrower on every side than the
// one under it, from the building's own bays; the top tier takes the building's roof. POST /2d/tower writes a new
// building (no model, so it answers at once), saved as a Stack version, and opens it on Stack.
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { factoryJson, listAssets } from "@/lib/api";
import { displayStem, useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { openAsset } from "@/app/open";
import { COMMIT } from "@/screens/viewer/classes";
import {
  DEFAULT_TIER_FLOORS,
  DEFAULT_TIERS,
  FLOOR_CHOICES,
  sourceBays,
  TIER_CHOICES,
  tiersText,
  towerBody,
  toweredText,
  towerName,
  towerProblem,
  type TowerSource,
} from "@/screens/home/building";

const SELECT =
  "hit-44 h-8 cursor-pointer rounded-[9px] border border-input bg-white px-2 text-[0.82rem] text-ink tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring";
const LABEL = "w-28 shrink-0 text-[0.78rem] font-semibold text-ink-3";

export function TowerPanel({ stem, meta }: { stem: string; meta: TowerSource }) {
  const [tiers, setTiers] = useState(DEFAULT_TIERS);
  const [floors, setFloors] = useState(DEFAULT_TIER_FLOORS);
  const [making, setMaking] = useState(false);
  const [error, setError] = useState("");
  const busy = useJobs((s) => s.busy) || making;
  const bays = sourceBays(meta);
  const problem = towerProblem(bays, tiers, floors);
  const ready = !problem && !busy;

  const make = async () => {
    if (!ready) return;
    setMaking(true);
    setError("");
    try {
      // One listing: a name nobody uses yet, and the building's project, which the tower is filed in too.
      const list = await listAssets();
      const taken = new Set((list.assets || []).map((a) => a.id));
      const base = towerName(stem);
      let name = base;
      for (let n = 2; taken.has(name); n++) name = `${base}-${n}`;
      const project = list.assets?.find((a) => a.id === stem)?.project ?? null;
      const out = await factoryJson<{ size?: { width: number; length: number; height: number }; tiers?: unknown[] }>("/2d/tower", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(towerBody(name, stem, tiers, floors, project)),
      });
      toast(toweredText(displayStem(name), out), "Export →", () => openAsset(name, "export"));
      openAsset(name, "stack");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMaking(false);
    }
  };

  return (
    <>
      <p className="m-0 text-[0.8rem] leading-snug text-ink-3">
        Stack this building up into a tower: each tier one window bay narrower on every side than the one under it. The top tier takes the building's roof. It is made as a new building.
      </p>
      <label className="flex min-h-8 items-center gap-3">
        <span className={LABEL}>Tiers</span>
        <select aria-label="Tiers" value={tiers} onChange={(e) => (setTiers(Number(e.target.value)), setError(""))} className={SELECT}>
          {TIER_CHOICES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <label className="flex min-h-8 items-center gap-3">
        <span className={LABEL}>Floors in each tier</span>
        <select aria-label="Floors in each tier" value={floors} onChange={(e) => (setFloors(Number(e.target.value)), setError(""))} className={SELECT}>
          {FLOOR_CHOICES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
      {bays && !problem ? <p className="m-0 text-[0.8rem] text-ink-2">Tiers, bottom up: {tiersText(bays, tiers)}.</p> : null}
      {problem || error ? (
        <p role="alert" className="m-0 text-[0.8rem] font-medium text-danger-text">
          {problem || error}
        </p>
      ) : null}
      <Button
        type="button"
        variant="default"
        aria-disabled={!ready}
        onClick={() => void make()}
        className={cn(COMMIT, "self-start aria-disabled:opacity-50", busy && "cursor-wait")}
      >
        {making ? "Making the tower…" : "Make tower"}
      </Button>
    </>
  );
}
