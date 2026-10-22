// The Rig stage's controls and the Rig check. RigToggle is the dock's toggle row, and on Rig it is the whole legacy
// rig bar (renderFlow's rig branch, index.html 2914-2999): the status button that opens the check, Check this
// version, the built badge, the body plan, Advanced and Rig it / Re-rig. RigPanel is the check itself (legacy
// rig-inspect.js renderPanel, 878-946, with its clicks 970-1035, open/close 1049-1100 and index.html 3175-3196).
// Copy and rules live in ./rig.ts and ./inspect.ts, the model side in ./overlay.ts; this file only lays them out.
// The clip row of the legacy bar is the HUD's clip picker (Hud.tsx ClipPicker), not repeated here.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { LoaderCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { headOut, outUrl } from "@/lib/api";
import { flowEntry, pickTake } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { cn } from "@/lib/utils";
import { TOGGLE } from "@/screens/viewer/classes";
import type { PanelToggleProps } from "@/screens/viewer/panels/PanelToggle";
import { openViewerPanel } from "@/screens/viewer/panels/store";
import { ViewerPanel } from "@/screens/viewer/panels/ViewerPanel";
import type { ViewerPanelProps } from "@/screens/viewer/types";
import { pauseClipAt, restPose, resumeClip } from "@/viewer/clipSeek";
import { loadFirstGlb } from "@/viewer/loadGlb";
import { useViewerTools } from "@/viewer/tools";
import { checkSheet, loadRigCheck, loadRigPlans, setChoices, sheetFor, useRig } from "./api";
import { recheckRig, renderSheet, rigAsset } from "./actions";
import {
  boneCardRows,
  boneDepth,
  boneFound,
  bonesTitle,
  builtBadge,
  canShow,
  checkBody,
  compareView,
  fitRows,
  headBlock,
  HEAD_SEGMENTS,
  INSPECT_COPY,
  legendParts,
  MODES,
  momentTitle,
  MOTION_HELP,
  motionNotes,
  pausedText,
  planRows,
  rigChecks,
  roleHex,
  ROLE_WORDS,
  SHEET_COPY,
  SHEET_FILE,
  sheetView,
  sideOf,
  stepGroups,
  stickRoles,
  stretchOf,
  tidy,
  viewTitle,
  VIEWS,
  worstClip,
  type CardRow,
  type HeadMode,
  type PairCell,
  type RigInspectDoc,
  type RigMode,
  type ShownCheck,
} from "./inspect";
import { bonePalette, RigOverlay, type ViewArg } from "./overlay";
import {
  advancedSummary,
  choicesOf,
  hasTears,
  madeBadgeTitle,
  originalUrl,
  planOptions,
  recheckView,
  RIG_ADVANCED,
  RIG_COPY,
  rigButton,
  rigCompare,
  rigKeptHead,
  rigStatus,
  rigTakePicked,
  shownChoice,
  type AdvancedKey,
} from "./rig";

const LEVEL_DOT: Record<string, string> = { pass: "bg-success", warn: "bg-signal", fail: "bg-destructive" };
// The check chips: ✓ / ! / × before the label (legacy CSS, index.html 627-632); warn is the signal yellow, fail red.
const CHIP: Record<string, string> = {
  pass: "border-hair text-ink-3 before:text-success-text before:content-['✓_']",
  warn: "border-signal bg-signal/25 text-signal-foreground before:content-['!_']",
  fail: "border-destructive/40 bg-destructive/10 text-danger-text before:content-['×_']",
};
const TONE: Record<string, string> = { "": "", warn: "rounded-[4px] bg-signal px-1 text-signal-foreground", fail: "font-medium text-danger-text" };
const LABEL = "mt-1.5 text-[length:calc(12px*var(--fs))] font-semibold tracking-[0.08em] text-ink-3 uppercase";
const MUTE = "m-0 text-[length:calc(13px*var(--fs))] text-ink-3";
const SMALL = "hit-44 h-[27px] rounded-[8px] px-2.5 text-[length:calc(13px*var(--fs))] font-semibold";
const SELECT = "hit-44 h-[29px] max-w-full rounded-[8px] border border-input bg-white px-2 text-[length:calc(13px*var(--fs))] text-ink outline-none focus-visible:ring-2 focus-visible:ring-ring";

/** Whether the rigged model and the mesh exist for this asset (legacy entry.live / flowEntry("mesh").live). */
function useRigFlow(p: ViewerPanelProps) {
  return { rigged: !!flowEntry(p.flow, "rig")?.live, meshLive: !!flowEntry(p.flow, "mesh")?.live };
}

/* ------------------------------------------------------------------ the bar (dock toggle) */

export function RigToggle(p: Omit<PanelToggleProps, "panel"> & { props: ViewerPanelProps }) {
  const props = p.props;
  const { stem } = props;
  const { rigged, meshLive } = useRigFlow(props);
  const st = useRig();
  const choices = choicesOf(st.choices, stem);
  const check = st.check && st.check.stem === stem ? st.check : null;
  const doc = rigged && check ? check.doc : null;
  const made = rigged && check ? check.made : null;

  useEffect(() => {
    void loadRigPlans();
  }, []);

  if (!meshLive) return <span className={cn(MUTE, "px-2 py-1")}>{RIG_COPY.noMesh}</span>;
  const status = doc ? rigStatus(doc) : null;
  const recheck = recheckView({ rigged, loaded: !!check, doc, checking: st.checking === stem });
  const options = planOptions(st.plans);
  const shown = shownChoice(choices.choice, made, options);
  const btn = rigButton(rigged, hasTears(doc), rigged && rigKeptHead(doc?.rig?.rest));
  return (
    <div className="flex max-w-full flex-wrap items-center gap-1">
      {status ? (
        <Button
          ref={p.ref}
          type="button"
          variant="hud-ghost"
          aria-expanded={p.open}
          aria-controls={p.controls}
          title={status.title}
          onClick={p.onToggle}
          className={cn(TOGGLE, "h-auto min-h-[29px] max-w-full gap-1.5 text-left whitespace-normal aria-expanded:bg-accent aria-expanded:text-accent-foreground")}
        >
          <span aria-hidden className={cn("size-2 shrink-0 rounded-full", LEVEL_DOT[status.level] ?? "bg-signal")} />
          {status.label}
        </Button>
      ) : null}
      {recheck.kind === "checking" ? <span className={cn(MUTE, "px-2 whitespace-nowrap")}>{RIG_COPY.checking}</span> : null}
      {recheck.kind === "button" ? (
        <Button type="button" variant="outline" title={recheck.title} onClick={() => void recheckRig(stem)} className={SMALL}>
          {RIG_COPY.checkVersion}
        </Button>
      ) : null}
      {rigged && made ? (
        <Badge variant="success" size="hud-sm" title={madeBadgeTitle(made.plan)}>
          {RIG_COPY.madeBadge}
        </Badge>
      ) : null}
      <select aria-label="Body plan" title={RIG_COPY.planTitle} value={shown} onChange={(e) => setChoices(stem, { choice: e.target.value })} className={SELECT}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
      <details
        open={choices.advancedOpen}
        onToggle={(e) => {
          const open = (e.currentTarget as HTMLDetailsElement).open;
          if (open !== choices.advancedOpen) setChoices(stem, { advancedOpen: open });
        }}
        className="group/adv flex max-w-full flex-wrap items-center gap-1.5"
      >
        <summary title={RIG_COPY.advancedTitle} className={cn(TOGGLE, "inline-flex cursor-pointer list-none items-center rounded-[8px] [&::-webkit-details-marker]:hidden")}>
          {advancedSummary(choices.advanced)}
        </summary>
        <div className="flex flex-wrap gap-1.5 py-1">
          {RIG_ADVANCED.map(([k, label, opts, help]) => (
            <label key={k} title={help} className="inline-flex items-center gap-1 text-[length:calc(12px*var(--fs))] text-ink-3">
              <span>{label}</span>
              <select
                value={choices.advanced[k]}
                onChange={(e) => setChoices(stem, { advanced: { ...choices.advanced, [k as AdvancedKey]: e.target.value } })}
                className={SELECT}
              >
                {opts.map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </details>
      <Button type="button" variant={btn.primary ? "default" : "outline"} title={btn.title} onClick={() => rigAsset(stem)} className={SMALL}>
        {btn.label}
      </Button>
    </div>
  );
}

/* ------------------------------------------------------------------ the panel */

interface PanelState {
  mode: RigMode;
  bone: number;
  skel: boolean;
  filter: string;
  /** The check row shown in full. */
  open: string;
  head: HeadMode;
  paused: { clip: string; t: number } | null;
  tab: "checks" | "sheet";
  allClips: boolean;
  details: boolean;
}

const S0: PanelState = { mode: "bones", bone: -1, skel: true, filter: "", open: "", head: "rigged", paused: null, tab: "checks", allClips: false, details: false };

export function RigPanel(p: ViewerPanelProps) {
  const { stem, open, view, tools } = p;
  const { rigged } = useRigFlow(p);
  const st = useRig();
  const check = st.check && st.check.stem === stem ? st.check : null;
  const doc = rigged && check ? check.doc : null;
  const model = useViewerTools((s) => s.model);
  const [S, setS] = useState<PanelState>(S0);
  const sRef = useRef(S);
  sRef.current = S;
  const patch = (x: Partial<PanelState>) => setS((s) => ({ ...s, ...x }));
  const [overlay, setOverlay] = useState<RigOverlay | null>(null);
  const ovRef = useRef<RigOverlay | null>(null);
  ovRef.current = overlay;
  const panelEl = useRef<HTMLDivElement>(null);
  const planRef = useRef<HTMLDivElement>(null);
  const motionRef = useRef<HTMLDivElement>(null);
  const pendingShow = useRef<string | null>(null);
  const selfClip = useRef<string | null>(null);
  const wasSpin = useRef(false);

  // The checks for this version: on the stage, after a pick or a job (the flow and the docs re-read), and per asset.
  const flowVersion = p.flowVersion;
  const docs = p.docs;
  useEffect(() => {
    if (!rigged) return;
    void loadRigCheck(stem);
  }, [stem, rigged, flowVersion, docs]);

  // A check left to open on (the finish toast's Inspect, a recheck with the panel open): open once this asset's
  // checks are in (legacy rigFocus).
  const focus = st.focus;
  useEffect(() => {
    if (!focus || focus.stem !== stem || !check) return;
    useRig.setState({ focus: null });
    if (!check.doc) return;
    openViewerPanel("rig");
    if (focus.key) {
      patch({ tab: "checks", open: focus.key });
      pendingShow.current = focus.key;
    }
  }, [focus, stem, check]);

  // Opening: stop Spin (the hover card stays put) and start over; closing gives Spin back if it was on.
  const viewRef = useRef(view);
  viewRef.current = view;
  useEffect(() => {
    if (!open) return;
    wasSpin.current = viewRef.current.spin;
    if (viewRef.current.spin) viewRef.current.onSpin(false);
    setS((s) => ({ ...S0, tab: pendingShow.current ? "checks" : s.tab, open: pendingShow.current ?? "", details: s.details }));
    sheetFor(stem);
    return () => {
      pendingShow.current = null;
      if (wasSpin.current) viewRef.current.onSpin(true);
      wasSpin.current = false;
      // A held moment or the original model on screen: play on (legacy close() resumed).
      const s = sRef.current;
      if (s.paused || s.head !== "rigged") {
        viewRef.current.onPlaying(true);
        resumeClip(viewRef.current.clip);
      }
    };
  }, [open, stem]);

  // The model side: bound while open, to the rigged model on screen. Another file on screen (an LOD) loads the
  // rigged one first; a model that is not skinned has no colour modes or skeleton, and the panel still reads.
  const riggedFile = (check?.info?.rigged as string | undefined) || "rigged.glb";
  const triedLoad = useRef("");
  useEffect(() => {
    if (!open || !doc || !model) return;
    const root = tools.root();
    if (!root) return;
    if (!model.skinned) {
      const key = `${stem}|${riggedFile}`;
      if (triedLoad.current !== key) {
        triedLoad.current = key;
        tools.loadFile(riggedFile).catch((err: unknown) => toast(err instanceof Error ? err.message : String(err)));
      }
      return;
    }
    const ov = RigOverlay.bind(root, doc, tools);
    if (!ov) return;
    ov.panel = () => panelEl.current;
    ov.onPick = (i) => pick(i, false);
    setOverlay(ov);
    return () => {
      ov.dispose();
      setOverlay(null);
    };
    // pick reads the state through sRef; tools is the stable handle.
  }, [open, doc, model, tools, stem, riggedFile]);

  useEffect(() => {
    overlay?.set({ mode: S.mode, bone: S.bone, skel: S.skel });
  }, [overlay, S.mode, S.bone, S.skel]);

  // Clay and Wire swap the model's materials: the check closes, as legacy closed on Wire and Texture.
  const looks = `${view.clay}|${view.wire}`;
  const seenLooks = useRef(looks);
  useEffect(() => {
    if (seenLooks.current === looks) return;
    seenLooks.current = looks;
    if (open) p.onClose();
  }, [looks, open, p]);

  // A clip played from the HUD: back to the rig as it moves (legacy clipChanged).
  const seenClip = useRef(view.clip);
  useEffect(() => {
    if (seenClip.current === view.clip) return;
    seenClip.current = view.clip;
    if (selfClip.current === view.clip) {
      selfClip.current = null;
      return;
    }
    if (!open) return;
    ovRef.current?.clearGhost();
    patch({ paused: null, head: "rigged" });
  }, [view.clip, open]);

  /* ---------------------------------------------------------------- the model's hands (legacy hooks) */

  const playAt = (name: string, t: number) =>
    new Promise<void>((resolve) => {
      selfClip.current = name === view.clip ? null : name;
      view.onClip(name);
      view.onPlaying(false);
      // After the Animator has started the clip (its effects run on the next commit), hold it still.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          pauseClipAt(name, t);
          resolve();
        }),
      );
    });
  const resume = () => {
    view.onPlaying(true);
    requestAnimationFrame(() => resumeClip(view.clip));
  };
  const toRest = () => {
    view.onPlaying(false);
    restPose();
    overlay?.pose();
  };

  const headUI = useOriginal(p, doc);

  async function setHead(m: HeadMode): Promise<void> {
    const ov = ovRef.current;
    if (!ov || m === sRef.current.head) return;
    ov.clearGhost();
    patch({ head: m });
    if (m === "rigged") {
      resume();
      return;
    }
    toRest();
    patch({ paused: null });
    ov.setView("top");
    const url = headUI.url;
    const ok = await ov.setGhost(m, async () => (await loadFirstGlb([url])).gltf.scene);
    if (!ok && sRef.current.head === m) {
      patch({ head: "rigged" });
      resume();
    }
  }

  function frame(box: "all" | number, id: ViewArg) {
    const ov = ovRef.current;
    if (!ov) return;
    ov.frameBox(box === "all" ? ov.posedBox(-1) : ov.boneBox(box), id);
  }

  function focusBone(n: string | undefined, v: ViewArg, mode: RigMode = "heat"): Partial<PanelState> {
    const i = ovRef.current ? ovRef.current.names.indexOf(n || "") : -1;
    if (i < 0) return {};
    frame(i, v);
    return { bone: i, mode, skel: true };
  }

  async function atMoment(name: string, t: number, tear: boolean, bone?: string): Promise<Partial<PanelState>> {
    if (sRef.current.head !== "rigged") await setHead("rigged");
    await playAt(name, t || 0);
    const out: Partial<PanelState> = { paused: { clip: name, t: t || 0 } };
    const names = ovRef.current?.names ?? [];
    if (tear) {
      Object.assign(out, { mode: "split", skel: false });
      frame("all", "free");
    } else if (bone && names.includes(bone)) {
      // The stretched part, from the side, where legs and their bends show.
      const i = names.indexOf(bone);
      Object.assign(out, { bone: i, mode: "heat" });
      frame(i, sideOf(bone));
    }
    return out;
  }

  async function moment(name: string) {
    const c = doc?.motion?.clips?.[name];
    if (!c) return;
    patch(c.tears > 0 ? await atMoment(name, c.tear_t ?? c.worst_t ?? 0, true) : await atMoment(name, c.worst_t ?? 0, false, c.worst_bone));
  }

  // A head that wasn't straightened: the rig as painted, from the top, with the head bone lit against the spine.
  async function showTurn(): Promise<Partial<PanelState>> {
    if (sRef.current.head !== "rigged") await setHead("rigged");
    toRest();
    const out = { paused: null, ...focusBone("head", "top", "texture") };
    ovRef.current?.setView("top");
    return out;
  }

  async function worstClipMoment(c: ShownCheck, start = false): Promise<Partial<PanelState> | "motion"> {
    const name = worstClip(c);
    if (!name) return "motion";
    const m = doc?.motion?.clips?.[name];
    const out = await atMoment(name, start ? 0 : m?.worst_t || 0, false, start ? "" : m?.worst_bone);
    if (start) {
      Object.assign(out, { mode: "texture", skel: true });
      ovRef.current?.setView("side");
    }
    return out;
  }

  /** Show me: turn on the view that proves a check, or scroll to the part of the panel that does (legacy SHOW). */
  async function show(c: ShownCheck) {
    if (!doc) return;
    if (c.key !== "rest" && sRef.current.head !== "rigged") await setHead("rigged");
    const ov = ovRef.current;
    let to: Partial<PanelState> | "plan" | "motion" = {};
    switch (c.key) {
      case "planner":
        to = "plan";
        break;
      case "rest":
        if (doc.rig?.rest?.straightened) {
          patch({ mode: "texture" });
          void setHead("both");
        } else to = await showTurn();
        break;
      case "head":
        to = focusBone("head", "top");
        break;
      case "fit":
        to = focusBone(c.bone, sideOf(c.bone));
        break;
      case "moved":
      case "pose":
      case "pred":
        to = { skel: true };
        break;
      case "seam":
        ov?.setView("front");
        to = { mode: "split", skel: false };
        break;
      case "folded":
        to = { mode: "dropped" };
        break;
      case "motion":
        to = c.level === "fail" && c.clip ? await atMoment(c.clip, c.t ?? 0, true) : "motion";
        break;
      case "stretch": {
        const m = c.clip ? doc.motion?.clips?.[c.clip] : undefined;
        to = m && c.clip ? await atMoment(c.clip, m.worst_t ?? 0, false, m.worst_bone) : "motion";
        break;
      }
      case "far":
      case "floor":
      case "fbx":
        to = "motion";
        break;
      case "facing":
        ov?.setView("side");
        to = { skel: true };
        break;
      case "template":
        ov?.setView("side");
        to = { skel: true, mode: "bones" };
        break;
      case "built":
        ov?.setView("side");
        to = { skel: true, mode: "texture" };
        break;
      // The bone with the most far weight, or the sole's worst owner, as One bone.
      case "a2": {
        const n = ((c.worst || [])[0] || [])[0];
        to = n && ov?.names.includes(n) ? focusBone(n, sideOf(n)) : { mode: "bones" };
        break;
      }
      case "foot": {
        const n = ((c.owners || [])[0] || [])[0];
        if (n && ov?.names.includes(n)) to = focusBone(n, "side");
        else {
          ov?.setView("side");
          to = { mode: "bones" };
        }
        break;
      }
      case "parts":
        ov?.setView("side");
        to = { mode: "bones", skel: true };
        break;
      case "cshape":
      case "slide":
        to = await worstClipMoment(c);
        break;
      case "loop_seam":
        to = await worstClipMoment(c, true);
        break;
    }
    if (to === "plan") planRef.current?.scrollIntoView({ block: "nearest" });
    else if (to === "motion") motionRef.current?.scrollIntoView({ block: "nearest" });
    else patch(to);
  }

  // From the list a second click lets go of the bone; from the model it always picks.
  function pick(i: number, toggle: boolean) {
    const s = sRef.current;
    const bone = toggle && s.bone === i ? -1 : i;
    patch({ bone, ...(bone >= 0 && s.mode !== "bones" && s.mode !== "heat" ? { mode: "heat" as const } : {}) });
  }

  const names = overlay?.names ?? doc?.bones.map((b) => b.name) ?? [];
  const showable = (c: ShownCheck) => !!doc && !!overlay && canShow(c, { doc, names: overlay.names, headUI: !!headUI.url, dropped: !!overlay.dropped });

  // A finding the panel was opened on: its view, once the model side is bound.
  useEffect(() => {
    const key = pendingShow.current;
    if (!open || !doc || !key) return;
    const c = rigChecks(doc).find((x) => x.key === key);
    if (!c) {
      pendingShow.current = null;
      return;
    }
    if (!overlay) return; // the model side binds first (or the rigged model is still loading)
    pendingShow.current = null;
    if (showable(c)) void show(c);
  });

  // The Sheet tab looks for the contact sheet the first time it shows for this asset.
  useEffect(() => {
    if (!open || S.tab !== "sheet") return;
    const sh = useRig.getState().sheet;
    if (sh.stem !== stem || sh.ok === null) void checkSheet(stem);
  }, [open, S.tab, stem]);

  return (
    <ViewerPanel panel="rig" id={p.id} open={open} onClose={p.onClose} className="max-h-[calc(100dvh-210px)]">
      <div ref={panelEl} className="contents">
        {doc ? (
          <PanelBody
            p={p}
            doc={doc}
            S={S}
            patch={patch}
            overlay={overlay}
            names={names}
            headUI={headUI}
            showable={showable}
            onShow={(c) => void show(c)}
            onMoment={(n) => void moment(n)}
            onResume={() => {
              patch({ paused: null });
              resume();
            }}
            onHead={(m) => void setHead(m)}
            onView={(id) => ovRef.current?.setView(id)}
            onPick={(i) => pick(i, true)}
            planRef={planRef}
            motionRef={motionRef}
          />
        ) : null}
      </div>
    </ViewerPanel>
  );
}

/** The original model's URL when it can be shown (legacy originalModel + headUI), HEAD-checked once per URL. */
const originalSeen = new Map<string, Promise<boolean>>();
function useOriginal(p: ViewerPanelProps, doc: RigInspectDoc | null): { url: string } {
  const url = doc ? originalUrl(p.stem, doc.rig as never, rigTakePicked(flowEntry(p.flow, "rig")), flowEntry(p.flow, "mesh")?.pick) : "";
  const [ok, setOk] = useState<string>("");
  useEffect(() => {
    if (!url) return;
    let live = true;
    const rel = decodeURIComponent(url.replace(/^\/out\/[^/]+\//, ""));
    if (!originalSeen.has(url)) originalSeen.set(url, headOut(p.stem, rel));
    void originalSeen.get(url)!.then((yes) => live && setOk(yes ? url : ""));
    return () => {
      live = false;
    };
  }, [url, p.stem]);
  return { url: url && ok === url && doc?.rig?.rest?.measured ? url : "" };
}

/* ------------------------------------------------------------------ the panel's body (layout only) */

interface BodyProps {
  p: ViewerPanelProps;
  doc: RigInspectDoc;
  S: PanelState;
  patch: (x: Partial<PanelState>) => void;
  overlay: RigOverlay | null;
  names: string[];
  headUI: { url: string };
  showable: (c: ShownCheck) => boolean;
  onShow: (c: ShownCheck) => void;
  onMoment: (clip: string) => void;
  onResume: () => void;
  onHead: (m: HeadMode) => void;
  onView: (id: ViewArg) => void;
  onPick: (i: number) => void;
  planRef: React.RefObject<HTMLDivElement | null>;
  motionRef: React.RefObject<HTMLDivElement | null>;
}

function PanelBody(b: BodyProps) {
  const { doc, S, patch, p } = b;
  const badge = builtBadge(doc);
  return (
    <>
      {badge ? (
        <Badge variant="success" size="hud-sm" title={badge.title} className="h-auto min-h-[25px] whitespace-normal">
          {badge.text}
        </Badge>
      ) : null}
      <div role="group" aria-label={INSPECT_COPY.label} className="flex gap-0.5">
        <Seg on={S.tab !== "sheet"} title={INSPECT_COPY.checksTitle} onClick={() => patch({ tab: "checks" })}>
          {INSPECT_COPY.checks}
        </Seg>
        <Seg on={S.tab === "sheet"} title={INSPECT_COPY.sheetTitle} onClick={() => patch({ tab: "sheet" })}>
          {INSPECT_COPY.sheet}
        </Seg>
      </div>
      {S.tab === "sheet" ? <SheetTab stem={p.stem} /> : <ChecksTab {...b} />}
    </>
  );
}

function Seg({ on, title, onClick, children, disabled }: { on: boolean; title: string; onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <Button type="button" variant="hud-ghost" aria-pressed={on} title={title} onClick={onClick} disabled={disabled} className={cn(TOGGLE, "h-[27px] px-2.5 text-[length:calc(13px*var(--fs))]")}>
      {children}
    </Button>
  );
}

function ChecksTab(b: BodyProps) {
  const { doc, S, patch, overlay, p } = b;
  const groups = stepGroups(doc);
  const pal = useMemo(() => bonePalette(b.names.length).map((c) => `#${c.getHexString()}`), [b.names.length]);
  const nice = overlay?.nice ?? b.names;
  const legend = legendParts(S.mode, { bone: S.bone >= 0 ? nice[S.bone] || "" : "", doc, dropped: !!overlay?.dropped, splitPoints: overlay?.splitPoints ?? 0 });
  const compare = compareView(rigCompare(flowEntry(p.flow, "rig"), doc), doc, S.allClips);
  const st = useRig();
  const checking = st.checking === p.stem;
  const head = headBlock(doc.rig?.rest);
  const roles = stickRoles(overlay?.roles ?? doc.bones.map((x, i) => x.role || (doc.roles || [])[i] || ""));
  const clips = Object.entries(doc.motion?.clips || {});
  const notes = motionNotes(doc, overlay?.splitPoints ?? 0);
  const bone = S.bone >= 0 ? b.names[S.bone] : "";
  return (
    <>
      <h3 className={LABEL}>{INSPECT_COPY.howRigged}</h3>
      <div className="flex flex-col gap-2">
        {groups.map((g) => (
          <div key={g.id} className="flex flex-col gap-1">
            <div className="text-[length:calc(13px*var(--fs))] font-semibold text-ink">{g.title}</div>
            {g.checks.map((c) => (
              <CheckRow
                key={c.key}
                c={c}
                doc={doc}
                open={S.open === c.key}
                onToggle={() => patch({ open: S.open === c.key ? "" : c.key })}
                head={c.key === "rest" ? head : null}
                headUI={!!b.headUI.url}
                headMode={S.head}
                onHead={b.onHead}
                showable={b.showable(c)}
                onShow={() => b.onShow(c)}
                stem={p.stem}
              />
            ))}
          </div>
        ))}
      </div>

      <h3 className={LABEL} title={MOTION_HELP}>
        {INSPECT_COPY.inMotion}
      </h3>
      <div ref={b.motionRef} className="flex flex-col gap-1.5">
        {!clips.length ? (
          <>
            <p className={MUTE}>{INSPECT_COPY.notMeasured}</p>
            {checking ? (
              <p className={MUTE}>{INSPECT_COPY.checking}</p>
            ) : (
              <Button type="button" variant="outline" title={INSPECT_COPY.measureTitle} onClick={() => void recheckRig(p.stem)} className={cn(SMALL, "w-fit")}>
                {INSPECT_COPY.measure}
              </Button>
            )}
          </>
        ) : (
          <>
            <table className="w-full border-collapse text-[length:calc(13px*var(--fs))] tabular-nums">
              <thead>
                <tr title={MOTION_HELP} className="text-left text-ink-3">
                  <th className="font-medium">{INSPECT_COPY.clip}</th>
                  <th className="font-medium">{INSPECT_COPY.tears}</th>
                  <th className="font-medium">{INSPECT_COPY.stretch}</th>
                </tr>
              </thead>
              <tbody>
                {clips.map(([n, c]) => (
                  <tr key={n} className={cn(S.paused?.clip === n && "bg-accent/60")}>
                    <td>
                      <button type="button" title={momentTitle(n, c.tears)} onClick={() => b.onMoment(n)} className="hit-44 rounded-sm text-left text-primary hover:underline focus-visible:ring-2 focus-visible:ring-ring outline-none">
                        {tidy(n)}
                      </button>
                    </td>
                    <td className={cn(c.tears ? "font-semibold text-danger-text" : "")}>{c.tears}</td>
                    <td>{stretchOf(c).toFixed(1)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {S.paused ? (
              <div className={MUTE}>
                {pausedText(S.paused.clip, S.paused.t)}
                <button type="button" onClick={b.onResume} className="hit-44 font-medium text-primary hover:underline">
                  {INSPECT_COPY.play}
                </button>
              </div>
            ) : null}
            {notes.map((l) => (
              <p key={l} className={MUTE}>
                {l}
              </p>
            ))}
          </>
        )}
      </div>

      {compare ? (
        <>
          <h3 className={LABEL}>{compare.heading}</h3>
          {compare.before ? <p className={MUTE}>{compare.before}</p> : null}
          {compare.rows.length ? (
            <table className="w-full text-[length:calc(13px*var(--fs))] tabular-nums">
              <tbody>
                {compare.rows.map((r) => (
                  <tr key={r.label}>
                    <td className="pr-2 text-ink-3">{r.label}</td>
                    <td>{typeof r.value === "string" ? r.value : <Pair cell={r.value} />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {compare.more ? (
            <button type="button" onClick={() => patch({ allClips: !S.allClips })} className="hit-44 w-fit text-[length:calc(13px*var(--fs))] font-medium text-primary hover:underline">
              {compare.more}
            </button>
          ) : null}
          <Button
            type="button"
            variant="outline"
            onClick={() => pickTake(p.stem, "rig", compare.takeId).catch((err: unknown) => toast(err instanceof Error ? err.message : String(err)))}
            className={cn(SMALL, "w-fit")}
          >
            {compare.back}
          </Button>
        </>
      ) : null}

      <h3 className={LABEL}>{INSPECT_COPY.colour}</h3>
      <div className="flex flex-wrap gap-0.5">
        {MODES.map(([id, label, title]) => (
          <Seg key={id} on={S.mode === id} title={title} onClick={() => patch({ mode: id })} disabled={!overlay}>
            {label}
          </Seg>
        ))}
      </div>
      <p className={MUTE}>
        {legend.lead ? legend.leadTone === "bold" ? <b className="font-semibold text-ink">{legend.lead}</b> : <span className={TONE[legend.leadTone]}>{legend.lead}</span> : null}
        {legend.rest}
      </p>
      {legend.ramp ? <div aria-hidden className="h-2 rounded-sm bg-[linear-gradient(90deg,hsl(240,100%,50%),hsl(180,100%,50%),hsl(120,100%,50%),hsl(60,100%,50%),hsl(0,100%,50%))]" /> : null}

      <h3 className={LABEL}>{INSPECT_COPY.show}</h3>
      <div className="flex flex-wrap gap-0.5">
        <Seg on={S.skel} title={INSPECT_COPY.skeletonTitle} onClick={() => patch({ skel: !S.skel })} disabled={!overlay}>
          {INSPECT_COPY.skeleton}
        </Seg>
        {VIEWS.map(([id, label]) => (
          <Button key={id} type="button" variant="hud-ghost" title={viewTitle(label)} onClick={() => b.onView(id)} disabled={!overlay} className={cn(TOGGLE, "h-[27px] px-2.5 text-[length:calc(13px*var(--fs))]")}>
            {label}
          </Button>
        ))}
        {b.headUI.url && doc.rig?.rest?.straightened ? (
          <Seg on={S.head === "both"} title={INSPECT_COPY.originalTitle} onClick={() => b.onHead(S.head === "both" ? "rigged" : "both")}>
            {INSPECT_COPY.original}
          </Seg>
        ) : null}
      </div>

      <h3 className={LABEL}>{bonesTitle(b.names.length)}</h3>
      <input
        type="search"
        aria-label={INSPECT_COPY.findBone}
        placeholder={INSPECT_COPY.findBone}
        value={S.filter}
        spellCheck={false}
        onChange={(e) => patch({ filter: e.target.value.trim() })}
        className="h-8 w-full rounded-lg border border-input bg-white px-2.5 text-[length:calc(13px*var(--fs))] text-ink outline-none focus-visible:ring-2 focus-visible:ring-ring"
      />
      <div className="flex max-h-[220px] shrink-0 flex-col overflow-y-auto rounded-lg border border-hair">
        {b.names.map((n, i) => (
          <button
            key={n + i}
            type="button"
            title={n}
            hidden={!boneFound(S.filter, n, nice[i] || n)}
            onClick={() => b.onPick(i)}
            aria-pressed={i === S.bone}
            className="flex items-center gap-1.5 py-[3px] pr-1.5 text-left font-mono text-[11px] whitespace-nowrap text-ink hover:bg-muted aria-pressed:bg-accent aria-pressed:text-accent-foreground"
            style={{ paddingLeft: 6 + Math.min(boneDepth(doc.bones, i), 8) * 8 }}
          >
            <span aria-hidden className="inline-block size-2.5 shrink-0 rounded-[2px]" style={{ background: pal[i] }} />
            <span className="flex-1 truncate">{nice[i] || n}</span>
            <i className="not-italic text-ink-3">{overlay?.owned[i] || ""}</i>
          </button>
        ))}
      </div>
      {S.bone < 0 || !bone ? (
        <p className={MUTE}>{INSPECT_COPY.pickBone}</p>
      ) : (
        <div className="rounded-lg border border-hair px-2.5 py-2">
          <div title={bone} className="mb-1 flex items-center gap-1.5 text-[length:calc(13px*var(--fs))] font-semibold text-ink">
            <span aria-hidden className="inline-block size-2.5 rounded-[2px]" style={{ background: pal[S.bone] }} />
            {nice[S.bone]}
          </div>
          <Rows rows={boneCardRows({ name: bone, doc, owned: overlay?.owned[S.bone] ?? 0, clip: p.view.clip, nice: (raw) => nice[b.names.indexOf(raw)] || "" })} />
        </div>
      )}

      <h3 className={LABEL}>{INSPECT_COPY.bodyPlan}</h3>
      <div ref={b.planRef} className="flex flex-col gap-1.5">
        {doc.planner?.image ? <img alt="" src={outUrl(p.stem, doc.planner.image, false)} className="max-h-40 w-fit max-w-full rounded-md" /> : null}
        <Rows rows={planRows(doc)} />
      </div>

      <details open={S.details} onToggle={(e) => patch({ details: (e.currentTarget as HTMLDetailsElement).open })}>
        <summary className={cn(LABEL, "hit-44 cursor-pointer")}>{INSPECT_COPY.skeletonFit}</summary>
        <Rows rows={fitRows(doc)} extra={roles.length ? { at: 1, label: "Stick colours", node: <Sticks roles={roles} /> } : undefined} />
      </details>

      {(doc.notes || []).length ? (
        <>
          <h3 className={LABEL}>{INSPECT_COPY.riggerDid}</h3>
          <ul className="m-0 list-disc pl-4 text-[length:calc(13px*var(--fs))] text-ink-2">
            {doc.notes!.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

function Pair({ cell }: { cell: PairCell }) {
  const tone = (t: PairCell["aTone"]) => (t === "better" ? "font-semibold text-success-text" : t === "worse" ? TONE.warn : "");
  return (
    <>
      <span className={tone(cell.aTone)}>{cell.a}</span> → <span className={tone(cell.bTone)}>{cell.b}{cell.mark}</span>
    </>
  );
}

function Sticks({ roles }: { roles: string[] }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-0.5">
      {roles.map((r) => (
        <span key={r} className="inline-flex items-center gap-1">
          <span aria-hidden className="inline-block size-2.5 rounded-[2px]" style={{ background: roleHex(r) }} />
          {ROLE_WORDS[r] || r}
        </span>
      ))}
    </span>
  );
}

function Rows({ rows, extra }: { rows: CardRow[]; extra?: { at: number; label: string; node: ReactNode } }) {
  const items: ReactNode[] = rows.map((r) => (
    <tr key={r.label}>
      <td className="pr-2 align-top text-ink-3">{r.label}</td>
      <td className={cn("align-top", r.tone ? TONE[r.tone] : "")}>{r.value}</td>
    </tr>
  ));
  if (extra)
    items.splice(
      extra.at,
      0,
      <tr key={extra.label}>
        <td className="pr-2 align-top text-ink-3">{extra.label}</td>
        <td>{extra.node}</td>
      </tr>,
    );
  return (
    <table className="w-full text-[length:calc(13px*var(--fs))] tabular-nums">
      <tbody>{items}</tbody>
    </table>
  );
}

function CheckRow(o: {
  c: ShownCheck;
  doc: RigInspectDoc;
  open: boolean;
  onToggle: () => void;
  head: ReturnType<typeof headBlock>;
  headUI: boolean;
  headMode: HeadMode;
  onHead: (m: HeadMode) => void;
  showable: boolean;
  onShow: () => void;
  stem: string;
}) {
  const { c } = o;
  const body = o.open ? checkBody(c, o.doc) : null;
  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        aria-expanded={o.open}
        title={c.text}
        onClick={o.onToggle}
        className={cn("hit-44 rounded-lg border px-2 py-1 text-left text-[length:calc(13px*var(--fs))] leading-snug outline-none focus-visible:ring-2 focus-visible:ring-ring", CHIP[c.level] ?? CHIP.warn)}
      >
        {c.label}
      </button>
      {body ? (
        <div className="flex flex-col gap-1 pl-2 text-[length:calc(13px*var(--fs))] text-ink-2">
          {body.text ? <p className="m-0">{body.text}</p> : null}
          {body.why ? (
            <p className="m-0">
              <b className="font-semibold">{INSPECT_COPY.why}</b> {body.why}
            </p>
          ) : null}
          {body.did ? (
            <p className="m-0">
              <b className="font-semibold">{INSPECT_COPY.did}</b> {body.did}
            </p>
          ) : null}
          {body.fix ? (
            <p className="m-0">
              <b className="font-semibold">{INSPECT_COPY.fix}</b> {body.fix}
            </p>
          ) : null}
          {body.note ? <p className={MUTE}>{body.note}</p> : null}
          {o.head ? (
            <div className="flex flex-col gap-1">
              {o.head.segments && o.headUI ? (
                <div role="group" className="flex gap-0.5">
                  {HEAD_SEGMENTS.map(([id, l, t]) => (
                    <Seg key={id} on={o.headMode === id} title={t} onClick={() => o.onHead(id)}>
                      {l}
                    </Seg>
                  ))}
                </div>
              ) : null}
              <div>{o.head.read}</div>
              {o.head.kept ? <p className={MUTE}>{o.head.kept}</p> : null}
              {o.head.button ? (
                <Button type="button" variant="outline" title={o.head.button.title} onClick={() => rigAsset(o.stem, { straighten: o.head!.button!.straighten })} className={cn(SMALL, "w-fit")}>
                  {o.head.button.label}
                </Button>
              ) : null}
            </div>
          ) : null}
          {o.showable ? (
            <Button type="button" variant="outline" onClick={o.onShow} className={cn(SMALL, "w-fit")}>
              {INSPECT_COPY.showMe}
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function SheetTab({ stem }: { stem: string }) {
  const sheet = useRig((s) => s.sheet);
  const v = sheetView(sheet, stem);
  if (v.kind === "looking") return <p className={MUTE}>{SHEET_COPY.looking}</p>;
  const url = `${outUrl(stem, SHEET_FILE, false)}?v=${sheet.v}`;
  return (
    <div className="flex flex-col gap-2">
      {v.kind === "none" ? (
        <p className={MUTE}>{SHEET_COPY.none}</p>
      ) : (
        <>
          <a href={url} target="_blank" rel="noopener" title={SHEET_COPY.openTitle} className="block overflow-hidden rounded-lg border border-hair">
            <img alt={SHEET_COPY.alt} src={url} className="block w-full" />
          </a>
          <p className={MUTE}>{SHEET_COPY.caption}</p>
        </>
      )}
      <Button type="button" variant="outline" disabled={sheet.busy} aria-busy={sheet.busy} onClick={() => void renderSheet(stem)} className={cn(SMALL, "w-fit gap-1.5")}>
        {sheet.busy ? <LoaderCircle className="animate-spin" /> : null}
        {v.button}
      </Button>
      {sheet.error ? <p className={cn(MUTE, "font-medium text-danger-text")}>{sheet.error}</p> : null}
    </div>
  );
}

