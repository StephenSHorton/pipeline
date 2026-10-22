// The Make panel, as the approved design (makepanel/project/Mokei.dc.html): a glass panel with the photo chip,
// one text field, Add photo and Make; the kind switch under it; one route line; quiet chips for what is automatic
// (head direction, height, file name, detail) with an Options disclosure to change them.
import { useEffect, useRef, useState, useSyncExternalStore, type DragEvent, type FormEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  arrayBufferToBase64,
  drop2d,
  factoryJson,
  saveIntentQuiet,
  uploadImage,
  type AssetOutput,
  type Facing,
  type FacingSource,
} from "@/lib/api";
import { displayStem, jobChipText, useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { onJobChip, openAsset } from "@/app/open";
import { cn } from "@/lib/utils";
import { startDrawThen3d, startHomeJob } from "./chain";
import { KIND_TAB } from "./Library";
import {
  drawFirstBody,
  drawTimeNote,
  FACING_CYCLE,
  FACING_SHORT,
  flatName,
  freshStem,
  photoDrawBody,
  guessSize,
  intentBody,
  isThreeD,
  kindDef,
  KINDS,
  loadKind,
  NO_PICTURE,
  POLYS,
  readIntent,
  saveKind,
  SIZES,
  slugStem,
  spriteDrawBody,
  STEPS,
  autoName,
  type FacingPick,
  type FlatFlags,
  type MakeChoice,
  type Poly,
  type Step,
} from "./make";

interface Photo {
  file: File;
  url: string;
  key: string;
}

const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

/** One upload per picture (name + size), so a second Make of the same photo does not upload it again. */
const uploads = new Map<string, Promise<string>>();
function uploadOnce(p: Photo): Promise<string> {
  let u = uploads.get(p.key);
  if (!u) {
    u = uploadImage(p.file).then((r) => r.path);
    u.catch(() => uploads.delete(p.key));
    uploads.set(p.key, u);
  }
  return u;
}

/** Phone width: the shorter placeholder. */
const NARROW = "(max-width: 639px)";
function useNarrow(): boolean {
  return useSyncExternalStore(
    (fn) => {
      const m = matchMedia(NARROW);
      m.addEventListener("change", fn);
      return () => m.removeEventListener("change", fn);
    },
    () => matchMedia(NARROW).matches,
    () => false,
  );
}

export function MakePanel() {
  const [kind, setKindState] = useState<AssetOutput>(loadKind);
  const [text, setText] = useState("");
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [options, setOptions] = useState(false);
  const [sizeIdx, setSizeIdx] = useState<number | null>(null);
  const [facingPick, setFacingPick] = useState<FacingPick>("auto");
  const [name, setName] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const [poly, setPoly] = useState<Poly>("medium");
  const [steps, setSteps] = useState<Step[]>([]);
  const [stepText, setStepText] = useState<Partial<Record<Step, string>>>({});
  const [line, setLine] = useState<{ text: string; state: "" | "running" | "error" } | null>(null);
  const [starting, setStarting] = useState(false);
  const [drop, setDrop] = useState(false);
  const [flags, setFlags] = useState<FlatFlags | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const narrow = useNarrow();

  const job = useJobs((s) => s.job);
  const watching = useJobs((s) => s.watching);
  const busy = useJobs((s) => s.busy) || starting;

  const cur = kindDef(kind);
  const hasPhoto = !!photo;
  const hasText = text.trim().length > 0;
  const hasInput = hasPhoto || hasText;
  const ps1 = kind === "ps1";

  // GET /2d/status for the draw time note; again after every job (legacy refreshAudioFlags on jobEnded).
  useEffect(() => {
    let live = true;
    factoryJson<FlatFlags>("/2d/status").then((f) => live && setFlags(f)).catch(() => {});
    return () => {
      live = false;
    };
  }, [job?.state]);

  useEffect(() => () => void (photo && URL.revokeObjectURL(photo.url)), [photo]);

  const setKind = (k: AssetOutput) => {
    setKindState(k);
    saveKind(k);
    setLine(null);
  };

  const addFiles = (files: FileList | File[] | null) => {
    const f = Array.from(files || []).find((x) => IMAGE_TYPES.includes(x.type) || /\.(jpe?g|png|webp)$/i.test(x.name));
    if (!f) {
      if (files && Array.from(files).length) setLine({ text: "That is not a picture. Use a JPEG, PNG or WebP.", state: "error" });
      return;
    }
    setPhoto({ file: f, url: URL.createObjectURL(f), key: `file:${f.name}:${f.size}` });
    setLine(null);
  };

  const removePhoto = () => setPhoto(null);

  /* --------------------------------------------- the automatic values */

  const guess = guessSize(text, photo?.file.name);
  const size = SIZES[sizeIdx ?? guess];
  const sizeText = `${size.label} · ${size.inches} in`;

  // A photo is a reference, never a view: every 3D kind draws all four views (from the photo when there is one),
  // so the head's facing always comes from the drawing (side-on, head right) unless you pick it.
  const facing: Facing = facingPick !== "auto" ? facingPick : "right";
  const facingSource: FacingSource | undefined = facingPick !== "auto" ? "user" : undefined;
  const facingText = facingPick !== "auto" ? FACING_SHORT[facingPick] : "right (our drawing)";

  const autoSlug = autoName(text, photo?.file.name);
  const nameSlug = name !== null ? slugStem(name) : autoSlug;
  const nameText = (ps1 ? nameSlug.replace(/-/g, "_") : nameSlug) || (hasPhoto ? "from the photo" : "auto");

  const polyDef = POLYS.find((p) => p.id === poly)!;

  let routeLine: string;
  if (hasPhoto) {
    routeLine =
      kind === "sprite"
        ? "Using your photo as the picture."
        : `We draw every angle from your photo: side, front, back and other side.${hasText ? " Your words steer the drawing." : ""} Then the ${cur.name} is built from the drawing.`;
  } else if (hasText) {
    routeLine = isThreeD(kind)
      ? `No photo, so we draw it first: side, front, back and other side.${drawTimeNote(flags)} Then the ${cur.name} is built from the drawing.`
      : `No photo, so we draw one picture from your words.${drawTimeNote(flags, steps)}`;
  } else
    routeLine = isThreeD(kind)
      ? "Add a photo, describe it, or both. We draw every angle, then build it."
      : "Every sprite starts from a picture. No photo? Describe it and we draw one first.";

  const chips: string[] = !hasInput
    ? ps1
      ? ["Auto size", "Auto name"]
      : ["Auto name"]
    : ps1
      ? [`Head ${facingText}`, sizeText, nameText]
      : kind === "sprite"
        ? [nameText, ...steps.map((s) => STEPS.find((x) => x.id === s)!.label)]
        : [`Detail ${polyDef.label}`, nameText];

  const makeReady = hasInput && !busy;

  /* --------------------------------------------- Make */

  const choice = (): MakeChoice => ({
    kind,
    prompt: text.trim(),
    facing,
    facingSource,
    height: size.inches,
    name: name !== null ? slugStem(name) : isThreeD(kind) ? autoSlug : "",
    poly,
    steps,
    stepText,
    subject: sizeIdx !== null && SIZES[sizeIdx].subject ? SIZES[sizeIdx].subject : undefined,
  });

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (busy) return;
    if (!hasInput) {
      setLine({ text: isThreeD(kind) ? NO_PICTURE : "describe the thing or attach a picture", state: "error" });
      return;
    }
    setStarting(true);
    try {
      if (kind === "sprite") await makeSprite();
      else if (photo) await make3dFromPhoto(photo);
      else await make3dByDrawing();
    } finally {
      setStarting(false);
    }
  };

  async function make3dFromPhoto(p: Photo) {
    const c = choice();
    setLine({ text: "Uploading the photo…", state: "running" });
    let path: string;
    try {
      path = await uploadOnce(p);
    } catch (err) {
      setLine({ text: `Upload failed: ${err instanceof Error ? err.message : err}`, state: "error" });
      return;
    }
    setLine(null);
    const stem = await freshStem(nameSlug || flatName(c.prompt, p.file.name));
    void startDrawThen3d(
      photoDrawBody(c, stem, path),
      { path: "/2d/draw", kind: "flat2d", title: `Drawing ${displayStem(stem)} from your photo` },
      readIntent(c, { drawn: true }),
    );
  }

  async function make3dByDrawing() {
    const c = choice();
    const stem = await freshStem(nameSlug || flatName(c.prompt));
    void startDrawThen3d(drawFirstBody(c, stem), { path: "/2d/draw", kind: "flat2d", title: `Drawing ${displayStem(stem)}` }, readIntent(c, { drawn: true }));
  }

  async function makeSprite() {
    const c = choice();
    const stem = await freshStem(nameSlug || flatName(c.prompt, photo?.file.name));
    const label = displayStem(stem);
    if (photo) {
      setLine({ text: "Opening the picture.", state: "running" });
      try {
        await drop2d(stem, arrayBufferToBase64(await photo.file.arrayBuffer()));
      } catch (err) {
        setLine({ text: err instanceof Error ? err.message : String(err), state: "error" });
        return;
      }
      setLine(null);
      if (!c.steps.length) {
        await saveIntentQuiet(stem, intentBody(readIntent(c)));
        toast(`Sprite ready: ${label}`);
        openAsset(stem, "flat");
        return;
      }
    }
    void startHomeJob(spriteDrawBody(c, stem, !!photo), { path: "/2d/draw", kind: "flat2d", title: photo ? `Finishing ${label}` : `Drawing ${label}` });
  }

  /* --------------------------------------------- drag and drop */

  const onDrag = (e: DragEvent) => {
    if (!Array.from(e.dataTransfer.types).includes("Files")) return;
    e.preventDefault();
    setDrop(true);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrop(false);
    addFiles(e.dataTransfer.files);
  };

  // The job, on Home too (legacy #home-job): every state, hidden only while you watch it run.
  const showChip = !!job && !(watching && (job.state === "running" || !!job.next));
  const quietBtn = "h-auto min-h-11 rounded-xl border-input bg-white px-[0.85rem] text-[0.8rem] font-normal text-ink-2 hover:bg-white hover:text-ink";

  return (
    <section aria-labelledby="mk-h" className="w-full max-w-[38rem]">
      <h1 id="mk-h" className="m-0 mb-[0.35rem] text-center text-[1.65rem] font-bold tracking-[-0.025em]">
        Make something
      </h1>
      <p className="m-0 mb-[1.4rem] text-center text-[0.95rem] text-ink-3">{isThreeD(kind) ? "A photo, a few words, or both. We draw every angle." : "A photo or a few words. We draw the picture."}</p>

      <form
        onSubmit={(e) => void submit(e)}
        onDragEnter={onDrag}
        onDragOver={onDrag}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrop(false);
        }}
        onDrop={onDrop}
        aria-label="Describe or add a photo, then Make"
        className={cn(
          "rounded-[20px] border bg-[var(--glass-bg)] p-2 shadow-[var(--glass-shadow)] [-webkit-backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))] [backdrop-filter:blur(var(--glass-blur))_saturate(var(--glass-saturate))] has-[#mk-desc:focus-visible]:ring-2 has-[#mk-desc:focus-visible]:ring-ring/50",
          drop ? "border-primary ring-3 ring-ring/20" : "border-[var(--glass-border)]",
        )}
      >
        {photo ? (
          <div className="flex items-center gap-[0.6rem] px-2 pt-[0.4rem] pb-[0.2rem]">
            <div className="flex min-w-0 items-center gap-[0.6rem] rounded-xl bg-muted p-[0.3rem]">
              <img src={photo.url} alt="" className="size-10 shrink-0 rounded-[9px] bg-input object-cover" />
              <span className="truncate text-[0.8rem] font-medium text-ink-2">{photo.file.name}</span>
              <Button variant="ghost" onClick={removePhoto} aria-label="Remove photo" className="size-11 shrink-0 rounded-[10px] text-ink-3 hover:bg-transparent hover:text-ink">
                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M6 6l12 12M18 6 6 18" />
                </svg>
              </Button>
            </div>
          </div>
        ) : null}

        <label htmlFor="mk-desc" className="sr-only">
          Describe what to make
        </label>
        <Input
          id="mk-desc"
          type="text"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setLine(null);
          }}
          placeholder={hasPhoto ? "Add notes, or just press Make" : narrow ? "Describe it: a speckled hen" : "Describe it: a speckled hen, a brass lantern"}
          title="One object per picture. Make each piece of a set as its own asset."
          autoComplete="off"
          spellCheck={false}
          className="block h-auto min-h-[52px] rounded-none border-0 bg-transparent px-[0.85rem] pt-[0.9rem] pb-[0.7rem] text-[1.05rem] text-ink shadow-none placeholder:text-muted-foreground focus-visible:border-0 focus-visible:ring-0 md:text-[1.05rem] dark:bg-transparent"
        />

        <div className="flex items-center justify-between gap-2 px-[0.15rem] pb-[0.1rem]">
          <Button
            variant="outline"
            onClick={() => fileRef.current?.click()}
            aria-pressed={hasPhoto}
            className="h-auto min-h-11 gap-[0.45rem] rounded-xl border-input bg-white px-[0.85rem] text-[0.85rem] font-medium text-ink-2 hover:bg-white hover:text-ink aria-pressed:bg-white"
          >
            <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
              <path d="M4 8h3l2-2.5h6L17 8h3v11H4z" />
              <circle cx="12" cy="13" r="3.5" />
            </svg>
            {hasPhoto ? "Change photo" : "Add photo"}
          </Button>
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            hidden
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <Button
            type="submit"
            aria-disabled={!makeReady}
            className={cn(
              "h-auto min-h-11 gap-2 rounded-xl px-[1.1rem] text-[0.9rem] font-semibold",
              makeReady
                ? "shadow-[0_1px_2px_rgba(37,99,235,0.25),0_6px_16px_rgba(37,99,235,0.2)] hover:bg-blue-deep"
                : cn("bg-muted text-ink-3 hover:bg-muted", busy ? "cursor-wait" : "cursor-not-allowed"),
            )}
          >
            {cur.make}
            <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          </Button>
        </div>
      </form>

      <Tabs value={kind} onValueChange={(v) => typeof v === "string" && setKind(v as AssetOutput)} className="mt-[0.9rem] items-center">
        <TabsList
          variant="hud"
          aria-label="What are you making"
          className="mx-auto box-border flex w-fit max-w-full flex-wrap justify-center gap-[0.35rem] rounded-[14px] border border-[var(--topbar-border)] bg-[var(--topbar-bg)] p-[0.3rem]"
        >
          {KINDS.map((k) => (
            <TabsTrigger key={k.id} variant="hud" value={k.id} title={k.hint} className={KIND_TAB}>
              {k.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="mt-4 flex flex-col items-center gap-2">
        <p aria-live="polite" className="m-0 max-w-[32rem] text-center text-[0.85rem] leading-normal text-ink-3">
          {routeLine}
        </p>
        <div className="flex flex-wrap items-center justify-center gap-[0.35rem]">
          {chips.map((c) => (
            <Badge key={c} variant="secondary" className="h-[26px] rounded-full px-[0.6rem] text-[0.75rem] font-medium">
              {c}
            </Badge>
          ))}
          <Button
            variant="ghost"
            onClick={() => setOptions((o) => !o)}
            aria-expanded={options}
            aria-controls="mk-options"
            className="h-auto min-h-11 rounded-[10px] px-[0.6rem] text-[0.8rem] font-semibold text-blue-deep hover:bg-transparent hover:text-ink aria-expanded:bg-transparent aria-expanded:text-blue-deep"
          >
            {options ? "Hide options" : "Options"}
          </Button>
        </div>

        {options ? (
          <div id="mk-options" className="flex flex-wrap justify-center gap-[0.4rem]">
            {ps1 ? (
              <>
                <Button variant="outline" className={quietBtn} title="Height in inches: Person 72, Animal 58, Bust 30, Prop 24" onClick={() => setSizeIdx(((sizeIdx ?? guess) + 1) % SIZES.length)}>
                  Size <strong className="font-semibold">{sizeText}</strong>
                </Button>
                <Button
                  variant="outline"
                  className={quietBtn}
                  title="As you look at the picture, not the animal's own left or right."
                  onClick={() => setFacingPick(FACING_CYCLE[(FACING_CYCLE.indexOf(facingPick) + 1) % FACING_CYCLE.length])}
                >
                  Head points <strong className="font-semibold">{facingText}</strong>
                </Button>
              </>
            ) : null}
            {kind === "model" || kind === "stack" ? (
              <Button variant="outline" className={quietBtn} title={polyDef.hint} onClick={() => setPoly(POLYS[(POLYS.indexOf(polyDef) + 1) % POLYS.length].id)}>
                Detail <strong className="font-semibold">{polyDef.label}</strong>
              </Button>
            ) : null}
            {editingName ? (
              <label className={cn(quietBtn, "inline-flex items-center gap-1.5 border focus-within:ring-2 focus-within:ring-ring/50")}>
                Name
                <input
                  autoFocus
                  value={name ?? (autoSlug ? displayStem(autoSlug) : "")}
                  onChange={(e) => setName(e.target.value)}
                  onBlur={() => {
                    setEditingName(false);
                    if (name !== null && !slugStem(name)) setName(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === "Escape") {
                      e.preventDefault();
                      if (e.key === "Escape") setName(null);
                      (e.target as HTMLInputElement).blur();
                    }
                  }}
                  placeholder={hasPhoto ? "from the photo" : "auto"}
                  className="w-36 border-0 bg-transparent font-semibold text-ink outline-none"
                />
              </label>
            ) : (
              <Button variant="outline" className={quietBtn} title="The file name. Click to change it." onClick={() => setEditingName(true)}>
                Name <strong className="font-semibold">{nameText}</strong>
              </Button>
            )}
            {kind === "sprite"
              ? STEPS.map((s) => {
                  const on = steps.includes(s.id);
                  return (
                    <Button
                      key={s.id}
                      variant="outline"
                      aria-pressed={on}
                      className={cn(quietBtn, on && "border-primary bg-secondary font-semibold hover:bg-secondary")}
                      title={`After the picture: ${s.label.toLowerCase()}`}
                      onClick={() => setSteps((cur) => (on ? cur.filter((x) => x !== s.id) : [...cur, s.id]))}
                    >
                      {s.label}
                    </Button>
                  );
                })
              : null}
            {kind === "sprite" && steps.length ? (
              <div className="flex w-full flex-col items-center gap-[0.4rem]">
                {STEPS.filter((s) => steps.includes(s.id)).map((s) => (
                  <label key={s.id} className="w-full max-w-[24rem]">
                    <span className="sr-only">{s.label}</span>
                    <Input
                      value={stepText[s.id] ?? ""}
                      onChange={(e) => setStepText((t) => ({ ...t, [s.id]: e.target.value }))}
                      placeholder={s.placeholder}
                      spellCheck={false}
                      className="h-11 rounded-xl border-input bg-white px-3 text-[0.85rem] md:text-[0.85rem]"
                    />
                  </label>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {line ? (
          <p role={line.state === "error" ? "alert" : "status"} className={cn("m-0 text-center text-[0.82rem]", line.state === "error" ? "font-medium text-danger-text" : "text-ink-3")}>
            {line.text}
          </p>
        ) : null}

        {showChip && job ? (
          <Button
            variant="secondary"
            onClick={onJobChip}
            data-state={job.state}
            title={job.state === "running" ? "Show progress" : undefined}
            className="h-auto min-h-11 max-w-full gap-[0.6rem] rounded-xl bg-[var(--glass-bg)] py-[0.4rem] pr-[0.7rem] pl-[0.45rem] shadow-[var(--glass-shadow)] hover:bg-white"
          >
            <Badge variant={job.state === "running" || job.next ? "signal" : job.state === "ok" ? "success" : job.state === "error" ? "danger" : "slate"} className="h-6 rounded-lg px-[0.55rem] text-[0.72rem] font-bold">
              {job.state === "running" || job.next ? "Running" : job.state === "ok" ? "Done" : job.state === "error" ? "Error" : "Cancelled"}
            </Badge>
            <span role="status" className="truncate text-[0.82rem] font-normal text-ink-2">
              {jobChipText(job)}
            </span>
          </Button>
        ) : null}
      </div>
    </section>
  );
}
