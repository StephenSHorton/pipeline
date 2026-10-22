// One typed client for the Go host and the factory (proxied through the host, same origin).
// Contract: scratchpad map-backend.md. Every URL is relative; the page never calls :8766 directly.

/* ---------------------------------------------------------------- types */

/** The DEGRADED grammar: one factory fallback. */
export interface DegradedEntry {
  stage?: string;
  code?: string;
  message?: string;
}

export interface HealthFeatures {
  isolate?: boolean;
  shape?: boolean;
  shape_weights?: boolean;
  paint?: boolean;
  paint_pbr?: boolean;
  shape_mv?: boolean;
  shape_mv_venv?: boolean;
  shape_mv_weights?: boolean;
  unirig?: boolean;
  remesh?: boolean;
  draw?: { backend?: string; licence?: string; model?: string; error?: string };
}

export interface Health {
  ok: boolean;
  service: string;
  backend?: string;
  python?: string;
  root?: string;
  gpu?: string;
  busy: boolean;
  job_id: string | null;
  features?: HealthFeatures;
}

export interface VersionInfo {
  ok: boolean;
  version: string;
  raw: string;
}

export interface PathsInfo {
  ok: boolean;
  dataDir: string;
  version: string;
  raw: string;
}

export type AssetOutput = "ps1" | "model" | "stack" | "sprite";

export interface Asset {
  id: string;
  title: string;
  has_mesh: boolean;
  has_stack: boolean;
  has_flat: boolean;
  has_ideas: boolean;
  has_psx: boolean;
  output: AssetOutput | null;
  input: "input.png" | "input.jpg" | null;
  /** First existing of preview.png, input.*, flat/source.png, stack/preview.png, ideas/idea-01.png (relative to out/<id>/). */
  preview: string | null;
  mtime: number;
}

export interface AssetsResponse {
  ok: boolean;
  root: string;
  assets: Asset[];
  busy: boolean;
  job: { job_id: string; stem: string; state: JobState } | null;
}

export interface RenameResponse {
  ok: true;
  stem: string;
  from: string;
  renamed: boolean;
  path: string;
}

export interface DeleteResponse {
  ok: true;
  deleted: true;
  stem: string;
  path: string;
}

export interface UploadResult {
  ok: boolean;
  path: string;
  bytes?: number;
  stem?: string;
  error?: string;
}

export type JobState = "running" | "ok" | "error" | "cancelled";
export type JobKind = "rig" | "render" | "flat2d" | "ideas" | "";

export interface JobStarted {
  ok: boolean;
  accepted: true;
  job_id: string;
  state: "running";
  stem: string;
  kind?: JobKind;
  status_url: string;
}

export interface JobStage {
  key: string;
  label: string;
}

export interface JobProgress {
  stages: JobStage[];
  stage: string;
  stage_index: number;
  stage_count: number;
  sub?: number;
  fraction?: number;
}

export interface StageTime {
  started?: number;
  ended?: number;
}

/** GET /status/<id>. Generate and flat2d/ideas jobs carry extra fields; the page reads these. */
export interface JobSnapshot {
  ok: boolean;
  job_id: string;
  state: JobState;
  stem?: string;
  kind?: JobKind;
  started?: number;
  finished?: number | null;
  returncode?: number | null;
  error?: string | null;
  now?: number;
  log_tail?: string;
  progress?: JobProgress | null;
  degraded?: DegradedEntry[];
  result?: JobResult | null;
  preview_url?: string;
  stage_times?: Record<string, StageTime>;
  stage_est?: Record<string, number>;
  estimate_s?: number;
  prompt?: string;
  count?: number;
  revise?: boolean;
  source?: string;
  quality?: string;
  poly?: string;
  intent?: Intent;
  glb_path?: string;
}

export interface JobResult {
  glb_path?: string;
  degraded?: DegradedEntry[];
  items?: Array<{ file: string; seed?: number; keep?: boolean; used?: boolean; prompt?: string }>;
  center?: string;
  notes?: string[];
  hear?: string;
  pieces?: unknown;
  revise?: boolean;
  seed?: number;
  redrawn?: boolean;
  ran?: string[];
}

export interface CancelResponse {
  ok: true;
  job_id: string;
  state: "cancelling" | JobState;
}

export type Facing = "camera" | "right" | "left" | "away";
export type FacingSource = "drawn" | "detected" | "user" | "cues";

export interface Intent {
  output: AssetOutput;
  facing?: Facing;
  height?: number;
  faces?: number;
  atlas?: number;
  up?: "z" | "y";
  name?: string;
  poly?: "high" | "medium" | "low";
  steps?: Array<"pieces" | "motion" | "say" | "sound">;
  subject?: string;
  kind?: string;
  facing_source?: FacingSource;
  rig?: { plan?: string; plan_hint?: string; canonical?: boolean };
}

/** POST /generate body (map-backend 3.5). The page sends quality psx/textured/stack, poly and intent always. */
export interface GenerateSpec {
  image: string;
  images?: string[];
  stem?: string;
  quality: "draft" | "textured" | "pbr" | "stack" | "psx";
  poly?: "high" | "medium" | "low";
  seed?: number;
  prompt?: string;
  facing?: Facing;
  psx_height?: number;
  psx_faces?: number;
  psx_atlas?: number;
  psx_up?: "z" | "y";
  psx_name?: string;
  intent?: Intent;
}

/** POST /2d/draw body. A description with no picture for a 3D kind posts {kind:"flat2d", views:"4", intent} (the four-view
 * sheet, then /generate on flat/source.png); kind:"ideas" is the legacy idea grid. */
export interface DrawSpec {
  kind: "flat2d" | "ideas";
  stem: string;
  prompt: string;
  source?: "drop";
  steps?: Array<"pieces" | "motion" | "say" | "sound">;
  pieces?: unknown;
  motion?: unknown;
  say?: unknown;
  sound?: unknown;
  video?: string;
  frames?: string[];
  revise?: boolean;
  count?: number;
  views?: "1" | "4";
  intent?: Intent;
}

export interface FacingDetectResponse {
  ok: true;
  how: "vlm" | "unavailable" | string;
  head: Facing | null;
  profile?: string | null;
  confidence: number;
  why?: string;
  model?: string;
  low: boolean;
  accept: number;
  image: string;
}

export interface RigPlan {
  id: string;
  label: string;
  blurb: string;
  class: "humanoid" | "quadruped" | "multileg" | "avian" | string;
  aliases: string[];
  built: boolean;
  fixed: boolean;
  clips: string[];
}

export interface Lod {
  id: string;
  file: string;
  label: string;
  faces?: number;
  vertices?: number;
  has_uvs?: boolean;
  kind?: string;
}

export interface LodsDoc {
  stem: string;
  lods: Lod[];
}

export interface InspectDoc {
  path?: string;
  ok?: boolean;
  meshes?: number;
  vertices?: number;
  faces?: number;
  materials?: number;
  textures?: number;
  has_uvs?: boolean;
  errors?: string[];
  degraded?: DegradedEntry[];
  [k: string]: unknown;
}

export interface RigDoc {
  class?: string;
  plan?: string;
  clips?: string[];
  fbx?: string;
  fbx_note?: string;
  fbx_check?: Record<string, unknown>;
  template?: { mode?: string; rig?: string; plan?: string; built?: boolean; dir?: string };
  clips_failed?: string[];
  source?: string;
  degraded?: DegradedEntry[];
  [k: string]: unknown;
}

export interface DegradedDoc {
  degraded: DegradedEntry[];
}

export interface WorkflowInfo {
  ok: boolean;
  stem: string;
  hasMesh: boolean;
  hasFlat: boolean;
  workflow: "2d" | "3d" | "";
  choose: boolean;
}

/* ---------------------------------------------------------------- core fetch */

/** The factory said no (or the sidecar is down: 503 with the start error). */
export class ApiError extends Error {
  status: number;
  payload: unknown;
  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

type FlowDirtyListener = (path: string) => void;
const flowDirty = new Set<FlowDirtyListener>();
/** Called after any successful POST to /2d/* or /paint/* (legacy queueFlowRefresh). Returns an unsubscribe. */
export function onFlowDirty(fn: FlowDirtyListener): () => void {
  flowDirty.add(fn);
  return () => flowDirty.delete(fn);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * JSON from the host/factory. Network failures retry 24 x 250 ms while the sidecar comes up, then
 * "Factory failed to start". A non-JSON body becomes {error: text}. !ok throws ApiError with
 * data.error || text || statusText || "HTTP <n>".
 */
export async function factoryJson<T = unknown>(path: string, opts?: RequestInit): Promise<T> {
  let res: Response | undefined;
  let last: unknown;
  for (let i = 0; i < 24; i++) {
    try {
      res = await fetch(path, opts);
      last = null;
      break;
    } catch (e) {
      last = e;
      await sleep(250);
    }
  }
  if (!res) throw new Error("Factory failed to start", { cause: last });
  const text = await res.text();
  let data: unknown = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text.trim() };
    }
  }
  if (!res.ok) {
    const msg = (data as { error?: string } | null)?.error || text.trim() || res.statusText || `HTTP ${res.status}`;
    throw new ApiError(msg, res.status, data);
  }
  if (opts?.method === "POST" && /^\/(2d|paint)\//.test(path)) for (const fn of flowDirty) fn(path);
  return data as T;
}

function postJson<T>(path: string, body: unknown, method = "POST"): Promise<T> {
  return factoryJson<T>(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
}

/* ---------------------------------------------------------------- URLs */

/** The one stem encoder. Stems match ^[A-Za-z0-9][A-Za-z0-9._-]*$, but always encode. */
export const enc = (stem: string) => encodeURIComponent(stem);
const encRel = (rel: string) => rel.split("/").map(encodeURIComponent).join("/");
const bust = (t?: number | string | false) => (t === false ? "" : `t=${t ?? Date.now()}`);
const withQuery = (url: string, q: string) => (q ? `${url}${url.includes("?") ? "&" : "?"}${q}` : url);

/** /out/<stem>/<rel> (Go file server, no-store). t: cache-bust value, false for none, default now. */
export function outUrl(stem: string, rel: string, t?: number | string | false): string {
  return withQuery(`/out/${enc(stem)}/${encRel(rel)}`, bust(t));
}

/** A root-relative path the factory gave back (in/<file>, out/<id>/<shot>) as a URL. */
export function rootUrl(rel: string): string {
  return "/" + encRel(rel.replace(/^\/+/, ""));
}

/** GET /glb/<stem>?lod= (model/gltf-binary). */
export function glbUrl(stem: string, lod?: string, t?: number | string | false): string {
  const q = [lod ? `lod=${encodeURIComponent(lod)}` : "", bust(t)].filter(Boolean).join("&");
  return withQuery(`/glb/${enc(stem)}`, q);
}

/** The legacy mesh fallback chain (meshUrls), in order. */
export function meshUrls(stem: string, preferLod?: string): string[] {
  const t = Date.now();
  const lod = preferLod || "lookdev";
  return [
    glbUrl(stem, lod, t),
    glbUrl(stem, "lookdev", t),
    glbUrl(stem, undefined, t),
    outUrl(stem, "mesh_lookdev.glb", t),
    outUrl(stem, "mesh_hero.glb", t),
    outUrl(stem, "mesh.glb", t),
    outUrl(stem, "textured.glb", t),
  ];
}

/** GET /2d/<stem>/<rel> (sources, parts, sheets, sounds; served by the factory, not /out). */
export function flatUrl(stem: string, rel: string, t?: number | string | false): string {
  return withQuery(`/2d/${enc(stem)}/${encRel(rel)}`, bust(t));
}

/* ---------------------------------------------------------------- out/ readers (a miss is null, never an error) */

export async function readOutJson<T>(stem: string, rel: string, t?: number | string | false): Promise<T | null> {
  try {
    const r = await fetch(outUrl(stem, rel, t));
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

export async function headOut(stem: string, rel: string): Promise<boolean> {
  try {
    return (await fetch(outUrl(stem, rel), { method: "HEAD" })).ok;
  } catch {
    return false;
  }
}

/** Every file the workflow stages see under out/<stem>/ (relative paths), from GET /flow/<stem>; null when the flow
 * cannot be read. Used instead of probing (HEAD or GET) for optional files, so an asset without a rig or a degraded.json
 * does not log a 404 per file. */
export async function flowFiles(stem: string): Promise<Set<string> | null> {
  try {
    const r = await fetch(`/flow/${enc(stem)}`);
    if (!r.ok) return null;
    const doc = (await r.json()) as { stages?: Array<{ files?: string[] }> };
    return new Set((doc.stages ?? []).flatMap((st) => st.files ?? []));
  } catch {
    return null;
  }
}

/** out/<stem>/inspect.json; the gallery busts with ?t=<asset mtime>. */
export const readInspect = (stem: string, t?: number | string) => readOutJson<InspectDoc>(stem, "inspect.json", t);
export const readRig = (stem: string) => readOutJson<RigDoc>(stem, "rig/rig.json");
export const readDegraded = (stem: string) => readOutJson<DegradedDoc>(stem, "degraded.json");
export const readIntent = (stem: string) => readOutJson<Intent & { made?: number }>(stem, "intent.json");
export const readLods = (stem: string) => readOutJson<LodsDoc>(stem, "lods.json");

/* ---------------------------------------------------------------- Wails */

const FQN = "github.com/StephenSHorton/pipeline/desktop/internal";
export const WAILS = {
  UPLOAD: `${FQN}/factory.Service.SaveUpload`,
  CHECK: `${FQN}/update.Service.CheckForUpdate`,
  INSTALL: `${FQN}/update.Service.DownloadAndInstall`,
  VERSION: `${FQN}/update.Service.GetVersion`,
  DATADIR: `${FQN}/factory.Service.DataDir`,
  OPEN_DIR: `${FQN}/factory.Service.OpenDataDir`,
  OPEN_EXPORT: `${FQN}/factory.Service.OpenExport`,
} as const;

interface WailsGlobal {
  Call?: { ByName?: (method: string, ...args: unknown[]) => Promise<unknown> };
  Events?: { On?: (name: string, cb: (ev: { data: unknown }) => void) => () => void };
}
declare global {
  interface Window {
    wails?: WailsGlobal;
  }
}

/** Call a bound Go method by its fully qualified name. null when the Wails runtime is absent (plain browser on :8777). */
export function wailsByName<T = unknown>(method: string, ...args: unknown[]): Promise<T> | null {
  const call = typeof window !== "undefined" ? window.wails?.Call : undefined;
  if (!call || typeof call.ByName !== "function") return null;
  return call.ByName(method, ...args) as Promise<T>;
}

/** Wails first, HTTP when the runtime is missing or the call throws. */
export async function viaWailsOr<T>(method: string, args: unknown[], http: () => Promise<T>): Promise<T> {
  const p = wailsByName<T>(method, ...args);
  if (p) {
    try {
      return await p;
    } catch {
      /* fall through */
    }
  }
  return http();
}

/** Subscribe to a Wails event (e.g. update:installing). No-op outside the app. */
export function onWailsEvent(name: string, cb: (data: unknown) => void): () => void {
  const on = typeof window !== "undefined" ? window.wails?.Events?.On : undefined;
  if (typeof on !== "function") return () => {};
  return on(name, (ev) => cb(Array.isArray(ev?.data) ? ev.data[0] : ev?.data));
}

/* ---------------------------------------------------------------- host routes */

/** "/version", then Wails GetVersion (raw string, no "v"). */
export async function getVersion(): Promise<string> {
  try {
    const j = await factoryJson<VersionInfo & { label?: string }>("/version");
    return j.version || j.label || "";
  } catch (err) {
    const p = wailsByName<string>(WAILS.VERSION);
    if (!p) throw err;
    return String(await p);
  }
}

export const getPaths = () => factoryJson<PathsInfo>("/paths");

export async function getDataDir(): Promise<string> {
  return viaWailsOr(WAILS.DATADIR, [], async () => (await getPaths()).dataDir);
}

export async function openDataDir(): Promise<void> {
  await viaWailsOr(WAILS.OPEN_DIR, [], () => postJson("/open-data-dir", {}));
}

/** Legacy openExport: in the app a failed Wails call is the answer (no second open over HTTP); outside it, POST. */
export async function openExport(stem: string): Promise<void> {
  const via = wailsByName(WAILS.OPEN_EXPORT, stem);
  if (via) await via;
  else await factoryJson(`/open-export?stem=${enc(stem)}`, { method: "POST" });
}

export const getWorkflow = (stem: string) => factoryJson<WorkflowInfo>(`/control/workflow?stem=${enc(stem)}`);

/* ---------------------------------------------------------------- upload */

export function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)));
  }
  return btoa(binary);
}

/**
 * Save a picture under in/. Base64 JSON to the Go /upload (never a raw File body); if that throws,
 * the Wails SaveUpload binding. Returns .path (root-relative, for /generate image and /facing/detect).
 */
export async function uploadImage(file: Blob & { name?: string }): Promise<UploadResult> {
  const name = file.name || "upload.png";
  const buf = await file.arrayBuffer();
  if (!buf.byteLength) throw new Error("could not read image");
  const b64 = arrayBufferToBase64(buf);
  try {
    return await factoryJson<UploadResult>(`/upload?name=${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, data: b64 }),
    });
  } catch (err) {
    const viaWails = wailsByName<UploadResult>(WAILS.UPLOAD, name, b64);
    if (!viaWails) throw err;
    const res = await viaWails;
    if (!res || !res.ok || !res.path) throw new Error((res && res.error) || "upload failed");
    return res;
  }
}

/* ---------------------------------------------------------------- factory routes */

export const getHealth = () => factoryJson<Health>("/health");
export const listAssets = () => factoryJson<AssetsResponse>("/assets");
export const renameAsset = (stem: string, name: string) => postJson<RenameResponse>(`/assets/${enc(stem)}`, { name }, "PATCH");
export const deleteAsset = (stem: string) => factoryJson<DeleteResponse>(`/assets/${enc(stem)}`, { method: "DELETE" });

/** POST /generate: 202 {job_id}; 409 when the slot is taken (jobs.ts follows the running one). */
export const generate = (spec: GenerateSpec) => postJson<JobStarted>("/generate", spec);
export const draw2d = (spec: DrawSpec) => postJson<JobStarted>("/2d/draw", spec);
/** POST /2d/drop {stem, data: base64}: the picture becomes out/<stem>/flat/source.png. */
export const drop2d = (stem: string, data: string) => postJson<Record<string, unknown>>("/2d/drop", { stem, data });
export const flat2dStatus = () => factoryJson<Record<string, unknown>>("/2d/status");
export const saveIntent = (stem: string, intent: Intent) => postJson<{ ok: boolean; stem: string; intent: Intent }>(`/intent/${enc(stem)}`, intent);

/** POST /intent/<stem>; a failure only warns (legacy saveIntent). */
export async function saveIntentQuiet(stem: string, intent: Intent): Promise<void> {
  try {
    await saveIntent(stem, intent);
  } catch (err) {
    console.warn("intent", err);
  }
}

export const jobStatus = (jobId: string) => factoryJson<JobSnapshot>(`/status/${enc(jobId)}`);
export const cancelJob = (jobId: string) => postJson<CancelResponse>(`/cancel/${enc(jobId)}`, {});

/** POST /facing/detect: synchronous (about 11 s cold), holds the job slot. */
export const detectFacing = (body: { image: string } | { stem: string }) => postJson<FacingDetectResponse>("/facing/detect", body);

export const rigPlans = () => factoryJson<{ ok: boolean; plans: RigPlan[] }>("/rig/plans");
export const startRig = (stem: string, body: Record<string, unknown> = {}) => postJson<JobStarted>(`/rig/${enc(stem)}`, body);
