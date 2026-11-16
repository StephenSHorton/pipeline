// The app update (legacy updateState / renderUpdateStatus / checkForUpdates, index.html 6696-6813). One store
// for the Settings Updates row and the update overlay, so an install that starts while you are on Home or on
// an asset still covers the window (legacy B2: #update-overlay was hidden under body.is-settings).
import { create } from "zustand";
import { getVersion, WAILS } from "@/lib/api";
import { desktopCall } from "./api";

export type UpdateStatus = "boot" | "idle" | "checking" | "uptodate" | "error" | "downloading";

export interface UpdateState {
  /** "boot" until the first check starts: the hint reads the bare "Checking…" of the static page. */
  status: UpdateStatus;
  /** Raw version without "v" ("1.4.2"), "dev", or "" while unknown. */
  version: string;
  /** The version being installed, "" when the event or the info carried none. */
  updatingTo: string;
  error: string;
  /** The button text survives a step into "downloading" (legacy returns before resetting it). */
  buttonLabel: string;
}

export const CHECK_LABEL = "Check for updates";
export const OUTSIDE_WAILS = "Check for updates runs in the desktop app.";

export const useUpdate = create<UpdateState>(() => ({
  status: "boot",
  version: "",
  updatingTo: "",
  error: "",
  buttonLabel: CHECK_LABEL,
}));

/* ---------------------------------------------------------------- pure */

/** The version as the store keeps it (legacy applyVersion): strip "Pipeline ", drop a leading v; "dev" stays. */
export function rawVersion(label: unknown): string {
  let v = String(label ?? "").trim();
  if (v.startsWith("Pipeline ")) v = v.slice(9).trim();
  if (!v) return "";
  return v === "dev" ? "dev" : v.replace(/^v/i, "");
}

/** "dev", or the version with one "v" (legacy renderUpdateStatus label; unknown reads as dev). */
export function versionLabel(version: string): string {
  const v = version || "dev";
  return v === "dev" || v.startsWith("v") ? v : "v" + v;
}

/** `Updating to v1.2…`, or "Updating…" when the version is unknown (legacy showUpdateOverlay). */
export function overlayTitle(version: string): string {
  return version ? `Updating to v${String(version).replace(/^v/i, "")}…` : "Updating…";
}

/** The update:installing payload: a string, or [string] (onWailsEvent already unwraps the array). */
export function installingVersion(data: unknown): string {
  const raw = Array.isArray(data) ? data[0] : data;
  return typeof raw === "string" ? raw : "";
}

export type HintTone = "" | "running" | "ok" | "error";

export interface UpdateView {
  hint: string;
  tone: HintTone;
  buttonLabel: string;
  buttonDisabled: boolean;
}

/** The Updates row, every legacy state (index.html 6708-6747). */
export function updateView(s: UpdateState): UpdateView {
  const label = versionLabel(s.version);
  switch (s.status) {
    case "boot":
      return { hint: "Checking…", tone: "", buttonLabel: s.buttonLabel, buttonDisabled: false };
    case "downloading":
      return {
        hint: s.updatingTo ? `${overlayTitle(s.updatingTo)} Pipeline will restart.` : "Updating… Pipeline will restart.",
        tone: "running",
        buttonLabel: s.buttonLabel,
        buttonDisabled: true,
      };
    case "checking":
      return { hint: `Pipeline ${label} · Checking…`, tone: "running", buttonLabel: "Checking…", buttonDisabled: true };
    case "error":
      return { hint: s.error || "Check failed", tone: "error", buttonLabel: CHECK_LABEL, buttonDisabled: false };
    case "uptodate":
      return { hint: `Pipeline ${label} · Up to date`, tone: "ok", buttonLabel: CHECK_LABEL, buttonDisabled: false };
    default:
      return { hint: `Pipeline ${label}`, tone: "", buttonLabel: CHECK_LABEL, buttonDisabled: false };
  }
}

/** Move to a status; the button text follows legacy (Checking… while checking, kept while downloading). */
function go(status: UpdateStatus, patch: Partial<UpdateState> = {}): void {
  useUpdate.setState((s) => {
    const next = { ...s, ...patch, status };
    if (status !== "downloading") next.buttonLabel = updateView(next).buttonLabel;
    return next;
  });
}

/* ---------------------------------------------------------------- actions */

/** An install is under way (from a check or from the host's update:installing event): row + overlay. */
export function markInstalling(version: string): void {
  go("downloading", { updatingTo: version });
}

/**
 * Check for an update (legacy checkForUpdates). Manual: outside Wails says so, errors show in the row.
 * Silent (boot): any failure, or no runtime, leaves the row idle. An update downloads and installs at once.
 */
export async function checkForUpdates(manual: boolean): Promise<void> {
  go("checking", { error: "" });
  try {
    const via = desktopCall<{ version?: string } | null>(WAILS.CHECK);
    if (!via) {
      if (manual) go("error", { error: OUTSIDE_WAILS });
      else go("idle");
      return;
    }
    const info = await via;
    if (!info) {
      go("uptodate");
      return;
    }
    markInstalling(info.version || "");
    const inst = desktopCall(WAILS.INSTALL, info);
    if (inst) await inst;
  } catch (e) {
    if (manual) go("error", { error: String(e && (e as Error).message ? (e as Error).message : e) });
    else go("idle");
  }
}

/** Store the app version (from /version or Wails GetVersion) for the row. */
export function setVersion(label: unknown): void {
  const v = rawVersion(label);
  if (v && !useUpdate.getState().version) useUpdate.setState({ version: v });
}

interface EventsApi {
  On?: (name: string, cb: (ev: { data?: unknown } | undefined) => void) => unknown;
}

function wailsEvents(): EventsApi | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as { wails?: { Events?: EventsApi }; runtime?: { Events?: EventsApi } };
  return w.wails?.Events || w.runtime?.Events;
}

/** Subscribe to update:installing now, or on wails:ready when the runtime loads late. */
function watchInstalling(): void {
  const on = (ev: { data?: unknown } | undefined) => markInstalling(installingVersion(ev?.data));
  const now = wailsEvents();
  if (typeof now?.On === "function") {
    now.On("update:installing", on);
    return;
  }
  if (typeof document === "undefined") return;
  document.addEventListener(
    "wails:ready",
    () => {
      const late = wailsEvents();
      if (typeof late?.On === "function") late.On("update:installing", on);
    },
    { once: true },
  );
}

let started = false;

/**
 * Boot (legacy 6824-6839 and 5903-5919): read the version, run the silent check, listen for update:installing.
 * Safe to call more than once (App, the overlay and Settings all make sure it ran); only the first call acts.
 */
export function startUpdateWatch(): void {
  if (started) return;
  started = true;
  void getVersion()
    .then(setVersion)
    .catch(() => {});
  watchInstalling();
  void checkForUpdates(false);
}

/** Tests only: forget that the watch ran and reset the store. */
export function resetUpdateForTest(): void {
  started = false;
  useUpdate.setState({ status: "boot", version: "", updatingTo: "", error: "", buttonLabel: CHECK_LABEL });
}
