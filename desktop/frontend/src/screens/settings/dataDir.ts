// The Factory folder line (legacy refreshSettingsPanel / openFactoryFolder, index.html 6758-6822, 5894-5901).
// Module state, like the legacy element: a path read on an earlier visit stays until a newer read replaces it.
import { create } from "zustand";
import { openDataDirHere, readDataDir } from "./api";

export const DIR_PENDING = "…";
export const DIR_UNAVAILABLE = "Factory folder unavailable";

export interface DirLine {
  text: string;
  /** "error" after Open failed (red); a fresh path clears it. */
  tone: "" | "error";
}

export const useDataDir = create<DirLine>(() => ({ text: DIR_PENDING, tone: "" }));

/** Wails DataDir, else GET /paths. An empty answer changes nothing; a failure says so only over "…". */
export async function refreshDataDir(read: () => Promise<string> = readDataDir): Promise<void> {
  try {
    const dir = await read();
    if (dir) useDataDir.setState({ text: String(dir), tone: "" });
  } catch {
    if (useDataDir.getState().text === DIR_PENDING) useDataDir.setState({ text: DIR_UNAVAILABLE });
  }
}

/** Wails OpenDataDir, else POST /open-data-dir. The error replaces the line, in red. */
export async function openFolder(open: () => Promise<void> = openDataDirHere): Promise<void> {
  try {
    await open();
  } catch (err) {
    useDataDir.setState({ text: (err as Error)?.message || String(err), tone: "error" });
  }
}
