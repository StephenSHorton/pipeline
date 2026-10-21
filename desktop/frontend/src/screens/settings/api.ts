// Settings' host calls. The Wails runtime script loads in a plain browser too (vite dev, :8777 in Chrome): its
// Call.ByName is there and every call resolves to "" there, so "is the runtime loaded" is not "are we in the
// app". These helpers call Go only inside the desktop webview, so the browser gets the outside-Wails path:
// "Check for Updates runs in the desktop app.", the folder from /paths, Open over HTTP.
import { factoryJson, getPaths, WAILS, wailsByName } from "@/lib/api";

/** The native bridge the Wails runtime itself looks for (WKWebView, WebView2, or an injected invoke). */
export function inDesktop(): boolean {
  if (typeof window === "undefined") return false;
  const w = window as unknown as {
    webkit?: { messageHandlers?: { external?: { postMessage?: unknown } } };
    chrome?: { webview?: { postMessage?: unknown } };
    wails?: { invoke?: unknown };
  };
  return (
    typeof w.webkit?.messageHandlers?.external?.postMessage === "function" ||
    typeof w.chrome?.webview?.postMessage === "function" ||
    typeof w.wails?.invoke === "function"
  );
}

/** A bound Go method, or null outside the desktop app (legacy wailsByName's null). */
export function desktopCall<T = unknown>(method: string, ...args: unknown[]): Promise<T> | null {
  return inDesktop() ? wailsByName<T>(method, ...args) : null;
}

/** Wails DataDir; when it is missing or empty, GET /paths. */
export async function readDataDir(): Promise<string> {
  const via = desktopCall<string>(WAILS.DATADIR);
  if (via) {
    const dir = await via.catch(() => "");
    if (dir) return String(dir);
  }
  return (await getPaths()).dataDir || "";
}

/** Wails OpenDataDir, else POST /open-data-dir. */
export async function openDataDirHere(): Promise<void> {
  const via = desktopCall(WAILS.OPEN_DIR);
  if (via) {
    await via;
    return;
  }
  await factoryJson("/open-data-dir", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
}
