// Global toast + confirm. The hosts (src/app/Overlays.tsx) mount at the App root, outside every screen,
// so they show on Home and Settings too (legacy B2: its #toast/#ask lived inside the hidden workspace).
import { create } from "zustand";

export interface Toast {
  id: number;
  text: string;
  actLabel?: string;
  act?: () => void;
}

export interface ConfirmAsk {
  id: number;
  message: string;
  yesLabel: string;
  /** "Delete" when yesLabel is Delete, else "Confirm" (legacy kicker). */
  kicker: string;
  resolve: (ok: boolean) => void;
}

interface UiState {
  toast: Toast | null;
  ask: ConfirmAsk | null;
  hideToast: () => void;
  answer: (ok: boolean) => void;
}

let seq = 0;
let toastTimer: ReturnType<typeof setTimeout> | undefined;

export const useUi = create<UiState>((set, get) => ({
  toast: null,
  ask: null,
  hideToast: () => {
    clearTimeout(toastTimer);
    set({ toast: null });
  },
  answer: (ok) => {
    const a = get().ask;
    if (!a) return;
    set({ ask: null });
    a.resolve(ok);
  },
}));

/** One toast at a time, 12 s, with an optional action button (e.g. "Open"). */
export function toast(text: string, actLabel?: string, act?: () => void): void {
  clearTimeout(toastTimer);
  useUi.setState({ toast: { id: ++seq, text, actLabel, act } });
  toastTimer = setTimeout(() => useUi.setState({ toast: null }), 12000);
}

/** Ask before a destructive step. Resolves false on No / Escape. A newer ask cancels an older one. */
export function askConfirm(message: string, yesLabel = "Delete"): Promise<boolean> {
  const prev = useUi.getState().ask;
  if (prev) prev.resolve(false);
  return new Promise((resolve) => {
    useUi.setState({ ask: { id: ++seq, message, yesLabel, kicker: yesLabel === "Delete" ? "Delete" : "Confirm", resolve } });
  });
}
