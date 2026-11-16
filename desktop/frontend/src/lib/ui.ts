// Global toast, confirm and text prompt. The hosts (src/app/Overlays.tsx) mount at the App root, outside every screen,
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

export interface TextAsk {
  id: number;
  title: string;
  placeholder: string;
  initial: string;
  confirmLabel: string;
  resolve: (text: string | null) => void;
}

interface UiState {
  toast: Toast | null;
  ask: ConfirmAsk | null;
  askingText: TextAsk | null;
  hideToast: () => void;
  answer: (ok: boolean) => void;
  answerText: (text: string | null) => void;
}

let seq = 0;
let toastTimer: ReturnType<typeof setTimeout> | undefined;

export const useUi = create<UiState>((set, get) => ({
  toast: null,
  ask: null,
  askingText: null,
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
  answerText: (text) => {
    const a = get().askingText;
    if (!a) return;
    set({ askingText: null });
    const t = text?.trim();
    a.resolve(t ? t : null);
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

/** Ask for a line of text (a project name). Resolves the trimmed text, or null on Cancel / Escape / empty. A newer ask
 * cancels an older one. */
export function askText(title: string, opts: { placeholder?: string; initial?: string; confirmLabel?: string } = {}): Promise<string | null> {
  const prev = useUi.getState().askingText;
  if (prev) prev.resolve(null);
  return new Promise((resolve) => {
    useUi.setState({
      askingText: { id: ++seq, title, placeholder: opts.placeholder ?? "", initial: opts.initial ?? "", confirmLabel: opts.confirmLabel ?? "Save", resolve },
    });
  });
}
