// The update overlay (legacy #update-overlay, index.html 2277-2283, 6698-6706): it covers the whole window
// while an update downloads and installs, and cannot be dismissed (the app restarts). Mounted once at the App
// root, so it shows on Home, on an asset and on Settings alike (legacy B2: it was hidden on Settings).
import { useEffect, useRef } from "react";
import { useUi } from "@/lib/ui";
import { overlayTitle, startUpdateWatch, useUpdate } from "./update";

export function UpdateOverlay() {
  const status = useUpdate((s) => s.status);
  const version = useUpdate((s) => s.updatingTo);
  const card = useRef<HTMLDivElement>(null);
  const open = status === "downloading";

  // Mounting the overlay is enough to get the boot check and the update:installing listener.
  useEffect(() => {
    startUpdateWatch();
  }, []);

  // No way out: an open confirm is cancelled (its focus trap would pull focus back and Enter would press its
  // Delete under the overlay), focus stays on the card, every key is swallowed, clicks stop at the backdrop.
  useEffect(() => {
    if (!open) return;
    useUi.getState().answer(false);
    const grab = () => {
      (document.activeElement as HTMLElement | null)?.blur?.();
      card.current?.focus();
    };
    grab();
    // The closing confirm hands focus back to its trigger a frame later; take it again.
    const raf = requestAnimationFrame(grab);
    const trap = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      card.current?.focus();
    };
    const keep = (e: FocusEvent) => {
      if (e.target !== card.current) card.current?.focus();
    };
    document.addEventListener("keydown", trap, true);
    document.addEventListener("keyup", trap, true);
    document.addEventListener("focusin", keep, true);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("keydown", trap, true);
      document.removeEventListener("keyup", trap, true);
      document.removeEventListener("focusin", keep, true);
    };
  }, [open]);

  if (!open) return null;
  return (
    // Above the toast and confirm (z-50); the blur keeps the page behind from reading through the glass card.
    <div className="fixed inset-0 z-[60] grid place-items-center bg-[rgba(15,23,42,0.4)] px-4 backdrop-blur-md animate-in fade-in-0">
      <div
        ref={card}
        tabIndex={-1}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="update-overlay-title"
        aria-describedby="update-overlay-step"
        className="glass-panel w-[min(26rem,100%)] rounded-xl p-6 text-ink outline-none animate-in fade-in-0 zoom-in-95"
      >
        <div className="flex items-center gap-2 text-[0.75rem] font-semibold tracking-wide text-ink-3 uppercase">
          <span aria-hidden="true" className="size-2 rounded-full bg-signal animate-pulse motion-reduce:animate-none" />
          Update
        </div>
        <h2 id="update-overlay-title" className="mt-2 text-xl leading-snug font-semibold tracking-tight" aria-live="polite">
          {overlayTitle(version)}
        </h2>
        <div id="update-overlay-step" className="mt-1.5 text-sm text-ink-3">
          Pipeline will restart.
        </div>
      </div>
    </div>
  );
}
