import { AlertDialog } from "@base-ui/react/alert-dialog";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUi } from "@/lib/ui";
import { UpdateOverlay } from "@/screens/settings/UpdateOverlay";

// Mounted once at the App root, above every screen: toasts and confirms show on Home too.
export function Overlays() {
  return (
    <>
      <ToastHost />
      <ConfirmHost />
      <UpdateOverlay />
    </>
  );
}

function ToastHost() {
  const t = useUi((s) => s.toast);
  const hide = useUi((s) => s.hideToast);
  if (!t) return null;
  return (
    // 24 px off the bottom: the versions panel moved to the left column, so nothing below the stage to clear. Phones
    // are the exception: Concept, Stack and Export put the panel last, along the bottom edge, so 80 px clears it.
    <div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4 max-sm:bottom-20">
      <div
        key={t.id}
        role="status"
        aria-live="polite"
        className="glass-panel pointer-events-auto flex max-w-[40rem] items-center gap-3 rounded-xl py-2.5 pr-2 pl-4 text-sm text-ink animate-in fade-in-0 slide-in-from-bottom-2"
      >
        <span className="min-w-0 flex-1 whitespace-pre-line">{t.text}</span>
        {t.actLabel ? (
          <Button
            size="sm"
            onClick={() => {
              hide();
              t.act?.();
            }}
          >
            {t.actLabel}
          </Button>
        ) : null}
        <Button size="icon-sm" variant="ghost" aria-label="Dismiss" title="Dismiss" onClick={hide} className="hit-44">
          <X />
        </Button>
      </div>
    </div>
  );
}

function ConfirmHost() {
  const ask = useUi((s) => s.ask);
  const answer = useUi((s) => s.answer);
  return (
    <AlertDialog.Root open={!!ask} onOpenChange={(open) => !open && answer(false)}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-50 bg-[rgba(15,23,42,0.18)] data-open:animate-in data-open:fade-in-0" />
        <AlertDialog.Popup className="glass-panel fixed top-1/2 left-1/2 z-50 w-[min(26rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-xl p-5 text-ink outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95">
          <div className="text-[0.75rem] font-semibold tracking-wide text-ink-3 uppercase">{ask?.kicker}</div>
          <AlertDialog.Description className="mt-2 text-sm whitespace-pre-line">{ask?.message}</AlertDialog.Description>
          <div className="mt-5 flex justify-end gap-2">
            <Button variant="outline" onClick={() => answer(false)}>
              Cancel
            </Button>
            <Button variant={ask?.yesLabel === "Delete" ? "destructive" : "default"} onClick={() => answer(true)} autoFocus>
              {ask?.yesLabel}
            </Button>
          </div>
        </AlertDialog.Popup>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
