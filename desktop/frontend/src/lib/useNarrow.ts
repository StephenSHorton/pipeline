// Phone width (under Tailwind's sm, 640 px), as a live hook. Shared by the viewer's panel rail (the right dock's
// panels join the left rail on phones) and the versions panel (folded by default on phones).
import { useSyncExternalStore } from "react";

export const NARROW = "(max-width: 639px)";

export function useNarrow(): boolean {
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
