// The HUD's shared class strings, in their own file so the panels can use them without importing Hud.tsx.
// PANEL drops the backdrop blur on purpose (see Hud.tsx): every viewer panel is a Card size="hud" with cn(PANEL, …).
export const PANEL = "pointer-events-auto rounded-[14px] [-webkit-backdrop-filter:none] [backdrop-filter:none]";
export const TOGGLE =
  "hit-44 h-[29px] rounded-[8px] px-[11px] text-[length:calc(14.2px*var(--fs))] font-semibold text-ink-3 hover:bg-muted aria-pressed:bg-accent aria-pressed:text-accent-foreground";
