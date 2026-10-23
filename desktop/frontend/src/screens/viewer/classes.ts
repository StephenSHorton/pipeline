// The HUD's shared class strings, in their own file so the panels can use them without importing Hud.tsx.
// PANEL drops the backdrop blur on purpose (see Hud.tsx): every viewer panel is a Card size="hud" with cn(PANEL, …).
// Without the blur a 0.9 white card turns grey over a dark canvas (the Stack stage), so the fill is near opaque.
export const PANEL = "pointer-events-auto rounded-[14px] [--glass-bg:rgba(255,255,255,0.97)] [-webkit-backdrop-filter:none] [backdrop-filter:none]";

// Three control looks, one rule (every panel, the dock and the bars follow it):
// - TOGGLE: a switch that stays on or off (aria-pressed), or opens a panel (aria-expanded). A white chip with a
//   hairline at rest, so it reads as a button before it is pressed; blue when on.
// - ACTION: a one-shot that runs something (Undo, Save, Re-rig). Same chip and height, darker text, never blue.
//   Use it with variant="outline" and keep it out of a toggle group.
// - TRAY + TRAY_ITEM: pick one of several (Checks/Sheet, colour modes, Voxel/Plane/Contour). The Mokei segmented
//   tray (the Tabs hud look): a grey track, the picked one a white pill.
// Use TOGGLE and TRAY_ITEM with variant="hud-ghost".
const CHIP =
  "hit-44 h-[29px] cursor-pointer rounded-[8px] border border-input bg-white px-[11px] text-[length:calc(14.2px*var(--fs))] font-semibold shadow-[0_1px_1px_rgba(30,41,90,0.04)] hover:border-[#c8d2e3] hover:bg-[#f5f7fc]";
export const TOGGLE = `${CHIP} text-ink-3 hover:text-ink aria-pressed:border-[#bcd2fb] aria-pressed:bg-accent aria-pressed:text-accent-foreground aria-expanded:border-[#bcd2fb] aria-expanded:bg-accent aria-expanded:text-accent-foreground`;
export const ACTION = `${CHIP} text-ink hover:text-ink`;
// A panel's commit (Save): the ACTION chip in the primary colour.
export const COMMIT = `${CHIP} border-transparent bg-primary text-primary-foreground hover:border-transparent hover:bg-primary/85 hover:text-primary-foreground`;
export const TRAY = "inline-flex max-w-full flex-wrap items-center gap-0.5 rounded-[10px] bg-[rgba(226,232,242,0.85)] p-[3px]";
export const TRAY_ITEM =
  "hit-44 h-[27px] cursor-pointer rounded-[8px] border-0 px-[10px] text-[length:calc(13.5px*var(--fs))] font-semibold text-ink-3 hover:bg-white/60 hover:text-ink aria-pressed:bg-white aria-pressed:text-ink aria-pressed:shadow-[0_1px_3px_rgba(30,41,90,0.12)]";
// An image tile that is the current pick (a version, a part, a swatch): a ring, nothing else.
export const PICKED = "ring-2 ring-primary";
// Panel text: the small uppercase section label, muted body copy and a bold group title, shared by every panel.
export const LABEL = "mt-1.5 text-[length:calc(11.5px*var(--fs))] font-semibold tracking-[0.08em] text-ink-3 uppercase";
export const MUTE = "m-0 text-[length:calc(12.5px*var(--fs))] leading-snug text-ink-3";
