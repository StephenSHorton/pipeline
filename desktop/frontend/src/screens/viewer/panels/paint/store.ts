// The brush state the Paint panel and the sprite stack share (legacy paintOn / paintDrop / paintErase, #paint-color
// and #paint-radius, index.html 4872-4874, 5384-5399): module-level, because legacy's stack pane drove the same
// mode with its own Paint toggle ("Same paint mode as the left pane.", #stack-paint, index.html 2191).
// The stack half (stack-lab.js beginStroke / paintSphere / undoStroke / getSheetBlob) is not ported yet: a stack
// view registers itself with setStackBrush and the brush, Undo and Save reach it as they reached stackLab.
import type { Vector3 } from "three";
import { create } from "zustand";
import { BRUSH, BRUSH_TEXT, PAINT_COLOR } from "./paint";

export interface PaintState {
  /** Paint mode: the brush is the armed viewer tool (#paint aria-pressed). */
  on: boolean;
  /** The next click samples a colour instead of painting (#paint-drop). */
  drop: boolean;
  /** The brush restores the original colours (#paint-erase). */
  erase: boolean;
  /** #paint-color, "#rrggbb". */
  color: string;
  /** #paint-radius: a fraction of the mesh's diagonal. */
  radius: number;
  /** #paint-radius-val. */
  radiusText: string;
  /** Save is running (legacy paintSaveBtn.disabled). */
  saving: boolean;
}

export const PAINT0: PaintState = { on: false, drop: false, erase: false, color: PAINT_COLOR, radius: BRUSH.value, radiusText: BRUSH_TEXT, saving: false };

export const usePaint = create<PaintState>(() => ({ ...PAINT0 }));

/** The sprite stack's side of the brush: the calls legacy made on stackLab. */
export interface StackBrush {
  beginStroke(): void;
  /** Paint (or with erase, restore) the voxels within `radius` of a world point. */
  paintSphere(point: Vector3, radius: number, rgb: [number, number, number], erase: boolean): void;
  undoStroke(): void;
  /** The stack/sheet.png strip to save, or null with no stack loaded. */
  getSheetBlob(): Promise<Blob | null>;
  /** Show the brush on the stack at a point, or hide it (null). */
  setBrush?(point: Vector3 | null, radius?: number): void;
}

let stack: StackBrush | null = null;

/** A stack view registers while it is on screen (null when it goes). */
export function setStackBrush(brush: StackBrush | null): void {
  stack = brush;
}

export function stackBrush(): StackBrush | null {
  return stack;
}
