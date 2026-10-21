// The Audio panel in 2D (legacy audio cubby, index.html 2016-2051; gaps A4: the old page could not open it in 2D).
// The fields follow the selected piece; Speak and Sound write the asset row, as they always did.
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { flatUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import { SPEAKERS, type AudioFields } from "./flat";

export const PANEL = "pointer-events-auto rounded-[14px] [-webkit-backdrop-filter:none] [backdrop-filter:none]";
export const FIELD =
  "w-full rounded-lg border border-input bg-white px-2.5 py-1.5 text-sm text-ink outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

export interface Bound {
  rel: string;
  t: number;
}

export interface AudioPanelProps {
  id: string;
  stem: string;
  who: string;
  fields: AudioFields;
  onFields: (f: AudioFields) => void;
  speak: { disabled: boolean; title: string };
  sound: { disabled: boolean; title: string };
  onSpeak: () => void;
  onSound: () => void;
  status: string;
  voice: Bound;
  sfx: Bound;
  onClose: () => void;
}

export function AudioPanel(p: AudioPanelProps) {
  return (
    <Card
      id={p.id}
      size="hud"
      role="region"
      aria-label="Audio"
      className={cn(PANEL, "absolute bottom-3 left-3 z-10 max-h-[calc(100%-24px)] w-[min(320px,calc(100%-24px))] gap-2.5 overflow-y-auto px-4 pt-3 pb-3.5")}
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="m-0 text-[0.95rem] font-semibold text-ink">Audio</h2>
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Close Audio" title="Close Audio" onClick={p.onClose} className="hit-44 text-ink-3">
          <X />
        </Button>
      </div>
      <span className="text-[0.8rem] font-medium text-ink-3" aria-live="polite">
        {p.who}
      </span>
      <Field label="What it says">
        <textarea
          rows={2}
          placeholder="One short spoken line."
          value={p.fields.line}
          onChange={(e) => p.onFields({ ...p.fields, line: e.target.value })}
          className={cn(FIELD, "resize-none")}
        />
      </Field>
      <Field label="Speaker">
        <select value={p.fields.speaker} onChange={(e) => p.onFields({ ...p.fields, speaker: e.target.value })} className={cn(FIELD, "h-11")}>
          {SPEAKERS.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </Field>
      <Field label="The sound it makes">
        <textarea
          rows={2}
          placeholder="One shot. A footstep, a bolt, a latch."
          value={p.fields.effect}
          onChange={(e) => p.onFields({ ...p.fields, effect: e.target.value })}
          className={cn(FIELD, "resize-none")}
        />
      </Field>
      <div className="flex gap-2">
        {/* A disabled button keeps its title readable on a wrapper. */}
        <span title={p.speak.title} className="flex-1">
          <Button type="button" variant="outline" disabled={p.speak.disabled} title={p.speak.title} onClick={p.onSpeak} className="min-h-11 w-full bg-white">
            Speak
          </Button>
        </span>
        <span title={p.sound.title} className="flex-1">
          <Button type="button" variant="outline" disabled={p.sound.disabled} title={p.sound.title} onClick={p.onSound} className="min-h-11 w-full bg-white">
            Sound
          </Button>
        </span>
      </div>
      {/* Always in the tree, so the first message is announced. */}
      <p role="status" aria-live="polite" className="m-0 text-[0.8rem] text-ink-3 empty:hidden">
        {p.status}
      </p>
      <Player label="The spoken line" stem={p.stem} bound={p.voice} />
      <Player label="The sound" stem={p.stem} bound={p.sfx} />
    </Card>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[0.78rem] font-semibold text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function Player({ label, stem, bound }: { label: string; stem: string; bound: Bound }) {
  if (!bound.rel) return null;
  return <audio aria-label={label} controls src={flatUrl(stem, bound.rel, bound.t)} className="h-10 w-full" />;
}
