// The animation principles side by side (pipeline#8), over the Stack stage: the walk and the punch without and with
// them. The top row is the skeleton a video model follows (drawn as the skeleton video is), the bottom row this
// walking character hung on the same frames, all four on one clock at the cycles' own 24 fps, on the spot.
import { X } from "lucide-react";
import { useEffect, useRef } from "react";
import { AmbientLight, DirectionalLight, Group, MeshLambertMaterial, OrthographicCamera, PlaneGeometry, Mesh, MeshBasicMaterial, Scene, WebGLRenderer } from "three";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { PANEL } from "@/screens/viewer/classes";
import { columnFrame, COMPARE_COLUMNS, COMPARE_COPY, compareCycles, drawSkeleton, skeletonAt } from "./character/compare";
import type { CharacterDoc } from "./character/parts";
import { bodyFromRig } from "./character/skeleton";
import { createWalkView, type WalkView } from "./character/walkView";

const FPS = 24;
/** The camera's look down on the walkers (radians): low, so the side view matches the skeleton's. */
const ELEVATION = (12 * Math.PI) / 180;

export function PrinciplesCompare({ doc, strips, onClose }: { doc: CharacterDoc; strips: Map<string, ArrayLike<number>>; onClose: () => void }) {
  const skel = useRef<(HTMLCanvasElement | null)[]>([]);
  const stage = useRef<HTMLCanvasElement>(null);
  const frames = COMPARE_COLUMNS.map((c) => columnFrame(doc.height, c.cycle));
  const aspect = frames[0].width / frames[0].height;

  useEffect(() => {
    const body = bodyFromRig(doc.rig, doc.height);
    const cycles = compareCycles(body);
    const canvas = stage.current!;
    const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0x0c0d0b);
    const scene = new Scene();
    scene.add(new AmbientLight(0xffffff, 1.6));
    const sun = new DirectionalLight(0xffffff, 1.4);
    sun.position.set(0.4, 1, 0.8);
    scene.add(sun);
    const root = new Group();
    scene.add(root);
    const material = new MeshLambertMaterial();
    const H = Math.max(1, doc.height - 1);
    const W = frames[0].width;
    // Four columns of one width each, left to right; rig x is three's x, rig y three's -z (walkView's frame).
    const walks: WalkView[] = COMPARE_COLUMNS.map((c, i) =>
      createWalkView({
        parent: root, doc, strips, pitch: 1, material, look: "cubes",
        color: (byte, layer) => (byte / 255) * (0.55 + 0.45 * (layer / H)),
        cycle: c.cycle, principles: c.principles, spot: { x: (i - 1.5) * W + frames[i].x, y: 0, h: 0 },
      }),
    );
    // the ground under each walker, so a drop or a lunge reads against it
    const groundGeo = new PlaneGeometry(4 * W, 0.35 * doc.height);
    const groundMat = new MeshBasicMaterial({ color: 0x23251f });
    const ground = new Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    scene.add(ground);
    const F = frames[0];
    const cam = new OrthographicCamera(-2 * W, 2 * W, F.height / 2, -F.height / 2, -10 * doc.height, 10 * doc.height);
    const mid = F.height / 2 - F.ground;
    cam.position.set(0, mid + Math.sin(ELEVATION) * 4 * doc.height, Math.cos(ELEVATION) * 4 * doc.height);
    cam.lookAt(0, mid, 0);

    const ctxs = skel.current.map((c) => c?.getContext("2d") ?? null);
    const fit = () => {
      const r = canvas.getBoundingClientRect();
      renderer.setSize(Math.max(1, r.width), Math.max(1, r.height), false);
      for (const c of skel.current) {
        if (!c) continue;
        const b = c.getBoundingClientRect();
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        c.width = Math.max(1, Math.round(b.width * dpr));
        c.height = Math.max(1, Math.round(b.height * dpr));
      }
    };
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);
    fit();

    let k = 0;
    let acc = 0;
    let last = performance.now();
    let raf = 0;
    const draw = () => {
      ctxs.forEach((ctx, i) => ctx && drawSkeleton(ctx, skeletonAt(cycles[i], k), frames[i]));
      renderer.render(scene, cam);
    };
    const loop = (now: number) => {
      acc = Math.min(acc + (now - last) / 1000, 4 / FPS);
      last = now;
      let moved = false;
      while (acc >= 1 / FPS) {
        acc -= 1 / FPS;
        k += 1;
        for (const w of walks) w.tick(1 / FPS);
        moved = true;
      }
      if (moved) draw();
      raf = requestAnimationFrame(loop);
    };
    draw();
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      for (const w of walks) w.dispose();
      material.dispose();
      groundGeo.dispose();
      groundMat.dispose();
      renderer.dispose();
    };
    // one build per character; the frames follow from it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, strips]);

  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center overflow-y-auto bg-black/55 p-4">
      <Card size="hud" role="dialog" aria-label={COMPARE_COPY.title} className={cn(PANEL, "my-auto w-[min(860px,100%)] [--glass-bg:#fff] gap-2.5 px-4 pt-3 pb-4")}>
        <div className="flex items-center justify-between gap-2">
          <h2 className="m-0 text-[0.95rem] font-semibold text-ink">{COMPARE_COPY.title}</h2>
          <Button type="button" variant="ghost" size="icon-sm" aria-label={COMPARE_COPY.close} title={COMPARE_COPY.close} onClick={onClose} className="hit-44 text-ink-3">
            <X />
          </Button>
        </div>
        <p className="m-0 max-w-[80ch] text-[0.8rem] leading-snug text-ink-3">{COMPARE_COPY.blurb}</p>
        <div className="grid grid-cols-4 gap-x-0 gap-y-1" data-compare="principles">
          {COMPARE_COLUMNS.map((c) => (
            <div key={c.label} className={cn("text-center text-[0.8rem] font-semibold", c.principles ? "text-ink" : "text-ink-3")}>
              {c.label}
            </div>
          ))}
          <div className="col-span-4 text-[0.72rem] font-semibold tracking-[0.06em] text-ink-3 uppercase">{COMPARE_COPY.skeleton}</div>
          {COMPARE_COLUMNS.map((c, i) => (
            <canvas
              key={c.label}
              ref={(el) => {
                skel.current[i] = el;
              }}
              aria-label={`${c.label}: skeleton`}
              style={{ aspectRatio: String(aspect) }}
              className={cn("block w-full bg-black", i % 2 ? "border-l border-white/10" : "", i === 2 ? "border-l-2 border-white/40" : "")}
            />
          ))}
          <div className="col-span-4 mt-1 text-[0.72rem] font-semibold tracking-[0.06em] text-ink-3 uppercase">{COMPARE_COPY.walker}</div>
          <canvas ref={stage} aria-label={COMPARE_COPY.walker} style={{ aspectRatio: String(4 * aspect) }} className="col-span-4 block w-full" />
        </div>
      </Card>
    </div>
  );
}
