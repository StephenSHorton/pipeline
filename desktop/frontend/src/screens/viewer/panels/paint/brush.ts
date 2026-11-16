// The mesh half of the brush: the albedo texture copied to a canvas and stamped through the uv under the pointer
// (legacy ensurePaintCanvas, stampUV, beginPaintStroke, the Undo click and the drop pick, index.html 5400-5461,
// 5481-5487, 5546-5555). The canvas and texture makers are injected so brush.test.ts runs without a DOM.
// Not copied from legacy: the canvas outliving the model it was taken from (Save then uploaded a stale texture
// after a LOD switch or reload). The panel calls forget() on every model change.
import { CanvasTexture, SRGBColorSpace, type Material, type Mesh, type Object3D, type Texture } from "three";
import { dropPixel, pixelRadius, rgbToHex, uvToCanvas } from "./paint";

/** The 2D context calls the brush makes. */
export interface PaintCtx {
  fillStyle: unknown;
  drawImage(image: unknown, dx: number, dy: number): void;
  save(): void;
  restore(): void;
  beginPath(): void;
  arc(x: number, y: number, r: number, start: number, end: number): void;
  clip(): void;
  fill(): void;
  getImageData(x: number, y: number, w: number, h: number): { data: ArrayLike<number> };
}

export interface PaintCanvas {
  width: number;
  height: number;
  getContext(id: "2d"): PaintCtx | null;
  toBlob(done: (blob: Blob | null) => void, type?: string): void;
}

export interface BrushDeps {
  makeCanvas(width: number, height: number): PaintCanvas;
  /** A texture showing `canvas`, set up like the map it replaces. */
  makeTexture(canvas: PaintCanvas, from: Texture): Texture;
}

export const domBrushDeps: BrushDeps = {
  makeCanvas(width, height) {
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c as unknown as PaintCanvas;
  },
  makeTexture(canvas, from) {
    const t = new CanvasTexture(canvas as unknown as HTMLCanvasElement);
    t.colorSpace = from.colorSpace || SRGBColorSpace;
    t.flipY = from.flipY;
    // Legacy dropped these: a pixel LOD's Nearest filtering went soft on the first stroke.
    t.wrapS = from.wrapS;
    t.wrapT = from.wrapT;
    t.magFilter = from.magFilter;
    t.minFilter = from.minFilter;
    t.generateMipmaps = from.generateMipmaps;
    t.channel = from.channel;
    t.needsUpdate = true;
    return t;
  },
};

type Mapped = Material & { map?: Texture | null };
const materialsOf = (o: Object3D): Mapped[] => {
  const m = o as Mesh;
  if (!m.isMesh || !m.material) return [];
  return (Array.isArray(m.material) ? m.material : [m.material]) as Mapped[];
};

export class MeshBrush {
  private canvas: PaintCanvas | null = null;
  private ctx: PaintCtx | null = null;
  private texture: Texture | null = null;
  private source: Texture | null = null;
  private orig: PaintCanvas | null = null;
  private undoSnap: PaintCanvas | null = null;

  constructor(private deps: BrushDeps = domBrushDeps) {}

  /** A stroke or drop has taken the texture: Save uploads it as textured_paint.glb. */
  get painted(): boolean {
    return !!this.canvas;
  }

  /**
   * Take the texture on first use: the last mesh under root with a map. None (an untextured mesh, or Clay on, whose
   * material has no map) is false: mesh paint does nothing then, as in legacy.
   */
  ensure(root: Object3D | null): boolean {
    if (this.canvas) return true;
    if (!root) return false;
    let map: Texture | null = null;
    root.traverse((o) => {
      for (const m of materialsOf(o)) if (m.map) map = m.map;
    });
    const src = map as Texture | null;
    const img = src?.image as { width?: number; height?: number } | null | undefined;
    if (!src || !img) return false;
    const canvas = this.deps.makeCanvas(img.width || 1024, img.height || 1024);
    const ctx = canvas.getContext("2d");
    if (!ctx) return false;
    ctx.drawImage(img, 0, 0);
    this.orig = this.copy(canvas);
    this.canvas = canvas;
    this.ctx = ctx;
    this.source = src;
    this.texture = this.deps.makeTexture(canvas, src);
    root.traverse((o) => {
      for (const m of materialsOf(o)) {
        if (m.map !== src) continue;
        m.map = this.texture;
        m.needsUpdate = true;
      }
    });
    return true;
  }

  /** One dab at uv: the brush colour, or with erase the original texture, inside a circle (legacy stampUV). */
  stamp(root: Object3D | null, uv: { x: number; y: number } | null, radiusWorld: number, diagonal: number, rgb: [number, number, number], erase: boolean): void {
    if (!this.ensure(root) || !uv) return;
    const { canvas, ctx, texture } = this;
    if (!canvas || !ctx || !texture) return;
    const c = uvToCanvas(uv, canvas.width, canvas.height, texture.flipY);
    const pr = pixelRadius(radiusWorld, diagonal, canvas.width);
    ctx.save();
    ctx.beginPath();
    ctx.arc(c.x, c.y, pr, 0, Math.PI * 2);
    ctx.clip();
    if (erase && this.orig) {
      ctx.drawImage(this.orig, 0, 0);
    } else {
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.beginPath();
      ctx.arc(c.x, c.y, pr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    texture.needsUpdate = true;
  }

  /** Snapshot for Undo, one level deep (legacy beginPaintStroke). */
  begin(root: Object3D | null): void {
    if (this.ensure(root) && this.canvas) this.undoSnap = this.copy(this.canvas);
  }

  /** Back to the snapshot of the last stroke; Undo again repeats it (legacy single-level Undo). */
  undo(): void {
    if (!this.undoSnap || !this.ctx) return;
    this.ctx.drawImage(this.undoSnap, 0, 0);
    if (this.texture) this.texture.needsUpdate = true;
  }

  /** The dropper: the texture's colour at uv as "#rrggbb", or null without a texture. */
  sample(root: Object3D | null, uv: { x: number; y: number } | null): string | null {
    if (!uv || !this.ensure(root) || !this.canvas || !this.ctx || !this.texture) return null;
    const p = dropPixel(uv, this.canvas.width, this.canvas.height, this.texture.flipY);
    const px = this.ctx.getImageData(p.x, p.y, 1, 1).data;
    return rgbToHex(px[0], px[1], px[2]);
  }

  /** The painted texture as a PNG for POST /paint/mesh, or null when nothing took it. */
  toBlob(): Promise<Blob | null> {
    const canvas = this.canvas;
    if (!canvas) return Promise.resolve(null);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }

  /** The model it painted is gone: drop the canvas, the snapshot and both textures (no material uses them now). */
  forget(): void {
    this.texture?.dispose();
    this.source?.dispose();
    this.canvas = this.ctx = this.orig = this.undoSnap = null;
    this.texture = this.source = null;
  }

  private copy(from: PaintCanvas): PaintCanvas {
    const c = this.deps.makeCanvas(from.width, from.height);
    c.getContext("2d")?.drawImage(from, 0, 0);
    return c;
  }
}
