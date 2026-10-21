import AppKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

enum LiftError: Error, CustomStringConvertible {
    case usage
    case load(String)
    case write(String)
    var description: String {
        switch self {
        case .usage: return "usage: lift-subject <input-image> <output.png>"
        case .load(let p): return "cannot load image: \(p)"
        case .write(let p): return "cannot write png: \(p)"
        }
    }
}

func loadCIImage(_ url: URL) -> CIImage? {
    CIImage(contentsOf: url, options: [.applyOrientationProperty: true])
}

func writePNG(_ image: CIImage, to url: URL) throws {
    let ctx = CIContext(options: [.workingColorSpace: NSNull(), .useSoftwareRenderer: false])
    let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    let color = image.matchedToWorkingSpace(from: image.colorSpace ?? cs) ?? image
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
    try ctx.writePNGRepresentation(of: color, to: url, format: .RGBA8, colorSpace: cs)
}

func visionLift(_ src: CIImage) -> CIImage? {
    let handler = VNImageRequestHandler(ciImage: src, options: [:])
    let request = VNGenerateForegroundInstanceMaskRequest()
    do {
        try handler.perform([request])
    } catch {
        fputs("vision: perform failed: \(error)\n", stderr)
        return nil
    }
    guard let obs = request.results?.first else {
        fputs("vision: no foreground instance\n", stderr)
        return nil
    }
    do {
        let buffer = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: handler)
        var mask = CIImage(cvPixelBuffer: buffer)
        if mask.extent.size != src.extent.size {
            let sx = src.extent.width / mask.extent.width
            let sy = src.extent.height / mask.extent.height
            mask = mask.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        }
        // Feather the mask a couple of pixels so the silhouette is not crunchy.
        mask = mask.clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.2])
            .cropped(to: src.extent)
        let clear = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: src.extent)
        return src.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: clear,
            kCIInputMaskImageKey: mask,
        ])
    } catch {
        fputs("vision: mask failed: \(error)\n", stderr)
        return nil
    }
}

func cornerFloodLift(_ src: CIImage) -> CIImage {
    // Fallback for studio shots: treat pixels similar to the four corners as background.
    let ctx = CIContext()
    let rect = src.extent.integral
    let w = Int(rect.width), h = Int(rect.height)
    var bgra = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
    guard let cg = ctx.createCGImage(src, from: rect),
          let bitmap = CGContext(data: &bgra, width: w, height: h, bitsPerComponent: 8,
                                 bytesPerRow: w * 4, space: cs, bitmapInfo: info.rawValue)
    else { return src }
    bitmap.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    // premultipliedFirst + byteOrder32Little is BGRA: [b,g,r,a]
    func rgb(_ x: Int, _ y: Int) -> (Int, Int, Int) {
        let i = (y * w + x) * 4
        return (Int(bgra[i + 2]), Int(bgra[i + 1]), Int(bgra[i + 0]))
    }
    let corners = [rgb(0, 0), rgb(w - 1, 0), rgb(0, h - 1), rgb(w - 1, h - 1)]
    let br = corners.map { $0.0 }.reduce(0, +) / 4
    let bg = corners.map { $0.1 }.reduce(0, +) / 4
    let bb = corners.map { $0.2 }.reduce(0, +) / 4
    let thresh = 38

    var seen = [UInt8](repeating: 0, count: w * h)
    var queue: [Int] = []
    func push(_ x: Int, _ y: Int) {
        if x < 0 || y < 0 || x >= w || y >= h { return }
        let idx = y * w + x
        if seen[idx] != 0 { return }
        let (r, g, b) = rgb(x, y)
        if abs(r - br) + abs(g - bg) + abs(b - bb) > thresh * 3 { return }
        seen[idx] = 1
        queue.append(idx)
    }
    for x in 0 ..< w { push(x, 0); push(x, h - 1) }
    for y in 0 ..< h { push(0, y); push(w - 1, y) }
    var q = 0
    while q < queue.count {
        let idx = queue[q]; q += 1
        let x = idx % w, y = idx / w
        push(x - 1, y); push(x + 1, y); push(x, y - 1); push(x, y + 1)
    }

    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let idx = y * w + x
            let i = idx * 4
            let (r, g, b) = rgb(x, y)
            let a: UInt8 = seen[idx] == 1 ? 0 : 255
            rgba[i] = UInt8(r); rgba[i + 1] = UInt8(g); rgba[i + 2] = UInt8(b); rgba[i + 3] = a
        }
    }
    let lifted = CIImage(bitmapData: Data(rgba), bytesPerRow: w * 4, size: CGSize(width: w, height: h),
                         format: .RGBA8, colorSpace: cs)
    // CIImage bitmap is top-left; flip to CI's bottom-left.
    return lifted.transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -CGFloat(h)))
}

let argv = Array(CommandLine.arguments.dropFirst())
guard argv.count >= 2 else {
    fputs(LiftError.usage.description + "\n", stderr)
    exit(2)
}
let inURL = URL(fileURLWithPath: argv[0]).absoluteURL
let outURL = URL(fileURLWithPath: argv[1]).absoluteURL
guard let src = loadCIImage(inURL) else {
    fputs(LiftError.load(inURL.path).description + "\n", stderr)
    exit(1)
}

var lifted = visionLift(src)
var method = "vision"
if lifted == nil {
    fputs("lift-subject: vision produced no mask, falling back to corner flood-fill\n", stderr)
    lifted = cornerFloodLift(src)
    method = "flood"
}
do {
    try writePNG(lifted!, to: outURL)
    print("lift-subject: \(method) -> \(outURL.path)")
} catch {
    fputs("lift-subject: \(error)\n", stderr)
    exit(1)
}
