import AppKit
import Foundation
import ImageIO
import Metal
import ModelIO
import SceneKit
import SceneKit.ModelIO
import UniformTypeIdentifiers

enum PreviewError: Error, CustomStringConvertible {
    case usage
    case load(String)
    case metal
    case write(String)
    var description: String {
        switch self {
        case .usage: return "usage: preview-glb <in.glb|obj> <out.png>"
        case .load(let p): return "cannot load mesh: \(p)"
        case .metal: return "no Metal device"
        case .write(let p): return "cannot write png: \(p)"
        }
    }
}

let argv = Array(CommandLine.arguments.dropFirst())
guard argv.count >= 2 else {
    fputs(PreviewError.usage.description + "\n", stderr)
    exit(2)
}
let inURL = URL(fileURLWithPath: argv[0]).absoluteURL
let outURL = URL(fileURLWithPath: argv[1]).absoluteURL

func loadScene(_ url: URL) -> SCNScene? {
    if url.pathExtension.lowercased() == "glb" || url.pathExtension.lowercased() == "gltf" {
        // ModelIO may or may not speak glTF depending on OS; try, then fall back to SceneKit.
        let asset = MDLAsset(url: url)
        if asset.count > 0 {
            return SCNScene(mdlAsset: asset)
        }
        if let s = try? SCNScene(url: url, options: nil) {
            return s
        }
        return nil
    }
    let asset = MDLAsset(url: url)
    guard asset.count > 0 else { return nil }
    return SCNScene(mdlAsset: asset)
}

guard let scene = loadScene(inURL) else {
    fputs(PreviewError.load(inURL.path).description + "\n", stderr)
    exit(1)
}

// The viewer's look and framing (#356; the numbers are in scripts/workbench/playbook.md "Viewer and thumbnail
// defaults"): flat Lambert like the app's diorama (trimesh's OBJ material has Ks 0.4 / Ns 1, a broad sheen that
// washed every thumbnail out), the orbit home's angle (az 36, el 26, Y up), orthographic, the model filling ~68%.
let FILL: Float = 0.68
let HOME_AZ: Float = 36 * .pi / 180
let HOME_EL: Float = 26 * .pi / 180

let clay = SCNMaterial()
clay.diffuse.contents = NSColor(calibratedRed: 0.45, green: 0.52, blue: 0.40, alpha: 1)
scene.rootNode.enumerateChildNodes { node, _ in
    guard let geom = node.geometry else { return }
    if geom.materials.isEmpty || geom.materials.allSatisfy({ $0.diffuse.contents == nil }) {
        geom.materials = [clay]
    }
    for m in geom.materials {
        m.lightingModel = .lambert
        m.specular.contents = NSColor.black
        m.emission.contents = NSColor.black
    }
}

// Every vertex in world space (the bounding box's corners framed the pig at 50%).
var pts: [SIMD3<Float>] = []
scene.rootNode.enumerateChildNodes { node, _ in
    guard let geom = node.geometry else { return }
    let world = node.simdWorldTransform
    for src in geom.sources(for: .vertex) where src.usesFloatComponents && src.bytesPerComponent == 4 && src.componentsPerVector >= 3 {
        src.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for i in 0..<src.vectorCount {
                let o = src.dataOffset + i * src.dataStride
                let x = raw.load(fromByteOffset: o, as: Float.self)
                let y = raw.load(fromByteOffset: o + 4, as: Float.self)
                let z = raw.load(fromByteOffset: o + 8, as: Float.self)
                let w = world * SIMD4<Float>(x, y, z, 1)
                pts.append(SIMD3<Float>(w.x, w.y, w.z))
            }
        }
    }
}
guard !pts.isEmpty else {
    fputs(PreviewError.load(inURL.path).description + "\n", stderr)
    exit(1)
}
var lo = pts[0], hi = pts[0]
for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
let center = (lo + hi) / 2
let diag = simd_length(hi - lo)
let dir = SIMD3<Float>(sin(HOME_AZ) * cos(HOME_EL), sin(HOME_EL), cos(HOME_AZ) * cos(HOME_EL))
let right = SIMD3<Float>(cos(HOME_AZ), 0, -sin(HOME_AZ))
let up = simd_cross(dir, right)
var half: Float = 1e-6
for p in pts {
    let d = p - center
    half = max(half, abs(simd_dot(d, right)), abs(simd_dot(d, up)))
}

let camera = SCNCamera()
camera.usesOrthographicProjection = true
camera.orthographicScale = Double(half / FILL)
camera.zNear = 0.001
camera.zFar = Double(max(1000, diag * 20))
let camNode = SCNNode()
camNode.camera = camera
camNode.simdPosition = center + dir * max(diag * 2, 1)
camNode.simdLook(at: center, up: SIMD3<Float>(0, 1, 0), localFront: SIMD3<Float>(0, 0, -1))
scene.rootNode.addChildNode(camNode)

// A key from above and to the camera's right, the way Mokei's sun falls on the diorama, and a soft fill.
let key = SCNNode()
key.light = SCNLight()
key.light?.type = .directional
key.light?.intensity = 900
key.light?.color = NSColor.white
key.simdPosition = center + SIMD3<Float>(sin(HOME_AZ + 0.7), 1.3, cos(HOME_AZ + 0.7)) * max(diag, 1)
key.simdLook(at: center, up: SIMD3<Float>(0, 1, 0), localFront: SIMD3<Float>(0, 0, -1))
scene.rootNode.addChildNode(key)
let fill = SCNNode()
fill.light = SCNLight()
fill.light?.type = .ambient
fill.light?.intensity = 420
scene.rootNode.addChildNode(fill)

scene.background.contents = NSColor(calibratedWhite: 0.12, alpha: 1)

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs(PreviewError.metal.description + "\n", stderr)
    exit(1)
}
let renderer = SCNRenderer(device: device, options: nil)
renderer.scene = scene
renderer.pointOfView = camNode
renderer.autoenablesDefaultLighting = false

let size = CGSize(width: 1024, height: 1024)
let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fputs(PreviewError.write(outURL.path).description + "\n", stderr)
    exit(1)
}
do {
    try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try png.write(to: outURL)
    print("preview-glb: \(outURL.path)")
} catch {
    fputs("preview-glb: \(error)\n", stderr)
    exit(1)
}
