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

// Clay look for untextured meshes
let clay = SCNMaterial()
clay.diffuse.contents = NSColor(calibratedRed: 0.45, green: 0.52, blue: 0.40, alpha: 1)
clay.roughness.contents = 0.55
clay.lightingModel = .physicallyBased
scene.rootNode.enumerateChildNodes { node, _ in
    guard let geom = node.geometry else { return }
    if geom.materials.isEmpty || geom.materials.allSatisfy({ $0.diffuse.contents == nil }) {
        geom.materials = [clay]
    }
}

var minV = SCNVector3(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
var maxV = SCNVector3(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
scene.rootNode.enumerateChildNodes { node, _ in
    let (locMin, locMax) = node.boundingBox
    let corners = [
        SCNVector3(locMin.x, locMin.y, locMin.z),
        SCNVector3(locMin.x, locMin.y, locMax.z),
        SCNVector3(locMin.x, locMax.y, locMin.z),
        SCNVector3(locMin.x, locMax.y, locMax.z),
        SCNVector3(locMax.x, locMin.y, locMin.z),
        SCNVector3(locMax.x, locMin.y, locMax.z),
        SCNVector3(locMax.x, locMax.y, locMin.z),
        SCNVector3(locMax.x, locMax.y, locMax.z),
    ]
    for c in corners {
        let w = node.convertPosition(c, to: nil)
        minV.x = min(minV.x, w.x); minV.y = min(minV.y, w.y); minV.z = min(minV.z, w.z)
        maxV.x = max(maxV.x, w.x); maxV.y = max(maxV.y, w.y); maxV.z = max(maxV.z, w.z)
    }
}
let center = SCNVector3((minV.x + maxV.x) / 2, (minV.y + maxV.y) / 2, (minV.z + maxV.z) / 2)
let dx = maxV.x - minV.x, dy = maxV.y - minV.y, dz = maxV.z - minV.z
let diag = CGFloat(sqrt(dx * dx + dy * dy + dz * dz))
let dist = max(diag * 1.4, 0.5)

let camera = SCNCamera()
camera.zNear = 0.01
camera.zFar = max(1000, dist * 20)
camera.fieldOfView = 45
let camNode = SCNNode()
camNode.camera = camera
camNode.position = SCNVector3(center.x + dist * 0.65, center.y - dist * 0.95, center.z + dist * 0.4)
camNode.look(at: center)
scene.rootNode.addChildNode(camNode)

let key = SCNNode()
key.light = SCNLight()
key.light?.type = .area
key.light?.intensity = 1200
key.light?.color = NSColor.white
key.position = SCNVector3(center.x + dist, center.y - dist * 0.3, center.z + dist)
scene.rootNode.addChildNode(key)
let fill = SCNNode()
fill.light = SCNLight()
fill.light?.type = .ambient
fill.light?.intensity = 250
scene.rootNode.addChildNode(fill)

scene.background.contents = NSColor(calibratedWhite: 0.12, alpha: 1)

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs(PreviewError.metal.description + "\n", stderr)
    exit(1)
}
let renderer = SCNRenderer(device: device, options: nil)
renderer.scene = scene
renderer.pointOfView = camNode
renderer.autoenablesDefaultLighting = true

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
