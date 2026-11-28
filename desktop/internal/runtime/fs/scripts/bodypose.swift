// bodypose: read a skeleton from each frame with Apple Vision (built into macOS).
// usage: bodypose <frames-dir> <out.json>
// Frames are PNGs sorted by name. Writes 2D joints (normalised, origin top-left)
// and, where Vision finds one, 3D joints (metres, relative to the root).
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count == 3 else { print("usage: bodypose <frames-dir> <out.json>"); exit(2) }
let dir = URL(fileURLWithPath: args[1])
let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension.lowercased() == "png" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

var frames: [[String: Any]] = []
for f in files {
    guard let img = NSImage(contentsOf: f),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let h = VNImageRequestHandler(cgImage: cg, options: [:])
    let r2 = VNDetectHumanBodyPoseRequest()
    let r3 = VNDetectHumanBodyPose3DRequest()
    try? h.perform([r2, r3])
    var entry: [String: Any] = ["file": f.lastPathComponent]
    if let obs = r2.results?.first, let pts = try? obs.recognizedPoints(.all) {
        var j: [String: [Double]] = [:]
        for (k, p) in pts where p.confidence > 0.1 {
            j[k.rawValue.rawValue] = [Double(p.location.x), Double(1 - p.location.y), Double(p.confidence)]
        }
        entry["joints2d"] = j
    }
    if let obs = r3.results?.first, let pts = try? obs.recognizedPoints(.all) {
        var j: [String: [Double]] = [:]
        for (k, p) in pts {
            let m = p.localPosition
            let w = p.position
            j[k.rawValue.rawValue] = [Double(w.columns.3.x), Double(w.columns.3.y), Double(w.columns.3.z),
                                      Double(m.columns.3.x), Double(m.columns.3.y), Double(m.columns.3.z)]
        }
        entry["joints3d"] = j
        entry["height_m"] = Double(obs.bodyHeight)
    }
    frames.append(entry)
}
let data = try JSONSerialization.data(withJSONObject: ["frames": frames], options: [.prettyPrinted, .sortedKeys])
try data.write(to: URL(fileURLWithPath: args[2]))
let n2 = frames.filter { $0["joints2d"] != nil }.count
let n3 = frames.filter { $0["joints3d"] != nil }.count
print("frames \(frames.count)  with 2D skeleton \(n2)  with 3D skeleton \(n3)")
