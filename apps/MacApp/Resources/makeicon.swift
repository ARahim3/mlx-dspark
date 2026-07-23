import AppKit

// Draw a 1024×1024 app-icon master: a rounded-rect ("squircle"-ish) with a diagonal
// blue→violet gradient and a white lightning bolt — the same spark the menu-bar glyph uses,
// so the app reads as one thing across the Dock and the menu bar.

let size = 1024.0
let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()

let ctx = NSGraphicsContext.current!.cgContext

// macOS icons sit on a rounded rect inset from the canvas edge (the system adds the shadow).
let inset = size * 0.09
let rect = CGRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)
let radius = rect.width * 0.235
let squircle = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
squircle.addClip()

// Diagonal gradient background.
let gradient = NSGradient(colors: [
    NSColor(calibratedRed: 0.29, green: 0.36, blue: 0.95, alpha: 1),   // indigo
    NSColor(calibratedRed: 0.55, green: 0.30, blue: 0.92, alpha: 1),   // violet
])!
gradient.draw(in: rect, angle: -55)

// A soft top-left highlight for a little depth.
let highlight = NSGradient(colors: [
    NSColor(white: 1, alpha: 0.20),
    NSColor(white: 1, alpha: 0.0),
])!
highlight.draw(in: rect, relativeCenterPosition: NSPoint(x: -0.4, y: 0.5))

// Lightning bolt, centred. Points defined in a 0–1 box then scaled.
let cx = size / 2, cy = size / 2
let s = size * 0.30                        // bolt half-height-ish scale
func p(_ x: Double, _ y: Double) -> NSPoint {
    NSPoint(x: cx + x * s, y: cy + y * s)   // y up (AppKit)
}
let bolt = NSBezierPath()
bolt.move(to: p(0.28, 1.05))
bolt.line(to: p(-0.55, 0.02))
bolt.line(to: p(-0.05, 0.02))
bolt.line(to: p(-0.28, -1.05))
bolt.line(to: p(0.55, 0.14))
bolt.line(to: p(0.05, 0.14))
bolt.close()

// Subtle drop shadow under the bolt.
ctx.setShadow(offset: CGSize(width: 0, height: -size * 0.012),
              blur: size * 0.03,
              color: NSColor(white: 0, alpha: 0.28).cgColor)
NSColor.white.setFill()
bolt.fill()

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("could not encode icon")
}
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon_1024.png"
try! png.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
