import AppKit

// Draw a 1024×1024 app-icon master derived from the repo wordmark: "mlx" in black,
// "-dspark" in #4D6BFE, white card. The icon is the wordmark's monogram — a heavy,
// tightly-tracked "ds" with the same two-color break (d black, s spark blue), on the same
// white rounded card. No gradient, no glyph clip-art: the brand *is* the typography.

let size = 1024.0
let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()

let ctx = NSGraphicsContext.current!.cgContext

// macOS icons sit on a rounded rect inset from the canvas edge (the system adds the shadow).
let inset = size * 0.09
let rect = CGRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)
let radius = rect.width * 0.235
let card = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
card.addClip()

// The wordmark's white, with a whisper of cool gradient so the card doesn't read as a hole
// in dark backgrounds.
let background = NSGradient(colors: [
    NSColor.white,
    NSColor(calibratedRed: 0.955, green: 0.962, blue: 0.995, alpha: 1),
])!
background.draw(in: rect, angle: -90)

// The wordmark colors.
let ink = NSColor(calibratedRed: 0.04, green: 0.04, blue: 0.05, alpha: 1)           // near-black
let spark = NSColor(calibratedRed: 0x4D / 255.0, green: 0x6B / 255.0, blue: 0xFE / 255.0,
                    alpha: 1)                                                        // #4D6BFE

// "ds" — heavy weight and the tight tracking of the wordmark (letter-spacing -1 at 64pt
// scales to about -0.016em).
let fontSize = size * 0.52
let font = NSFont.systemFont(ofSize: fontSize, weight: .heavy)
let mark = NSMutableAttributedString()
mark.append(NSAttributedString(string: "d", attributes: [
    .font: font, .foregroundColor: ink, .kern: -fontSize * 0.035,
]))
mark.append(NSAttributedString(string: "s", attributes: [
    .font: font, .foregroundColor: spark,
]))

// Center on the glyphs' actual ink, not the line box — "d" has an ascender and "s" doesn't,
// so line-box centering would sit the mark visually low.
let line = CTLineCreateWithAttributedString(mark)
let bounds = CTLineGetImageBounds(line, ctx)
ctx.textPosition = CGPoint(x: size / 2 - bounds.midX, y: size / 2 - bounds.midY)
CTLineDraw(line, ctx)

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("could not encode icon")
}
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon_1024.png"
try! png.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
