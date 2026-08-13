import SwiftUI

/// The app's visual vocabulary. One brand color, used with one meaning.
///
/// `spark` is the wordmark blue (#4D6BFE) and it means exactly one thing everywhere it
/// appears: *speculation paying off* — drafter-accepted tokens, speedup figures, the live
/// rate. The rest of the palette is semantic, not decorative: purple is a free lookup draft,
/// gray is a plain baseline step, green is a verified/lossless result, orange marks the knee
/// and warnings. A chart, the status bar, and the menu bar all read with the same key.
enum Theme {
    /// #4D6BFE — the `dspark` half of the wordmark.
    static let spark = Color(red: 0x4D / 255, green: 0x6B / 255, blue: 0xFE / 255)
    /// Free n-gram lookup drafts.
    static let lookup = Color.purple
    /// Plain baseline steps — no speculation.
    static let plain = Color.secondary
    /// Verified / lossless / on-disk.
    static let verified = Color.green
    /// The qmm knee, and anything needing attention.
    static let warning = Color.orange

    /// Card fill — one level above the window background.
    static let cardFill = AnyShapeStyle(.quaternary.opacity(0.25))
    /// Hairline that separates a card from the window without shouting.
    static let cardStroke = AnyShapeStyle(.separator.opacity(0.55))

    /// Color for a round's draft source (`RoundEvent.source`).
    static func source(_ source: String) -> Color {
        switch source {
        case "lookup": return lookup
        case "plain":  return .secondary
        default:       return spark
        }
    }
}
