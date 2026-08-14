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

// MARK: - Text zoom

/// App-wide text scale for *content* (chat messages, race lanes — prose, code, math).
/// Chrome (buttons, labels, charts) deliberately does not scale: the knob exists so model
/// output is legible in a recording or from across a desk, not to resize the UI.
private struct TextZoomKey: EnvironmentKey {
    static let defaultValue: Double = 1.0
}

extension EnvironmentValues {
    var textZoom: Double {
        get { self[TextZoomKey.self] }
        set { self[TextZoomKey.self] = newValue }
    }
}

/// The appearance override: follow the system, or pin light/dark (e.g. to record a video in
/// a look that doesn't match the desktop).
enum Appearance: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }

    var label: String {
        switch self {
        case .system: return "System"
        case .light:  return "Light"
        case .dark:   return "Dark"
        }
    }

    var symbol: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light:  return "sun.max"
        case .dark:   return "moon"
        }
    }
}
