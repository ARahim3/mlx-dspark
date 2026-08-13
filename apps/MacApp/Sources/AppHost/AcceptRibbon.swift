import AppCore
import SwiftUI

/// The app's signature element: the engine's heartbeat as a strip of round ticks.
///
/// Each tick is one speculative round — height is how many tokens it committed, color is where
/// the draft came from (spark blue = drafter, purple = free lookup hit, gray = plain step).
/// It lives in the status bar, so speculation is *visible* whenever anything generates —
/// including when the client is Claude Code and not this window. No other local-LLM app can
/// draw this, because no other app measures it.
struct AcceptRibbon: View {
    let rounds: [RoundEvent]
    var maxTicks: Int = 90

    var body: some View {
        Canvas { context, size in
            let ticks = Array(rounds.suffix(maxTicks))
            guard !ticks.isEmpty else { return }
            let slot = size.width / CGFloat(maxTicks)
            let barWidth = max(1.5, slot * 0.6)
            // Committed tokens per round: 1 (plain step) up to cap+1. Normalize against the
            // window's own max so the ribbon always uses its full height.
            let peak = CGFloat(max(ticks.map(\.committed).max() ?? 1, 2))
            // Right-aligned: the newest round hugs the right edge and history slides left.
            let start = size.width - CGFloat(ticks.count) * slot
            for (i, round) in ticks.enumerated() {
                let h = max(2, size.height * CGFloat(round.committed) / peak)
                let rect = CGRect(x: start + CGFloat(i) * slot,
                                  y: size.height - h,
                                  width: barWidth,
                                  height: h)
                context.fill(Path(roundedRect: rect, cornerRadius: 0.75),
                             with: .color(Theme.source(round.source).opacity(0.85)))
            }
        }
        .frame(maxWidth: 260)
        .frame(height: 14)
        .help("Live speculative rounds — height is tokens committed; blue drafted, "
              + "purple free lookup, gray plain.")
        .accessibilityLabel("Live speculation activity")
    }
}
