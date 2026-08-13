import AppCore
import SwiftUI

/// The menu-bar item's label: a small glyph plus the live rate when generating.
///
/// Showing tok/s here is the whole appeal — a glance tells you the model is working and how
/// fast, with no window. When idle it collapses to just the glyph so it isn't noise.
struct MenuBarLabel: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: glyph)
            if model.liveTokensPerSec > 0 {
                Text("\(Int(model.liveTokensPerSec))")
                    .monospacedDigit()
            }
        }
    }

    private var glyph: String {
        switch model.phase {
        case .ready:  return "bolt.fill"
        case .failed: return "bolt.slash"
        default:      return "bolt"
        }
    }
}

/// The popover behind the menu-bar item. A compact status readout plus the two things you'd
/// open the app for from here: bring the window forward, or quit (which stops the engine).
struct MenuBarPanel: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 7) {
                Circle().fill(statusColor).frame(width: 7, height: 7)
                Text(AppIdentity.displayName).font(.headline)
                Spacer()
            }

            if let health = model.health {
                VStack(alignment: .leading, spacing: 6) {
                    row("Model", health.model)
                    if let drafter = health.drafter {
                        row("Drafter", drafter.components(separatedBy: "/").last ?? drafter)
                    }
                    row("Mode", health.mode)
                    if let memory = model.memoryLine {
                        row("Memory", memory)
                    }
                }

                Divider()

                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(model.liveTokensPerSec, specifier: "%.0f")")
                        .font(.system(size: 26, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(model.liveTokensPerSec > 0 ? Theme.spark : .secondary)
                        .contentTransition(.numericText())
                        .animation(.easeOut(duration: 0.2), value: model.liveTokensPerSec)
                    Text("tok/s").foregroundStyle(.secondary)
                    Spacer()
                    if let stats = model.stats, stats.rounds > 0 {
                        VStack(alignment: .trailing, spacing: 1) {
                            Text("accept \(stats.meanAcceptLen, specifier: "%.2f")")
                            Text("\(stats.rounds) rounds").foregroundStyle(.secondary)
                        }
                        .font(.caption)
                    }
                }

                AcceptRibbon(rounds: model.rounds, maxTicks: 60)
                    .frame(maxWidth: .infinity)
            } else {
                Text(model.statusLine).font(.callout).foregroundStyle(.secondary)
            }

            if let update = model.appUpdate {
                Label("App v\(update.version) available — brew upgrade",
                      systemImage: "arrow.down.circle")
                    .font(.caption)
                    .foregroundStyle(Theme.spark)
            }

            Divider()

            HStack {
                Button("Open Window") {
                    NSApp.activate(ignoringOtherApps: true)
                    openWindow(id: "main")
                    // A window that was closed (not just hidden) needs bringing back explicitly;
                    // activation alone won't recreate it.
                    for window in NSApp.windows where window.canBecomeMain {
                        window.makeKeyAndOrderFront(nil)
                    }
                }
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
            }
        }
        .padding(14)
        .frame(width: 260)
    }

    private var statusColor: Color {
        switch model.phase {
        case .ready:  return Theme.verified
        case .failed: return .red
        default:      return Theme.warning
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).lineLimit(1).truncationMode(.middle)
        }
        .font(.callout)
    }
}
