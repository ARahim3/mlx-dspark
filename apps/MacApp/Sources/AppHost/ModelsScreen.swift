import AppCore
import SwiftUI

/// Which models this Mac can actually run, and which are already on disk.
///
/// The design borrows LM Studio's most-praised idea — answer "will this fit?" *before* someone
/// downloads 15 GB — and adds the thing only this project has to show: the **pair**. Every row
/// names the target and the drafter that auto-resolves for it, because a speculative setup is
/// two models or it is nothing.
struct ModelsScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if let ram = model.doctorReport?.environment.ramGB {
                    Text("\(model.doctorReport?.environment.device ?? "This Mac") · \(ram, specifier: "%.0f") GB")
                        .font(.caption).foregroundStyle(.secondary)
                }

                ForEach(model.models) { row in
                    ModelRowView(row: row, isLoaded: row.target == model.model)
                }

                if model.models.isEmpty {
                    ContentUnavailableView("No models listed", systemImage: "shippingbox")
                        .frame(height: 200)
                }
            }
            .padding(16)
        }
        .task { await model.refreshDiagnostics() }
    }
}

struct ModelRowView: View {
    let row: ModelRow
    let isLoaded: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(row.shortTarget).font(.headline)
                    if isLoaded {
                        Text("loaded")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(.tint.opacity(0.18), in: Capsule())
                    }
                }

                // The pairing — this app's domain language.
                if let drafter = row.shortDrafter {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.triangle.merge").imageScale(.small)
                        Text(drafter).font(.caption.monospaced())
                    }
                    .foregroundStyle(.secondary)
                }

                HStack(spacing: 10) {
                    badge(fitsLabel, systemImage: fitsSymbol, tint: fitsTint)
                    badge(stateLabel, systemImage: stateSymbol, tint: row.ready ? .green : .secondary)
                    if let ram = row.ram {
                        Text(ram).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }

    private var fitsLabel: String {
        switch row.fits {
        case true?:  return "fits this Mac"
        case false?: return "too large"
        default:     return "unknown"
        }
    }

    private var fitsSymbol: String {
        switch row.fits {
        case true?:  return "checkmark.circle.fill"
        case false?: return "exclamationmark.triangle.fill"
        default:     return "questionmark.circle"
        }
    }

    private var fitsTint: Color {
        switch row.fits {
        case true?:  return .green
        case false?: return .orange
        default:     return .secondary
        }
    }

    /// Deliberately says which *half* is missing: with speculative decoding a model can be
    /// half-downloaded in a way that matters, and "not downloaded" would hide that.
    private var stateLabel: String {
        if row.ready { return "ready" }
        if row.targetInstalled { return "drafter not downloaded" }
        if row.drafterInstalled { return "target not downloaded" }
        return "not downloaded"
    }

    private var stateSymbol: String {
        row.ready ? "internaldrive.fill" : "arrow.down.circle"
    }

    private func badge(_ text: String, systemImage: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage).imageScale(.small).foregroundStyle(tint)
            Text(text)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}
