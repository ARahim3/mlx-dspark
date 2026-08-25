import AppCore
import SwiftUI

/// Everything runnable, in three layers: the measured pairs the registry vouches for, whatever
/// is already on this disk, and any Hugging Face repo the user cares to type.
///
/// The design keeps LM Studio's most-praised idea — answer "will this fit?" *before* someone
/// downloads 15 GB — and adds the thing only this project has to show: the **pair**. A row
/// names the target and the drafter that auto-resolves for it, because a speculative setup is
/// two models or it is nothing. Anything without a pair still runs: `--mode auto` gives every
/// model drafter-free lookup speculation.
struct ModelsScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let error = model.modelSwitchError {
                    SwapErrorBanner(message: error) { model.modelSwitchError = nil }
                }

                if model.isModelLoading {
                    // The screen most downloads start from is also where they should be
                    // stoppable — the same progress + Cancel the empty chat shows.
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Loading \(model.model.components(separatedBy: "/").last ?? model.model)…")
                                .font(.callout.weight(.medium))
                        }
                        if let dl = model.downloadProgress {
                            DownloadProgressRow(progress: dl)
                        }
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.5)))
                }

                header

                AnyModelField()

                sectionTitle("Measured pairs",
                             note: "Benchmarked drafter pairings — the speedups this project vouches for.")
                // Not-downloaded rows stay clickable: they confirm, then download-and-load.
                // They used to be disabled outright — which left a fresh machine with every
                // pair greyed out and no way to download from this screen at all (issue #15).
                ForEach(model.models) { row in
                    // `model.model` is the SELECTED target, set before a load succeeds — on
                    // its own it left a "loaded" badge on a model whose load failed (PR #18).
                    ModelRowView(row: row, isLoaded: model.isServerReady && row.target == model.model,
                                 canLoad: row.target != model.model && !model.isModelLoading) {
                        Task { await model.switchModel(to: row.target) }
                    }
                }
                if model.models.isEmpty {
                    ContentUnavailableView("No models listed", systemImage: "shippingbox")
                        .frame(height: 160)
                }

                let onDisk = model.installedModels.filter { !$0.isDrafter }
                if !onDisk.isEmpty {
                    sectionTitle("On this Mac",
                                 note: "Already downloaded. Anything here loads instantly; "
                                     + "unpaired models run with lookup speculation.")
                    ForEach(onDisk) { installed in
                        InstalledRowView(installed: installed,
                                         isLoaded: model.isServerReady && installed.repo == model.model)
                    }
                }

                let drafters = model.installedModels.filter(\.isDrafter)
                if !drafters.isEmpty {
                    DisclosureGroup {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(drafters) { drafter in
                                HStack {
                                    Text(drafter.shortRepo).font(.caption.monospaced())
                                    Spacer()
                                    Text(drafter.size).font(.caption).foregroundStyle(.secondary)
                                    RevealButton(path: drafter.path)
                                }
                            }
                        }
                        .padding(.top, 6)
                    } label: {
                        Text("Drafters on disk (\(drafters.count)) — resolved automatically")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(.top, 2)
                }
            }
            .padding(16)
        }
        .task { await model.refreshDiagnostics() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                if let ram = model.doctorReport?.environment.ramGB {
                    Text("\(model.doctorReport?.environment.device ?? "This Mac") · \(ram, specifier: "%.0f") GB memory")
                }
                if let disk = model.diskUsage {
                    Text("· \(disk.total) of models on disk")
                }
            }
            .font(.caption).foregroundStyle(.secondary)

            Text("Switching swaps the model in place — the server and its port stay up, so "
                 + "connected agents keep working.")
                .font(.caption).foregroundStyle(.secondary)

            // The badges are M4 Pro measurements. Say how this Mac compares instead of
            // letting an M5 Max owner read them as a ceiling (or an M1 owner as a promise).
            if let bw = model.bandwidth, let scale = bw.scale, abs(scale - 1.0) > 0.05 {
                Text(String(format: "Speedup badges were measured on an M4 Pro (273 GB/s). "
                            + "This Mac's memory bandwidth is %.1f× that, so absolute tok/s "
                            + "scale roughly with it; the ratios carry over.", scale))
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func sectionTitle(_ title: String, note: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.headline)
            Text(note).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.top, 6)
    }
}

/// Load any Hugging Face repo — the engine serves anything MLX-compatible.
struct AnyModelField: View {
    @EnvironmentObject private var model: AppModel
    @State private var repo: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                TextField("org/model — any MLX model on Hugging Face",
                          text: $repo)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout.monospaced())
                    .onSubmit(load)
                Button("Load") { load() }
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmed.isEmpty || trimmed == model.model)
            }
            Text("Downloads on first load. A known target gets its drafter pair; anything "
                 + "else runs with drafter-free lookup speculation.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(12)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.cardStroke, lineWidth: 1))
    }

    private var trimmed: String {
        repo.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func load() {
        let target = trimmed
        guard !target.isEmpty else { return }
        repo = ""
        Task { await model.switchModel(to: target) }
    }
}

struct SwapErrorBanner: View {
    let message: String
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.warning)
            VStack(alignment: .leading, spacing: 2) {
                Text("That model didn't load — the previous one was restored.")
                    .font(.callout.weight(.medium))
                Text(message).font(.caption).foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Spacer()
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark").imageScale(.small)
            }
            .buttonStyle(.borderless)
        }
        .padding(12)
        .background(Theme.warning.opacity(0.10), in: RoundedRectangle(cornerRadius: 9))
    }
}

struct ModelRowView: View {
    let row: ModelRow
    let isLoaded: Bool
    var canLoad: Bool = false
    var onLoad: () -> Void = {}
    @State private var confirmingDownload = false

    var body: some View {
        Button {
            // A ready pair loads immediately; a missing one confirms first — the size is
            // right on the row, and a mis-click must not start a multi-gigabyte fetch
            // (the accident behind issue #15; cancel exists now, but asking is cheaper).
            if row.ready { onLoad() } else { confirmingDownload = true }
        } label: {
            content
        }
        .buttonStyle(.plain)
        .disabled(!canLoad)
        .confirmationDialog(
            "Download \(row.shortTarget)\(row.ram.map { " (\($0))" } ?? "")?",
            isPresented: $confirmingDownload, titleVisibility: .visible
        ) {
            Button("Download and load") { onLoad() }
        } message: {
            Text(row.fits == false
                 ? "This model looks too large for this Mac — it may not run once downloaded."
                 : "Downloads the missing pieces, then loads the pair. You can cancel mid-download.")
        }
    }

    private var content: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(row.shortTarget).font(.headline)
                    if isLoaded { LoadedBadge() }
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
                    badge(stateLabel, systemImage: stateSymbol,
                          tint: row.ready ? Theme.verified : .secondary)
                    if let ram = row.ram {
                        Text(ram).font(.caption).foregroundStyle(.secondary)
                    }
                    // Physics, not a promise: bandwidth ÷ weight bytes is the most a plain
                    // decode of these weights can do on this Mac. Speculation multiplies it.
                    if let ceiling = row.ceilingTps {
                        Text(String(format: "~%.0f tok/s plain ceiling here", ceiling))
                            .font(.caption).foregroundStyle(.secondary)
                            .help("This Mac's memory bandwidth divided by the model's weight "
                                  + "bytes — the single-stream roofline. Speculative decoding "
                                  + "is what beats it.")
                    }
                }
            }
            Spacer()
            if let speedup = row.speedup {
                Text(speedup)
                    .font(.system(.callout, design: .rounded).monospacedDigit())
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.spark)
                    .help("Measured on an M4 Pro; yours will differ")
            }
            if canLoad {
                Image(systemName: row.ready ? "arrow.right.circle" : "arrow.down.circle")
                    .foregroundStyle(.tint)
                    .help(row.ready ? "Load this model" : "Download and load this pair")
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
            isLoaded ? AnyShapeStyle(Theme.spark.opacity(0.45)) : Theme.cardStroke, lineWidth: 1))
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
        case true?:  return Theme.verified
        case false?: return Theme.warning
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

/// A model that's already on disk — load it, reveal it, or reclaim the space.
struct InstalledRowView: View {
    @EnvironmentObject private var model: AppModel
    let installed: InstalledModel
    let isLoaded: Bool
    @State private var confirmingDelete = false

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(installed.shortRepo).font(.body.weight(.medium))
                    if isLoaded { LoadedBadge() }
                }
                HStack(spacing: 8) {
                    Text(installed.size).font(.caption).foregroundStyle(.secondary)
                    if installed.registryID != nil {
                        HStack(spacing: 3) {
                            Image(systemName: "arrow.triangle.merge").imageScale(.small)
                            Text("drafter pair available")
                        }
                        .font(.caption)
                        .foregroundStyle(Theme.spark)
                    } else {
                        Text("lookup speculation").font(.caption).foregroundStyle(.secondary)
                    }
                    if let label = installed.sourceLabel {
                        Text(label).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            if !isLoaded {
                Button("Load") { Task { await model.switchModel(to: installed.repo) } }
                    .controlSize(.small)
            }
            RevealButton(path: installed.path)
            // LM Studio's downloads and the user's own model folders are not our files —
            // we read them, we never offer to delete them.
            if !installed.isExternal {
                Button {
                    confirmingDelete = true
                } label: {
                    Image(systemName: "trash").imageScale(.small)
                }
                .buttonStyle(.borderless)
                .disabled(isLoaded)
                .help(isLoaded ? "Can't delete the loaded model" : "Move to Trash")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
            isLoaded ? AnyShapeStyle(Theme.spark.opacity(0.45)) : Theme.cardStroke, lineWidth: 1))
        .confirmationDialog("Move \(installed.shortRepo) to the Trash?",
                            isPresented: $confirmingDelete, titleVisibility: .visible) {
            Button("Move to Trash (\(installed.size))", role: .destructive) {
                delete()
            }
        } message: {
            Text("Recoverable from the Trash. Loading it again later re-downloads it.")
        }
    }

    private func delete() {
        let url = URL(fileURLWithPath: installed.path)
        do {
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
            Task { await model.refreshDiagnostics() }
        } catch {
            model.modelSwitchError = "Couldn't move \(installed.shortRepo) to the Trash: "
                + error.localizedDescription
        }
    }
}

struct LoadedBadge: View {
    var body: some View {
        Text("loaded")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(.tint.opacity(0.18), in: Capsule())
            .foregroundStyle(.tint)
    }
}

struct RevealButton: View {
    let path: String

    var body: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
        } label: {
            Image(systemName: "magnifyingglass").imageScale(.small)
        }
        .buttonStyle(.borderless)
        .help("Reveal in Finder")
    }
}
