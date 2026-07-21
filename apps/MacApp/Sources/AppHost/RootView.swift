import AppCore
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            content
            Divider()
            StatusBar()
            if model.showLogs {
                Divider()
                LogPane()
            }
        }
        .frame(minWidth: 720, minHeight: 520)
        .task { await model.boot() }
    }

    @ViewBuilder
    private var content: some View {
        switch model.phase {
        case .launching, .settingUp, .startingServer:
            SetupView()
        case .ready:
            ChatView()
        case .failed:
            FailureView()
        }
    }
}

// MARK: - Setup

/// The onboarding checklist. Note the wording: the user is told an *engine* is being set up,
/// never that a Python environment is being built. That is the whole point of the vendored-uv
/// runtime — the implementation detail stays invisible.
struct SetupView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Setting up \(AppIdentity.displayName)")
                    .font(.system(size: 22, weight: .semibold))
                Text("This happens once. The first run downloads the engine, which takes a few minutes.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }

            VStack(spacing: 0) {
                ForEach(model.setupSteps) { step in
                    SetupRow(step: step)
                    if step.id != model.setupSteps.last?.id { Divider().padding(.leading, 34) }
                }
            }
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))

            if model.phase == .startingServer {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text(model.statusLine).foregroundStyle(.secondary).font(.callout)
                }
            }
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct SetupRow: View {
    let step: SetupStep

    var body: some View {
        HStack(spacing: 12) {
            icon.frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title).font(.body)
                if !detailText.isEmpty {
                    Text(detailText)
                        .font(.caption)
                        .foregroundStyle(state == .failedish ? .red : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }

    private enum Kind { case pending, running, done, failedish }
    private var state: Kind {
        switch step.state {
        case .pending: return .pending
        case .running: return .running
        case .done:    return .done
        case .failed:  return .failedish
        }
    }

    private var detailText: String {
        if case .failed(let message) = step.state { return message }
        return step.detail
    }

    @ViewBuilder
    private var icon: some View {
        switch state {
        case .pending:
            Image(systemName: "circle").foregroundStyle(.tertiary)
        case .running:
            ProgressView().controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failedish:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }
}

// MARK: - Chat

struct ChatView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            if let pairing = model.pairingLine {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.merge").imageScale(.small)
                    Text(pairing).font(.caption.monospaced())
                    Spacer()
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20).padding(.vertical, 8)
                .background(.quaternary.opacity(0.25))
                Divider()
            }
            ScrollView {
                Text(model.output.isEmpty ? "Ask something to see the engine run." : model.output)
                    .font(.system(.body, design: .default))
                    .textSelection(.enabled)
                    .foregroundStyle(model.output.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(20)
            }

            if let stats = model.lastStats {
                StatsStrip(stats: stats).padding(.horizontal, 20).padding(.bottom, 10)
            }

            Divider()
            HStack(spacing: 10) {
                TextField("Message", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .onSubmit { model.send() }
                if model.isGenerating {
                    Button("Stop") { model.cancelGeneration() }
                } else {
                    Button("Send") { model.send() }
                        .keyboardShortcut(.return, modifiers: [])
                        .disabled(!model.isServerReady)
                }
            }
            .padding(14)
        }
    }
}

/// The per-turn speculative-decoding numbers. In the real app this becomes the live accept
/// ribbon; here it proves the `x_mlx_dspark` block survives the round trip.
struct StatsStrip: View {
    let stats: SpecInfo

    var body: some View {
        HStack(spacing: 16) {
            metric(String(format: "%.1f", stats.tokensPerSec), "tok/s")
            metric(String(format: "%.2f", stats.acceptLen), "accept")
            if let cap = stats.cap { metric("\(cap)", "cap") }
            metric("\(stats.targetForwards)", "verifies")
            if let lookup = stats.lookupRounds, lookup > 0 {
                metric("\(lookup)", "lookup")
            }
            Spacer()
            Text(stats.mode.uppercased())
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 7).padding(.vertical, 3)
                .background(.tint.opacity(0.15), in: Capsule())
        }
        .font(.callout)
    }

    private func metric(_ value: String, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(value).monospacedDigit().fontWeight(.medium)
            Text(label).foregroundStyle(.secondary).font(.caption)
        }
    }
}

// MARK: - Chrome

struct StatusBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(model.isServerReady ? Color.green : Color.orange)
                .frame(width: 7, height: 7)
            Text(model.statusLine).font(.caption).foregroundStyle(.secondary)
            Spacer()
            if model.liveTokensPerSec > 0 {
                Text("\(model.liveTokensPerSec, specifier: "%.1f") tok/s")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Button(model.showLogs ? "Hide Log" : "Log") {
                model.showLogs.toggle()
            }
            .buttonStyle(.link)
            .font(.caption)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
    }
}

struct LogPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(model.logLines.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .frame(height: 170)
            .background(.quaternary.opacity(0.3))
            .onChange(of: model.logLines.count) { _, count in
                proxy.scrollTo(count - 1, anchor: .bottom)
            }
        }
    }
}

struct FailureView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 32)).foregroundStyle(.orange)
            Text("Something went wrong").font(.title3.weight(.semibold))
            Text(model.errorMessage ?? "Unknown error")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .frame(maxWidth: 460)
            Button("Try Again") { Task { await model.boot() } }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(28)
    }
}
