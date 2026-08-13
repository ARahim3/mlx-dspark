import AppCore
import Foundation
import SwiftUI

/// Point a coding agent at the local model. The v0.6.0 payoff, made visible.
///
/// The principle throughout: **the raw config is always shown**, never hidden behind a magic
/// button. A one-click apply is a convenience on top of something the user can see and paste
/// themselves — because when a magic button fails, a user with no visible config is stuck.
struct AgentsScreen: View {
    @EnvironmentObject private var model: AppModel
    @State private var integrations: [Integration] = []
    @State private var baseURL: String = ""
    @State private var loadError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                if let loadError {
                    Label(loadError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange).font(.callout)
                }
                ForEach(integrations) { integration in
                    AgentCard(integration: integration)
                }
            }
            .padding(16)
        }
        .task { await load() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Run your coding agent on this model")
                .font(.title3.weight(.semibold))
            if !baseURL.isEmpty {
                HStack(spacing: 6) {
                    Text("Serving at").foregroundStyle(.secondary)
                    Text(baseURL).font(.callout.monospaced())
                    CopyButton(text: baseURL)
                }
                .font(.callout)
            }
        }
    }

    private func load() async {
        guard let client = model.apiClient else { return }
        do {
            let list = try await client.integrations()
            integrations = list.integrations
            baseURL = list.baseURL
            loadError = nil
        } catch {
            loadError = "Could not load integration config: \(error.localizedDescription)"
        }
    }
}

struct AgentCard: View {
    let integration: Integration
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button { withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() } } label: {
                HStack(spacing: 10) {
                    ProtocolBadge(proto: integration.proto)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(integration.name).font(.headline)
                        Text(integration.summary).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .foregroundStyle(.secondary).imageScale(.small)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(integration.setup) { step in
                        SetupStepView(step: step)
                    }
                    if let note = integration.note {
                        Label(note, systemImage: "info.circle")
                            .font(.caption).foregroundStyle(.secondary)
                            .padding(.top, 2)
                    }
                    RoundTripTest()
                }
                .padding(.top, 12)
            }
        }
        .padding(14)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.cardStroke, lineWidth: 1))
    }
}

/// Proof the endpoint works before the user leaves to configure their agent: one real
/// completion through the same server the agent will hit, with the measured rate.
struct RoundTripTest: View {
    @EnvironmentObject private var model: AppModel

    enum TestState: Equatable {
        case idle, running
        case passed(String)
        case failed(String)
    }

    @SwiftUI.State private var state: TestState = .idle

    var body: some View {
        HStack(spacing: 10) {
            Button {
                run()
            } label: {
                Label("Test the endpoint", systemImage: "checkmark.circle")
                    .font(.caption)
            }
            .buttonStyle(.link)
            .disabled(state == .running || !model.isServerReady)

            switch state {
            case .idle:
                EmptyView()
            case .running:
                ProgressView().controlSize(.small)
            case .passed(let detail):
                Label(detail, systemImage: "checkmark.seal.fill")
                    .font(.caption).foregroundStyle(Theme.verified)
            case .failed(let message):
                Label(message, systemImage: "xmark.circle")
                    .font(.caption).foregroundStyle(Theme.warning)
                    .textSelection(.enabled)
            }
            Spacer()
        }
    }

    private func run() {
        guard let client = model.apiClient else { return }
        state = .running
        Task {
            do {
                var stats: SpecInfo?
                var text = ""
                for try await event in client.streamChat(
                    model: model.model,
                    messages: [["role": "user", "content": "Reply with the single word: ready"]],
                    maxTokens: 16, enableThinking: false
                ) {
                    switch event {
                    case .delta(let piece): text += piece
                    case .reasoning: break        // thinking counts as a live round trip too
                    case .finished(let info): stats = info
                    }
                }
                let rate = stats.map { String(format: " · %.0f tok/s", $0.tokensPerSec) } ?? ""
                state = text.isEmpty ? .failed("The server answered with no text.")
                                     : .passed("Round trip OK\(rate)")
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }
}

struct ProtocolBadge: View {
    let proto: String

    var body: some View {
        Text(proto == "anthropic" ? "Anthropic" : "OpenAI")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background((proto == "anthropic" ? Color.orange : Color.green).opacity(0.16),
                        in: Capsule())
            .foregroundStyle(proto == "anthropic" ? .orange : .green)
    }
}

struct SetupStepView: View {
    let step: Integration.SetupStep
    @State private var writeResult: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(step.label).font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Spacer()
                if let path = step.path {
                    Text(path).font(.caption2.monospaced()).foregroundStyle(.tertiary)
                }
            }

            switch step.kind {
            case .command:
                CodeBlock(text: step.command ?? "")
            case .file:
                CodeBlock(text: step.content ?? "")
            case .env:
                CodeBlock(text: (step.copyText ?? ""))
            case .fields:
                FieldsView(fields: step.fields ?? [:])
            }

            HStack(spacing: 10) {
                if let copy = step.copyText {
                    CopyButton(text: copy, label: "Copy")
                }
                // Writing a file is offered only where it is safe: a merge-free provider file a
                // user drops in. Never for env exports (those belong in the user's own shell)
                // and never blindly appended to a config that may already have content.
                if step.kind == .file, let path = step.path, let content = step.content,
                   canOfferWrite(path) {
                    Button("Save to \(shortPath(path))") {
                        writeResult = writeFile(path: path, content: content)
                    }
                    .buttonStyle(.link).font(.caption)
                }
                if let writeResult {
                    Text(writeResult).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// Only offer to create a file we won't clobber. If it already exists the user may have
    /// their own providers in it, so we show the content to merge by hand instead of
    /// overwriting — a lost config is a far worse outcome than a copy-paste.
    private func canOfferWrite(_ path: String) -> Bool {
        let expanded = (path as NSString).expandingTildeInPath
        return !FileManager.default.fileExists(atPath: expanded)
    }

    private func shortPath(_ path: String) -> String {
        (path as NSString).lastPathComponent
    }

    private func writeFile(path: String, content: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
            return "Saved ✓"
        } catch {
            return "Couldn't save: \(error.localizedDescription)"
        }
    }
}

struct FieldsView: View {
    let fields: [String: String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(fields.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                HStack(spacing: 8) {
                    Text(key).font(.caption).foregroundStyle(.secondary)
                        .frame(width: 74, alignment: .leading)
                    Text(value).font(.caption.monospaced()).textSelection(.enabled)
                    CopyButton(text: value)
                    Spacer()
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }
}

struct CodeBlock: View {
    let text: String

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Text(text)
                .font(.system(size: 11, design: .monospaced))
                .textSelection(.enabled)
                .padding(10)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }
}

struct CopyButton: View {
    let text: String
    var label: String?
    @State private var copied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            copied = true
            Task { try? await Task.sleep(nanoseconds: 1_200_000_000); copied = false }
        } label: {
            let content = Label(copied ? "Copied" : (label ?? "Copy"),
                                systemImage: copied ? "checkmark" : "doc.on.doc")
                .font(.caption)
            if label == nil {
                content.labelStyle(.iconOnly)
            } else {
                content.labelStyle(.titleAndIcon)
            }
        }
        .buttonStyle(.link)
    }
}
