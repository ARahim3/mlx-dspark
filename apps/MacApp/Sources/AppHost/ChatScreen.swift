import AppCore
import SwiftUI

struct ChatScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 18) {
                        if model.messages.isEmpty { EmptyChat() }
                        ForEach(model.messages) { message in
                            MessageView(message: message,
                                        isStreaming: model.isGenerating
                                            && message.id == model.messages.last?.id)
                                .id(message.id)
                        }
                    }
                    .padding(20)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.messages.last?.text) { _, _ in
                    if let last = model.messages.last?.id {
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }

            Divider()
            Composer()
        }
    }
}

struct EmptyChat: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ask anything.").font(.title3.weight(.medium))
            Text(model.detail.showsLab
                 ? "Every round is measured — open the Lab to watch acceptance while it generates."
                 : "Running locally on your Mac.")
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 40)
    }
}

struct MessageView: View {
    let message: ChatMessage
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message.role == .user ? "You" : "Assistant")
                .font(.caption.weight(.semibold))
                .foregroundStyle(message.role == .user
                                 ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tint))

            if message.text.isEmpty && isStreaming {
                ProgressView().controlSize(.small)
            } else if message.role == .assistant {
                // Assistant output is markdown (a coding model emits fenced code constantly);
                // the user's own message stays plain so their literal text isn't reinterpreted.
                MarkdownText(text: message.text)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let stats = message.stats { StatsStrip(stats: stats) }
        }
    }
}

/// Per-turn speculative-decoding numbers, under the message that produced them.
struct StatsStrip: View {
    let stats: SpecInfo

    var body: some View {
        HStack(spacing: 14) {
            item(String(format: "%.1f", stats.tokensPerSec), "tok/s")
            item(String(format: "%.2f", stats.acceptLen), "per round")
            if let cap = stats.cap { item("\(cap)", "cap") }
            item("\(stats.targetForwards)", "verifies")
            if let lookup = stats.lookupRounds, lookup > 0 {
                item("\(lookup)", "free drafts")
            }
            Text(stats.mode.uppercased())
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(.tint.opacity(0.15), in: Capsule())
            Spacer()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.top, 2)
    }

    private func item(_ value: String, _ label: String) -> some View {
        HStack(spacing: 3) {
            Text(value).monospacedDigit().fontWeight(.medium).foregroundStyle(.primary)
            Text(label)
        }
    }
}

struct Composer: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message", text: $model.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .font(.body)
                .onSubmit(model.send)

            if model.isGenerating {
                Button("Stop", systemImage: "stop.fill") { model.cancelGeneration() }
                    .labelStyle(.iconOnly)
            } else {
                Button("Send", systemImage: "arrow.up") { model.send() }
                    .labelStyle(.iconOnly)
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(!model.isServerReady
                              || model.prompt.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            if !model.messages.isEmpty {
                Button("Clear", systemImage: "trash") { model.clearChat() }
                    .labelStyle(.iconOnly)
            }
        }
        .buttonStyle(.borderless)
        .padding(14)
    }
}
