import AppCore
import SwiftUI

struct ChatScreen: View {
    @EnvironmentObject private var model: AppModel
    /// How far the content bottom sits below the viewport bottom. Near zero = reader is at
    /// the end and wants to follow the stream; large = they scrolled up to read and the
    /// stream must not yank them back down.
    @State private var gapBelow: CGFloat = 0

    private static let bottomID = "chat-bottom"

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { viewport in
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 18) {
                            if model.messages.isEmpty { EmptyChat() }
                            ForEach(model.messages) { message in
                                MessageView(message: message,
                                            isStreaming: model.isGenerating
                                                && message.id == model.messages.last?.id)
                            }
                            Color.clear.frame(height: 1).id(Self.bottomID)
                        }
                        .padding(20)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(GeometryReader { content in
                            Color.clear.preference(
                                key: ChatScrollGapKey.self,
                                value: content.frame(in: .global).maxY
                                    - viewport.frame(in: .global).maxY)
                        })
                    }
                    .onPreferenceChange(ChatScrollGapKey.self) { gapBelow = $0 }
                    .onChange(of: model.messages.last?.text) { _, _ in
                        // Follow the stream only while the reader is already at the bottom.
                        if gapBelow < 140 {
                            proxy.scrollTo(Self.bottomID, anchor: .bottom)
                        }
                    }
                    .onChange(of: model.messages.count) { _, _ in
                        // A new message (sending, or switching sessions) always re-pins.
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(Self.bottomID, anchor: .bottom)
                        }
                    }
                }
            }

            if let error = model.chatError {
                ChatErrorBanner(message: error) { model.chatError = nil }
            }

            Divider()
            Composer()
        }
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                SessionMenu()
                ChatSettingsButton()
            }
        }
    }
}

private struct ChatScrollGapKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

/// Saved conversations. Titles come from the first message; nothing to name, nothing to file.
struct SessionMenu: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Menu {
            Button("New Chat") { model.newChat() }
                .keyboardShortcut("n")
            if !model.sessions.isEmpty {
                Divider()
                ForEach(model.sessions) { session in
                    Button {
                        model.selectSession(session.id)
                    } label: {
                        if session.id == model.currentSessionID {
                            Label(session.title, systemImage: "checkmark")
                        } else {
                            Text(session.title)
                        }
                    }
                }
                Divider()
                Button("Delete This Chat", role: .destructive) {
                    if let id = model.currentSessionID { model.deleteSession(id) }
                }
                .disabled(model.currentSessionID == nil)
            }
        } label: {
            Label("Chats", systemImage: "clock.arrow.circlepath")
        }
        .help("Saved chats")
    }
}

struct ChatSettingsButton: View {
    @EnvironmentObject private var model: AppModel
    @State private var showing = false

    var body: some View {
        Button {
            showing.toggle()
        } label: {
            Label("Chat settings", systemImage: "slider.horizontal.3")
        }
        .help("System prompt, sampling, thinking")
        .popover(isPresented: $showing, arrowEdge: .bottom) {
            ChatSettingsPanel()
                .environmentObject(model)
        }
    }
}

/// Generation preferences. Everything defaults to "the model's own default" — the server
/// already reads `generation_config.json`, so the honest default is to send nothing.
struct ChatSettingsPanel: View {
    @EnvironmentObject private var model: AppModel
    @State private var maxTokensText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                Text("System prompt").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                TextEditor(text: $model.chatSettings.systemPrompt)
                    .font(.callout)
                    .frame(height: 64)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
            }

            VStack(alignment: .leading, spacing: 6) {
                Toggle(isOn: Binding(
                    get: { model.chatSettings.temperature != nil },
                    set: { model.chatSettings.temperature = $0 ? 0.7 : nil }
                )) {
                    Text("Custom temperature")
                }
                if let temperature = model.chatSettings.temperature {
                    HStack {
                        Slider(value: Binding(
                            get: { temperature },
                            set: { model.chatSettings.temperature = $0 }
                        ), in: 0...1.5)
                        Text(String(format: "%.2f", temperature))
                            .font(.caption.monospacedDigit()).frame(width: 34)
                    }
                    Text("0 is greedy — with speculation on, output is byte-identical to "
                         + "plain decoding. Sampling stays lossless too.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Max response tokens")
                    TextField("model default", text: $maxTokensText)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 110)
                        .onChange(of: maxTokensText) { _, text in
                            model.chatSettings.maxTokens = Int(text)
                        }
                }
                Text("Leave empty for the server default. Thinking counts against this budget.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            Toggle(isOn: $model.chatSettings.thinking) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allow thinking")
                    Text("Reasoning models spend real tokens thinking before answering. "
                         + "Off is faster; the answer quality trade is the model's.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .padding(16)
        .frame(width: 340)
        .onAppear {
            maxTokensText = model.chatSettings.maxTokens.map(String.init) ?? ""
        }
    }
}

struct ChatErrorBanner: View {
    let message: String
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.warning)
            Text(message)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark").imageScale(.small)
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(Theme.warning.opacity(0.10))
    }
}

struct EmptyChat: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Ask anything.").font(.title3.weight(.medium))
            if let pairing = model.pairingLine {
                Text("\(pairing) — the drafter guesses ahead, the target verifies every "
                     + "token, so this is faster with identical output.")
                    .foregroundStyle(.secondary)
            }
            if model.detail.showsLab {
                Text("Open the Lab to watch acceptance live, or race the decoders on the "
                     + "same prompt.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 40)
    }
}

struct MessageView: View {
    let message: ChatMessage
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(message.role == .user ? "You" : "Assistant")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(message.role == .user
                                     ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tint))
                if message.role == .assistant, !message.text.isEmpty {
                    CopyButton(text: ThinkingSplit.parse(message.text).answer)
                        .controlSize(.small)
                }
            }

            if message.text.isEmpty && isStreaming {
                ProgressView().controlSize(.small)
            } else if message.role == .assistant {
                AssistantContent(text: message.text)
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

/// Assistant output: a reasoning trace (if any) as a collapsible card, then the answer as
/// markdown. Splitting `<think>` out here means the answer leads instead of trailing a wall of
/// the model reasoning to itself. A coding model emits fenced code constantly, so the answer
/// goes through the markdown renderer; the user's own message stays literal (handled upstream).
struct AssistantContent: View {
    let text: String

    var body: some View {
        let split = ThinkingSplit.parse(text)
        VStack(alignment: .leading, spacing: 8) {
            if let reasoning = split.reasoning {
                ThinkingCard(reasoning: reasoning, thinking: split.thinking)
            }
            if !split.answer.isEmpty {
                MarkdownText(text: split.answer)
            } else if split.reasoning == nil {
                // No think block and no answer yet — render whatever text there is verbatim.
                MarkdownText(text: text)
            }
        }
    }
}

/// Per-turn speculative-decoding numbers, under the message that produced them.
struct StatsStrip: View {
    let stats: SpecInfo

    var body: some View {
        HStack(spacing: 14) {
            item(String(format: "%.1f", stats.tokensPerSec), "tok/s", tint: Theme.spark)
            item(String(format: "%.2f", stats.acceptLen), "per round")
            if let cap = stats.cap { item("\(cap)", "cap") }
            item("\(stats.targetForwards)", "verifies")
            if let lookup = stats.lookupRounds, lookup > 0 {
                item("\(lookup)", "free drafts", tint: Theme.lookup)
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

    private func item(_ value: String, _ label: String, tint: Color? = nil) -> some View {
        HStack(spacing: 3) {
            Text(value).monospacedDigit().fontWeight(.medium)
                .foregroundStyle(tint ?? .primary)
            Text(label)
        }
    }
}

struct Composer: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .font(.body)
                    .onSubmit(model.send)

                if model.isGenerating {
                    Button("Stop", systemImage: "stop.fill") { model.cancelGeneration() }
                        .labelStyle(.iconOnly)
                        .help("Stop generating")
                } else {
                    Button("Send", systemImage: "arrow.up") { model.send() }
                        .labelStyle(.iconOnly)
                        .keyboardShortcut(.return, modifiers: [])
                        .disabled(!model.isServerReady
                                  || model.prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            Text("⏎ send · ⌥⏎ new line")
                .font(.caption2)
                .foregroundStyle(.quaternary)
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }
}
