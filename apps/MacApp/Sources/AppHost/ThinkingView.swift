import SwiftUI

/// Splits a reasoning model's output into its `<think>…</think>` trace and the actual answer.
///
/// Qwen-class models emit their chain of thought inline, wrapped in `<think>` tags, ahead of
/// the answer. Shown raw it's a wall of the model talking to itself before it gets to the
/// point. This pulls it apart so the answer leads and the reasoning collapses out of the way.
///
/// Handles the streaming mid-thought case: while `</think>` hasn't arrived, everything after
/// `<think>` is reasoning and the answer is still empty — so the card shows "Thinking…" live,
/// then collapses once the answer starts.
struct ThinkingSplit {
    let reasoning: String?
    let answer: String
    /// True while the model is still inside the think block (no closing tag yet).
    let thinking: Bool

    static func parse(_ text: String) -> ThinkingSplit {
        guard let open = text.range(of: "<think>") else {
            // Prefilled-opener templates (Qwen3-2507 / Qwen3.8-class) generate only the
            // closer — the opener sits in the prompt. The engine splits that into the
            // reasoning channel now, but an older engine (or a saved session) streams it
            // raw: everything before a dangling </think> is reasoning.
            if let close = text.range(of: "</think>") {
                return ThinkingSplit(reasoning: trimmed(String(text[..<close.lowerBound])),
                                     answer: leadingTrimmed(String(text[close.upperBound...])),
                                     thinking: false)
            }
            return ThinkingSplit(reasoning: nil, answer: text, thinking: false)
        }
        let afterOpen = text[open.upperBound...]
        if let close = afterOpen.range(of: "</think>") {
            let reasoning = String(afterOpen[..<close.lowerBound])
            let answer = String(afterOpen[close.upperBound...])
            return ThinkingSplit(reasoning: trimmed(reasoning),
                                 answer: leadingTrimmed(answer), thinking: false)
        }
        // Still thinking: everything after <think> is reasoning, no answer yet.
        return ThinkingSplit(reasoning: trimmed(String(afterOpen)), answer: "", thinking: true)
    }

    private static func trimmed(_ s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    private static func leadingTrimmed(_ s: String) -> String {
        String(s.drop(while: { $0 == "\n" || $0 == " " }))
    }
}

/// The collapsible reasoning card. Expanded and live while the model is thinking, then
/// collapses itself the instant the answer begins — the reasoning stays one click away.
struct ThinkingCard: View {
    let reasoning: String
    let thinking: Bool
    @Environment(\.textZoom) private var zoom
    @State private var expanded: Bool
    /// Once the user opens or closes it themselves, stop auto-collapsing on their behalf.
    @State private var userToggled = false

    init(reasoning: String, thinking: Bool) {
        self.reasoning = reasoning
        self.thinking = thinking
        _expanded = State(initialValue: thinking)   // open while actively thinking
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                userToggled = true
                withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    if thinking {
                        ProgressView().controlSize(.mini)
                        Text("Thinking…")
                    } else {
                        Image(systemName: "brain")
                        Text("Reasoning")
                    }
                    Spacer()
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .imageScale(.small)
                }
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                Text(reasoning)
                    .font(.system(size: 11 * zoom))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 6)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
        .onChange(of: thinking) { _, stillThinking in
            // Auto-collapse when thinking finishes — unless the reader has taken control.
            if !stillThinking && !userToggled {
                withAnimation(.easeOut(duration: 0.2)) { expanded = false }
            }
        }
    }
}
