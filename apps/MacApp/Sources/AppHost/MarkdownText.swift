import SwiftUI

/// A lightweight markdown renderer for assistant messages.
///
/// Deliberately *not* a full markdown engine. The one thing a local *coding* model produces
/// that raw text ruins is a fenced code block — indentation collapses, ``**`` litters the
/// output, and a copy button is sorely missed. So this splits the message into paragraphs and
/// code blocks, renders code blocks as monospace panels with copy, and hands the prose to
/// SwiftUI's built-in inline markdown (which handles ``**bold**``, ``*italic*``, ``code`` and
/// links well enough). Headers and list bullets get a light touch on top.
///
/// It also has to render *partial* markdown: during streaming a code fence may be open with no
/// closing ``` yet. An unterminated fence is treated as a code block to the end of what's
/// arrived, so code renders as code the moment it starts, not only once it finishes.
struct MarkdownText: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(MarkdownBlock.parse(text).enumerated()), id: \.offset) { _, block in
                switch block {
                case .prose(let lines):
                    ProseBlock(lines: lines)
                case .code(let language, let code):
                    CodeCard(language: language, code: code)
                }
            }
        }
    }
}

enum MarkdownBlock {
    case prose([String])
    case code(language: String?, code: String)

    /// Split on ``` fences. An unclosed fence (mid-stream) runs to the end.
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var prose: [String] = []
        var inCode = false
        var codeLines: [String] = []
        var language: String?

        func flushProse() {
            if !prose.isEmpty { blocks.append(.prose(prose)); prose = [] }
        }

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                if inCode {
                    blocks.append(.code(language: language, code: codeLines.joined(separator: "\n")))
                    codeLines = []
                    language = nil
                    inCode = false
                } else {
                    flushProse()
                    language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                    language = (language?.isEmpty ?? true) ? nil : language
                    inCode = true
                }
                continue
            }
            if inCode {
                codeLines.append(line)
            } else {
                prose.append(line)
            }
        }
        // Mid-stream: an open fence renders as code up to whatever has arrived.
        if inCode { blocks.append(.code(language: language, code: codeLines.joined(separator: "\n"))) }
        flushProse()
        return blocks
    }
}

struct ProseBlock: View {
    let lines: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                lineView(line)
            }
        }
    }

    @ViewBuilder
    private func lineView(_ line: String) -> some View {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            Spacer().frame(height: 2)
        } else if let header = headerLevel(trimmed) {
            Text(inline(String(trimmed.drop(while: { $0 == "#" || $0 == " " }))))
                .font(header == 1 ? .title3.bold() : header == 2 ? .headline : .subheadline.bold())
                .padding(.top, 2)
        } else if let bullet = bulletBody(trimmed) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("•").foregroundStyle(.secondary)
                Text(inline(bullet)).frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            Text(inline(line)).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func headerLevel(_ line: String) -> Int? {
        guard line.hasPrefix("#") else { return nil }
        let hashes = line.prefix(while: { $0 == "#" }).count
        return (1...4).contains(hashes) && line.dropFirst(hashes).hasPrefix(" ") ? hashes : nil
    }

    private func bulletBody(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    /// Inline markdown via AttributedString, falling back to plain text if it can't parse
    /// (a stray unbalanced `*` mid-stream, say — better plain than an exception).
    private func inline(_ s: String) -> AttributedString {
        (try? AttributedString(
            markdown: s,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(s)
    }
}

struct CodeCard: View {
    let language: String?
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code")
                    .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                Spacer()
                CopyButton(text: code)
            }
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(.quaternary.opacity(0.4))

            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 12, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(.quaternary.opacity(0.22))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary.opacity(0.5)))
    }
}
