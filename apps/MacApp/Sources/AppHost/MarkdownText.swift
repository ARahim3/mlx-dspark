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
                case .math(let latex):
                    MathBlock(latex: latex)
                case .table(let header, let rows):
                    TableBlock(header: header, rows: rows)
                }
            }
        }
    }
}

enum MarkdownBlock {
    case prose([String])
    case code(language: String?, code: String)
    case math(String)
    case table(header: [String]?, rows: [[String]])

    /// Split on ``` fences, `$$`/`\[ … \]` display-math blocks, and `|`-row tables. An
    /// unclosed fence or math block (mid-stream) runs to the end.
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var prose: [String] = []
        var inCode = false
        var inMath = false
        var mathCloser = "$$"
        var codeLines: [String] = []
        var mathLines: [String] = []
        var tableLines: [String] = []
        var language: String?

        func flushProse() {
            if !prose.isEmpty { blocks.append(.prose(prose)); prose = [] }
        }
        func flushTable() {
            if !tableLines.isEmpty { blocks.append(makeTable(tableLines)); tableLines = [] }
        }
        func openMath(body: String, closer: String) {
            flushProse()
            // One-line `$$ x $$` / `\[ x \]`, or an opener whose block spans several lines.
            if body.hasSuffix(closer), body.count >= closer.count {
                blocks.append(.math(String(body.dropLast(closer.count))))
            } else {
                inMath = true
                mathCloser = closer
                if !body.isEmpty { mathLines.append(body) }
            }
        }

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if inMath {
                if trimmed.hasSuffix(mathCloser) {
                    let body = trimmed.dropLast(mathCloser.count)
                        .trimmingCharacters(in: .whitespaces)
                    if !body.isEmpty { mathLines.append(body) }
                    blocks.append(.math(mathLines.joined(separator: " ")))
                    mathLines = []
                    inMath = false
                } else {
                    mathLines.append(line)
                }
                continue
            }
            if !inCode, trimmed.hasPrefix("|") {
                // Consecutive pipe rows form a table (separator row detected in makeTable).
                if tableLines.isEmpty { flushProse() }
                tableLines.append(trimmed)
                continue
            }
            flushTable()
            if !inCode, trimmed.hasPrefix("$$") {
                openMath(body: String(trimmed.dropFirst(2)), closer: "$$")
                continue
            }
            if !inCode, trimmed.hasPrefix("\\[") {
                // LaTeX display math — models emit this form as often as `$$`.
                openMath(body: String(trimmed.dropFirst(2))
                    .trimmingCharacters(in: .whitespaces), closer: "\\]")
                continue
            }
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
        // Mid-stream: an open fence/block renders up to whatever has arrived.
        if inCode { blocks.append(.code(language: language, code: codeLines.joined(separator: "\n"))) }
        if inMath { blocks.append(.math(mathLines.joined(separator: " "))) }
        flushTable()
        flushProse()
        return blocks
    }

    /// `| a | b |` rows → a table block. A `|---|:--:|` second row marks row 1 as the header.
    static func makeTable(_ lines: [String]) -> MarkdownBlock {
        func cells(_ line: String) -> [String] {
            var l = line
            if l.hasPrefix("|") { l.removeFirst() }
            if l.hasSuffix("|") { l.removeLast() }
            return l.components(separatedBy: "|")
                .map { $0.trimmingCharacters(in: .whitespaces) }
        }
        var rows = lines.map(cells)
        var header: [String]?
        if rows.count >= 2, !rows[1].isEmpty, rows[1].allSatisfy({ cell in
            !cell.isEmpty && cell.allSatisfy { "-: ".contains($0) }
        }) {
            header = rows[0]
            rows.removeSubrange(0...1)
        }
        return .table(header: header, rows: rows)
    }
}

/// Display math, centered and set slightly larger in a serif — visibly "an equation", even
/// though it is Unicode translation rather than TeX typesetting (see `MathText`).
struct MathBlock: View {
    let latex: String
    @Environment(\.textZoom) private var zoom

    var body: some View {
        Text(MathText.unicode(latex))
            .font(.system(size: 15 * zoom, design: .serif))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 6)
    }
}

struct ProseBlock: View {
    let lines: [String]
    @Environment(\.textZoom) private var zoom

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
            // Explicit sizes rather than .title3/.headline so the zoom scales headers with
            // the body; at zoom 1.0 these match the text styles they replace.
            Text(inline(String(trimmed.drop(while: { $0 == "#" || $0 == " " }))))
                .font(.system(size: (header == 1 ? 15 : header == 2 ? 13 : 12) * zoom).bold())
                .padding(.top, 2)
        } else if let bullet = bulletBody(trimmed) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("•").foregroundStyle(.secondary)
                Text(inline(bullet)).frame(maxWidth: .infinity, alignment: .leading)
            }
            .font(.system(size: 13 * zoom))
        } else if let (marker, body) = numberedBody(trimmed) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker).foregroundStyle(.secondary).monospacedDigit()
                Text(inline(body)).frame(maxWidth: .infinity, alignment: .leading)
            }
            .font(.system(size: 13 * zoom))
        } else {
            Text(inline(line)).font(.system(size: 13 * zoom))
                .frame(maxWidth: .infinity, alignment: .leading)
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

    /// "1. body" / "12) body" → ("1.", "body"). Models number lists constantly; rendering the
    /// marker separately keeps the numbers aligned and the body wrapping cleanly.
    private func numberedBody(_ line: String) -> (String, String)? {
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let rest = line.dropFirst(digits.count)
        guard let punct = rest.first, punct == "." || punct == ")",
              rest.dropFirst().first == " " else { return nil }
        return ("\(digits)\(punct)", String(rest.dropFirst(2)))
    }

    private func inline(_ s: String) -> AttributedString { inlineMarkdown(s) }
}

/// Inline markdown via AttributedString, falling back to plain text if it can't parse
/// (a stray unbalanced `*` mid-stream, say — better plain than an exception).
/// `$…$` / `\(…\)` math spans are translated to Unicode first (see `MathText`).
func inlineMarkdown(_ s: String) -> AttributedString {
    let text = s.contains("$") || s.contains("\\(") ? MathText.inlineReplaced(s) : s
    return (try? AttributedString(
        markdown: text,
        options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
        ?? AttributedString(text)
}

/// A markdown table, rendered as a real grid: header row shaded, zebra rows, cells taking
/// inline markdown + math, the whole thing horizontally scrollable when it outgrows the
/// bubble. Missing trailing cells (a common model slip) render empty rather than crashing
/// the row.
struct TableBlock: View {
    let header: [String]?
    let rows: [[String]]
    @Environment(\.textZoom) private var zoom

    var body: some View {
        let columns = max(header?.count ?? 0, rows.map(\.count).max() ?? 0)
        if columns > 0 {
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
                    if let header {
                        GridRow {
                            ForEach(0..<columns, id: \.self) { i in
                                cell(i < header.count ? header[i] : "", header: true)
                                    .background(.quaternary.opacity(0.4))
                            }
                        }
                    }
                    ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                        GridRow {
                            ForEach(0..<columns, id: \.self) { i in
                                cell(i < row.count ? row[i] : "", header: false)
                                    .background(index.isMultiple(of: 2)
                                                ? Color.clear
                                                : Color.secondary.opacity(0.06))
                            }
                        }
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary.opacity(0.5)))
        }
    }

    private func cell(_ text: String, header: Bool) -> some View {
        Text(inlineMarkdown(text))
            .font(.system(size: 12 * zoom).weight(header ? .semibold : .regular))
            .textSelection(.enabled)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct CodeCard: View {
    let language: String?
    let code: String
    @Environment(\.textZoom) private var zoom

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
                Text(SyntaxHighlight.highlight(code, size: 12 * zoom))
                    .font(.system(size: 12 * zoom, design: .monospaced))
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
