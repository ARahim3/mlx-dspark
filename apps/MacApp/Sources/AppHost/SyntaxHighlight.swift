import SwiftUI

/// A deliberately small syntax highlighter for chat code blocks.
///
/// Not a grammar per language — a single tokenizer that gets comments, strings, numbers and a
/// shared keyword set right across the languages a local coding model actually emits. That is
/// ~90% of what makes code read as code; a real per-language engine would be a dependency and
/// a maintenance surface for the remaining 10%. Runs per render on streaming text, so it stays
/// regex-free on the hot path apart from four precompiled patterns.
enum SyntaxHighlight {

    /// Union of keywords across Python/Swift/JS-TS/Rust/Go/C-ish/shell/SQL. A word that is a
    /// keyword in one language and an identifier in another (`in`, `for`) still reads fine
    /// highlighted — editors with mislabeled languages do the same.
    private static let keywords: Set<String> = [
        // control flow
        "if", "else", "elif", "for", "while", "do", "switch", "case", "default", "break",
        "continue", "return", "yield", "guard", "defer", "match", "when", "throw", "throws",
        "try", "except", "catch", "finally", "raise", "goto",
        // declarations
        "func", "def", "fn", "function", "class", "struct", "enum", "protocol", "interface",
        "trait", "impl", "extension", "type", "typealias", "var", "let", "const", "static",
        "final", "mut", "public", "private", "protected", "internal", "override", "abstract",
        "async", "await", "lambda", "init", "deinit", "new", "delete",
        // modules
        "import", "from", "package", "module", "use", "using", "namespace", "export", "require",
        // values & operators-as-words
        "true", "false", "True", "False", "nil", "null", "None", "undefined", "self", "this",
        "super", "in", "is", "as", "not", "and", "or", "where", "with", "pass", "global",
        "nonlocal", "assert", "del",
        // SQL (upper-case common forms)
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "INSERT", "UPDATE", "DELETE",
        // shell
        "echo", "cd", "fi", "then", "esac", "done", "local", "source", "sudo",
    ]

    private static let pattern: NSRegularExpression = {
        // Order matters: comments swallow anything after their marker, strings swallow
        // would-be comments inside them, and only then do numbers/words get a look.
        let comment = #"(//[^\n]*|#[^\n]*|--[^\n]*|/\*[\s\S]*?\*/)"#
        let string = #"("""[\s\S]*?"""|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`[^`\n]*`)"#
        let number = #"(\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b|\b0x[0-9a-fA-F_]+\b)"#
        let word = #"([A-Za-z_][A-Za-z0-9_]*)"#
        return try! NSRegularExpression(pattern: [comment, string, number, word]
            .joined(separator: "|"))
    }()

    static func highlight(_ code: String, size: Double = 12) -> AttributedString {
        var result = AttributedString(code)
        let ns = code as NSString
        for match in pattern.matches(in: code, range: NSRange(location: 0, length: ns.length)) {
            let (group, color): (Int, Color?) =
                match.range(at: 1).location != NSNotFound ? (1, .secondary)          // comment
                : match.range(at: 2).location != NSNotFound ? (2, Theme.verified)     // string
                : match.range(at: 3).location != NSNotFound ? (3, Theme.warning)      // number
                : (4, nil)                                                            // word
            let nsRange = match.range(at: group)
            guard nsRange.location != NSNotFound,
                  let range = Range(nsRange, in: code),
                  let attrRange = attributedRange(range, in: code, of: result) else { continue }
            if let color {
                result[attrRange].foregroundColor = color
            } else if keywords.contains(String(code[range])) {
                result[attrRange].foregroundColor = Theme.spark
                // Same size as the surrounding CodeCard text (which zooms), or the semibold
                // keywords would stay put while everything else scales.
                result[attrRange].font = .system(size: size, design: .monospaced).weight(.semibold)
            }
        }
        return result
    }

    private static func attributedRange(_ range: Range<String.Index>, in source: String,
                                        of attributed: AttributedString)
        -> Range<AttributedString.Index>? {
        guard let lower = AttributedString.Index(range.lowerBound, within: attributed),
              let upper = AttributedString.Index(range.upperBound, within: attributed)
        else { return nil }
        return lower..<upper
    }
}
