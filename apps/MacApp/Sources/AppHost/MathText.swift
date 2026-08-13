import SwiftUI

/// LaTeX-ish math → readable Unicode, for chat.
///
/// Models answer math in LaTeX (`$…$`, `$$…$$`) because that's what their training rewards;
/// shown raw it reads like line noise. This is deliberately *not* a TeX engine (no layout, no
/// nested fraction stacking — that would mean a web view or a typesetting dependency): it
/// translates the vocabulary that actually appears in chat — operators, Greek, fractions,
/// roots, super/subscripts — into Unicode that reads correctly in a sentence. `\frac{d}{dx}
/// \sin x = \cos x` becomes `d/dx sin x = cos x`, which is the point: legible, not typeset.
enum MathText {

    private static let symbols: [String: String] = [
        // operators & relations
        "int": "∫", "iint": "∬", "oint": "∮", "sum": "Σ", "prod": "Π", "infty": "∞",
        "pm": "±", "mp": "∓", "times": "×", "div": "÷", "cdot": "·", "ast": "∗",
        "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
        "approx": "≈", "sim": "∼", "simeq": "≃", "equiv": "≡", "propto": "∝",
        "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "implies": "⇒",
        "mapsto": "↦", "partial": "∂", "nabla": "∇", "in": "∈", "notin": "∉",
        "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩", "emptyset": "∅",
        "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
        "angle": "∠", "perp": "⊥", "parallel": "∥", "degree": "°", "circ": "∘",
        "ldots": "…", "cdots": "⋯", "dots": "…", "prime": "′", "hbar": "ℏ", "ell": "ℓ",
        // greek
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
        "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
        "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
        "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
        "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
        "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
        "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
        // named functions render as their own name
        "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
        "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
        "log": "log", "ln": "ln", "lg": "lg", "exp": "exp", "lim": "lim",
        "min": "min", "max": "max", "sup": "sup", "inf": "inf", "det": "det", "mod": "mod",
        // spacing / layout commands that carry no glyph
        "left": "", "right": "", "displaystyle": "", "textstyle": "", "limits": "",
        "quad": "  ", "qquad": "   ", ",": " ", ";": " ", "!": "", " ": " ",
    ]

    private static let superscripts: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
        "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ", "x": "ˣ", "T": "ᵀ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "k": "ᵏ",
        "m": "ᵐ", "t": "ᵗ", "*": "*", "'": "′",
    ]

    private static let subscripts: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
        "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "n": "ₙ", "i": "ᵢ", "j": "ⱼ", "x": "ₓ", "a": "ₐ", "e": "ₑ", "k": "ₖ", "m": "ₘ",
        "t": "ₜ", "s": "ₛ",
    ]

    /// Translate one math span (the text between `$…$` / `$$…$$`, delimiters stripped).
    static func unicode(_ latex: String) -> String {
        var out = ""
        var s = Substring(latex)

        func bracedArgument() -> String {
            // Consume `{…}` (balanced) or a single token; returns the *translated* content.
            while s.first == " " { s.removeFirst() }
            guard s.first == "{" else {
                guard let c = s.first else { return "" }
                s.removeFirst()
                return String(c)
            }
            s.removeFirst()
            var depth = 1
            var inner = ""
            while let c = s.first {
                s.removeFirst()
                if c == "{" { depth += 1 }
                if c == "}" { depth -= 1; if depth == 0 { break } }
                inner.append(c)
            }
            return unicode(inner)
        }

        func scripted(_ table: [Character: Character], fallback: String) -> String {
            let arg = bracedArgument()
            let mapped = arg.map { table[$0] }
            if mapped.allSatisfy({ $0 != nil }) && !arg.isEmpty {
                return String(mapped.compactMap { $0 })
            }
            return arg.count == 1 ? fallback + arg : "\(fallback)(\(arg))"
        }

        while let c = s.first {
            if c == "\\" {
                s.removeFirst()
                let name = s.prefix(while: { $0.isLetter })
                if name.isEmpty {                      // \, \; \! \{ \} …
                    if let punct = s.first {
                        s.removeFirst()
                        out += symbols[String(punct)] ?? String(punct)
                    }
                    continue
                }
                s.removeFirst(name.count)
                switch String(name) {
                case "frac", "dfrac", "tfrac":
                    let num = bracedArgument()
                    let den = bracedArgument()
                    let wrap = { (t: String) in t.count > 1 && !t.allSatisfy(\.isNumber)
                        ? "(\(t))" : t }
                    out += "\(wrap(num))/\(wrap(den))"
                case "sqrt":
                    out += "√(\(bracedArgument()))"
                case "text", "mathrm", "mathbf", "mathit", "operatorname", "boldsymbol",
                     "mathcal", "mathbb", "vec", "hat", "bar", "tilde", "overline":
                    out += bracedArgument()
                case "begin", "end":                   // environments: drop the wrapper
                    _ = bracedArgument()
                default:
                    var glyph = symbols[String(name)] ?? String(name)
                    // TeX puts space around named operators; without it `\frac{1}{a}\cos x`
                    // reads as "1/acos x".
                    if glyph.first?.isLetter == true, let last = out.last,
                       last.isLetter || last.isNumber || last == ")" {
                        glyph = " " + glyph
                    }
                    out += glyph
                }
            } else if c == "^" {
                s.removeFirst()
                out += scripted(superscripts, fallback: "^")
            } else if c == "_" {
                s.removeFirst()
                out += scripted(subscripts, fallback: "_")
            } else if c == "{" || c == "}" {
                s.removeFirst()                        // grouping braces carry no glyph
            } else if c == "&" {
                s.removeFirst()                        // alignment marker
                out += " "
            } else {
                s.removeFirst()
                out.append(c)
            }
        }
        // TeX collapses whitespace; mirror that so `\sin x` and `\sin  x` read the same.
        return out.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    /// Replace inline `$…$` and `\(...\)` spans in a prose line with their Unicode form.
    /// Currency stays untouched: a span only counts as math when it has a closing `$` on the
    /// same line and doesn't look like `$5` followed by prose.
    static func inlineReplaced(_ line: String) -> String {
        var result = ""
        var rest = Substring(line)
        while let open = rest.firstIndex(of: "$") {
            let afterOpen = rest.index(after: open)
            guard let close = rest[afterOpen...].firstIndex(of: "$"),
                  close > afterOpen else { break }
            let span = String(rest[afterOpen..<close])
            result += rest[..<open]
            // Math, or money? "$120 in 1.5 hours" must not be eaten: require some math-y
            // character (a backslash, ^, _, =, or a letter adjacent to the delimiters).
            let mathy = span.contains("\\") || span.contains("^") || span.contains("_")
                || span.contains("=") || (span.first?.isLetter ?? false)
                || (span.last?.isLetter ?? false)
            result += mathy ? unicode(span) : "$" + span + "$"
            rest = rest[rest.index(after: close)...]
        }
        result += rest
        return result
            .replacingOccurrences(of: "\\(", with: "")
            .replacingOccurrences(of: "\\)", with: "")
    }
}
