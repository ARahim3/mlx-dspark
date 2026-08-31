import Foundation

/// One speculative-decoding round, as reported live by the engine's `/events` stream.
///
/// This is the atom the Lab is built from. A round is the unit at which speculation succeeds
/// or fails: the drafter proposes `drafted` tokens, the target accepts `accepted` of them, and
/// `committed` tokens (accepted + one free bonus token) join the output.
public struct RoundEvent: Decodable, Sendable, Identifiable, Equatable {
    public let seq: Int
    /// Per-request id, so a UI can separate one generation from the next.
    public let req: String
    public let mode: String
    /// Round index within its request.
    public let i: Int
    public let drafted: Int
    public let accepted: Int
    public let committed: Int
    public let cap: Int?
    /// `drafter` · `lookup` (free n-gram draft) · `plain` (baseline step, or auto-cap parked).
    public let source: String
    public let ms: Double
    /// Milliseconds since the engine started, for time-axis plotting.
    public let t: Double

    public var id: Int { seq }

    /// Tokens per second implied by this single round.
    public var tokensPerSecond: Double {
        ms > 0 ? Double(committed) / (ms / 1000.0) : 0
    }

    /// A round that drafted nothing — baseline decoding, a lookup miss, or the auto-cap
    /// controller parked because speculation was losing on this content.
    public var isPlainStep: Bool { drafted == 0 }
}

/// Aggregates over every round the engine has run.
public struct RoundStats: Decodable, Sendable, Equatable {
    public let rounds: Int
    public let committedTokens: Int
    public let meanAcceptLen: Double
    /// Fraction of all drafted tokens the target accepted.
    public let draftAcceptance: Double
    public let lookupRounds: Int
    /// **d₀, d₁, d₂ …** — the probability the drafter's i-th proposed token survives
    /// verification. This is the curve the project's entire speedup rests on, and the reason
    /// the Lab exists: no other local-LLM app can show it.
    public let positionAcceptance: [Double]
    /// How many drafts were offered at each position (the denominator behind the curve —
    /// without it a 100% at position 6 looks impressive when it's one lucky sample).
    public let positionOffered: [Int]
    /// Committed-tokens-per-round → number of rounds.
    public let acceptHistogram: [String: Int]

    enum CodingKeys: String, CodingKey {
        case rounds
        case committedTokens = "committed_tokens"
        case meanAcceptLen = "mean_accept_len"
        case draftAcceptance = "draft_acceptance"
        case lookupRounds = "lookup_rounds"
        case positionAcceptance = "position_acceptance"
        case positionOffered = "position_offered"
        case acceptHistogram = "accept_histogram"
    }
}

/// Prefill progress for one request, off the `/events` stream (engine ≥ the release after
/// 0.17.2; issue #29). The engine emits one of these per evaluated prefill chunk — the only
/// signal during the minutes a long cold prompt spends before the first token — and a final
/// `done` event on every exit path (success, error, cancel), so a display can always clear.
/// Positions are absolute prompt tokens, so `processed` starts at the reused prefix length.
public struct PrefillEvent: Decodable, Sendable, Equatable {
    /// Always `"prefill"` — the discriminator (round events carry no `type` key).
    public let type: String
    public let req: String
    public let mode: String
    public let processed: Int
    public let total: Int
    public let done: Bool?

    public var isDone: Bool { done == true }
    public var fraction: Double { total > 0 ? Double(processed) / Double(total) : 0 }
}

/// An event off the `/events` stream: a round, a periodic aggregate refresh, or prefill
/// progress.
public enum TelemetryEvent: Sendable {
    case round(RoundEvent)
    case stats(RoundStats)
    case prefill(PrefillEvent)
}

// MARK: - Calibration

/// This machine's measured cost curves — what `--max-draft auto` computes once and caches.
///
/// Plotting `verifyMs` is the single most distinctive thing this app can show: the convex
/// knee that explains why cap 2 wins on M-series, the 16–32 plateau that makes long lookup
/// drafts pay, and (on bf16 targets) the 2× cliff at width 2.
public struct Calibration: Decodable, Sendable {
    public let available: Bool
    public let reason: String?
    public let mode: String?
    public let target: String?
    public let drafter: String?
    /// Verify-forward cost in ms, keyed by verify width.
    public let verifyMs: [String: Double]?
    /// Drafter cost per round in ms — keyed by cap for dspark, one flat number for dflash.
    public let drafterMs: DrafterCost?
    public let roundOverheadMs: Double?
    public let recommendation: Recommendation?
    public let controller: Controller?

    public enum DrafterCost: Decodable, Sendable {
        case byCap([String: Double])
        case flat(Double)

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let dict = try? container.decode([String: Double].self) {
                self = .byCap(dict)
            } else {
                self = .flat(try container.decode(Double.self))
            }
        }
    }

    public struct Recommendation: Decodable, Sendable {
        /// Width at which the quantized-matmul cost curve leaves its cheap region.
        public let kneeWidth: Int
        public let dflashFullBlockViable: Bool
        public let recommend: String

        enum CodingKeys: String, CodingKey {
            case kneeWidth = "knee_width"
            case dflashFullBlockViable = "dflash_full_block_viable"
            case recommend
        }
    }

    public struct Controller: Decodable, Sendable {
        public let cap: Int
        /// Live per-position acceptance estimate driving the cap choice.
        public let p: Double
        public let rounds: Int
        public let kneeWidth: Int
        /// Predicted tok/s at each cap, from the cost model alone.
        public let predictedRates: [String: Double]

        enum CodingKeys: String, CodingKey {
            case cap, p, rounds
            case kneeWidth = "knee_width"
            case predictedRates = "predicted_rates"
        }
    }

    enum CodingKeys: String, CodingKey {
        case available, reason, mode, target, drafter, recommendation, controller
        case verifyMs = "verify_ms"
        case drafterMs = "drafter_ms"
        case roundOverheadMs = "round_overhead_ms"
    }

    /// Verify curve as sorted (width, ms) pairs, ready to plot.
    public var verifyCurve: [(width: Int, ms: Double)] {
        (verifyMs ?? [:])
            .compactMap { key, value in Int(key).map { (width: $0, ms: value) } }
            .sorted { $0.width < $1.width }
    }

    /// Drafter cost for one round at `cap` drafted tokens, if measured.
    public func drafterCost(cap: Int) -> Double? {
        switch drafterMs {
        case .byCap(let dict): return dict[String(cap)]
        case .flat(let value): return value
        case nil:              return nil
        }
    }

    /// Expected tokens committed per round at `cap` if each drafted position survives with
    /// probability `p` — the bonus token plus the geometric survival of each draft slot.
    public static func expectedCommitted(cap: Int, p: Double) -> Double {
        (0...max(0, cap)).reduce(0.0) { $0 + pow(p, Double($1)) }
    }

    /// Predicted tok/s at `cap` for acceptance `p` — the same cost model the engine's
    /// controller prices caps with, computed client-side so the acceptance can be explored
    /// interactively (the engine only reports rates at its current live estimate).
    public func predictedRate(cap: Int, p: Double) -> Double? {
        guard let verify = verifyMs?[String(cap + 1)] else { return nil }
        let draft: Double
        if cap == 0 {
            draft = 0                             // cap 0 = a plain step, no drafter run
        } else if let cost = drafterCost(cap: cap) {
            draft = cost
        } else {
            return nil
        }
        let ms = verify + draft + (roundOverheadMs ?? 0)
        guard ms > 0 else { return nil }
        return Self.expectedCommitted(cap: cap, p: p) / ms * 1000
    }
}
