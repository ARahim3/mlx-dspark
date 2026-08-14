import Foundation

/// One competitor in a race.
public struct RaceArm: Codable, Sendable, Hashable, Identifiable {
    public let mode: String
    public let cap: Int?

    public var id: String { cap.map { "\(mode)-\($0)" } ?? mode }

    public init(mode: String, cap: Int? = nil) {
        self.mode = mode
        self.cap = cap
    }

    public var label: String { cap.map { "\(mode) cap \($0)" } ?? mode }
}

/// How one arm finished.
public struct RaceResult: Decodable, Sendable, Identifiable {
    public let index: Int
    public let label: String
    public let mode: String
    public let cap: Int?
    public let tokens: Int
    public let seconds: Double
    public let tokensPerSec: Double
    public let acceptLen: Double
    public let targetForwards: Int
    public let lookupRounds: Int

    public var id: Int { index }

    enum CodingKeys: String, CodingKey {
        case index, label, mode, cap, tokens, seconds
        case tokensPerSec = "tokens_per_sec"
        case acceptLen = "accept_len"
        case targetForwards = "target_forwards"
        case lookupRounds = "lookup_rounds"
    }
}

/// Whether the arms produced the same tokens.
///
/// The point of the whole feature: speculation is supposed to be a speed change, not a
/// behaviour change, and this is the app *checking* that rather than asserting it.
public struct RaceVerdict: Decodable, Sendable {
    public let comparable: Bool
    public let identical: Bool?
    public let reference: String?
    public let divergences: [Divergence]?
    public let detail: String

    public struct Divergence: Decodable, Sendable, Identifiable {
        public let arm: Int
        public let label: String
        public let firstDiff: Int
        public let lengthOnly: Bool
        /// Approximate: measured on a clean re-run of the shared prefix, which is *not* the
        /// cache either arm actually had. Never treat it as a pass/fail — see the server's
        /// `_logit_margin` docstring and NOTES "Race verdict: why diverged is not wrong".
        public let margin: Double?

        public var id: Int { arm }

        enum CodingKeys: String, CodingKey {
            case arm, label, margin
            case firstDiff = "first_diff"
            case lengthOnly = "length_only"
        }
    }

    /// Content divergences only — a length difference means the arms agreed everywhere they
    /// overlapped and one simply emitted more before the limit.
    public var contentDivergences: [Divergence] {
        (divergences ?? []).filter { !$0.lengthOnly }
    }
}

/// One token as it arrived, with the offset from its own arm's start.
///
/// Keeping `tMs` is what makes an honest replay possible: the arms must run sequentially (MLX
/// is thread-affine, so two decoders cannot share the process), but their timings are real, so
/// the UI can replay them side by side afterwards at true relative speed.
public struct RaceToken: Sendable {
    public let index: Int
    public let text: String
    public let tMs: Double
}

public enum RaceEvent: Sendable {
    case started(arms: [RaceArm], maxTokens: Int)
    case armStarted(index: Int, label: String)
    case token(RaceToken)
    case armFinished(RaceResult)
    case verdict(RaceVerdict)
    case failed(String)
}

extension APIClient {
    /// Run the same prompt through several decode strategies, streamed.
    public func streamRace(prompt: String, arms: [RaceArm],
                           maxTokens: Int = 200,
                           thinking: Bool? = nil) -> AsyncThrowingStream<RaceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var payload: [String: Any] = [
                        "prompt": prompt,
                        "max_tokens": maxTokens,
                        "arms": arms.map { arm -> [String: Any] in
                            var dict: [String: Any] = ["mode": arm.mode]
                            if let cap = arm.cap { dict["cap"] = cap }
                            return dict
                        },
                    ]
                    // nil = the server's own default; the Lab toggle always sends a value so
                    // the UI state and the race configuration can't drift apart.
                    if let thinking { payload["thinking"] = thinking }
                    var request = URLRequest(url: baseURL.appendingPathComponent("admin/race"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    if let apiKey, !apiKey.isEmpty {
                        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    }
                    request.httpBody = try JSONSerialization.data(withJSONObject: payload)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var detail = ""
                        for try await line in bytes.lines { detail += line }
                        throw APIError.badStatus(http.statusCode, detail)
                    }

                    let decoder = JSONDecoder()
                    var event = ""
                    for try await line in bytes.lines {
                        if line.hasPrefix("event: ") {
                            event = String(line.dropFirst(7))
                            continue
                        }
                        guard line.hasPrefix("data: "),
                              let data = String(line.dropFirst(6)).data(using: .utf8)
                        else { continue }

                        switch event {
                        case "start":
                            struct Start: Decodable {
                                let arms: [RaceArm]
                                let max_tokens: Int
                            }
                            if let s = try? decoder.decode(Start.self, from: data) {
                                continuation.yield(.started(arms: s.arms, maxTokens: s.max_tokens))
                            }
                        case "arm_start":
                            struct ArmStart: Decodable { let index: Int; let label: String }
                            if let a = try? decoder.decode(ArmStart.self, from: data) {
                                continuation.yield(.armStarted(index: a.index, label: a.label))
                            }
                        case "token":
                            struct Tok: Decodable { let index: Int; let text: String; let t_ms: Double }
                            if let t = try? decoder.decode(Tok.self, from: data) {
                                continuation.yield(.token(RaceToken(index: t.index,
                                                                    text: t.text, tMs: t.t_ms)))
                            }
                        case "arm_done":
                            if let r = try? decoder.decode(RaceResult.self, from: data) {
                                continuation.yield(.armFinished(r))
                            }
                        case "verdict":
                            if let v = try? decoder.decode(RaceVerdict.self, from: data) {
                                continuation.yield(.verdict(v))
                            }
                        case "error":
                            struct Failure: Decodable { let message: String }
                            let message = (try? decoder.decode(Failure.self, from: data))?.message
                            continuation.yield(.failed(message ?? "race failed"))
                        case "done":
                            continuation.finish()
                            return
                        default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
