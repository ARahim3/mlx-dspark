import AppCore
import Foundation
import SwiftUI

/// Drives one race and, afterwards, replays it.
///
/// The arms **must** run one at a time — MLX arrays are thread- and stream-affine, so two
/// decoders cannot share the process. That is a real constraint, not a shortcut, so the live
/// view shows panes filling in turn and says which arm is running.
///
/// The replay is what makes the difference land. Each token carries the millisecond offset
/// from its own arm's start, so playing every arm back on one clock shows them pulling apart
/// at their true relative speeds. Nothing is simulated: the timings are the ones just measured.
@MainActor
final class RaceModel: ObservableObject {

    struct Lane: Identifiable {
        let index: Int
        var label: String
        var text: String = ""
        var tokens: [RaceToken] = []
        var result: RaceResult?
        var isRunning = false

        var id: Int { index }
    }

    enum Phase: Equatable { case idle, running, done, failed(String) }

    @Published var phase: Phase = .idle
    @Published var lanes: [Lane] = []
    @Published var verdict: RaceVerdict?
    @Published var isReplaying = false
    /// Text shown during replay, keyed by lane index.
    @Published var replayText: [Int: String] = [:]

    @Published var prompt: String = RacePrompt.presets[0].text
    @Published var maxTokens: Int = 200
    @Published var selectedArms: [RaceArm] = []

    private var task: Task<Void, Never>?
    private var replayTask: Task<Void, Never>?

    /// Sensible arms for whatever is loaded: the drafter mode at the default cap, at the cap
    /// the cost model likes, and the plain baseline for reference.
    func defaultArms(available: [String]) -> [RaceArm] {
        var arms: [RaceArm] = []
        if let drafterMode = available.first(where: { $0 == "dspark" || $0 == "dflash" }) {
            arms.append(RaceArm(mode: drafterMode, cap: 2))
            arms.append(RaceArm(mode: drafterMode, cap: 4))
        } else if available.contains("lookup") {
            arms.append(RaceArm(mode: "lookup"))
        }
        arms.append(RaceArm(mode: "baseline"))
        return arms
    }

    func run(client: APIClient) {
        guard phase != .running, !selectedArms.isEmpty else { return }
        cancel()
        lanes = selectedArms.enumerated().map { Lane(index: $0.offset, label: $0.element.label) }
        verdict = nil
        replayText = [:]
        isReplaying = false
        phase = .running

        task = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.streamRace(prompt: prompt, arms: selectedArms,
                                                         maxTokens: maxTokens) {
                    switch event {
                    case .started:
                        break
                    case .armStarted(let index, let label):
                        guard index < self.lanes.count else { break }
                        self.lanes[index].label = label
                        self.lanes[index].isRunning = true
                    case .token(let token):
                        guard token.index < self.lanes.count else { break }
                        self.lanes[token.index].text += token.text
                        self.lanes[token.index].tokens.append(token)
                    case .armFinished(let result):
                        guard result.index < self.lanes.count else { break }
                        self.lanes[result.index].result = result
                        self.lanes[result.index].isRunning = false
                    case .verdict(let v):
                        self.verdict = v
                    case .failed(let message):
                        self.phase = .failed(message)
                        return
                    }
                }
                if case .running = self.phase { self.phase = .done }
            } catch {
                if !Task.isCancelled { self.phase = .failed(error.localizedDescription) }
            }
        }
    }

    /// Play every lane back on one clock, at the speeds actually measured.
    func replay() {
        guard phase == .done, !lanes.isEmpty else { return }
        replayTask?.cancel()
        isReplaying = true
        replayText = Dictionary(uniqueKeysWithValues: lanes.map { ($0.index, "") })

        replayTask = Task { [weak self] in
            guard let self else { return }
            // One merged timeline so the lanes stay in true relative order; a per-lane timer
            // would drift apart and quietly misrepresent the result.
            let timeline = self.lanes
                .flatMap { lane in lane.tokens.map { (lane: lane.index, token: $0) } }
                .sorted { $0.token.tMs < $1.token.tMs }
            guard !timeline.isEmpty else {
                self.isReplaying = false
                return
            }

            let started = Date()
            for entry in timeline {
                if Task.isCancelled { break }
                let due = entry.token.tMs / 1000.0
                let elapsed = Date().timeIntervalSince(started)
                if due > elapsed {
                    try? await Task.sleep(nanoseconds: UInt64((due - elapsed) * 1_000_000_000))
                }
                if Task.isCancelled { break }
                self.replayText[entry.lane, default: ""] += entry.token.text
            }
            self.isReplaying = false
        }
    }

    func cancel() {
        task?.cancel()
        replayTask?.cancel()
        isReplaying = false
        if phase == .running { phase = .idle }
    }

    /// Fastest lane vs the slowest, for the headline number.
    var speedup: (fastest: RaceResult, slowest: RaceResult)? {
        let results = lanes.compactMap(\.result)
        guard results.count >= 2,
              let fastest = results.max(by: { $0.tokensPerSec < $1.tokensPerSec }),
              let slowest = results.min(by: { $0.tokensPerSec < $1.tokensPerSec }),
              fastest.index != slowest.index
        else { return nil }
        return (fastest, slowest)
    }
}

/// Prompts chosen to show the *different regimes* speculation lives in, since acceptance is
/// content-dependent: fresh prose accepts least, code more, and re-emitting text already in
/// context is where lookup drafting runs away with it.
struct RacePrompt: Identifiable {
    let name: String
    let text: String
    var id: String { name }

    static let presets: [RacePrompt] = [
        RacePrompt(name: "Code",
                   text: "Write a Python function that checks whether a number is prime."),
        RacePrompt(name: "Math",
                   text: "A train travels 120 km in 1.5 hours. What is its average speed in "
                         + "m/s? Show your work."),
        RacePrompt(name: "Chat",
                   text: "Explain how rainbows form."),
        RacePrompt(name: "Copy-heavy edit",
                   text: "Here is a function:\n\ndef add(a, b):\n    return a + b\n\n"
                         + "Rewrite it exactly, but add type hints and a docstring."),
    ]
}
