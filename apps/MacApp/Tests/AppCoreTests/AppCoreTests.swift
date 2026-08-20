import Foundation
import Testing
@testable import AppCore

// swift-testing, not XCTest: XCTest ships only with full Xcode, and this project builds with
// Command Line Tools alone. Keeping the app buildable+testable without Xcode is deliberate.

@Suite("Line buffering")
struct LineBufferTests {
    /// Pipe reads split wherever the kernel feels like it; a line must survive being cut in half.
    @Test func reassemblesLinesAcrossChunks() {
        let buffer = LineBuffer()
        #expect(buffer.consume(Data("hel".utf8)) == [])
        #expect(buffer.consume(Data("lo\nwor".utf8)) == ["hello"])
        #expect(buffer.consume(Data("ld\n".utf8)) == ["world"])
    }

    @Test func handlesMultipleLinesInOneChunk() {
        let buffer = LineBuffer()
        #expect(buffer.consume(Data("a\nb\nc\n".utf8)) == ["a", "b", "c"])
    }

    @Test func keepsOnlyTheMostRecentLines() {
        let tail = LineTail(limit: 3)
        for line in ["1", "2", "3", "4", "5"] { tail.append(line) }
        #expect(tail.joined() == "3\n4\n5")
    }
}

@Suite("Termination gate")
struct TerminationGateTests {
    /// The bug that made the first app launch hang: Foundation does not call a
    /// terminationHandler assigned after the process already exited, so a short command
    /// (`uv --version`) signalled before anyone awaited. Both orders must resume exactly once.
    @Test func resumesWhenSignalArrivesBeforeWait() async {
        let gate = TerminationGate()
        gate.signal()
        await gate.wait()                    // must not hang
    }

    @Test func resumesWhenSignalArrivesAfterWait() async {
        let gate = TerminationGate()
        Task {
            try? await Task.sleep(nanoseconds: 20_000_000)
            gate.signal()
        }
        await gate.wait()
    }

    @Test func repeatedSignalsAreHarmless() async {
        let gate = TerminationGate()
        gate.signal()
        gate.signal()
        await gate.wait()
    }
}

@Suite("Server supervision")
struct ServerSupervisorTests {
    /// Binding a fixed 8080 is how every one of these apps first collides with the user's
    /// other tools.
    @Test func freePortIsUnprivileged() throws {
        let port = try ServerSupervisor.freePort()
        #expect(port > 1024)
        #expect(port <= 65535)
    }

    @Test func freePortVariesAcrossCalls() throws {
        let ports = try (0..<4).map { _ in try ServerSupervisor.freePort() }
        #expect(Set(ports).count > 1)
    }
}

@Suite("Engine source resolution")
struct EngineSourceTests {
    @Test func defaultsToLatestPyPIRelease() {
        #expect(RuntimeBootstrapper.EngineSource.fromEnvironment([:]) == .pypi)
    }

    @Test func environmentOverridePointsAtWorkingTree() {
        let source = RuntimeBootstrapper.EngineSource
            .fromEnvironment(["MLXDSPARK_ENGINE_SOURCE": "/Users/x/Codes/dspark"])
        #expect(source == .localSourceTree(URL(fileURLWithPath: "/Users/x/Codes/dspark")))
    }

    /// A dev runtime and a released one must not share a fingerprint, or switching between
    /// them would silently keep the wrong engine installed — the same-version-rebuild trap.
    @Test func fingerprintsDifferBySource() {
        let pypi = RuntimeBootstrapper.EngineSource.pypi
        let local = RuntimeBootstrapper.EngineSource.localSourceTree(URL(fileURLWithPath: "/tmp/x"))
        #expect(pypi.fingerprint != local.fingerprint)
    }
}

@Suite("Engine stats block")
struct SpecInfoTests {
    /// `x_mlx_dspark` is the engine's own extension to the OpenAI response — the numbers this
    /// project exists to produce. Decoding it is not optional.
    @Test func decodesFullBlock() throws {
        let json = """
        {"mode":"dspark","accept_len":3.19,"tokens_per_sec":60.9,
         "target_forwards":63,"cap":3,"lookup_rounds":4}
        """
        let info = try JSONDecoder().decode(SpecInfo.self, from: Data(json.utf8))
        #expect(info.mode == "dspark")
        #expect(abs(info.acceptLen - 3.19) < 0.001)
        #expect(info.cap == 3)
        #expect(info.lookupRounds == 4)
    }

    /// Baseline and lookup responses omit cap/lookup_rounds entirely.
    @Test func decodesWithOptionalFieldsAbsent() throws {
        let json = """
        {"mode":"baseline","accept_len":1.0,"tokens_per_sec":27.9,"target_forwards":200}
        """
        let info = try JSONDecoder().decode(SpecInfo.self, from: Data(json.utf8))
        #expect(info.cap == nil)
        #expect(info.lookupRounds == nil)
    }
}

@Suite("Health payload")
struct HealthInfoTests {
    /// Verbatim from a live `mlx-dspark serve` on 2026-07-21 — decoding must not drift from
    /// what the engine actually sends.
    @Test func decodesLiveServerPayload() throws {
        let json = """
        {"status": "ok", "model": "Qwen3-4B-8bit", "mode": "dspark",
         "target": "mlx-community/Qwen3-4B-8bit",
         "drafter": "deepseek-ai/dspark_qwen3_4b_block7",
         "context_window": 40960, "max_output_tokens": 32768}
        """
        let health = try JSONDecoder().decode(HealthInfo.self, from: Data(json.utf8))
        #expect(health.status == "ok")
        #expect(health.mode == "dspark")
        #expect(health.drafter == "deepseek-ai/dspark_qwen3_4b_block7")
        #expect(health.contextWindow == 40960)
    }

    /// Drafter-free modes (lookup, baseline) report no drafter; the UI must not require one.
    @Test func decodesWithoutDrafter() throws {
        let json = #"{"status":"ok","model":"X","mode":"lookup"}"#
        let health = try JSONDecoder().decode(HealthInfo.self, from: Data(json.utf8))
        #expect(health.drafter == nil)
        #expect(health.contextWindow == nil)
        // reasoning keys are optional: older engines don't report them
        #expect(health.reasoningBudget == nil)
        #expect(health.enableThinking == nil)
        #expect(health.adminConfig == nil)
    }

    /// The reasoning keys a current engine reports — the Settings card's data source and
    /// its `admin_config` capability gate.
    @Test func decodesReasoningKeys() throws {
        let json = """
        {"status": "ok", "model": "X", "mode": "dflash",
         "reasoning_budget": 8192, "reasoning_budget_message": "Wrap it up.",
         "enable_thinking": false, "admin_config": true,
         "supports_reasoning_budget": true}
        """
        let health = try JSONDecoder().decode(HealthInfo.self, from: Data(json.utf8))
        #expect(health.reasoningBudget == 8192)
        #expect(health.reasoningBudgetMessage == "Wrap it up.")
        #expect(health.enableThinking == false)
        #expect(health.adminConfig == true)
        #expect(health.supportsReasoningBudget == true)
        // absent on older engines and in the no-model state — the per-chat control hides
        #expect((try JSONDecoder().decode(
            HealthInfo.self,
            from: Data(#"{"status":"no_model"}"#.utf8))).supportsReasoningBudget == nil)
    }
}

@Suite("Reasoning payload")
struct ReasoningPayloadTests {
    /// Thinking is explicit true/false BOTH ways — ON must win over a template whose own
    /// default is off, so it can never be encoded as null (null merely clears the override).
    @Test func thinkingIsExplicitBothWays() {
        let on = APIClient.reasoningPayload(thinkingEnabled: true, budget: 8192, message: nil)
        #expect(on["enable_thinking"] as? Bool == true)
        let off = APIClient.reasoningPayload(thinkingEnabled: false, budget: 8192, message: nil)
        #expect(off["enable_thinking"] as? Bool == false)
    }

    /// Budget 0 is the checkbox-off state and must pass through as a real 0 (the engine's
    /// "disabled"), not be dropped or nulled.
    @Test func zeroBudgetPassesThrough() {
        let p = APIClient.reasoningPayload(thinkingEnabled: true, budget: 0, message: nil)
        #expect(p["reasoning_budget"] as? Int == 0)
    }

    /// nil message = engine default (JSON null); "" = the engine's explicit
    /// close-with-no-message mode — the two must stay distinguishable on the wire.
    @Test func messageNilVersusEmptyVersusText() {
        let none = APIClient.reasoningPayload(thinkingEnabled: true, budget: 1, message: nil)
        #expect(none["reasoning_budget_message"] is NSNull)
        let empty = APIClient.reasoningPayload(thinkingEnabled: true, budget: 1, message: "")
        #expect(empty["reasoning_budget_message"] as? String == "")
        let text = APIClient.reasoningPayload(thinkingEnabled: true, budget: 1,
                                              message: "I have to answer now.")
        #expect(text["reasoning_budget_message"] as? String == "I have to answer now.")
    }
}

@Suite("Chat payload")
struct ChatPayloadTests {
    private let msgs = [["role": "user", "content": "hi"]]

    /// The three keys every request carries, and nothing else when the options are nil.
    @Test func minimalPayloadOmitsEveryOptional() {
        let p = APIClient.chatPayload(model: "m", messages: msgs)
        #expect(p["model"] as? String == "m")
        #expect(p["stream"] as? Bool == true)
        #expect((p["messages"] as? [[String: String]]) == msgs)
        for key in ["temperature", "max_tokens", "enable_thinking", "reasoning_effort",
                    "reasoning_budget", "top_p", "seed", "stop"] {
            #expect(p[key] == nil, "\(key) must be omitted when nil")
        }
    }

    /// Per-chat budget: nil = inherit (omitted); 0 = "unbounded this chat" and must
    /// survive as a real JSON 0, not be dropped; a value passes through.
    @Test func reasoningBudgetNilZeroAndValue() {
        #expect(APIClient.chatPayload(model: "m", messages: msgs)["reasoning_budget"] == nil)
        let zero = APIClient.chatPayload(model: "m", messages: msgs, reasoningBudget: 0)
        #expect(zero["reasoning_budget"] as? Int == 0)
        let capped = APIClient.chatPayload(model: "m", messages: msgs, reasoningBudget: 512)
        #expect(capped["reasoning_budget"] as? Int == 512)
    }

    /// The chat toggle's one-way veto: nil (allow) is omitted, false is a real false.
    @Test func enableThinkingOmittedOrFalse() {
        #expect(APIClient.chatPayload(model: "m", messages: msgs)["enable_thinking"] == nil)
        let off = APIClient.chatPayload(model: "m", messages: msgs, enableThinking: false)
        #expect(off["enable_thinking"] as? Bool == false)
    }

    /// The remaining optionals encode under their wire names; empty stop is dropped.
    @Test func optionalKeysEncodeUnderWireNames() {
        let p = APIClient.chatPayload(model: "m", messages: msgs, temperature: 0.7,
                                      maxTokens: 64, reasoningEffort: "low", topP: 0.9,
                                      seed: 7, stop: ["END"])
        #expect(p["temperature"] as? Double == 0.7)
        #expect(p["max_tokens"] as? Int == 64)
        #expect(p["reasoning_effort"] as? String == "low")
        #expect(p["top_p"] as? Double == 0.9)
        #expect(p["seed"] as? Int == 7)
        #expect(p["stop"] as? [String] == ["END"])
        let noStop = APIClient.chatPayload(model: "m", messages: msgs, stop: [])
        #expect(noStop["stop"] == nil)
    }
}

@Suite("Chat sessions")
struct ChatSessionTests {
    /// The per-conversation budget round-trips through the session files.
    @Test func reasoningBudgetRoundTripsThroughTheStore() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chatstore-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatStore(directory: dir)
        store.save(ChatSession(title: "budgeted",
                               messages: [ChatMessage(role: .user, text: "hi")],
                               reasoningBudget: 512))
        store.save(ChatSession(title: "inheriting",
                               messages: [ChatMessage(role: .user, text: "yo")]))
        let sessions = store.list()
        #expect(sessions.first(where: { $0.title == "budgeted" })?.reasoningBudget == 512)
        #expect(sessions.first(where: { $0.title == "inheriting" })?.reasoningBudget == nil)
    }

    /// Session files written by builds that predate the field must still decode (= inherit).
    @Test func oldSessionFilesDecodeWithNilBudget() throws {
        let json = """
        {"id": "00000000-0000-0000-0000-000000000001", "title": "old",
         "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
         "messages": []}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let session = try decoder.decode(ChatSession.self, from: Data(json.utf8))
        #expect(session.reasoningBudget == nil)
    }
}

@Suite("Coalescer")
@MainActor
struct CoalescerTests {
    /// Every wait is bounded (~2 s) so a regression fails fast instead of hanging the run.
    private func wait(until condition: @escaping @MainActor () -> Bool) async -> Bool {
        for _ in 0..<2000 {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
        return condition()
    }

    /// Latest-wins: schedule() calls arriving while a run is in flight coalesce into
    /// exactly one re-run, and that re-run observes the newest state — a stale earlier
    /// completion can never be the last word.
    @Test func coalescesMidFlightSchedulesIntoOneRerunWithNewestState() async {
        final class Box: @unchecked Sendable {
            var value = 0
            var seen: [Int] = []
            var started = 0
            var gate = false
        }
        let box = Box()
        let coalescer = Coalescer {
            box.started += 1
            while !box.gate { try? await Task.sleep(nanoseconds: 1_000_000) }
            box.seen.append(box.value)
        }
        box.value = 1
        coalescer.schedule()
        #expect(await wait { box.started == 1 })   // first run is genuinely in flight
        box.value = 2
        coalescer.schedule()                       // mid-flight: queues one re-run
        box.value = 3
        coalescer.schedule()                       // coalesces into the SAME re-run
        box.gate = true
        #expect(await wait { box.seen.count == 2 })
        try? await Task.sleep(nanoseconds: 20_000_000)   // would surface a spurious 3rd run
        #expect(box.seen.count == 2)               // one run + exactly one re-run
        #expect(box.seen.last == 3)                // the re-run saw the newest state
    }

    /// Schedules landing BEFORE the first run starts fold into it — no redundant re-run,
    /// and the single run still sees the newest state (inputs are read at send time).
    @Test func preRunSchedulesFoldIntoTheFirstRun() async {
        final class Box: @unchecked Sendable {
            var value = 0
            var seen: [Int] = []
        }
        let box = Box()
        let coalescer = Coalescer { box.seen.append(box.value) }
        box.value = 1
        coalescer.schedule()
        box.value = 2
        coalescer.schedule()
        #expect(await wait { !box.seen.isEmpty })
        try? await Task.sleep(nanoseconds: 20_000_000)
        #expect(box.seen == [2])
    }
}

@Suite("Install locations")
struct PathsTests {
    /// Everything installed must live outside the .app — that is what keeps notarization to
    /// two binaries instead of thousands of nested dylibs.
    @Test func runtimeLivesUnderApplicationSupport() {
        #expect(Paths.venvDir.path.contains("Application Support"))
        #expect(Paths.venvDir.path.contains(AppIdentity.bundleID))
        #expect(Paths.engineExecutable.path.hasSuffix("bin/mlx-dspark"))
    }
}

@Suite("Setup checklist")
struct SetupStepTests {
    /// Onboarding must never say "Python" — the runtime is an implementation detail the user
    /// should not have to know about.
    @Test func stepTitlesNeverMentionPython() {
        for id in SetupStepID.allCases {
            #expect(!id.title.lowercased().contains("python"))
        }
    }
}

@Suite("Engine version probe")
struct VersionProbeTests {
    /// Shell.capture merges stdout and stderr, so the probe output can carry import-time
    /// warnings around the version line. A polluted fingerprint never equals PyPI's version
    /// string — that made the "Engine X is available" banner permanent and re-ran the
    /// in-place upgrade on every launch. The sentinel parse must survive the noise.
    @Test func parsesVersionOutOfMergedWarnings() {
        let noisy = """
        [transformers] PyTorch was not found. Models won't be available.
        MLXDSPARK_VERSION=0.13.1
        """
        #expect(RuntimeBootstrapper.parseVersionProbe(noisy) == "0.13.1")
        #expect(RuntimeBootstrapper.parseVersionProbe("MLXDSPARK_VERSION=0.13.1\n") == "0.13.1")
        // warnings can also arrive AFTER the print (flushed on exit)
        let trailing = "MLXDSPARK_VERSION=0.14.0\nsome/late.py:1: UserWarning: whatever"
        #expect(RuntimeBootstrapper.parseVersionProbe(trailing) == "0.14.0")
    }

    @Test func rejectsOutputWithoutTheSentinel() {
        #expect(RuntimeBootstrapper.parseVersionProbe("0.13.1") == nil)
        #expect(RuntimeBootstrapper.parseVersionProbe("") == nil)
        #expect(RuntimeBootstrapper.parseVersionProbe("MLXDSPARK_VERSION=") == nil)
    }
}
