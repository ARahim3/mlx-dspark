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
