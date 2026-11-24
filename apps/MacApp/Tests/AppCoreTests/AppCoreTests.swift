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

@Suite("Model folders → engine environment")
struct ModelDirsTests {
    @Test func joinsExpandsAndDedupes() {
        let config = ServerConfig(modelDirs: [" ~/models ", "/Volumes/SSD/mlx", "", "/Volumes/SSD/mlx"])
        let home = NSHomeDirectory()
        #expect(config.modelDirsEnvironmentValue == "\(home)/models:/Volumes/SSD/mlx")
        #expect(ServerConfig.modelDirsEnvironmentKey == "MLX_DSPARK_MODEL_DIRS")
    }

    @Test func emptyMeansLeaveTheVariableAlone() {
        #expect(ServerConfig().modelDirsEnvironmentValue == nil)
        #expect(ServerConfig(modelDirs: ["", "  "]).modelDirsEnvironmentValue == nil)
    }
}

@Suite("LAN serving config")
struct LANConfigTests {
    @Test func wildcardListenerKeepsAppTrafficOnLoopback() {
        let lan = ServerConfig(host: "0.0.0.0")
        #expect(lan.servesLAN && lan.clientHost == "127.0.0.1")
        let local = ServerConfig()
        #expect(!local.servesLAN && local.clientHost == "127.0.0.1")
    }

    @Test func generatedKeysAreDistinctAndPasteable() {
        let a = LocalNetwork.generateAPIKey(), b = LocalNetwork.generateAPIKey()
        #expect(a != b && a.hasPrefix("mdk_") && a.count > 40)
        #expect(a.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" })
    }

    @Test func freePortWorksForBothListenerModes() throws {
        #expect(try ServerSupervisor.freePort(host: "127.0.0.1") > 0)
        #expect(try ServerSupervisor.freePort(host: "0.0.0.0") > 0)
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

    /// Issue #44: a crashed app left its engine running under launchd. Only this runtime's
    /// own `serve` with launchd as its parent is stopped — never a running app's engine,
    /// never the user's own install, never another command of this runtime.
    @Test func findsOnlyThisRuntimesOrphanedServe() {
        let runtime = "/Users/x/Library/Application Support/com.arahim.mlx-dspark/runtime/venv/bin"
        let engine = URL(fileURLWithPath: runtime + "/mlx-dspark")
        let listing = """
                1     0 /sbin/launchd
             4242     1 \(runtime)/python \(runtime)/mlx-dspark serve --host 127.0.0.1 --port 8484 --no-model
             4243  4100 \(runtime)/python \(runtime)/mlx-dspark serve --host 127.0.0.1 --port 8485 --no-model
             4244     1 /Users/x/.venv/bin/python3 /Users/x/.venv/bin/mlx-dspark serve --port 8080
             4245     1 \(runtime)/python \(runtime)/mlx-dspark doctor --json
             4246     1 \(runtime)/python \(runtime)/mlx-dspark serveX
            42470     1 \(runtime)/python \(runtime)/mlx-dspark serve
            garbage line
            """
        #expect(ServerSupervisor.orphanedEngines(in: listing, engine: engine) == [4242, 42470])
        #expect(ServerSupervisor.orphanedEngines(in: "", engine: engine).isEmpty)
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
        #expect(health.raceStock == nil)      // an engine without the stock race arm
    }

    /// The Lab's stock lane: the engine names the runtime the arm stands for, or sends null
    /// when the loaded target has none.
    @Test func decodesTheStockRaceRuntime() throws {
        let json = #"{"status":"ok","model":"Inkling-2-Mini","mode":"dspark","race_stock":"mlx-vlm 0.8.0"}"#
        let health = try JSONDecoder().decode(HealthInfo.self, from: Data(json.utf8))
        #expect(health.raceStock == "mlx-vlm 0.8.0")
        let none = #"{"status":"ok","model":"X","mode":"dspark","race_stock":null}"#
        #expect(try JSONDecoder().decode(HealthInfo.self, from: Data(none.utf8)).raceStock == nil)
    }

    /// A model's faster variant (Inkling 2's 8-bit attention + output head): the engine reports
    /// the bits in force and whether the loaded target has the option at all. The Weights
    /// picker shows only on `true`; an older engine reports neither.
    @Test func decodesTheTargetBitsOption() throws {
        let json = #"{"status":"ok","model":"Inkling-2-Mini","mode":"dflash","target_bits":8,"target_bits_supported":true}"#
        let health = try JSONDecoder().decode(HealthInfo.self, from: Data(json.utf8))
        #expect(health.targetBits == 8)
        #expect(health.targetBitsSupported == true)
        let other = #"{"status":"ok","model":"Qwen3-4B-8bit","mode":"dspark","target_bits":0,"target_bits_supported":false}"#
        #expect(try JSONDecoder().decode(HealthInfo.self, from: Data(other.utf8)).targetBitsSupported == false)
        let old = #"{"status":"ok","model":"X","mode":"dspark"}"#
        let before = try JSONDecoder().decode(HealthInfo.self, from: Data(old.utf8))
        #expect(before.targetBits == nil)
        #expect(before.targetBitsSupported == nil)
    }
}

@Suite("Model inventory rows")
struct ModelRowTests {
    /// `/admin/models`: a row with an opt-in faster variant carries what it buys and what it
    /// costs in the engine's words; other rows carry null, and older engines omit the key.
    @Test func decodesARowsFasterVariant() throws {
        let json = """
        [{"id": "inkling-2-mini", "target": "thinkingmachines/Inkling-2-Mini",
          "target_installed": true, "drafter_installed": true, "ready": true,
          "target_bits_option": {"bits": 8, "gain": "+30% plain decoding, +12% with the drafter",
                                 "caveat": "not the released weights"}},
         {"id": "qwen3-4b", "target": "mlx-community/Qwen3-4B-8bit",
          "target_installed": false, "drafter_installed": false, "ready": false,
          "target_bits_option": null},
         {"id": "from-an-older-engine", "target": "x/y",
          "target_installed": false, "drafter_installed": false, "ready": false}]
        """
        let rows = try JSONDecoder().decode([ModelRow].self, from: Data(json.utf8))
        #expect(rows[0].targetBitsOption?.bits == 8)
        #expect(rows[0].targetBitsOption?.gain == "+30% plain decoding, +12% with the drafter")
        #expect(rows[0].targetBitsOption?.caveat == "not the released weights")
        #expect(rows[1].targetBitsOption == nil)
        #expect(rows[2].targetBitsOption == nil)
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
