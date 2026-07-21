import AppCore
import Foundation
import SwiftUI

/// Coordinates the three long-lived pieces: the runtime, the server process, and the API.
///
/// This is the spike's whole state machine. It exists to prove the spine end to end —
/// bootstrap → serve → stream → measure — before any real screens are built on top.
@MainActor
final class AppModel: ObservableObject {

    // Runtime + server
    @Published var setupSteps: [SetupStep] = SetupStepID.allCases.map { SetupStep(id: $0) }
    @Published var serverState: ServerState = .idle
    @Published var phase: Phase = .launching
    @Published var errorMessage: String?

    // Chat
    @Published var prompt: String = "Write a Python function that checks if a string is a palindrome."
    @Published var output: String = ""
    @Published var isGenerating = false
    @Published var lastStats: SpecInfo?
    @Published var liveTokensPerSec: Double = 0

    // Logs
    @Published var logLines: [String] = []
    @Published var showLogs = false

    @Published var model: String = "mlx-community/Qwen3-4B-8bit"

    enum Phase: Equatable {
        case launching
        case settingUp
        case startingServer
        case ready
        case failed
    }

    let logStore = LogStore()
    private var bootstrapper: RuntimeBootstrapper?
    private var supervisor: ServerSupervisor?
    private var client: APIClient?
    private var generationTask: Task<Void, Never>?

    init() {
        logStore.subscribe { [weak self] line in
            Task { @MainActor in
                guard let self else { return }
                self.logLines.append(line.text)
                if self.logLines.count > 500 { self.logLines.removeFirst(100) }
            }
        }
    }

    // MARK: - Lifecycle

    func boot() async {
        phase = .settingUp
        errorMessage = nil

        let bootstrapper = RuntimeBootstrapper(logStore: logStore)
        self.bootstrapper = bootstrapper

        let engine: URL
        do {
            engine = try await bootstrapper.ensureRuntime { [weak self] steps in
                Task { @MainActor in self?.setupSteps = steps }
            }
        } catch {
            errorMessage = error.localizedDescription
            phase = .failed
            showLogs = true
            return
        }

        phase = .startingServer
        let supervisor = ServerSupervisor(engine: engine, logStore: logStore)
        self.supervisor = supervisor
        await supervisor.observeState { [weak self] state in
            Task { @MainActor in self?.serverState = state }
        }

        do {
            let port = try await supervisor.start(
                config: ServerConfig(model: model, mode: "auto", maxDraft: "auto"))
            client = APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
            phase = .ready
        } catch {
            errorMessage = error.localizedDescription
            phase = .failed
            showLogs = true
        }
    }

    func shutdown() async {
        generationTask?.cancel()
        await supervisor?.stop()
    }

    // MARK: - Generation

    func send() {
        guard let client, !isGenerating else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        output = ""
        lastStats = nil
        liveTokensPerSec = 0
        isGenerating = true

        // Rough live rate: characters are a stand-in for tokens until the telemetry stream
        // (backend B3) lands and gives us the real per-round numbers.
        let started = Date()
        var characters = 0

        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = client.streamChat(
                    model: self.model,
                    messages: [["role": "user", "content": text]],
                    maxTokens: 400)
                for try await event in stream {
                    switch event {
                    case .delta(let piece):
                        characters += piece.count
                        self.output += piece
                        let elapsed = Date().timeIntervalSince(started)
                        if elapsed > 0.4 {
                            self.liveTokensPerSec = Double(characters) / 4.0 / elapsed
                        }
                    case .finished(let info):
                        self.lastStats = info
                        if let info { self.liveTokensPerSec = info.tokensPerSec }
                    }
                }
            } catch {
                if !Task.isCancelled { self.errorMessage = error.localizedDescription }
            }
            self.isGenerating = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        isGenerating = false
    }

    // MARK: - Derived

    var statusLine: String {
        switch serverState {
        case .idle:     return "Idle"
        case .starting(let detail): return detail
        case .ready(let port, let health):
            return "\(health.model) · \(health.mode) · :\(port)"
        case .failed(let message): return message
        case .stopped:  return "Stopped"
        }
    }

    var isServerReady: Bool {
        if case .ready = serverState { return true }
        return false
    }

    var health: HealthInfo? {
        if case .ready(_, let health) = serverState { return health }
        return nil
    }

    /// "target ← drafter", the pairing that makes speculative decoding work. Naming both is
    /// something no other local-LLM app has to do, and it's worth showing rather than hiding.
    var pairingLine: String? {
        guard let health, let drafter = health.drafter else { return nil }
        let short = { (repo: String) in repo.components(separatedBy: "/").last ?? repo }
        return "\(short(health.target ?? health.model))  ←  \(short(drafter))"
    }
}
