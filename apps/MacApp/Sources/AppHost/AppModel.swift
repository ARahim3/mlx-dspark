import AppCore
import Foundation
import SwiftUI

/// Which screen the sidebar is showing.
enum Screen: String, CaseIterable, Identifiable {
    case chat, lab, agents, models, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat:     return "Chat"
        case .lab:      return "Lab"
        case .agents:   return "Agents"
        case .models:   return "Models"
        case .settings: return "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .chat:     return "bubble.left.and.bubble.right"
        case .lab:      return "chart.xyaxis.line"
        case .agents:   return "terminal"
        case .models:   return "shippingbox"
        case .settings: return "gearshape"
        }
    }
}

/// How much of the machinery to expose.
///
/// Straight from LM Studio's most-copied idea, and load-bearing here: mlx-dspark has five
/// modes, four quantizations, and knobs for cap / kv-bits / drafter-bits / long-draft ceiling
/// / confidence. Shown all at once that is unusable by anyone who didn't build it; hidden
/// entirely it stops being this project's app.
enum Detail: String, CaseIterable, Identifiable {
    case simple, advanced, developer

    var id: String { rawValue }
    var title: String { rawValue.capitalized }

    var blurb: String {
        switch self {
        case .simple:    return "Everything automatic. Just chat."
        case .advanced:  return "Cost curves, live acceptance, mode and cap."
        case .developer: return "Every knob, raw logs, server internals."
        }
    }

    var showsLab: Bool { self != .simple }
    var showsRawLogs: Bool { self == .developer }
}

/// Coordinates the runtime, the server process, and everything the screens read.
@MainActor
final class AppModel: ObservableObject {

    // MARK: Lifecycle / server
    @Published var setupSteps: [SetupStep] = SetupStepID.allCases.map { SetupStep(id: $0) }
    @Published var serverState: ServerState = .idle
    @Published var phase: Phase = .launching
    @Published var errorMessage: String?

    // MARK: Navigation
    /// Reopens where you left off.
    @Published var screen: Screen = Defaults.screen {
        didSet { Defaults.screen = screen }
    }
    /// Advanced, not Simple, is the right default *here*. LM Studio starts users in its
    /// simplest mode because its audience is everyone; this app's audience went looking for a
    /// speculative-decoding project, and the Lab is the reason to prefer it over LM Studio.
    /// Hiding it by default means most users never find it. Simple stays one click away.
    @Published var detail: Detail = Defaults.detail {
        didSet { Defaults.detail = detail }
    }

    // MARK: Chat
    @Published var prompt: String = ""
    @Published var messages: [ChatMessage] = []
    @Published var isGenerating = false
    @Published var liveTokensPerSec: Double = 0

    // MARK: Telemetry (Lab)
    @Published var rounds: [RoundEvent] = []
    @Published var stats: RoundStats?
    @Published var calibration: Calibration?

    /// The loaded model's self-description. Fetched from `/health` on start and after every hot
    /// swap, so it stays correct across a model change — the supervisor's own cached health
    /// only reflects the model the process *started* with, not one swapped in later.
    @Published var currentHealth: HealthInfo?

    // MARK: Models
    @Published var models: [ModelRow] = []
    @Published var doctorReport: DoctorReport?

    // MARK: Logs
    @Published var logLines: [String] = []
    @Published var showLogs = false

    /// The target the server is (or will be) loaded with. Persisted after onboarding.
    @Published var model: String = Defaults.selectedModel ?? "mlx-community/Qwen3-4B-8bit"

    // MARK: Onboarding
    /// Hardware + inventory, probed model-free before any server starts.
    @Published var onboarding: DoctorReport?

    enum Phase: Equatable {
        case launching, settingUp
        case onboarding            // first run only: choose a model before loading anything
        case startingServer, ready, failed
    }

    /// Rounds kept for the live charts. A few hundred is several seconds of the fastest
    /// decoding — enough to see shape, cheap enough to re-render every frame.
    private let liveWindow = 400

    let logStore = LogStore()
    private var bootstrapper: RuntimeBootstrapper?
    private var supervisor: ServerSupervisor?
    /// Exposed so feature models (the Race) can drive the engine without re-deriving the port.
    private(set) var apiClient: APIClient?
    private var client: APIClient? { apiClient }
    private var generationTask: Task<Void, Never>?
    private var telemetryTask: Task<Void, Never>?

    init() {
        logStore.subscribe { [weak self] line in
            Task { @MainActor in
                guard let self else { return }
                self.logLines.append(line.text)
                if self.logLines.count > 500 { self.logLines.removeFirst(100) }
            }
        }
    }

    // MARK: - Boot

    private var engineURL: URL?

    func boot() async {
        guard phase == .launching || phase == .failed else { return }
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
            return fail(error)
        }
        engineURL = engine

        // First run: choose a model before loading anything heavy. The machine check is
        // model-free, so it runs without a server and without committing to a multi-GB
        // download the user might not have meant.
        if Defaults.selectedModel == nil {
            phase = .onboarding
            onboarding = try? await DoctorProbe.run(engine: engine)
            if onboarding == nil {
                // The machine check couldn't run (offline before any weights, a broken engine
                // install). Don't strand the user on a spinner — fall through to the default
                // model, whose own load will surface a real error if something is truly wrong.
                await startServer(model: model)
            }
            return                                  // otherwise resumes on the user's pick
        }
        await startServer(model: model)
    }

    /// Load a target and go. Called from onboarding's pick, or straight from boot on a return
    /// visit. The first-ever load of a model downloads it, which is why the "starting" state is
    /// worded for minutes, not seconds.
    func startServer(model: String) async {
        guard let engine = engineURL else { return }
        self.model = model
        Defaults.selectedModel = model
        phase = .startingServer

        let supervisor = ServerSupervisor(engine: engine, logStore: logStore)
        self.supervisor = supervisor
        await supervisor.observeState { [weak self] state in
            Task { @MainActor in self?.serverState = state }
        }

        do {
            let port = try await supervisor.start(
                config: ServerConfig(model: model, mode: "auto", maxDraft: "auto"))
            let client = APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
            self.apiClient = client
            currentHealth = try? await client.health()
            phase = .ready
            startTelemetry()
            await refreshDiagnostics()
        } catch {
            fail(error)
        }
    }

    /// Switch to a different target — an in-place hot swap via `/admin/load`, not a restart.
    ///
    /// The server and its port survive, so anything else pointed at it (a Claude Code session)
    /// keeps working across the change. The engine frees the old model before loading the new
    /// one (release-then-load), so peak memory stays at one model. The loading screen shows
    /// meanwhile; telemetry resets because the new model's numbers are its own.
    func switchModel(to target: String) async {
        guard target != model, let client = apiClient else { return }
        generationTask?.cancel()
        self.model = target
        Defaults.selectedModel = target
        rounds = []
        stats = nil
        calibration = nil
        phase = .startingServer
        logStore.note("switching model → \(target)")
        do {
            _ = try await client.loadModel(target)
            // Re-point health at the new model and restart the telemetry stream (the old one
            // ended when the engine it was streaming from was torn down).
            currentHealth = try? await client.health()
            phase = .ready
            startTelemetry()
            await refreshDiagnostics()
        } catch {
            fail(error)
        }
    }

    private func fail(_ error: Error) {
        errorMessage = error.localizedDescription
        phase = .failed
        showLogs = true
    }

    func shutdown() async {
        generationTask?.cancel()
        telemetryTask?.cancel()
        await supervisor?.stop()
    }

    // MARK: - Telemetry

    /// Subscribe to the engine's round stream for the lifetime of the app.
    ///
    /// Not tied to a chat request on purpose — the stream reports every round the engine runs,
    /// so the Lab keeps updating even when the tokens are being generated for Claude Code or
    /// any other client pointed at this server.
    private func startTelemetry() {
        guard let client else { return }
        telemetryTask?.cancel()
        telemetryTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    for try await event in client.streamRounds() {
                        guard let self else { return }
                        switch event {
                        case .round(let round):
                            self.rounds.append(round)
                            if self.rounds.count > self.liveWindow {
                                self.rounds.removeFirst(self.rounds.count - self.liveWindow)
                            }
                            if round.ms > 0 { self.liveTokensPerSec = round.tokensPerSecond }
                        case .stats(let stats):
                            self.stats = stats
                        }
                    }
                } catch {
                    // The stream ends when the engine restarts or a socket drops; reconnect
                    // rather than leaving the Lab silently frozen.
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    func refreshDiagnostics() async {
        guard let client else { return }
        async let report = try? client.doctor()
        async let inventory = try? client.modelInventory()
        async let curves = try? client.calibration()
        doctorReport = await report
        models = await inventory ?? []
        calibration = await curves
    }

    /// Pull the latest aggregates (the SSE stream only pushes them periodically).
    func refreshStats() async {
        guard let client else { return }
        if let (_, latest) = try? await client.rounds(limit: 1) { stats = latest }
    }

    // MARK: - Chat

    func send() {
        guard let client, !isGenerating else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        messages.append(ChatMessage(role: .user, text: text))
        messages.append(ChatMessage(role: .assistant, text: ""))
        prompt = ""
        isGenerating = true

        let history = messages.dropLast().map {
            ["role": $0.role == .user ? "user" : "assistant", "content": $0.text]
        }

        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.streamChat(model: self.model,
                                                         messages: Array(history),
                                                         maxTokens: 1024) {
                    switch event {
                    case .delta(let piece):
                        self.messages[self.messages.count - 1].text += piece
                    case .finished(let info):
                        self.messages[self.messages.count - 1].stats = info
                        if let info { self.liveTokensPerSec = info.tokensPerSec }
                    }
                }
            } catch {
                if !Task.isCancelled { self.errorMessage = error.localizedDescription }
            }
            self.isGenerating = false
            await self.refreshStats()
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        isGenerating = false
    }

    func clearChat() {
        cancelGeneration()
        messages.removeAll()
    }

    // MARK: - Derived

    var health: HealthInfo? {
        // Prefer the freshly-fetched health (correct after a hot swap); fall back to the
        // supervisor's start-time snapshot before the first fetch lands.
        if let currentHealth { return currentHealth }
        if case .ready(_, let health) = serverState { return health }
        return nil
    }

    var isServerReady: Bool { health != nil }

    /// Which strategies can be raced with the currently loaded pair. `baseline` and `lookup`
    /// need only the target; a drafter mode needs the drafter this engine was loaded with, so
    /// dspark and dflash are never both on offer.
    var availableRaceArms: [String] {
        guard let mode = health?.mode else { return ["baseline", "lookup"] }
        return mode == "dspark" || mode == "dflash"
            ? [mode, "baseline", "lookup"] : ["baseline", "lookup"]
    }

    var statusLine: String {
        switch serverState {
        case .idle:                 return "Idle"
        case .starting(let detail): return detail
        case .ready(let port, let health):
            return "\(health.model) · \(health.mode) · :\(port)"
        case .failed(let message):  return message
        case .stopped:              return "Stopped"
        }
    }

    /// "target ← drafter" — the pairing that makes speculative decoding work. Naming both is
    /// something no other local-LLM app has to do, so it belongs in the chrome, not a submenu.
    var pairingLine: String? {
        guard let health, let drafter = health.drafter else { return nil }
        let short = { (repo: String) in repo.components(separatedBy: "/").last ?? repo }
        return "\(short(health.target ?? health.model))  ←  \(short(drafter))"
    }

    /// Rounds from the most recent request only — what the live charts should show.
    var currentRunRounds: [RoundEvent] {
        guard let last = rounds.last else { return [] }
        return rounds.filter { $0.req == last.req }
    }
}

/// Persisted UI preferences.
///
/// Plain `UserDefaults` rather than `@AppStorage` because these live on the model, not in a
/// view. Being real defaults keys also means they can be set from the command line, which is
/// how the app gets driven for screenshots and QA without a click.
enum Defaults {
    private static let store = UserDefaults.standard

    static var screen: Screen {
        get { store.string(forKey: "screen").flatMap(Screen.init(rawValue:)) ?? .chat }
        set { store.set(newValue.rawValue, forKey: "screen") }
    }

    static var detail: Detail {
        get { store.string(forKey: "detail").flatMap(Detail.init(rawValue:)) ?? .advanced }
        set { store.set(newValue.rawValue, forKey: "detail") }
    }

    /// Which Lab tab was last open.
    static var labTab: String {
        get { store.string(forKey: "labTab") ?? "Live" }
        set { store.set(newValue, forKey: "labTab") }
    }

    /// The target chosen during onboarding. `nil` means onboarding hasn't run — the signal
    /// that gates the model-pick flow, so it must only be set once a model is actually chosen.
    static var selectedModel: String? {
        get { store.string(forKey: "selectedModel") }
        set { store.set(newValue, forKey: "selectedModel") }
    }
}

struct ChatMessage: Identifiable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String
    var stats: SpecInfo?
}
