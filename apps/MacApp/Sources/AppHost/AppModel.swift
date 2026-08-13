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
        case .advanced:  return "The Lab: races, live acceptance, this Mac's cost curves."
        case .developer: return "Advanced, plus raw engine logs."
        }
    }

    var showsLab: Bool { self != .simple }
    var showsRawLogs: Bool { self == .developer }
}

/// Generation preferences the user can change per conversation. `nil` means "the model's own
/// default" — the server already reads `generation_config.json`, so absent beats guessed.
struct ChatSettings: Codable, Equatable {
    var systemPrompt: String = ""
    var temperature: Double?
    var maxTokens: Int?
    /// Reasoning models spend real tokens thinking; agentic/plain use often wants it off.
    var thinking: Bool = true
}

/// A log line with a stable identity, so trimming the buffer never reshuffles SwiftUI ids.
struct LogRow: Identifiable {
    let id: Int
    let text: String
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
    /// A generation that failed mid-stream. Rendered inline in Chat — a chat problem must
    /// never take over the whole window.
    @Published var chatError: String?
    @Published var chatSettings: ChatSettings = Defaults.chatSettings {
        didSet { Defaults.chatSettings = chatSettings }
    }

    // MARK: Chat sessions
    @Published var sessions: [ChatSession] = []
    @Published private(set) var currentSessionID: UUID?

    // MARK: Telemetry (Lab)
    @Published var rounds: [RoundEvent] = []
    @Published var stats: RoundStats?
    @Published var calibration: Calibration?
    /// MLX allocator state — what the loaded model holds resident, polled from `/metrics`.
    @Published var memory: EngineMemory?

    /// The loaded model's self-description. Fetched from `/health` on start and after every hot
    /// swap, so it stays correct across a model change — the supervisor's own cached health
    /// only reflects the model the process *started* with, not one swapped in later.
    @Published var currentHealth: HealthInfo?

    // MARK: Models
    @Published var models: [ModelRow] = []
    @Published var installedModels: [InstalledModel] = []
    @Published var diskUsage: DiskUsage?
    @Published var doctorReport: DoctorReport?
    /// A model swap that failed. Rendered inline on the Models screen — the server survives a
    /// bad load by design, so the app must too.
    @Published var modelSwitchError: String?

    // MARK: Logs
    @Published var logLines: [LogRow] = []
    @Published var showLogs = false

    /// The target the server is (or will be) loaded with. Persisted after onboarding.
    @Published var model: String = Defaults.selectedModel ?? "mlx-community/Qwen3-4B-8bit"

    // MARK: Onboarding
    /// Hardware + inventory, probed model-free before any server starts.
    @Published var onboarding: DoctorReport?

    /// A newer app release on GitHub, if any. Informational — updating is `brew upgrade`.
    @Published var appUpdate: AppUpdate.Available?
    /// One-time banner after onboarding: land in the Lab with the race ready to run.
    @Published var showLabWelcome = false

    enum Phase: Equatable {
        case launching, settingUp
        case onboarding            // first run only: choose a model before loading anything
        case startingServer, ready, failed
    }

    /// Rounds kept for the live charts. A few hundred is several seconds of the fastest
    /// decoding — enough to see shape, cheap enough to re-render every frame.
    private let liveWindow = 400

    let logStore = LogStore()
    private let chatStore = ChatStore()
    private var bootstrapper: RuntimeBootstrapper?
    private var supervisor: ServerSupervisor?
    /// Exposed so feature models (the Race) can drive the engine without re-deriving the port.
    private(set) var apiClient: APIClient?
    private var client: APIClient? { apiClient }
    private var generationTask: Task<Void, Never>?
    private var telemetryTask: Task<Void, Never>?
    private var memoryTask: Task<Void, Never>?
    private var idleDecayTask: Task<Void, Never>?
    private var logCounter = 0
    /// When the last round (or completion) arrived — drives the idle decay of the live rate.
    private var lastActivity = Date.distantPast
    /// Set while the first-ever run is in flight, so success can land in the Lab.
    private var isFirstRun = false

    init() {
        logStore.subscribe { [weak self] line in
            Task { @MainActor in
                guard let self else { return }
                self.logCounter += 1
                self.logLines.append(LogRow(id: self.logCounter, text: line.text))
                if self.logLines.count > 500 { self.logLines.removeFirst(100) }
            }
        }
        loadSessions()
    }

    // MARK: - Boot

    private var engineURL: URL?

    func boot() async {
        guard phase == .launching || phase == .failed else { return }
        phase = .settingUp
        errorMessage = nil

        // Non-blocking: a release note in Settings/menu bar, never a launch dependency.
        if AppIdentity.appVersion != "dev" {
            Task { [weak self] in
                if let update = await AppUpdate.check(current: AppIdentity.appVersion) {
                    self?.appUpdate = update
                }
            }
        }

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
            isFirstRun = true
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
            startMemoryPolling()
            await refreshDiagnostics()
            if isFirstRun {
                // The plan's "hooked" moment: straight from onboarding into a race the user
                // can run — not a blank chat that hides why this app exists.
                isFirstRun = false
                if detail.showsLab {
                    Defaults.labTab = "Race"
                    screen = .lab
                    showLabWelcome = true
                }
            }
        } catch {
            fail(error)
        }
    }

    /// Switch to a different target — an in-place hot swap via `/admin/load`, not a restart.
    ///
    /// The server and its port survive, so anything else pointed at it (a Claude Code session)
    /// keeps working across the change. The engine frees the old model before loading the new
    /// one (release-then-load), so peak memory stays at one model.
    ///
    /// Failure stays contained: the engine survives a bad load by design (400 + not-ready),
    /// so the app reports the error inline and reloads the previous model rather than
    /// declaring the whole app failed.
    func switchModel(to target: String) async {
        let target = target.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty, target != model, let client = apiClient else { return }
        generationTask?.cancel()
        let previous = model
        modelSwitchError = nil
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
            startMemoryPolling()
            await refreshDiagnostics()
        } catch {
            modelSwitchError = error.localizedDescription
            logStore.note("model swap failed: \(error.localizedDescription)")
            screen = .models                     // the inline error lives there
            // Release-then-load means the old model is already gone, so restore it — the
            // app (and anything on the port) should keep working on what it had.
            self.model = previous
            Defaults.selectedModel = previous
            do {
                _ = try await client.loadModel(previous)
                currentHealth = try? await client.health()
                phase = .ready
                startTelemetry()
                startMemoryPolling()
                await refreshDiagnostics()
            } catch {
                fail(error)                      // even the old model won't load — real failure
            }
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
        memoryTask?.cancel()
        idleDecayTask?.cancel()
        persistCurrentSession()
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
                            if round.ms > 0 { self.observeRate(round.tokensPerSecond) }
                            self.lastActivity = Date()
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
        startIdleDecay()
    }

    /// Smoothed rate accumulator behind `liveTokensPerSec`.
    private var rateEWMA: Double = 0
    private var lastRatePublish = Date.distantPast

    /// Per-round instantaneous rates jitter hard (rounds commit 1–5 tokens, so each one's
    /// implied tok/s swings) — published raw they read as a slot machine. Smooth with an
    /// EWMA and publish at most ~2×/s so the gauge reads like a needle settling; the exact
    /// final figure still lands from the completion's own stats.
    private func observeRate(_ rate: Double) {
        rateEWMA = rateEWMA == 0 ? rate : 0.8 * rateEWMA + 0.2 * rate
        let now = Date()
        if now.timeIntervalSince(lastRatePublish) > 0.5 || liveTokensPerSec == 0 {
            liveTokensPerSec = rateEWMA
            lastRatePublish = now
        }
    }

    /// Zero the live rate a few seconds after the last round, so the sidebar and menu bar
    /// read "idle" instead of showing the last generation's speed forever.
    private func startIdleDecay() {
        idleDecayTask?.cancel()
        idleDecayTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                guard let self else { return }
                if self.liveTokensPerSec > 0,
                   Date().timeIntervalSince(self.lastActivity) > 4 {
                    self.liveTokensPerSec = 0
                    self.rateEWMA = 0
                }
            }
        }
    }

    /// Poll the allocator numbers behind the memory gauge. Cheap on the server side (it reads
    /// the allocator, never the GPU), so a relaxed cadence is plenty.
    private func startMemoryPolling() {
        guard let client else { return }
        memoryTask?.cancel()
        memoryTask = Task { [weak self] in
            while !Task.isCancelled {
                if let memory = try? await client.engineMemory() {
                    self?.memory = memory
                }
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    func refreshDiagnostics() async {
        guard let client else { return }
        async let report = try? client.doctor()
        async let inventory = try? client.modelInventory()
        async let curves = try? client.calibration()
        doctorReport = await report
        if let inventory = await inventory {
            models = inventory.models
            installedModels = inventory.installed ?? []
            diskUsage = inventory.disk
        }
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

        chatError = nil
        messages.append(ChatMessage(role: .user, text: text))
        messages.append(ChatMessage(role: .assistant, text: ""))
        prompt = ""
        isGenerating = true
        persistCurrentSession()

        // History goes back *without* the reasoning traces: `<think>` blocks are the model
        // talking to itself, resending them just inflates prefill (which already dominates).
        var history: [[String: String]] = []
        let system = chatSettings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty { history.append(["role": "system", "content": system]) }
        history += messages.dropLast().map { message in
            let content = message.role == .assistant
                ? ThinkingSplit.parse(message.text).answer : message.text
            return ["role": message.role == .user ? "user" : "assistant", "content": content]
        }

        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.streamChat(
                    model: self.model,
                    messages: history,
                    temperature: self.chatSettings.temperature,
                    maxTokens: self.chatSettings.maxTokens,
                    enableThinking: self.chatSettings.thinking ? nil : false
                ) {
                    switch event {
                    case .delta(let piece):
                        // If channelled reasoning opened a synthetic think block, the first
                        // answer text closes it — from here on the message reads exactly
                        // like an inline-thinking model's output.
                        var text = self.messages[self.messages.count - 1].text
                        if text.hasPrefix("<think>"), !text.contains("</think>") {
                            text += "</think>\n"
                        }
                        self.messages[self.messages.count - 1].text = text + piece
                    case .reasoning(let piece):
                        // Muse-class models stream thinking as a separate channel; fold it
                        // into the same `<think>` form the thinking card already renders.
                        var text = self.messages[self.messages.count - 1].text
                        if !text.hasPrefix("<think>") { text = "<think>" + text }
                        self.messages[self.messages.count - 1].text = text + piece
                    case .finished(let info):
                        self.messages[self.messages.count - 1].stats = info
                        // The turn's true mean rate, replacing the smoothed live estimate.
                        if let info {
                            self.liveTokensPerSec = info.tokensPerSec
                            self.rateEWMA = info.tokensPerSec
                        }
                        self.lastActivity = Date()
                    }
                }
            } catch {
                if !Task.isCancelled { self.chatError = error.localizedDescription }
            }
            self.isGenerating = false
            self.persistCurrentSession()
            await self.refreshStats()
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        isGenerating = false
        persistCurrentSession()
    }

    // MARK: - Chat sessions

    private func loadSessions() {
        sessions = chatStore.list()
        if let saved = Defaults.currentSession,
           let session = sessions.first(where: { $0.id == saved }) {
            currentSessionID = session.id
            messages = session.messages
        } else if let latest = sessions.first {
            currentSessionID = latest.id
            messages = latest.messages
        }
        // No sessions yet: stay unsaved until the first message, so quitting a fresh app
        // doesn't litter empty files.
    }

    func newChat() {
        cancelGeneration()
        persistCurrentSession()
        currentSessionID = nil
        Defaults.currentSession = nil
        messages = []
        chatError = nil
        screen = .chat
    }

    func selectSession(_ id: UUID) {
        guard id != currentSessionID else { return }
        cancelGeneration()
        persistCurrentSession()
        guard let session = sessions.first(where: { $0.id == id }) else { return }
        currentSessionID = session.id
        Defaults.currentSession = session.id
        messages = session.messages
        chatError = nil
    }

    func deleteSession(_ id: UUID) {
        chatStore.delete(id)
        sessions.removeAll { $0.id == id }
        if id == currentSessionID {
            currentSessionID = nil
            Defaults.currentSession = nil
            messages = []
        }
    }

    /// Write the live conversation back to disk (and materialize the session on first use).
    private func persistCurrentSession() {
        guard !messages.isEmpty else { return }
        var session: ChatSession
        if let id = currentSessionID, let existing = sessions.first(where: { $0.id == id }) {
            session = existing
        } else {
            session = ChatSession()
            currentSessionID = session.id
            Defaults.currentSession = session.id
        }
        session.messages = messages
        session.updatedAt = Date()
        session.retitleFromContent()
        chatStore.save(session)
        sessions.removeAll { $0.id == session.id }
        sessions.insert(session, at: 0)
    }

    var currentSessionTitle: String {
        guard let id = currentSessionID,
              let session = sessions.first(where: { $0.id == id }) else { return "New chat" }
        return session.title
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

    /// The loaded model's resident footprint, ready for the chrome. Peak is deliberately not
    /// shown here — a gauge that never goes down reads as a leak.
    var memoryLine: String? {
        guard let gb = memory?.activeGB, gb > 0.05 else { return nil }
        return String(format: "%.1f GB", gb)
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

    static var currentSession: UUID? {
        get { store.string(forKey: "currentSession").flatMap(UUID.init(uuidString:)) }
        set { store.set(newValue?.uuidString, forKey: "currentSession") }
    }

    static var chatSettings: ChatSettings {
        get {
            guard let data = store.data(forKey: "chatSettings"),
                  let settings = try? JSONDecoder().decode(ChatSettings.self, from: data)
            else { return ChatSettings() }
            return settings
        }
        set { store.set(try? JSONEncoder().encode(newValue), forKey: "chatSettings") }
    }
}
