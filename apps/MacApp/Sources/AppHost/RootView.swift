import AppCore
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            switch model.phase {
            case .launching, .settingUp:
                SetupView()
            case .onboarding:
                OnboardingView()
            case .startingServer:
                LoadingView()
            case .ready:
                MainWindow()
            case .failed:
                FailureView()
            }
        }
        // The floor is deliberately modest: 900x600 blocked half-screen tiling and narrow
        // side-by-side recording layouts (user report). Below ~640 the sidebar can be
        // collapsed with the toolbar button; screens scroll rather than clip.
        .frame(minWidth: 640, minHeight: 440)
        .environment(\.textZoom, model.textZoom)
        .preferredColorScheme(model.appearance.colorScheme)
        .task { await model.boot() }
    }
}

// MARK: - Main window

struct MainWindow: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationSplitView {
            List(selection: $model.screen) {
                ForEach(visibleScreens) { screen in
                    Label(screen.title, systemImage: screen.symbol).tag(screen)
                }
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190, max: 240)
            .safeAreaInset(edge: .bottom) { SidebarFooter() }
        } detail: {
            VStack(spacing: 0) {
                switch model.screen {
                case .chat:     ChatScreen()
                case .lab:      LabScreen()
                case .agents:   AgentsScreen()
                case .models:   ModelsScreen()
                case .settings: SettingsScreen()
                }
                Divider()
                StatusBar()
                if model.showLogs {
                    Divider()
                    LogPane()
                }
            }
            .toolbar {
                // An update is the one thing worth a permanent spot in the chrome: the app
                // runs for days serving agents, and a note buried in Settings was never seen.
                // The pill only exists while there is something to act on.
                if model.hasUpdate {
                    ToolbarItem(placement: .primaryAction) { UpdatePill() }
                }
                // The model belongs in the chrome: hot-swapping is a headline feature, not a
                // settings-page errand. One click from any screen.
                ToolbarItem(placement: .primaryAction) { ModelPill() }
            }
        }
    }

    private var visibleScreens: [Screen] {
        Screen.allCases.filter { $0 != .lab || model.detail.showsLab }
    }
}

/// "Update available" in the window chrome — visible from every screen, gone once applied.
/// Engine updates apply in place from here; app updates hand you the brew line (installing
/// the app is Homebrew's / the DMG's job, see `AppUpdate`).
struct UpdatePill: View {
    @EnvironmentObject private var model: AppModel
    @State private var open = false

    var body: some View {
        Button {
            open.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "arrow.down.circle.fill")
                Text(title).font(.callout.weight(.semibold)).lineLimit(1)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(Theme.spark, in: Capsule())
        }
        .buttonStyle(.plain)
        .help("A newer release is available — click to see what and update")
        .popover(isPresented: $open, arrowEdge: .bottom) {
            UpdatePopover().environmentObject(model)
        }
    }

    private var title: String {
        switch (model.appUpdate, model.engineUpdateAvailable) {
        case (.some, .some): return "Updates available"
        case (.some(let app), nil): return "App v\(app.version) available"
        case (nil, .some(let engine)): return "Engine \(engine) available"
        case (nil, nil): return "Update available"
        }
    }
}

struct UpdatePopover: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Updates").font(.headline)
            if let engine = model.engineUpdateAvailable {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Engine \(engine)", systemImage: "arrow.triangle.2.circlepath")
                        .font(.callout.weight(.medium))
                    Text("Current: \(model.doctorReport?.environment.version ?? "—"). "
                         + "Installs in place; the model reloads (about a minute).")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(model.engineUpdating ? "Updating…" : "Update engine now") {
                        Task { await model.applyEngineUpdateNow() }
                    }
                    .disabled(model.engineUpdating)
                    .controlSize(.small)
                }
            }
            if let app = model.appUpdate {
                VStack(alignment: .leading, spacing: 4) {
                    Label("App v\(app.version)", systemImage: "app.badge")
                        .font(.callout.weight(.medium))
                    Text("Current: v\(AppIdentity.appVersion). Installed by Homebrew or a "
                         + "fresh DMG — the app can't replace itself.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        Text("brew upgrade --cask mlx-dspark")
                            .font(.caption.monospaced()).textSelection(.enabled)
                        CopyButton(text: "brew upgrade --cask mlx-dspark")
                        Button("Download / notes") {
                            if let url = URL(string: app.url) { NSWorkspace.shared.open(url) }
                        }
                        .controlSize(.small)
                    }
                }
            }
            Divider()
            HStack(spacing: 8) {
                if let when = model.lastUpdateCheck {
                    Text("Checked \(when.formatted(date: .omitted, time: .shortened))")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                Spacer()
                CheckForUpdatesButton()
            }
        }
        .padding(14)
        .frame(width: 360)
    }
}

/// "Check now" with its in-flight spinner and the one-line outcome — shared by the pill
/// popover, Settings → About and the menu-bar panel so all three behave the same.
struct CheckForUpdatesButton: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            if let message = model.updateCheckMessage, !model.updateCheckInFlight {
                Text(message).font(.caption).foregroundStyle(.secondary)
            }
            Button {
                Task { await model.checkForUpdates(manual: true) }
            } label: {
                if model.updateCheckInFlight {
                    HStack(spacing: 5) { ProgressView().controlSize(.mini); Text("Checking…") }
                } else {
                    Text("Check for updates")
                }
            }
            .controlSize(.small)
            .disabled(model.updateCheckInFlight)
        }
    }
}

/// Current model + one-click hot swap, from anywhere.
struct ModelPill: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Menu {
            Section("Measured pairs") {
                ForEach(model.models.filter(\.ready)) { row in
                    Button {
                        Task { await model.switchModel(to: row.target) }
                    } label: {
                        if row.target == model.model {
                            Label(row.shortTarget, systemImage: "checkmark")
                        } else {
                            Text(row.shortTarget)
                        }
                    }
                }
            }
            let extras = model.installedModels.filter { !$0.isDrafter && $0.registryID == nil }
            if !extras.isEmpty {
                Section("On this Mac") {
                    ForEach(extras) { installed in
                        Button {
                            Task { await model.switchModel(to: installed.repo) }
                        } label: {
                            if installed.repo == model.model {
                                Label(installed.shortRepo, systemImage: "checkmark")
                            } else {
                                Text(installed.shortRepo)
                            }
                        }
                    }
                }
            }
            Divider()
            if model.isServerReady {
                Button("Unload model") { Task { await model.unloadModel() } }
            }
            Button("All models…") { model.screen = .models }
        } label: {
            HStack(spacing: 6) {
                if model.isModelLoading {
                    ProgressView().controlSize(.mini)
                } else {
                    Circle()
                        .fill(model.isServerReady ? Theme.verified : Theme.warning)
                        .frame(width: 7, height: 7)
                }
                Text(title)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
            }
        }
        .menuIndicator(.visible)
        .help("Switch model — the server and its port stay up")
    }

    private var title: String {
        if model.isModelLoading {
            return "Loading \(model.model.components(separatedBy: "/").last ?? model.model)…"
        }
        return model.health?.model ?? "No model — choose"
    }
}

/// Ambient telemetry, always visible.
///
/// Straight from MTPLX, and cheap for how much it's liked: seeing tok/s move makes the speedup
/// feel real instead of being a number in a benchmark table you have to go looking for.
struct SidebarFooter: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            HStack(spacing: 6) {
                if model.isModelLoading {
                    ProgressView().controlSize(.mini)
                } else {
                    Circle()
                        .fill(model.isServerReady ? Theme.verified : Theme.warning)
                        .frame(width: 6, height: 6)
                }
                Text(model.isModelLoading
                     ? "Loading \(model.model.components(separatedBy: "/").last ?? model.model)…"
                     : model.health?.model ?? "No model")
                    .font(.caption).lineLimit(1).truncationMode(.middle)
            }
            if model.liveTokensPerSec > 0 {
                Text("\(model.liveTokensPerSec, specifier: "%.0f") tok/s")
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                    .fontWeight(.medium)
                    .foregroundStyle(Theme.spark)
                    .contentTransition(.numericText())
                    .animation(.easeOut(duration: 0.2), value: model.liveTokensPerSec)
            }
            if let stats = model.stats, stats.rounds > 0 {
                Text("accept \(stats.meanAcceptLen, specifier: "%.2f")")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Setup

/// The onboarding checklist.
///
/// Note the wording: the user is told an *engine* is being set up, never that a Python
/// environment is being built. That invisibility is the entire point of the vendored-uv
/// runtime — the implementation detail must not leak into the product.
struct SetupView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Setting up \(AppIdentity.displayName)")
                    .font(.system(size: 22, weight: .semibold))
                Text("This happens once. The first run downloads the engine, which takes a few minutes.")
                    .foregroundStyle(.secondary).font(.callout)
            }

            VStack(spacing: 0) {
                ForEach(model.setupSteps) { step in
                    SetupRow(step: step)
                    if step.id != model.setupSteps.last?.id { Divider().padding(.leading, 34) }
                }
            }
            .padding(.vertical, 4)
            .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))

            if model.phase == .startingServer {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text(model.statusLine).foregroundStyle(.secondary).font(.callout)
                }
            }
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct SetupRow: View {
    let step: SetupStep

    var body: some View {
        HStack(spacing: 12) {
            icon.frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title).font(.body)
                if !detailText.isEmpty {
                    Text(detailText)
                        .font(.caption)
                        .foregroundStyle(isFailed ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary))
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
    }

    private var isFailed: Bool { if case .failed = step.state { return true }; return false }

    private var detailText: String {
        if case .failed(let message) = step.state { return message }
        return step.detail
    }

    @ViewBuilder
    private var icon: some View {
        switch step.state {
        case .pending: Image(systemName: "circle").foregroundStyle(.tertiary)
        case .running: ProgressView().controlSize(.small)
        case .done:    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.verified)
        case .failed:  Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }
}

// MARK: - Chrome

struct StatusBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            if let progress = model.prefillProgress {
                ProgressView(value: progress.fraction)
                    .progressViewStyle(.linear)
                    .tint(Theme.spark)
                    .frame(width: 72)
                    .accessibilityLabel("Prefill progress")
                    .accessibilityValue("\(Int(progress.fraction * 100)) percent")
                Text("Prefill \(progress.processed.formatted()) / "
                     + "\(progress.total.formatted()) · \(Int(progress.fraction * 100))%")
                    .font(.caption.monospacedDigit())
                    .lineLimit(1).truncationMode(.middle)
            } else if let pairing = model.pairingLine {
                Image(systemName: "arrow.triangle.merge").imageScale(.small)
                Text(pairing).font(.caption.monospaced())
                    .lineLimit(1).truncationMode(.middle)
            } else {
                Text(model.statusLine).font(.caption)
                    .lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 12)
            AcceptRibbon(rounds: model.rounds)
            if let memory = model.memoryLine {
                HStack(spacing: 3) {
                    Image(systemName: "memorychip").imageScale(.small)
                    Text(memory).monospacedDigit()
                }
                .font(.caption)
                .help("Model memory held resident by the engine")
            }
            if let stats = model.stats, stats.rounds > 0 {
                Text("\(stats.rounds) rounds").font(.caption)
            }
            if model.detail.showsRawLogs {
                Button(model.showLogs ? "Hide Log" : "Log") { model.showLogs.toggle() }
                    .buttonStyle(.link).font(.caption)
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 14).padding(.vertical, 7)
    }
}

struct LogPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(model.logLines) { line in
                        Text(line.text)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                    }
                }
                .padding(8)
            }
            .frame(height: 160)
            .background(.quaternary.opacity(0.3))
            .onChange(of: model.logLines.last?.id) { _, last in
                if let last { proxy.scrollTo(last, anchor: .bottom) }
            }
        }
    }
}

/// Shown only while the server process itself spawns (`--no-model`, a few seconds) — model
/// loading happens *inside* the main window with inline progress, so a launch never parks the
/// whole app behind a full-screen wait anymore.
struct LoadingView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Starting the engine")
                        .font(.system(size: 20, weight: .semibold))
                }
                Text("A few seconds — the window opens before any model loads.")
                    .foregroundStyle(.secondary).font(.callout)
            }
            EngineLogTail()
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The engine's own recent log lines. On a first model download the honest progress lives
/// here (huggingface_hub writes it to stderr, which the supervisor captures) — a spinner
/// alone looks stuck for minutes.
struct EngineLogTail: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        let recent = Array(model.logLines.suffix(8))
        if !recent.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(recent) { line in
                    Text(line.text)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

struct FailureView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 32)).foregroundStyle(Theme.warning)
            Text("Something went wrong").font(.title3.weight(.semibold))
            Text(model.errorMessage ?? "Unknown error")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .frame(maxWidth: 460)
            Button("Try Again") { Task { await model.boot() } }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(28)
    }
}

// MARK: - Shared bits

/// A labelled number. Used everywhere; keeps figures on one baseline and stops digits from
/// jittering as values change.
struct Metric: View {
    let value: String
    let label: String
    var tint: Color?

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value)
                .font(.system(.title3, design: .rounded).monospacedDigit())
                .fontWeight(.medium)
                .foregroundStyle(tint ?? .primary)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct Card<Content: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                if let subtitle {
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
            }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.cardStroke, lineWidth: 1))
    }
}
