import AppCore
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            switch model.phase {
            case .launching, .settingUp, .startingServer:
                SetupView()
            case .ready:
                MainWindow()
            case .failed:
                FailureView()
            }
        }
        .frame(minWidth: 900, minHeight: 600)
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
        }
    }

    private var visibleScreens: [Screen] {
        Screen.allCases.filter { $0 != .lab || model.detail.showsLab }
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
                Circle()
                    .fill(model.isServerReady ? Color.green : Color.orange)
                    .frame(width: 6, height: 6)
                Text(model.health?.model ?? "starting…")
                    .font(.caption).lineLimit(1).truncationMode(.middle)
            }
            if model.liveTokensPerSec > 0 {
                Text("\(model.liveTokensPerSec, specifier: "%.0f") tok/s")
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                    .fontWeight(.medium)
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
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))

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
        case .done:    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:  Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }
}

// MARK: - Chrome

struct StatusBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            if let pairing = model.pairingLine {
                Image(systemName: "arrow.triangle.merge").imageScale(.small)
                Text(pairing).font(.caption.monospaced())
            } else {
                Text(model.statusLine).font(.caption)
            }
            Spacer()
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
                    ForEach(Array(model.logLines.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .frame(height: 160)
            .background(.quaternary.opacity(0.3))
            .onChange(of: model.logLines.count) { _, count in
                proxy.scrollTo(count - 1, anchor: .bottom)
            }
        }
    }
}

struct FailureView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 32)).foregroundStyle(.orange)
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
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }
}
