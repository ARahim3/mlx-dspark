import AppCore
import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                DetailLevelCard()
                if let report = model.doctorReport { MachineCard(report: report) }
                ServerCard()
            }
            .padding(16)
        }
        .task { await model.refreshDiagnostics() }
    }
}

/// Progressive disclosure — LM Studio's most-copied idea, and the thing that decides whether
/// this app is usable by anyone who didn't write it.
struct DetailLevelCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "How much to show") {
            Picker("", selection: $model.detail) {
                ForEach(Detail.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(model.detail.blurb).font(.callout).foregroundStyle(.secondary)
        }
    }
}

struct MachineCard: View {
    let report: DoctorReport

    var body: some View {
        Card(title: "This Mac",
             subtitle: report.ok ? "Everything checks out." : "Some things need attention.") {
            VStack(alignment: .leading, spacing: 7) {
                row("Chip", report.environment.device ?? report.environment.machine)
                if let ram = report.environment.ramGB {
                    row("Memory", String(format: "%.0f GB", ram))
                }
                row("macOS", report.environment.osVersion ?? "—")
                row("Engine", report.environment.version)
                let versions = ["mlx", "mlx_lm", "mlx_vlm"]
                    .compactMap { name -> String? in
                        guard let v = report.environment.packages[name] ?? nil else { return nil }
                        return "\(name) \(v)"
                    }
                row("Runtime", versions.joined(separator: " · "))

                ForEach(report.problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout).foregroundStyle(.orange)
                }

                // Letting macOS page weights out mid-generation is the classic silent
                // slowdown, so the fix is offered as a copyable command rather than advice.
                if let hint = report.environment.wiredLimitHint {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Large models run faster if the GPU can keep them resident:")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Text(hint).font(.caption.monospaced()).textSelection(.enabled)
                            Button("Copy") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(hint, forType: .string)
                            }
                            .buttonStyle(.link).font(.caption)
                        }
                    }
                    .padding(.top, 4)
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.callout).foregroundStyle(.secondary).frame(width: 78, alignment: .leading)
            Text(value).font(.callout).textSelection(.enabled)
            Spacer()
        }
    }
}

struct ServerCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "Local server",
             subtitle: "OpenAI- and Anthropic-compatible, on this machine only.") {
            if case .ready(let port, let health) = model.serverState {
                VStack(alignment: .leading, spacing: 8) {
                    endpoint("OpenAI", "http://127.0.0.1:\(port)/v1")
                    endpoint("Anthropic", "http://127.0.0.1:\(port)")
                    if let window = health.contextWindow {
                        Text("Context window \(window) tokens")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Point Claude Code at it with:  mlx-dspark claude")
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            } else {
                Text(model.statusLine).foregroundStyle(.secondary)
            }
        }
    }

    private func endpoint(_ label: String, _ url: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            Text(url).font(.callout.monospaced()).textSelection(.enabled)
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(url, forType: .string)
            }
            .buttonStyle(.link).font(.caption)
            Spacer()
        }
    }
}
