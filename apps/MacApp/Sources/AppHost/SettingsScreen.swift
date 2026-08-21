import AppCore
import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                DetailLevelCard()
                if model.detail != .simple { DecodingCard() }
                if let report = model.doctorReport { MachineCard(report: report) }
                ServerCard()
                AboutCard()
            }
            .padding(16)
        }
        .task { await model.refreshDiagnostics() }
    }
}

/// Versions and updates. Two version numbers on purpose: the app and the engine release
/// independently — the engine keeps itself on the latest release automatically, the app
/// updates through Homebrew (or a fresh DMG) and only *tells* you when one exists.
struct AboutCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "About") {
            VStack(alignment: .leading, spacing: 7) {
                row("App", AppIdentity.appVersion)
                row("Engine", model.doctorReport?.environment.version ?? "—")
                if let update = model.appUpdate {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("App v\(update.version) is available.",
                              systemImage: "arrow.down.circle.fill")
                            .font(.callout).foregroundStyle(Theme.spark)
                        HStack(spacing: 8) {
                            Text("brew upgrade --cask mlx-dspark")
                                .font(.caption.monospaced()).textSelection(.enabled)
                            CopyButton(text: "brew upgrade --cask mlx-dspark")
                            Button("Release notes") {
                                if let url = URL(string: update.url) { NSWorkspace.shared.open(url) }
                            }
                            .buttonStyle(.link).font(.caption)
                        }
                    }
                    .padding(.top, 4)
                } else {
                    Text("The engine stays on the latest release automatically; app updates "
                         + "are checked at launch.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let engineUpdate = model.engineUpdateAvailable {
                    HStack(spacing: 8) {
                        Label("Engine \(engineUpdate) is available.",
                              systemImage: "arrow.triangle.2.circlepath")
                            .font(.caption).foregroundStyle(.secondary)
                        Button(model.engineUpdating ? "Updating…" : "Update now") {
                            Task { await model.applyEngineUpdateNow() }
                        }
                        .buttonStyle(.link).font(.caption)
                        .disabled(model.engineUpdating)
                        Text("or it installs on the next launch")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            Text(value).font(.callout).textSelection(.enabled)
            Spacer()
        }
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

/// The engine-level knobs: decode mode and draft cap.
///
/// Both are speed dials, never behavior dials — the target verifies every token, so output is
/// byte-identical across all of them. Applying reloads the model in place (the CLI's
/// `--mode` / `--max-draft`, via `/admin/load` overrides); the port survives.
struct DecodingCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "Decoding", subtitle: model.decodingLine) {
            DecodingControls()

            Text("Output is byte-identical in every mode — these change speed, not text. "
                 + "Cap Auto calibrates this Mac once and adapts per round; pin a value if "
                 + "you've measured a better fixed cap for this model. Applying reloads the "
                 + "model in place (the server and its port stay up).")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Mode + cap pickers with Apply. Shared between Settings → Decoding and the chat toolbar's
/// settings popover, so the knobs live where the user already is.
struct DecodingControls: View {
    /// The popover is 340pt wide; both pickers are `.fixedSize()`, so the single-row layout
    /// designed for the Settings card pushes Apply past the popover's edge. Compact stacks
    /// the button on its own row instead.
    var compact = false

    @EnvironmentObject private var model: AppModel
    @State private var mode: String = "auto"
    @State private var cap: String = "auto"
    @State private var confidence: String = "off"
    @State private var contextWindow: String = "default"
    @State private var lookupDrafts: Bool = true
    @State private var kvBits: String = "default"
    @State private var applying = false

    /// Context presets as (tag, label, tokens). "default" = the model's own maximum.
    private static let contextPresets: [(tag: String, label: String, tokens: Int?)] = [
        ("default", "Model max", nil),
        ("8192", "8k", 8192), ("16384", "16k", 16384), ("32768", "32k", 32768),
        ("65536", "64k", 65536), ("131072", "128k", 131072), ("262144", "256k", 262144),
    ]

    private static func contextTag(_ value: Int?) -> String {
        guard let value, contextPresets.contains(where: { $0.tokens == value })
        else { return "default" }
        return String(value)
    }

    /// Health's kv_bits (0 = full precision) as a picker tag.
    private static func kvTag(_ value: Int?) -> String {
        guard let value, value == 4 || value == 8 else { return "default" }
        return String(value)
    }

    /// Health's 0.0/0.2/0.3/0.5 as picker tags ("off"/"0.2"/…). Values outside the preset
    /// list (a server started with an unusual --confidence-threshold) round to one decimal
    /// and appear as their own tag so the picker never lies about the running state.
    private static func confTag(_ value: Double?) -> String {
        guard let value, value > 0 else { return "off" }
        return String(format: "%.1f", value)
    }

    private var confOptions: [String] {
        var options = ["off", "0.2", "0.3", "0.5"]
        let current = Self.confTag(model.health?.confidenceThreshold)
        if !options.contains(current) { options.append(current) }
        return options
    }

    private var modes: [(id: String, label: String)] {
        // availableDecodingModes, NOT availableRaceArms: applying reloads the pair, so the
        // drafter mode stays selectable while Baseline/Lookup is running (it used to vanish).
        var options = [(id: "auto", label: "Auto")]
        for arm in model.availableDecodingModes {
            options.append((id: arm, label: arm == "dspark" ? "DSpark"
                            : arm == "dflash" ? "DFlash" : arm.capitalized))
        }
        return options
    }

    var body: some View {
        Group {
            if compact {
                // Three pickers no longer fit one 340pt popover row (the clipped-Apply
                // lesson, relearned the day Confidence landed): mode+cap on one row,
                // confidence on its own, Apply last.
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 14) {
                        modeCapPickers
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        confidencePicker
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        contextPicker
                        Spacer(minLength: 0)
                    }
                    if model.health?.kvBits != nil {
                        HStack(spacing: 14) {
                            kvPicker
                            Spacer(minLength: 0)
                        }
                    }
                    if model.health?.lookupDrafts != nil {
                        HStack(spacing: 14) {
                            lookupToggle
                            Spacer(minLength: 0)
                        }
                    }
                    HStack(spacing: 8) {
                        applyControl
                        if !applying, isDirty {
                            Text("Reloads the model in place.")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                        Spacer(minLength: 0)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 14) {
                        modeCapPickers
                        confidencePicker
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        contextPicker
                        if model.health?.kvBits != nil {
                            kvPicker
                        }
                        if model.health?.lookupDrafts != nil {
                            lookupToggle
                        }
                        applyControl
                        Spacer(minLength: 0)
                    }
                }
            }
        }
        // The pickers show what the engine is actually running, not a stale default — the
        // server reports both (`/health` mode + max_draft), so a reopened popover agrees
        // with what was applied.
        .onAppear {
            mode = model.health?.mode ?? "auto"
            cap = model.health?.maxDraft ?? "auto"
            confidence = Self.confTag(model.health?.confidenceThreshold)
            contextWindow = Self.contextTag(model.health?.contextWindow)
            lookupDrafts = model.health?.lookupDrafts ?? true
            kvBits = Self.kvTag(model.health?.kvBits)
        }
    }

    @ViewBuilder private var modeCapPickers: some View {
        Picker("Mode", selection: $mode) {
            ForEach(modes, id: \.id) { Text($0.label).tag($0.id) }
        }
        .fixedSize()

        Picker("Cap", selection: $cap) {
            Text("Auto").tag("auto")
            ForEach(1...8, id: \.self) { Text("\($0)").tag("\($0)") }
        }
        .fixedSize()
    }

    @ViewBuilder private var confidencePicker: some View {
        Picker("Confidence", selection: $confidence) {
            ForEach(confOptions, id: \.self) { Text($0 == "off" ? "Off" : $0).tag($0) }
        }
        .fixedSize()
        .help("Confidence-head early stop: the drafter truncates its own block when it "
              + "stops believing in it. Pays only where the verify curve still rises inside "
              + "the cap AND the drafter leaves acceptance headroom — e.g. the "
              + "Qwen3.6-35B-A3B MoE. Off is right where the drafter already accepts near "
              + "its ceiling (Qwen3.8-27B) or the curve is flat (8-bit targets).")
    }

    @ViewBuilder private var contextPicker: some View {
        Picker("Context", selection: $contextWindow) {
            ForEach(Self.contextPresets, id: \.tag) { Text($0.label).tag($0.tag) }
        }
        .fixedSize()
        .help("Cap the context window below the model's own maximum — a RAM lever: the "
              + "KV cache grows with every token of context (~84 KB/token on the "
              + "Qwen3.8-27B pair), so a long agent session at full context can add "
              + "many GB. Requests past the cap get a clear \"prompt is too long\", "
              + "which agent clients like Claude Code auto-compact on.")
    }

    /// Only rendered when `/health` reports `kv_bits` (engines below 0.13.1 also lack the
    /// `/admin/load` override, so the picker would silently do nothing — issue #17).
    @ViewBuilder private var kvPicker: some View {
        Picker("KV cache", selection: $kvBits) {
            Text("Full").tag("default")
            Text("8-bit").tag("8")
            Text("4-bit").tag("4")
        }
        .fixedSize()
        .help("Quantize the KV cache (the per-token memory that grows with context): 8-bit "
              + "halves it, 4-bit quarters it — the long-context RAM lever after the "
              + "context cap. Output stays lossless the same way the rest of the engine "
              + "is (the target verifies every token against its own kv-quantized "
              + "forward); quality at very long contexts is the usual KV-quantization "
              + "trade. Full = the model's own precision.")
    }

    /// Only rendered when `/health` reports the field (older engines don't) — a toggle
    /// that silently does nothing is worse than none.
    @ViewBuilder private var lookupToggle: some View {
        Toggle("Lookup drafts", isOn: $lookupDrafts)
            .fixedSize()
            .help("Hybrid n-gram drafts: a 4-gram match in the context supplies a free "
                  + "draft instead of running the drafter that round. Shipped per-pair at "
                  + "its measured best — OFF where extra verify rows cost more than the "
                  + "free draft saves (every MoE, the 4-bit 27B hybrids), on elsewhere. "
                  + "Flip it to A/B on your own content; it can't affect output, only speed.")
    }

    @ViewBuilder private var applyControl: some View {
        if applying {
            ProgressView().controlSize(.small)
        } else {
            Button("Apply") { apply() }
                .disabled(!model.isServerReady || !isDirty)
        }
    }

    private var isDirty: Bool {
        mode != (model.health?.mode ?? "auto") || cap != (model.health?.maxDraft ?? "auto")
            || confidence != Self.confTag(model.health?.confidenceThreshold)
            || contextWindow != Self.contextTag(model.health?.contextWindow)
            || lookupDrafts != (model.health?.lookupDrafts ?? lookupDrafts)
            || kvBits != Self.kvTag(model.health?.kvBits)
    }

    private func apply() {
        applying = true
        Task {
            await model.applyEngineSettings(
                mode: mode, cap: cap,
                confidence: confidence == "off" ? 0.0 : Double(confidence),
                contextWindow: contextWindow == "default" ? nil : Int(contextWindow),
                // Sent only when the user actually flipped it, so an untouched Apply
                // keeps riding the pair's measured default instead of pinning it.
                lookupDrafts: lookupDrafts == model.health?.lookupDrafts ? nil : lookupDrafts,
                // Same: unchanged -> nil (keep the server's setting); "default" -> explicit
                // 0 (full precision). Gated on health reporting the field at all.
                kvBits: kvBits == Self.kvTag(model.health?.kvBits) ? nil
                        : (kvBits == "default" ? 0 : Int(kvBits)))
            applying = false
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
                if let chip = report.environment.chip, let spec = chip.bandwidthGBs {
                    // The number that governs decode speed on a Mac — spec sheet next to
                    // what a microbench actually achieves here (~80–90% of spec is normal).
                    let measured = chip.bandwidthMeasuredGBs
                        .map { String(format: " · %.0f GB/s measured", $0) } ?? ""
                    row("Bandwidth", String(format: "%.0f GB/s spec", spec) + measured)
                }
                if let pressure = report.environment.memory?.pressure, pressure != "unknown" {
                    row("Pressure", pressure == "normal" ? "normal"
                        : pressure.uppercased() + " — generation runs slower until it clears")
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
    // String mirror, committed on submit/apply — a TextField(value:format:) with a live
    // clamp rewrites the text under the cursor (see the Race max-tokens field).
    @State private var portText: String =
        Defaults.enginePort == 0 ? "" : String(Defaults.enginePort)
    @State private var restarting = false

    var body: some View {
        Card(title: "Local server",
             subtitle: "OpenAI- and Anthropic-compatible, on this machine only.") {
            VStack(alignment: .leading, spacing: 8) {
                if case .ready(let port, let health) = model.serverState {
                    endpoint("OpenAI", "http://127.0.0.1:\(port)/v1")
                    endpoint("Anthropic", "http://127.0.0.1:\(port)")
                    if let window = health.contextWindow {
                        Text("Context window \(window) tokens")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Point Claude Code at it with:  mlx-dspark claude")
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                } else {
                    Text(model.statusLine).foregroundStyle(.secondary)
                }
                Divider()
                portRow
            }
        }
    }

    /// Fixed engine port (issue #16): external clients (Claude Code, Open WebUI, …) need a
    /// base URL that survives an app restart; the automatic port changes on every launch.
    @ViewBuilder private var portRow: some View {
        HStack(spacing: 8) {
            Text("Port").font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            TextField("automatic", text: $portText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 92)
                .onSubmit { commitPort() }
            Button(restarting ? "Restarting…" : "Apply & restart engine") {
                commitPort()
                restarting = true
                Task { await model.restartEngine(); restarting = false }
            }
            .font(.caption)
            .disabled(restarting)
            Spacer()
        }
        Text("A fixed port keeps external clients' base URL stable across launches "
             + "(default \(Defaults.defaultEnginePort); if something else holds it, the "
             + "app falls back to an automatic port). Blank = always automatic. "
             + "Ports 1024–65535.")
            .font(.caption).foregroundStyle(.secondary)
    }

    private func commitPort() {
        let value = Int(portText.trimmingCharacters(in: .whitespaces)) ?? 0
        Defaults.enginePort = (1024...65535).contains(value) ? value : 0
        portText = Defaults.enginePort == 0 ? "" : String(Defaults.enginePort)
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
