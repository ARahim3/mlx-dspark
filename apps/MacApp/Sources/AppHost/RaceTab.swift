import AppCore
import SwiftUI

/// Same prompt, several decode strategies, side by side — with the losslessness claim
/// *checked* rather than asserted.
struct RaceTab: View {
    @EnvironmentObject private var model: AppModel
    // App-owned (see MlxDsparkApp): the run survives switching tabs/screens.
    @EnvironmentObject private var race: RaceModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            RaceControls(race: race)

            if case .failed(let message) = race.phase {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange).font(.callout)
            }

            if !race.lanes.isEmpty {
                if let verdict = race.verdict { VerdictBanner(verdict: verdict) }
                if race.phase == .done { ResultsTable(race: race) }
                RaceLanes(race: race)
            } else {
                ContentUnavailableView(
                    "Race two decoders",
                    systemImage: "flag.checkered",
                    description: Text("Run the same prompt through speculative decoding and "
                                      + "plain decoding, then compare speed — and check the "
                                      + "output really is identical."))
                    .frame(height: 240)
            }
        }
        .onAppear {
            if race.selectedArms.isEmpty {
                race.selectedArms = race.defaultArms(available: model.availableRaceArms)
            }
        }
        // A hot swap can change what's raceable (a drafterless target has no dspark arm);
        // stale arms would 400 the next run.
        .onChange(of: model.availableRaceArms) { _, available in
            let valid = Set(available)
            if race.phase != .running,
               race.selectedArms.contains(where: { !valid.contains($0.mode) }) {
                race.selectedArms = race.defaultArms(available: available)
            }
        }
    }
}

struct RaceControls: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var race: RaceModel
    // String mirror of race.maxTokens, parsed on change and clamped only on commit. Binding
    // the field to the Int directly (value:format:) re-parses every keystroke, so a live
    // clamp rewrites the text under the cursor mid-entry ("500" passes through "5" -> 16).
    @State private var maxTokensText: String = ""

    /// Everything this pair could race: baseline first (the reference belongs on the left),
    /// then the drafter at a few caps, then lookup. The user toggles which of these run.
    private var candidateArms: [RaceArm] {
        var arms: [RaceArm] = [RaceArm(mode: "baseline")]
        if let drafterMode = model.availableRaceArms.first(where: { $0 == "dspark" || $0 == "dflash" }) {
            arms.append(contentsOf: [2, 3, 4, 6].map { RaceArm(mode: drafterMode, cap: $0) })
        }
        if model.availableRaceArms.contains("lookup") {
            arms.append(RaceArm(mode: "lookup"))
        }
        return arms
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                ForEach(RacePrompt.presets) { preset in
                    Button(preset.name) { race.prompt = preset.text }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(race.phase == .running)
                }
                Spacer()
                Toggle("Thinking", isOn: $race.thinking)
                    .toggleStyle(.checkbox)
                    .disabled(race.phase == .running)
                    .help("Let the model reason before answering. Off keeps a race about "
                          + "speed — thinking buries the answer both lanes are producing.")
                HStack(spacing: 4) {
                    Text("Max tokens").font(.callout).foregroundStyle(.secondary)
                    TextField("200", text: $maxTokensText)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 64)
                        .multilineTextAlignment(.trailing)
                        .disabled(race.phase == .running)
                        .onChange(of: maxTokensText) { _, text in
                            if let value = Int(text) { race.maxTokens = value }
                        }
                        .onSubmit(commitMaxTokens)
                }
            }

            TextField("Prompt", text: $race.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
                .disabled(race.phase == .running)

            HStack(spacing: 8) {
                // Toggle chips: pick the matchup. The server takes 2–4 arms; order is fixed
                // to the candidate order so baseline always lands in the left lane.
                ForEach(candidateArms) { arm in
                    ArmChip(arm: arm,
                            isOn: race.selectedArms.contains(arm),
                            disabled: race.phase == .running
                                || (!race.selectedArms.contains(arm) && race.selectedArms.count >= 4))
                        { toggle(arm) }
                }
                Spacer()
                if race.phase == .running {
                    ProgressView().controlSize(.small)
                    Button("Stop") { race.cancel() }
                } else {
                    if race.phase == .done {
                        Button(race.isReplaying ? "Replaying…" : "Replay side by side") {
                            race.replay()
                        }
                        .disabled(race.isReplaying)
                        .help("Play every arm back on one clock at the speeds just measured. "
                              + "The arms have to run one at a time, but the timings are real.")
                    }
                    Button("Run race") {
                        commitMaxTokens()
                        if let client = model.apiClient { race.run(client: client) }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.isServerReady || race.selectedArms.count < 2)
                }
            }
            if race.selectedArms.count < 2 {
                Text("Pick at least two arms to race.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .onAppear { maxTokensText = String(race.maxTokens) }
    }

    /// Free-form entry, sane bounds: the server clamps to its cap anyway, but a 0/negative
    /// value would silently fall back to the default. Applied on Enter and on Run, never
    /// per keystroke.
    private func commitMaxTokens() {
        let clamped = min(max(Int(maxTokensText) ?? race.maxTokens, 16), 4096)
        race.maxTokens = clamped
        maxTokensText = String(clamped)
    }

    private func toggle(_ arm: RaceArm) {
        if let index = race.selectedArms.firstIndex(of: arm) {
            race.selectedArms.remove(at: index)
        } else {
            race.selectedArms.append(arm)
            let order = candidateArms
            race.selectedArms.sort {
                (order.firstIndex(of: $0) ?? .max) < (order.firstIndex(of: $1) ?? .max)
            }
        }
    }
}

private struct ArmChip: View {
    let arm: RaceArm
    let isOn: Bool
    let disabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .imageScale(.small)
                Text(arm.label)
            }
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(isOn ? AnyShapeStyle(.tint.opacity(0.18)) : AnyShapeStyle(.quaternary.opacity(0.6)),
                        in: Capsule())
            .overlay(Capsule().strokeBorder(isOn ? AnyShapeStyle(.tint.opacity(0.55))
                                                 : AnyShapeStyle(.clear), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled && !isOn ? 0.4 : 1)
    }
}

/// The claim this project rests on, stated as a checked result.
struct VerdictBanner: View {
    let verdict: RaceVerdict

    private var identical: Bool { verdict.identical ?? false }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: identical ? "checkmark.seal.fill" : "arrow.triangle.branch")
                .foregroundStyle(identical ? .green : .yellow)
                .font(.title3)
            VStack(alignment: .leading, spacing: 3) {
                Text(identical ? "Identical output" : "Equally valid outputs")
                    .font(.headline)
                Text(verdict.detail)
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(verdict.contentDivergences) { divergence in
                    Text("\(divergence.label): first differs at token \(divergence.firstDiff)"
                         + (divergence.margin.map {
                             String(format: " · logit gap ≈ %.2f (approximate)", $0) } ?? ""))
                        .font(.caption.monospaced()).foregroundStyle(.tertiary)
                }
            }
            Spacer()
        }
        .padding(12)
        .background((identical ? Color.green : Color.yellow).opacity(0.10),
                    in: RoundedRectangle(cornerRadius: 9))
    }
}

struct ResultsTable: View {
    @ObservedObject var race: RaceModel

    var body: some View {
        let results = race.lanes.compactMap(\.result)
        let fastest = results.map(\.tokensPerSec).max() ?? 1

        VStack(alignment: .leading, spacing: 8) {
            if let (fast, slow) = race.speedup, slow.tokensPerSec > 0 {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(String(format: "%.2f×", fast.tokensPerSec / slow.tokensPerSec))
                        .font(.system(size: 30, weight: .semibold, design: .rounded))
                        .foregroundStyle(.tint)
                    Text("\(fast.label) vs \(slow.label)")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }

            ForEach(results) { result in
                HStack(spacing: 10) {
                    Text(result.label)
                        .font(.callout.weight(.medium))
                        .frame(width: 120, alignment: .leading)

                    GeometryReader { geometry in
                        let fraction = fastest > 0 ? result.tokensPerSec / fastest : 0
                        RoundedRectangle(cornerRadius: 4)
                            .fill(result.tokensPerSec == fastest
                                  ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary.opacity(0.45)))
                            .frame(width: max(2, geometry.size.width * fraction))
                    }
                    .frame(height: 16)

                    Text(String(format: "%.1f tok/s", result.tokensPerSec))
                        .font(.callout.monospacedDigit())
                        .frame(width: 92, alignment: .trailing)
                    Text(result.mode == "baseline" ? "—"
                         : String(format: "accept %.2f", result.acceptLen))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        .frame(width: 92, alignment: .trailing)
                    Text("\(result.targetForwards) verifies")
                        .font(.caption).foregroundStyle(.secondary)
                        .frame(width: 88, alignment: .trailing)
                }
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 9))
    }
}

struct RaceLanes: View {
    @ObservedObject var race: RaceModel

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ForEach(race.lanes) { lane in
                RaceLaneColumn(race: race, laneIndex: lane.index)
            }
        }
    }
}

/// One arm's output, rendered like a chat message (markdown, code blocks, math) and pinned
/// to the bottom while tokens stream — live and during replay alike.
struct RaceLaneColumn: View {
    @ObservedObject var race: RaceModel
    let laneIndex: Int

    private var lane: RaceModel.Lane? {
        race.lanes.indices.contains(laneIndex) ? race.lanes[laneIndex] : nil
    }

    private var displayText: String {
        guard let lane else { return "" }
        return race.isReplaying ? (race.replayText[laneIndex] ?? "") : lane.text
    }

    var body: some View {
        if let lane {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(lane.label).font(.caption.weight(.semibold))
                    if lane.isRunning { ProgressView().controlSize(.small) }
                    Spacer()
                    // Shown during replay too — the replay is the shot people record.
                    if let result = lane.result {
                        Text(String(format: "%.0f tok/s", result.tokensPerSec))
                            .font(.caption.monospacedDigit()).foregroundStyle(.tint)
                    }
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        MarkdownText(text: displayText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                        Color.clear.frame(height: 1).id("lane-end")
                    }
                    .frame(height: 340)
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
                    // Follow the stream like a chat window: every appended token keeps the
                    // tail in view. Replay resets the text, which naturally scrolls back up.
                    .onChange(of: displayText) { _, _ in
                        proxy.scrollTo("lane-end", anchor: .bottom)
                    }
                }
            }
        }
    }
}
