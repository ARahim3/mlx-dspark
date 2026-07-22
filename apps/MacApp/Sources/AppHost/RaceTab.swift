import AppCore
import SwiftUI

/// Same prompt, several decode strategies, side by side — with the losslessness claim
/// *checked* rather than asserted.
struct RaceTab: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var race = RaceModel()

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
    }
}

struct RaceControls: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var race: RaceModel

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
            }

            TextField("Prompt", text: $race.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
                .disabled(race.phase == .running)

            HStack(spacing: 10) {
                ForEach(race.selectedArms) { arm in
                    Text(arm.label)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(.quaternary.opacity(0.6), in: Capsule())
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
                        if let client = model.apiClient { race.run(client: client) }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.isServerReady || race.selectedArms.count < 2)
                }
            }
        }
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
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Text(lane.label).font(.caption.weight(.semibold))
                        if lane.isRunning { ProgressView().controlSize(.small) }
                        Spacer()
                        if let result = lane.result, !race.isReplaying {
                            Text(String(format: "%.0f tok/s", result.tokensPerSec))
                                .font(.caption.monospacedDigit()).foregroundStyle(.tint)
                        }
                    }
                    ScrollView {
                        Text(race.isReplaying ? (race.replayText[lane.index] ?? "") : lane.text)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    }
                    .frame(height: 240)
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }
}
