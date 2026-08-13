import AppCore
import Charts
import SwiftUI

/// The Lab — what makes this app worth having over any other local-LLM front end.
///
/// Every chart here is drawn from data the engine already produced and previously threw away:
/// the machine's measured verify-cost curve, and the per-round acceptance record. No
/// competitor shows either, because no competitor measures them.
struct LabScreen: View {
    @EnvironmentObject private var model: AppModel
    @State private var tab: Tab = Tab(rawValue: Defaults.labTab) ?? .live

    enum Tab: String, CaseIterable, Identifiable {
        case race = "Race", live = "Live", curves = "Curves"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tab) {
                ForEach(Tab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 280)
            .padding(12)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if model.showLabWelcome { LabWelcomeBanner() }
                    switch tab {
                    case .race:   RaceTab()
                    case .live:   LiveTab()
                    case .curves: CurvesTab()
                    }
                }
                .padding(16)
            }
        }
        .task { await model.refreshStats() }
        .onChange(of: tab) { _, new in Defaults.labTab = new.rawValue }
    }
}

/// One-time landing after onboarding: the model is loaded, this Mac has been measured, and
/// the fastest way to be convinced is to race the decoders — so that's where the user starts.
struct LabWelcomeBanner: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "bolt.fill").foregroundStyle(Theme.spark).font(.title3)
            VStack(alignment: .leading, spacing: 3) {
                Text("Your Mac has been measured.").font(.headline)
                Text("Run the race below: the same prompt through speculative and plain "
                     + "decoding, with the output checked for equality — not asserted. "
                     + "Curves shows the measurements this machine produced.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Button {
                model.showLabWelcome = false
            } label: {
                Image(systemName: "xmark").imageScale(.small)
            }
            .buttonStyle(.borderless)
        }
        .padding(12)
        .background(Theme.spark.opacity(0.08), in: RoundedRectangle(cornerRadius: 9))
        .overlay(RoundedRectangle(cornerRadius: 9)
            .strokeBorder(Theme.spark.opacity(0.35), lineWidth: 1))
    }
}

// MARK: - Live

struct LiveTab: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let stats = model.stats, stats.rounds > 0 {
            HStack(spacing: 24) {
                Metric(value: String(format: "%.2f", stats.meanAcceptLen), label: "tokens / round")
                Metric(value: String(format: "%.0f%%", stats.draftAcceptance * 100),
                       label: "drafts accepted")
                Metric(value: "\(stats.rounds)", label: "rounds")
                Metric(value: "\(stats.committedTokens)", label: "tokens")
                if stats.lookupRounds > 0 {
                    Metric(value: "\(stats.lookupRounds)", label: "free lookup drafts",
                           tint: .purple)
                }
                Spacer()
            }
            .padding(.bottom, 4)

            AcceptanceDecayCard(stats: stats)
            AcceptHistogramCard(stats: stats)
            RoundTimelineCard(rounds: model.currentRunRounds)
        } else {
            ContentUnavailableView(
                "No rounds yet",
                systemImage: "waveform.path.ecg",
                description: Text("Send a message in Chat — every speculative round shows up here live."))
                .frame(height: 260)
        }
    }
}

/// **d₀, d₁, d₂ …** — the probability the drafter's i-th proposal survives verification.
///
/// This single chart is the project in one picture: speculation only pays while this curve
/// stays high, and the shape of its decay is exactly what decides the optimal cap.
struct AcceptanceDecayCard: View {
    let stats: RoundStats

    private var points: [(position: Int, rate: Double, offered: Int)] {
        zip(stats.positionAcceptance.indices, zip(stats.positionAcceptance, stats.positionOffered))
            .map { (position: $0, rate: $1.0, offered: $1.1) }
            // Positions sampled only a handful of times are noise, not signal — showing a
            // confident-looking 0% from a single draft would misrepresent the drafter.
            .filter { $0.offered >= max(3, stats.rounds / 50) }
    }

    var body: some View {
        Card(title: "Acceptance by draft position",
             subtitle: "d₀, d₁, d₂ … — how far ahead the drafter stays right. The decay here is what sets the best cap.") {
            if points.isEmpty {
                Text("Not enough rounds yet.").font(.callout).foregroundStyle(.secondary)
            } else {
                Chart(points, id: \.position) { point in
                    BarMark(
                        x: .value("Position", "d\(point.position)"),
                        y: .value("Accepted", point.rate)
                    )
                    // One hue fading with depth: the color itself reads as the decay.
                    .foregroundStyle(Theme.spark.opacity(max(0.35, 1.0 - 0.13 * Double(point.position))))
                    .annotation(position: .top) {
                        Text("\(Int(point.rate * 100))%")
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                .chartYScale(domain: 0...1)
                .chartYAxis {
                    AxisMarks(format: Decimal.FormatStyle.Percent.percent.scale(100))
                }
                .chartLegend(.hidden)
                .frame(height: 190)

                Text(points.map { "d\($0.position) n=\($0.offered)" }.joined(separator: "   "))
                    .font(.caption2.monospaced()).foregroundStyle(.tertiary)
            }
        }
    }
}

struct AcceptHistogramCard: View {
    let stats: RoundStats

    private var bars: [(committed: Int, rounds: Int)] {
        stats.acceptHistogram
            .compactMap { key, value in Int(key).map { (committed: $0, rounds: value) } }
            .sorted { $0.committed < $1.committed }
    }

    var body: some View {
        Card(title: "Tokens committed per round",
             subtitle: "Every round commits at least one — the target's own token — so this never has a zero bucket.") {
            Chart(bars, id: \.committed) { bar in
                BarMark(
                    x: .value("Tokens", bar.committed),
                    y: .value("Rounds", bar.rounds)
                )
                .foregroundStyle(Theme.spark)
            }
            .chartXScale(domain: 0...(max(bars.map(\.committed).max() ?? 1, 1) + 1))
            .frame(height: 150)
        }
    }
}

/// Per-round throughput over the current request, coloured by where the draft came from.
struct RoundTimelineCard: View {
    let rounds: [RoundEvent]

    var body: some View {
        Card(title: "This run, round by round",
             subtitle: "Committed tokens per round. Purple rounds got their draft free from the n-gram lookup.") {
            if rounds.isEmpty {
                Text("No rounds in the current run.").font(.callout).foregroundStyle(.secondary)
            } else {
                Chart(rounds) { round in
                    BarMark(
                        x: .value("Round", round.i),
                        y: .value("Committed", round.committed)
                    )
                    .foregroundStyle(color(for: round))
                }
                .chartLegend(.hidden)
                .frame(height: 140)

                HStack(spacing: 14) {
                    legend(Theme.spark, "drafter")
                    legend(Theme.lookup, "lookup")
                    legend(.secondary, "plain step")
                    Spacer()
                }
                .font(.caption2)
            }
        }
    }

    private func color(for round: RoundEvent) -> Color {
        Theme.source(round.source)
    }

    private func legend(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 9, height: 9)
            Text(label).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Curves

struct CurvesTab: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let calibration = model.calibration, calibration.available {
            VerifyCurveCard(calibration: calibration)
            if let controller = calibration.controller {
                PredictedRatesCard(controller: controller)
            }
        } else {
            ContentUnavailableView(
                "Not calibrated yet",
                systemImage: "ruler",
                description: Text(model.calibration?.reason
                                  ?? "This machine hasn't been measured yet."))
                .frame(height: 260)
        }
    }
}

/// The measured cost of one verify forward at each width — **this Mac's**, not a benchmark
/// from someone else's.
///
/// Its shape is the reason every tuning decision in this project goes the way it does: the
/// knee is where MLX's quantized matmul leaves its cheap few-rows path, and drafting past it
/// costs more than the extra tokens are worth.
struct VerifyCurveCard: View {
    let calibration: Calibration

    var body: some View {
        let curve = calibration.verifyCurve
        let knee = calibration.recommendation?.kneeWidth

        Card(title: "Verify cost on this Mac",
             subtitle: "Milliseconds for one target forward at each verify width. Speculation pays only while the extra rows stay cheap.") {
            Chart {
                ForEach(curve, id: \.width) { point in
                    LineMark(x: .value("Width", point.width), y: .value("ms", point.ms))
                        .interpolationMethod(.monotone)
                        .foregroundStyle(Theme.spark)
                    PointMark(x: .value("Width", point.width), y: .value("ms", point.ms))
                        .foregroundStyle(Theme.spark)
                }
                if let knee, curve.contains(where: { $0.width == knee }) {
                    RuleMark(x: .value("Knee", knee))
                        .foregroundStyle(Theme.warning)
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        // Bottom, not top: a top annotation collides with the card's own
                        // subtitle at this chart height.
                        .annotation(position: .bottom, alignment: .center) {
                            Text("knee").font(.caption2).foregroundStyle(.orange)
                        }
                }
            }
            .chartXAxis { AxisMarks(values: curve.map(\.width)) }
            .frame(height: 200)

            if let recommendation = calibration.recommendation {
                HStack(spacing: 18) {
                    Metric(value: "\(recommendation.kneeWidth)", label: "knee width", tint: .orange)
                    Metric(value: recommendation.recommend, label: "best drafting strategy")
                    if let overhead = calibration.roundOverheadMs, overhead > 0 {
                        Metric(value: String(format: "%.1f ms", overhead), label: "per-round overhead")
                    }
                    Spacer()
                }
            }
        }
    }
}

/// What the cost model predicts before anything runs — the calibrator's forecast, which the
/// Live tab then measures against reality.
struct PredictedRatesCard: View {
    let controller: Calibration.Controller

    private var rates: [(cap: Int, rate: Double)] {
        controller.predictedRates
            .compactMap { key, value in Int(key).map { (cap: $0, rate: value) } }
            .sorted { $0.cap < $1.cap }
    }

    var body: some View {
        Card(title: "Predicted throughput by cap",
             subtitle: "From the measured curves alone, before generating a token. Cap 0 means \"don't speculate this round\".") {
            Chart(rates, id: \.cap) { point in
                BarMark(x: .value("Cap", point.cap), y: .value("tok/s", point.rate))
                    .foregroundStyle(point.cap == controller.cap ? Theme.spark : Color.secondary.opacity(0.45))
                    .annotation(position: .top) {
                        Text("\(Int(point.rate))")
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    }
            }
            .chartXScale(domain: -0.5...(Double(rates.map(\.cap).max() ?? 1) + 0.5))
            .frame(height: 160)

            HStack(spacing: 18) {
                Metric(value: "\(controller.cap)", label: "cap in use", tint: .accentColor)
                Metric(value: String(format: "%.2f", controller.p), label: "live acceptance estimate")
                Metric(value: "\(controller.rounds)", label: "rounds observed")
                Spacer()
            }
        }
    }
}
