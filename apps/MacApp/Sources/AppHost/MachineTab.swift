import AppCore
import Charts
import SwiftUI

/// Lab → This Mac: the roofline view. Is this machine actually saturated, and what would
/// change that?
///
/// Every "GPU %" gauge reads ~100% during decode while the chip waits on memory, so the only
/// honest utilization is against physics: bandwidth ÷ bytes-per-token is the most a plain
/// decode can do. The engine measures both halves exactly (a one-time bandwidth microbench;
/// the loaded model's byte footprint), judges the plain step against them, and reports how far
/// *above* that ceiling speculation is taking the live rate.
struct MachineTab: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let report = model.machine {
            RooflineCard(report: report, lastStats: model.messages.last(where: { $0.stats != nil })?.stats)
            if let verdict = report.verdict { VerdictCard(verdict: verdict) }
            MemoryCard(report: report)
        } else {
            ContentUnavailableView(
                "Not measured yet",
                systemImage: "gauge.with.dots.needle.33percent",
                description: Text("Load a model — the engine measures this Mac's memory "
                                  + "bandwidth once and reports the roofline here."))
                .frame(height: 260)
        }
    }
}

/// Ceiling vs achieved, plus the model's byte bill per token.
struct RooflineCard: View {
    let report: MachineReport
    /// The most recent chat turn's stats, for the "achieved" bar.
    let lastStats: SpecInfo?

    private var bars: [(label: String, value: Double, achieved: Bool)] {
        var out: [(String, Double, Bool)] = []
        if let ceiling = report.roofline?.atZero?.ceilingTps {
            out.append(("plain-decode ceiling", ceiling, false))
        }
        if let stats = lastStats {
            out.append(("last turn (\(stats.mode))", stats.displayTokensPerSec, true))
        }
        return out
    }

    var body: some View {
        Card(title: "Roofline on this Mac",
             subtitle: "A plain decode reads every weight byte per token, so memory bandwidth "
                 + "÷ bytes per token is the most it can do. Speculative decoding commits "
                 + "several tokens per weight read — that is how the live rate gets above the "
                 + "ceiling.") {
            HStack(spacing: 18) {
                if let bw = report.bandwidth.gbs {
                    Metric(value: String(format: "%.0f GB/s", bw),
                           label: bandwidthLabel)
                }
                if let ceiling = report.roofline?.atZero?.ceilingTps {
                    Metric(value: String(format: "%.0f tok/s", ceiling),
                           label: "plain-decode ceiling")
                }
                if let deep = report.roofline?.atContextWindow, let ceiling = deep.ceilingTps {
                    Metric(value: String(format: "%.0f tok/s", ceiling),
                           label: "ceiling at \(deep.context.formatted()) tokens of context")
                }
                if let baseline = report.baseline, let mbu = baseline.mbu {
                    Metric(value: String(format: "%.0f%%", mbu * 100),
                           label: "of bandwidth, plain step", tint: mbuTint(mbu))
                }
                Spacer()
            }

            if !bars.isEmpty {
                Chart(bars, id: \.label) { bar in
                    BarMark(x: .value("tok/s", bar.value), y: .value("", bar.label))
                        .foregroundStyle(bar.achieved ? Theme.spark : Color.secondary.opacity(0.5))
                        .annotation(position: .trailing) {
                            Text(String(format: "%.0f", bar.value))
                                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        }
                }
                .chartXAxisLabel("tok/s")
                .frame(height: CGFloat(40 + 30 * bars.count))
                if let stats = lastStats, let ratio = stats.rooflineRatio {
                    Text(ratio >= 1.0
                         ? String(format: "The last turn ran at %.1f× the single-stream "
                                  + "roofline at its context depth.", ratio)
                         : String(format: "The last turn ran at %.1f× the roofline — below "
                                  + "what a plain step would do on this content.", ratio))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            if let m = report.model {
                let weights = ByteFormat.gb(m.targetWeights.activeBytes) ?? "?"
                let moe = m.targetWeights.isMoe
                    ? " active of \(ByteFormat.gb(m.targetWeights.totalBytes) ?? "?") total "
                      + "(\(m.targetWeights.expertsPerTok ?? 0) of \(m.targetWeights.nExperts ?? 0) experts per token)"
                    : ""
                let kv = ByteFormat.kb(m.kvBytesPerToken).map { " · \($0) of KV cache per token of context" } ?? ""
                Text("\(m.target?.components(separatedBy: "/").last ?? "model"): \(weights) read per token\(moe)\(kv)"
                     + (m.targetWeights.activeIsEstimate ? " (active bytes estimated)" : ""))
                    .font(.caption2).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var bandwidthLabel: String {
        let spec = report.chip.bandwidthGBs.map { String(format: " · %.0f spec", $0) } ?? ""
        return (report.bandwidth.source == "measured" ? "measured bandwidth" : "bandwidth (spec)") + spec
    }

    private func mbuTint(_ mbu: Double) -> Color {
        mbu >= 0.75 ? Theme.verified : (mbu >= 0.5 ? Theme.warning : .red)
    }
}

/// The engine's judgement: headline, what it saw, what to try next.
struct VerdictCard: View {
    let verdict: MachineReport.Verdict

    var body: some View {
        Card(title: "Verdict", subtitle: "From the last request and what macOS reports right now.") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(verdict.level.uppercased())
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(tint.opacity(0.15), in: Capsule())
                        .foregroundStyle(tint)
                    Text(verdict.headline).font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(verdict.findings, id: \.self) { finding in
                    Label(finding, systemImage: "circle.fill")
                        .labelStyle(BulletLabelStyle())
                        .font(.callout).foregroundStyle(.secondary)
                }
                if !verdict.levers.isEmpty {
                    Text("Next levers").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                        .padding(.top, 2)
                    ForEach(verdict.levers, id: \.self) { lever in
                        Label(lever, systemImage: "arrow.turn.down.right")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var tint: Color {
        switch verdict.level {
        case "healthy":   return Theme.verified
        case "ok":        return Theme.spark
        case "attention": return Theme.warning
        case "problem":   return .red
        default:          return .secondary
        }
    }
}

private struct BulletLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            configuration.icon.font(.system(size: 5))
            configuration.title.fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Model memory next to what macOS sees — the pair that explains the fits-but-swaps cliff.
struct MemoryCard: View {
    let report: MachineReport

    var body: some View {
        let mem = report.memory
        Card(title: "Memory",
             subtitle: "When the model plus its KV cache approaches what macOS can keep "
                 + "resident, it compresses and swaps — and decode falls off a cliff. "
                 + "Pressure is macOS's own verdict.") {
            VStack(alignment: .leading, spacing: 7) {
                if let alloc = mem.allocator, let active = ByteFormat.gb(alloc.activeBytes) {
                    row("Model resident", active
                        + (ByteFormat.gb(alloc.peakBytes).map { " (peak \($0))" } ?? ""))
                }
                if let total = ByteFormat.gb(mem.totalBytes, digits: 0) {
                    // `free_percent` is kern.memorystatus_level — the share macOS could
                    // reclaim WITHOUT swapping (free + inactive/file-cache pages), which is
                    // what its pressure verdict keys off. It is NOT the inverse of Activity
                    // Monitor's "Used" (that counts app + wired + compressed), so it reads
                    // higher than "100 − used%". Label it as what it is.
                    row("Unified memory", total
                        + (mem.freePercent.map { " · \($0)% reclaimable without swapping (macOS memory level)" } ?? ""))
                }
                if let pressure = mem.pressure {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Pressure").font(.callout).foregroundStyle(.secondary)
                            .frame(width: 120, alignment: .leading)
                        Text(pressure.uppercased())
                            .font(.callout.weight(mem.isUnderPressure ? .semibold : .regular))
                            .foregroundStyle(pressure == "critical" ? .red
                                             : (pressure == "warn" ? Theme.warning : .primary))
                        Spacer()
                    }
                }
                if let used = ByteFormat.gb(mem.swapUsedBytes), let total = ByteFormat.gb(mem.swapTotalBytes) {
                    row("Swap", "\(used) of \(total) in use")
                }
                if let limit = mem.wiredLimitMB {
                    row("GPU wired limit", "\(limit) MB (raised)")
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.callout).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
            Text(value).font(.callout).textSelection(.enabled)
            Spacer()
        }
    }
}
