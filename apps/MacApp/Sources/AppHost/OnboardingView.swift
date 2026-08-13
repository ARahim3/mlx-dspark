import AppCore
import SwiftUI

/// First-run: check the Mac, then pick a model that fits it — before committing to a download.
///
/// The order is deliberate and taken from MTPLX's onboarding, the best in the field: tell the
/// user what their machine can do, recommend something that actually fits, and only then start
/// a multi-gigabyte download. Dropping someone straight into a chat with a hardcoded model
/// (what the app did before) hides both the choice and the cost.
struct OnboardingView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                if let report = model.onboarding {
                    MachineCheckCard(env: report.environment, problems: report.problems)
                    ModelPicker(report: report)
                } else {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text("Checking your Mac…").foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 30)
                }
            }
            .padding(28)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Welcome to \(AppIdentity.displayName)")
                .font(.system(size: 24, weight: .semibold))
            Text("Fast local models on Apple Silicon — and a look at exactly why they're fast. "
                 + "Pick one to start; you can switch any time.")
                .foregroundStyle(.secondary)
        }
    }
}

struct MachineCheckCard: View {
    let env: DoctorReport.Environment
    let problems: [String]

    var body: some View {
        Card(title: "Your Mac") {
            HStack(spacing: 22) {
                fact(env.device ?? env.machine, "chip")
                if let ram = env.ramGB { fact(String(format: "%.0f GB", ram), "memory") }
                if let os = env.osVersion { fact(os, "macOS") }
                fact(env.metalOK ? "ready" : "unavailable", "GPU",
                     tint: env.metalOK ? .green : .red)
                Spacer()
            }
            ForEach(problems, id: \.self) { problem in
                Label(problem, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout).foregroundStyle(.orange)
            }
        }
    }

    private func fact(_ value: String, _ label: String, tint: Color? = nil) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(.system(.body, design: .rounded)).fontWeight(.medium)
                .foregroundStyle(tint ?? .primary)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct ModelPicker: View {
    @EnvironmentObject private var model: AppModel
    let report: DoctorReport

    /// What to actually offer: things that fit this Mac, ready-to-run ones first (no download),
    /// then by descending RAM so the strongest model the machine can hold sits near the top.
    private var candidates: [ModelRow] {
        report.models
            .filter { $0.fits ?? true }
            .sorted { lhs, rhs in
                if lhs.ready != rhs.ready { return lhs.ready }        // installed first
                return (lhs.ramGB ?? 0) > (rhs.ramGB ?? 0)
            }
    }

    /// Recommend a *balanced* first model, not the biggest one that fits.
    ///
    /// The largest model a Mac can hold is the slowest and the heaviest thing to start on — a
    /// poor first impression. An 8–13 GB installed model is the sweet spot: comfortably fast,
    /// clearly benefits from speculation, and already on disk so the first run is instant.
    /// Falls back to any installed model, then to the smallest download.
    private var recommendedID: String? {
        let installed = candidates.filter(\.ready)
        let sweetSpot = installed
            .filter { ($0.ramGB ?? 0) >= 8 && ($0.ramGB ?? 0) <= 13 }
            .max { ($0.ramGB ?? 0) < ($1.ramGB ?? 0) }         // top of the comfortable band
        return sweetSpot?.id
            ?? installed.min { ($0.ramGB ?? 0) < ($1.ramGB ?? 0) }?.id
            ?? candidates.min { ($0.ramGB ?? 0) < ($1.ramGB ?? 0) }?.id
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Choose a model").font(.headline)
            Text("Ratios are best measured on an M4 Pro; yours will differ. Anything already "
                 + "downloaded starts instantly.")
                .font(.caption).foregroundStyle(.secondary)

            ForEach(candidates) { row in
                ModelPickRow(row: row, recommended: row.id == recommendedID) {
                    Task { await model.startServer(model: row.target) }
                }
            }

            if candidates.isEmpty {
                Text("No registry model fits this Mac's memory. You can still start any model "
                     + "from the Models screen once you're in.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }
}

struct ModelPickRow: View {
    let row: ModelRow
    let recommended: Bool
    let onPick: () -> Void

    var body: some View {
        Button(action: onPick) {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(row.shortTarget).font(.headline)
                        if recommended {
                            Text("recommended")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(.tint.opacity(0.18), in: Capsule())
                        }
                    }
                    if let drafter = row.shortDrafter {
                        HStack(spacing: 5) {
                            Image(systemName: "arrow.triangle.merge").imageScale(.small)
                            Text(drafter).font(.caption.monospaced())
                        }
                        .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if let speedup = row.speedup {
                    Text(speedup)
                        .font(.system(.callout, design: .rounded).monospacedDigit())
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.spark)
                }
                VStack(alignment: .trailing, spacing: 3) {
                    if let ram = row.ram {
                        Text(ram).font(.caption).foregroundStyle(.secondary)
                    }
                    Text(row.ready ? "ready" : "downloads on first use")
                        .font(.caption2)
                        .foregroundStyle(row.ready ? Theme.verified : .secondary)
                }
                Image(systemName: "chevron.right").foregroundStyle(.tertiary).imageScale(.small)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(recommended ? AnyShapeStyle(.tint.opacity(0.5))
                                              : AnyShapeStyle(.clear), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}
