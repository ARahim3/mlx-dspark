import Foundation

/// Runs `mlx-dspark doctor --json` to learn what this machine can do — **before** any model is
/// loaded.
///
/// Onboarding has a chicken-and-egg problem: the server's `/doctor` and `/admin/models` need a
/// running engine, but the engine needs a model, and choosing the model is the whole point of
/// onboarding. `mlx-dspark doctor --json` sidesteps it — the diagnostics are model-free Python,
/// so this reports hardware and the registry-vs-installed inventory with nothing loaded.
public enum DoctorProbe {
    public static func run(engine: URL) async throws -> DoctorReport {
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
        env.removeValue(forKey: "VIRTUAL_ENV")

        let collector = LineTail(limit: 400)
        let code = try await Shell.run(engine, ["doctor", "--json"], environment: env) { line in
            // doctor --json prints the report to stdout; anything on stderr (a transformers
            // import warning, say) would corrupt a naive full-buffer decode, so keep stdout only.
            if line.stream == .stdout { collector.append(line.text) }
        }
        let output = collector.joined()
        // doctor exits non-zero when the machine fails a hard check, but it STILL prints the
        // report — that report is exactly what onboarding needs to explain the failure. So
        // decode first and only treat a missing/garbled report as an error.
        guard let data = Self.extractJSON(output),
              let report = try? JSONDecoder().decode(DoctorReport.self, from: data) else {
            throw DoctorProbeError.unreadable(code: code, output: output)
        }
        return report
    }

    /// Pull the JSON object out of the captured output, tolerating a stray leading line.
    private static func extractJSON(_ output: String) -> Data? {
        guard let start = output.firstIndex(of: "{"),
              let end = output.lastIndex(of: "}") else { return nil }
        return String(output[start...end]).data(using: .utf8)
    }
}

public enum DoctorProbeError: LocalizedError {
    case unreadable(code: Int32, output: String)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let code, let output):
            let tail = output.suffix(200).trimmingCharacters(in: .whitespacesAndNewlines)
            return "Could not read the machine check (exit \(code)). \(tail)"
        }
    }
}
