import Foundation

// MARK: - /machine — the roofline view of this Mac
//
// Every "GPU %" gauge on a Mac lies about decode (a bandwidth-starved GPU reads busy). The
// engine instead compares what it measures against physics: a single-stream decode can never
// beat `bandwidth ÷ bytes-per-token`, and speculative decoding is the thing that legitimately
// does. `/machine` carries both halves — the chip's measured bandwidth and the loaded model's
// exact byte footprint — plus what macOS itself says about memory pressure.

/// A banner-worthy condition the engine reports on `/health.warnings` — live memory pressure,
/// or a load-time note (the context-window RAM estimate) that used to reach only its stderr.
public struct EngineWarning: Decodable, Sendable, Equatable, Identifiable {
    public let code: String
    /// `attention` or `problem`.
    public let level: String
    public let message: String
    public let action: String?

    public var id: String { code + message }

    public init(code: String, level: String, message: String, action: String?) {
        self.code = code
        self.level = level
        self.message = message
        self.action = action
    }
}

public struct MachineReport: Decodable, Sendable {
    public struct Chip: Decodable, Sendable {
        public let name: String?
        public let generation: String?
        public let family: String?
        public let gpuCores: Int?
        /// Spec-sheet unified-memory bandwidth (nil for a chip the table doesn't know).
        public let bandwidthGBs: Double?
        public let bandwidthSource: String?

        enum CodingKeys: String, CodingKey {
            case name, generation, family
            case gpuCores = "gpu_cores"
            case bandwidthGBs = "bandwidth_gb_s"
            case bandwidthSource = "bandwidth_source"
        }
    }

    public struct Bandwidth: Decodable, Sendable {
        /// The honest local number: a one-time 512 MB matvec microbench (`measured`), or the
        /// chip table when the microbench couldn't run (`theoretical`).
        public let gbs: Double?
        public let source: String?
        /// The M4 Pro every registry speedup badge was measured on.
        public let referenceGBs: Double?

        enum CodingKeys: String, CodingKey {
            case gbs = "gb_s"
            case source
            case referenceGBs = "reference_gb_s"
        }
    }

    public struct Memory: Decodable, Sendable {
        public let totalBytes: Int?
        /// `normal` · `warn` · `critical` · `unknown` — macOS's own verdict, the one Activity
        /// Monitor colours its gauge by. `warn`/`critical` is the usual cause of "mysteriously
        /// half speed".
        public let pressure: String?
        public let freePercent: Int?
        public let swapUsedBytes: Int?
        public let swapTotalBytes: Int?
        public let wiredLimitMB: Int?
        /// MLX allocator — what the loaded model itself holds resident.
        public let allocator: EngineMemory?

        enum CodingKeys: String, CodingKey {
            case pressure, allocator
            case totalBytes = "total_bytes"
            case freePercent = "free_percent"
            case swapUsedBytes = "swap_used_bytes"
            case swapTotalBytes = "swap_total_bytes"
            case wiredLimitMB = "wired_limit_mb"
        }

        public var isUnderPressure: Bool { pressure == "warn" || pressure == "critical" }
    }

    public struct Weights: Decodable, Sendable {
        public let totalBytes: Int
        /// Bytes one decode step actually reads: routed MoE experts count `top_k / n` of
        /// their size, the embedding gather is excluded — the roofline denominator.
        public let activeBytes: Int
        public let isMoe: Bool
        public let nExperts: Int?
        public let expertsPerTok: Int?
        public let activeIsEstimate: Bool

        enum CodingKeys: String, CodingKey {
            case totalBytes = "total_bytes"
            case activeBytes = "active_bytes"
            case isMoe = "is_moe"
            case nExperts = "n_experts"
            case expertsPerTok = "experts_per_tok"
            case activeIsEstimate = "active_is_estimate"
        }
    }

    public struct Model: Decodable, Sendable {
        public let target: String?
        public let drafter: String?
        public let mode: String?
        public let targetWeights: Weights
        public let drafterWeights: Weights?
        public let kvBytesPerToken: Int?
        public let contextWindow: Int?

        enum CodingKeys: String, CodingKey {
            case target, drafter, mode
            case targetWeights = "target_weights"
            case drafterWeights = "drafter_weights"
            case kvBytesPerToken = "kv_bytes_per_token"
            case contextWindow = "context_window"
        }
    }

    /// The ceiling at one context depth. The KV cache joins the per-token byte bill as context
    /// grows, so the ceiling falls with depth — the physics behind long-context slowdown.
    public struct Point: Decodable, Sendable {
        public let context: Int
        public let bytesPerToken: Int
        public let ceilingTps: Double?

        enum CodingKeys: String, CodingKey {
            case context
            case bytesPerToken = "bytes_per_token"
            case ceilingTps = "ceiling_tps"
        }
    }

    public struct Roofline: Decodable, Sendable {
        public let atZero: Point?
        public let atLastRequest: Point?
        public let atContextWindow: Point?

        enum CodingKeys: String, CodingKey {
            case atZero = "at_zero"
            case atLastRequest = "at_last_request"
            case atContextWindow = "at_context_window"
        }
    }

    /// Machine health from the calibration's measured plain step — no extra measurement.
    /// `mbu` ≥ 0.75 = the machine is saturated (software is done); below 0.5 = structural.
    public struct Baseline: Decodable, Sendable {
        public let stepMs: Double
        public let achievedGBs: Double
        public let mbu: Double?

        enum CodingKeys: String, CodingKey {
            case mbu
            case stepMs = "step_ms"
            case achievedGBs = "achieved_gb_s"
        }
    }

    public struct Verdict: Decodable, Sendable, Equatable {
        /// `info` · `healthy` · `ok` · `attention` · `problem`.
        public let level: String
        public let headline: String
        public let findings: [String]
        public let levers: [String]
    }

    /// The memory-pressure guard: when macOS reports pressure the engine frees its prefix-
    /// cache snapshots and returns MLX's retained buffers, at a round boundary, instead of
    /// letting the model swap. `lastShed` is why a turn re-prefilled.
    public struct Guard: Decodable, Sendable {
        public let enabled: Bool
        public let level: String?
        public let pending: String?
        public let sheds: Int?
        public let lastShed: Shed?

        public struct Shed: Decodable, Sendable, Equatable {
            public let level: String
            /// Unix seconds.
            public let at: Double
            public let freedBytes: Int
            public let action: String?

            enum CodingKeys: String, CodingKey {
                case level, at, action
                case freedBytes = "freed_bytes"
            }
        }

        enum CodingKeys: String, CodingKey {
            case enabled, level, pending, sheds
            case lastShed = "last_shed"
        }
    }

    public let chip: Chip
    public let bandwidth: Bandwidth
    public let memory: Memory
    public let model: Model?
    public let roofline: Roofline?
    public let baseline: Baseline?
    public let verdict: Verdict?
    /// Optional: engines without the guard don't report it. (`guard` is a Swift keyword.)
    public let memoryGuard: Guard?

    enum CodingKeys: String, CodingKey {
        case chip, bandwidth, memory, model, roofline, baseline, verdict
        case memoryGuard = "guard"
    }
}

/// `/admin/models.bandwidth` — this Mac relative to the reference M4 Pro, so the stamped
/// speedup badges can be read honestly on other chips.
public struct BandwidthInfo: Decodable, Sendable, Equatable {
    public let measuredGBs: Double?
    public let theoreticalGBs: Double?
    public let gbs: Double?
    public let source: String?
    public let referenceGBs: Double?
    /// 1.0 on an M4 Pro, ~2.2 on a top-binned M5 Max. Nil for an unknown chip.
    public let scale: Double?

    enum CodingKeys: String, CodingKey {
        case source, scale
        case measuredGBs = "measured_gb_s"
        case theoreticalGBs = "theoretical_gb_s"
        case gbs = "gb_s"
        case referenceGBs = "reference_gb_s"
    }
}

extension APIClient {
    /// Chip, measured bandwidth, what macOS sees, the loaded model's footprint and its
    /// roofline. Answers model-less too (chip/bandwidth/memory only).
    public func machine() async throws -> MachineReport {
        let (data, response) = try await session.data(for: request("machine"))
        try Self.check(response, data)
        return try JSONDecoder().decode(MachineReport.self, from: data)
    }
}

/// Byte formatting shared by the machine views.
public enum ByteFormat {
    public static func gb(_ bytes: Int?, digits: Int = 1) -> String? {
        guard let bytes else { return nil }
        return String(format: "%.\(digits)f GB", Double(bytes) / 1_073_741_824)
    }

    public static func kb(_ bytes: Int?) -> String? {
        guard let bytes else { return nil }
        return String(format: "%.0f KB", Double(bytes) / 1024)
    }
}
