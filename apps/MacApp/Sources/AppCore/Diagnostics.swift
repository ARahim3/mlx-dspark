import Foundation

/// What this Mac can do — the payload behind onboarding's "Checking your Mac" step.
public struct DoctorReport: Decodable, Sendable {
    public let ok: Bool
    /// Human-readable blockers, already phrased for display.
    public let problems: [String]
    public let environment: Environment
    public let models: [ModelRow]

    public struct Environment: Decodable, Sendable {
        public let version: String
        public let platform: String
        public let machine: String
        public let osVersion: String?
        public let appleSilicon: Bool
        public let metalOK: Bool
        public let device: String?
        public let metalError: String?
        public let ramGB: Double?
        public let iogpuWiredLimitMB: Int?
        /// A ready-to-run `sudo sysctl …` command, present only when raising the limit would
        /// actually help. Letting macOS page weights out mid-generation is the classic silent
        /// slowdown, so this is worth surfacing rather than burying.
        public let wiredLimitHint: String?
        public let packages: [String: String?]
        /// Chip family + spec-sheet bandwidth (+ the measured figure once a model has ever
        /// loaded here). Optional: older engines don't report it.
        public let chip: Chip?
        /// What macOS sees right now (pressure, swap). Optional, same reason.
        public let memory: MachineReport.Memory?

        public struct Chip: Decodable, Sendable {
            public let name: String?
            public let family: String?
            public let gpuCores: Int?
            public let bandwidthGBs: Double?
            public let bandwidthMeasuredGBs: Double?

            enum CodingKeys: String, CodingKey {
                case name, family
                case gpuCores = "gpu_cores"
                case bandwidthGBs = "bandwidth_gb_s"
                case bandwidthMeasuredGBs = "bandwidth_measured_gb_s"
            }
        }

        enum CodingKeys: String, CodingKey {
            case version, platform, machine, device, packages, chip, memory
            case osVersion = "os_version"
            case appleSilicon = "apple_silicon"
            case metalOK = "metal_ok"
            case metalError = "metal_error"
            case ramGB = "ram_gb"
            case iogpuWiredLimitMB = "iogpu_wired_limit_mb"
            case wiredLimitHint = "wired_limit_hint"
        }
    }
}

/// One registry model, annotated for *this* machine.
public struct ModelRow: Decodable, Sendable, Identifiable {
    public let id: String
    public let target: String
    /// The matched drafter. Naming the pair is this app's domain language — every other
    /// local-LLM app has only one model to show.
    public let dsparkDrafter: String?
    public let dflashDrafter: String?
    public let ram: String?
    public let ramGB: Double?
    /// Measured headline speedup for this pair (M4 Pro reference numbers) — the "why this
    /// one" a picker should answer.
    public let speedup: String?
    /// nil when RAM couldn't be determined — render as "unknown", never as "no".
    public let fits: Bool?
    public let targetInstalled: Bool
    public let drafterInstalled: Bool
    /// Runnable right now, with nothing left to download.
    public let ready: Bool
    /// Safetensors bytes of an already-downloaded target, and the plain-decode ceiling they
    /// imply at this Mac's bandwidth — the physics a picker can quote before loading. Nil
    /// until the weights are local (and on older engines).
    public let weightBytes: Int?
    public let ceilingTps: Double?

    enum CodingKeys: String, CodingKey {
        case id, target, ram, fits, ready, speedup
        case weightBytes = "weight_bytes"
        case ceilingTps = "ceiling_tps"
        case dsparkDrafter = "dspark_drafter"
        case dflashDrafter = "dflash_drafter"
        case ramGB = "ram_gb"
        case targetInstalled = "target_installed"
        case drafterInstalled = "drafter_installed"
    }

    public var shortTarget: String { target.components(separatedBy: "/").last ?? target }

    public var drafter: String? { dsparkDrafter ?? dflashDrafter }

    public var shortDrafter: String? {
        drafter.map { $0.components(separatedBy: "/").last ?? $0 }
    }
}

/// A model that is actually on this disk — HF hub cache or the plain-dir cache — regardless
/// of whether the registry knows it. The other half of the Models screen: the registry says
/// what we vouch for, this says what the user already has.
public struct InstalledModel: Decodable, Sendable, Identifiable {
    public let repo: String
    public let path: String
    public let sizeBytes: Int
    public let size: String
    /// "model" (loadable as a target) or "drafter" (one half of a pair; not a chat model).
    public let kind: String
    /// Registry pairing this repo resolves into (quant-agnostic) — nil means `--mode auto`
    /// will fall back to drafter-free lookup speculation.
    public let registryID: String?
    /// "cache" (our download), "lmstudio" (LM Studio's) or "model_dirs" (the user's own
    /// `MLX_DSPARK_MODEL_DIRS` folder, engine ≥ 0.17.1) — the latter two are loadable but not
    /// ours to delete. Optional: older engines don't report it.
    public let source: String?

    public var id: String { repo }

    enum CodingKeys: String, CodingKey {
        case repo, path, size, kind, source
        case sizeBytes = "size_bytes"
        case registryID = "registry_id"
    }

    public var shortRepo: String { repo.components(separatedBy: "/").last ?? repo }
    public var isDrafter: Bool { kind == "drafter" }
    public var isLMStudio: Bool { source == "lmstudio" }
    /// Someone else's files (LM Studio's, or a folder the user pointed the engine at): we
    /// read them, we never offer to delete them. Anything that isn't our own cache counts —
    /// a source this app doesn't know yet must default to "not ours", never to "deletable".
    public var isExternal: Bool { source != nil && source != "cache" }
    /// Short provenance caption for an external row.
    public var sourceLabel: String? {
        switch source {
        case "lmstudio": return "from LM Studio"
        case "model_dirs": return "from your model folder"
        case nil, "cache": return nil
        default: return "external"
        }
    }
}

public struct DiskUsage: Decodable, Sendable {
    public let totalBytes: Int
    public let total: String

    enum CodingKeys: String, CodingKey {
        case total
        case totalBytes = "total_bytes"
    }
}

public struct ModelInventory: Decodable, Sendable {
    public let models: [ModelRow]
    public let installed: [InstalledModel]?
    public let disk: DiskUsage?
    public let loaded: String?
    /// This Mac vs the reference M4 Pro (optional: older engines don't report it).
    public let bandwidth: BandwidthInfo?
}
