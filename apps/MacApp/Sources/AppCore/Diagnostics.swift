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

        enum CodingKeys: String, CodingKey {
            case version, platform, machine, device, packages
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

    enum CodingKeys: String, CodingKey {
        case id, target, ram, fits, ready, speedup
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
    /// "cache" (our download) or "lmstudio" (LM Studio's — loadable, but not ours to
    /// delete). Optional: older engines don't report it.
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
}
