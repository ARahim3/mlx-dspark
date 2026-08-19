import Foundation

// MARK: - Setup checklist model
//
// The onboarding "Setting up the engine" step renders these rows. The bootstrapper publishes
// full snapshots rather than deltas so the view is a pure function of the last snapshot and
// tests can assert on one value.

public enum SetupStepID: String, CaseIterable, Sendable {
    case uv, python, engine, verify

    /// User-facing wording. Note what these deliberately do *not* say: "Python". A user
    /// installing a Mac app should never have to know the engine is a Python package.
    public var title: String {
        switch self {
        case .uv:     return "Preparing installer"
        case .python: return "Engine runtime"
        case .engine: return "Inference engine"
        case .verify: return "Finishing up"
        }
    }
}

public enum SetupState: Equatable, Sendable {
    case pending
    case running
    case done
    case failed(String)
}

public struct SetupStep: Identifiable, Equatable, Sendable {
    public let id: SetupStepID
    public var state: SetupState
    public var detail: String

    public init(id: SetupStepID, state: SetupState = .pending, detail: String = "") {
        self.id = id
        self.state = state
        self.detail = detail
    }

    public var title: String { id.title }
}

/// What produced the current runtime. Compared on every launch.
struct RuntimeFingerprint: Codable, Equatable {
    var engineVersion: String
    var pythonVersion: String
    /// Set when the engine was installed from a local source tree (development builds), so a
    /// dev runtime is never mistaken for a released one.
    var source: String
}

public enum BootstrapError: LocalizedError {
    case uvMissing
    case verifyFailed(String)

    public var errorDescription: String? {
        switch self {
        case .uvMissing:
            return "The bundled installer is missing from the app. Reinstall \(AppIdentity.displayName)."
        case .verifyFailed(let detail):
            return "The engine installed but did not start correctly: \(detail)"
        }
    }
}

// MARK: - RuntimeBootstrapper

/// Builds and maintains the app-owned Python runtime.
///
/// The whole point of this type is that the user never learns any of this happened. It vendors
/// `uv`, lets uv fetch its own CPython (so **Homebrew is never required** — the failure mode
/// that MTPLX ships as a visible error string), and installs the engine into a venv under
/// Application Support, outside the .app bundle.
public actor RuntimeBootstrapper {

    public typealias ProgressHandler = @Sendable ([SetupStep]) -> Void

    private var steps: [SetupStep] = SetupStepID.allCases.map { SetupStep(id: $0) }
    private let bundle: Bundle
    private let logStore: LogStore?

    public init(bundle: Bundle = .main, logStore: LogStore? = nil) {
        self.bundle = bundle
        self.logStore = logStore
    }

    /// Where the engine is installed from.
    ///
    /// A local source tree wins when present, which is what makes development bearable — a
    /// release build has no such override and always tracks the latest released engine.
    public enum EngineSource: Equatable, Sendable {
        /// The newest release on PyPI, resolved at launch.
        case pypi
        case localSourceTree(URL)

        var fingerprint: String {
            switch self {
            case .pypi: return "pypi"
            case .localSourceTree(let url): return "local:\(url.path)"
            }
        }

        /// `MLXDSPARK_ENGINE_SOURCE=/path/to/repo` points the app at a working tree.
        public static func fromEnvironment(
            _ env: [String: String] = ProcessInfo.processInfo.environment
        ) -> EngineSource {
            if let path = env["MLXDSPARK_ENGINE_SOURCE"], !path.isEmpty {
                return .localSourceTree(URL(fileURLWithPath: path))
            }
            return .pypi
        }
    }

    /// The newest released engine version, or nil when PyPI is unreachable (offline launch).
    static func latestReleasedVersion(timeout: TimeInterval = 5) async -> String? {
        struct Payload: Decodable {
            struct Info: Decodable { let version: String }
            let info: Info
        }
        guard let url = URL(string: "https://pypi.org/pypi/\(AppIdentity.enginePackage)/json")
        else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let payload = try? JSONDecoder().decode(Payload.self, from: data)
        else { return nil }
        return payload.info.version
    }

    /// Version probe with a sentinel prefix. `Shell.capture` merges stdout and stderr, so a
    /// bare `print(__version__)` picks up any import-time warning the venv emits
    /// (transformers, urllib3, …) — the polluted fingerprint then never equals PyPI's
    /// version string, which made the "Engine X is available" banner permanent and re-ran
    /// the in-place upgrade on every launch (community report, app-v0.6.1).
    static let versionProbeCode =
        "import mlx_dspark; print('MLXDSPARK_VERSION=' + mlx_dspark.__version__)"

    /// The version from a `versionProbeCode` run: the last sentinel-prefixed line, or nil.
    static func parseVersionProbe(_ output: String) -> String? {
        let prefix = "MLXDSPARK_VERSION="
        let version = output.split(whereSeparator: \.isNewline)
            .last { $0.trimmingCharacters(in: .whitespaces).hasPrefix(prefix) }
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .map { String($0.dropFirst(prefix.count)) }
        guard let version, !version.isEmpty else { return nil }
        return version
    }

    /// A newer release discovered by the background check, recorded to install on the next
    /// launch. Stored rather than acted on immediately: upgrading the venv while the running
    /// server lazily imports from it risks mixing module versions inside one process.
    public static var pendingEngineUpdate: String? {
        get { UserDefaults.standard.string(forKey: "pendingEngineUpdate") }
        set { UserDefaults.standard.set(newValue, forKey: "pendingEngineUpdate") }
    }

    /// Background update check — run after the app is up, never on the launch path. Records a
    /// newer PyPI release for the next launch's in-place upgrade and returns it for the UI.
    public func checkForEngineUpdate(source: EngineSource = .fromEnvironment()) async -> String? {
        guard source == .pypi,
              let latest = await Self.latestReleasedVersion(),
              let current = currentFingerprint(),
              latest != current.engineVersion
        else { return nil }
        Self.pendingEngineUpdate = latest
        return latest
    }

    /// Ensure a working runtime exists, reinstalling if the fingerprint doesn't match.
    ///
    /// For the PyPI source this is also the auto-update path: a newer release found by the
    /// *background* check (`checkForEngineUpdate`) is applied here as an in-place upgrade.
    /// Offline, an existing runtime is used as-is — a network check must never brick (or
    /// delay) a working install.
    /// - Returns: the engine executable to drive.
    @discardableResult
    public func ensureRuntime(
        source: EngineSource = .fromEnvironment(),
        onProgress: ProgressHandler? = nil
    ) async throws -> URL {
        try Paths.ensureDirectories()

        // Fast path FIRST and network-free: a fingerprint-matching runtime boots instantly.
        // (This used to sit behind a blocking PyPI query, so every launch spent up to 5 s
        // showing the install checklist over a fully-cached runtime.)
        if let current = currentFingerprint(),
           current.source == source.fingerprint,
           current.pythonVersion == AppIdentity.pythonVersion,
           FileManager.default.isExecutableFile(atPath: Paths.engineExecutable.path) {
            let pending = source == .pypi ? Self.pendingEngineUpdate : nil
            if let pending, pending != current.engineVersion {
                if let upgraded = await upgradeInPlace(to: pending, onProgress: onProgress) {
                    return upgraded
                }
                // Upgrade failed (offline, resolver churn): keep the working runtime and
                // retry on a later launch — an update must never brick a working install.
                logStore?.note("engine update to \(pending) didn't complete; "
                               + "keeping \(current.engineVersion)")
            } else if pending != nil {
                Self.pendingEngineUpdate = nil     // already on it (e.g. a manual rebuild)
            }
            steps = SetupStepID.allCases.map { SetupStep(id: $0, state: .done, detail: "Ready") }
            onProgress?(steps)
            return Paths.engineExecutable
        }

        // Full (re)install: first run, a python bump, a source switch, or a broken venv.
        // Only here — where a rebuild happens regardless — is a blocking version query
        // acceptable.
        var latest: String?
        if source == .pypi {
            latest = Self.pendingEngineUpdate
            if latest == nil { latest = await Self.latestReleasedVersion() }
        }

        // --- uv ------------------------------------------------------------------
        set(.uv, .running, "Locating installer")
        onProgress?(steps)
        guard let uv = Paths.uvExecutable(bundle: bundle) else {
            set(.uv, .failed("not found"), "")
            onProgress?(steps)
            throw BootstrapError.uvMissing
        }
        let uvVersion = await Shell.capture(uv, ["--version"]).output
            .trimmingCharacters(in: .whitespacesAndNewlines)
        set(.uv, .done, uvVersion)
        onProgress?(steps)

        let env = childEnvironment()

        // --- venv (uv fetches a managed CPython if the machine has none) ----------
        set(.python, .running, "Preparing Python \(AppIdentity.pythonVersion)")
        onProgress?(steps)
        do {
            // --clear: reaching here means the fingerprint did not match, so whatever is on
            // disk is stale (wrong engine version, or a dev runtime where a released one is
            // wanted). Replacing it is the point; without this uv refuses with "a virtual
            // environment already exists" and the app can never self-heal.
            try await Shell.check(uv, ["venv", Paths.venvDir.path,
                                       "--python", AppIdentity.pythonVersion, "--clear"],
                                  environment: env) { [logStore] line in
                logStore?.append(line)
            }
        } catch {
            set(.python, .failed(error.localizedDescription), "")
            onProgress?(steps)
            throw error
        }
        set(.python, .done, "Python \(AppIdentity.pythonVersion)")
        onProgress?(steps)

        // --- engine --------------------------------------------------------------
        set(.engine, .running, "Downloading (this can take a few minutes)")
        onProgress?(steps)
        var installArgs = ["pip", "install", "--python", venvPython().path]
        switch source {
        case .pypi:
            // The exact version when the check succeeded, else let uv resolve the newest —
            // either way the runtime lands on the latest release.
            installArgs.append(latest.map { "\(AppIdentity.enginePackage)==\($0)" }
                               ?? AppIdentity.enginePackage)
        case .localSourceTree(let url):
            installArgs.append(contentsOf: ["--editable", url.path])
        }
        do {
            try await Shell.check(uv, installArgs, environment: env) { [weak self, logStore] line in
                logStore?.append(line)
                // uv's progress lines are noisy; surface only the package being resolved so
                // the row reads like progress rather than a build log.
                if line.text.contains("Downloading") || line.text.contains("Installed") {
                    Task { await self?.updateDetail(.engine, line.text.trimmed()) }
                }
            }
        } catch {
            set(.engine, .failed(error.localizedDescription), "")
            onProgress?(steps)
            throw error
        }
        set(.engine, .done, "\(AppIdentity.enginePackage) \(latest ?? "latest")")
        onProgress?(steps)

        // --- verify --------------------------------------------------------------
        set(.verify, .running, "Checking the engine")
        onProgress?(steps)
        let probe = await Shell.capture(
            venvPython(), ["-c", Self.versionProbeCode], environment: env)
        guard probe.code == 0, let installed = Self.parseVersionProbe(probe.output) else {
            let detail = probe.output.trimmingCharacters(in: .whitespacesAndNewlines)
            set(.verify, .failed(detail), "")
            onProgress?(steps)
            throw BootstrapError.verifyFailed(detail.isEmpty ? "no version reported" : detail)
        }
        // Fingerprint the version that actually installed (the probe's answer, not the
        // query's) — that is what the next launch's latest-check compares against.
        try writeFingerprint(RuntimeFingerprint(engineVersion: installed,
                                                pythonVersion: AppIdentity.pythonVersion,
                                                source: source.fingerprint))
        if source == .pypi { Self.pendingEngineUpdate = nil }
        set(.verify, .done, "Engine \(installed)")
        onProgress?(steps)

        return Paths.engineExecutable
    }

    /// Apply a pending engine update NOW — the UI-triggered path ("Update now" in Settings).
    /// The same in-place upgrade the next launch would run; stop the server first (upgrading
    /// the venv under a running engine risks mixing module versions in one process).
    /// Returns the engine executable on success, nil on failure (the working install is kept).
    public func applyPendingUpdate(onProgress: ProgressHandler? = nil) async -> URL? {
        guard let version = Self.pendingEngineUpdate else { return nil }
        return await upgradeInPlace(to: version, onProgress: onProgress)
    }

    /// Upgrade the engine inside the existing venv (`uv pip install pkg==version`) — no venv
    /// rebuild, nothing deleted, so any failure leaves the working install untouched.
    private func upgradeInPlace(to version: String, onProgress: ProgressHandler?) async -> URL? {
        guard let uv = Paths.uvExecutable(bundle: bundle) else { return nil }
        let requirement = "\(AppIdentity.enginePackage)==\(version)"
        let env = childEnvironment()
        set(.uv, .done, "Ready")
        set(.python, .done, "Python \(AppIdentity.pythonVersion)")
        set(.engine, .running, "Updating to \(requirement)")
        onProgress?(steps)
        do {
            // --refresh-package: force uv to revalidate this package's index metadata. A
            // cached index that predates the release makes `pkg==new` unsatisfiable, and the
            // launch-path retry then never converges (issue #16 item 3 — an update stuck
            // pending across relaunches until a manual venv install).
            try await Shell.check(uv, ["pip", "install", "--python", venvPython().path,
                                       "--refresh-package", AppIdentity.enginePackage,
                                       requirement],
                                  environment: env) { [logStore] line in
                logStore?.append(line)
            }
        } catch {
            // The caller keeps the working runtime; make the WHY visible in the log instead
            // of a bare "didn't complete".
            logStore?.note("engine update to \(version) failed: \(error.localizedDescription)")
            return nil
        }
        set(.engine, .done, requirement)
        set(.verify, .running, "Checking the engine")
        onProgress?(steps)
        let probe = await Shell.capture(
            venvPython(), ["-c", Self.versionProbeCode], environment: env)
        guard probe.code == 0, let installed = Self.parseVersionProbe(probe.output) else { return nil }
        try? writeFingerprint(RuntimeFingerprint(engineVersion: installed,
                                                 pythonVersion: AppIdentity.pythonVersion,
                                                 source: EngineSource.pypi.fingerprint))
        Self.pendingEngineUpdate = nil
        set(.verify, .done, "Engine \(installed)")
        onProgress?(steps)
        return Paths.engineExecutable
    }

    // MARK: - internals

    private func venvPython() -> URL {
        Paths.venvDir.appendingPathComponent("bin/python")
    }

    /// A deliberately minimal environment.
    ///
    /// An app launched from Finder inherits almost nothing, and that is the environment we want
    /// to test against — inheriting the developer's shell would hide breakage that every real
    /// user hits. `VIRTUAL_ENV` is cleared so uv never resolves against a venv the user happens
    /// to have active.
    private func childEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
        env["HOME"] = NSHomeDirectory()
        env.removeValue(forKey: "VIRTUAL_ENV")
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "PYTHONPATH")
        return env
    }

    private func set(_ id: SetupStepID, _ state: SetupState, _ detail: String) {
        guard let idx = steps.firstIndex(where: { $0.id == id }) else { return }
        steps[idx].state = state
        if !detail.isEmpty { steps[idx].detail = detail }
    }

    private func updateDetail(_ id: SetupStepID, _ detail: String) {
        guard let idx = steps.firstIndex(where: { $0.id == id }) else { return }
        steps[idx].detail = detail
    }

    private func currentFingerprint() -> RuntimeFingerprint? {
        guard let data = try? Data(contentsOf: Paths.runtimeFingerprint) else { return nil }
        return try? JSONDecoder().decode(RuntimeFingerprint.self, from: data)
    }

    private func writeFingerprint(_ fp: RuntimeFingerprint) throws {
        try JSONEncoder().encode(fp).write(to: Paths.runtimeFingerprint)
    }
}

extension String {
    func trimmed() -> String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
