import Darwin
import Foundation
import Security

public enum ServerState: Equatable, Sendable {
    case idle
    /// Loading weights. On a first run this is minutes, not seconds — the UI must say so.
    case starting(detail: String)
    case ready(port: Int, health: HealthInfo)
    case failed(String)
    case stopped
}

public struct ServerConfig: Equatable, Sendable {
    public var model: String?
    public var mode: String
    public var maxDraft: String?
    public var host: String
    public var apiKey: String?
    /// Fixed engine port (issue #16) so external OpenAI/Anthropic clients keep a stable
    /// base URL across launches. 0 = automatic (kernel-assigned, the old behavior).
    public var port: Int
    /// Whether `port` was chosen by the user (Settings) rather than being the app default.
    /// An explicit port that is taken is a hard error naming the fix; the DEFAULT fixed
    /// port falls back to automatic instead — a default must never block launch.
    public var portIsExplicit: Bool
    /// Extra folders the engine searches for MLX checkpoints before downloading anything
    /// (Settings → Model folders). Passed as the engine's `MLX_DSPARK_MODEL_DIRS` (engine ≥
    /// 0.17.1; older engines ignore the variable) — a GUI app is the only way most users can
    /// set an environment variable for the engine at all.
    public var modelDirs: [String]

    public init(model: String? = nil, mode: String = "auto", maxDraft: String? = "auto",
                host: String = "127.0.0.1", apiKey: String? = nil, port: Int = 0,
                portIsExplicit: Bool = true, modelDirs: [String] = []) {
        self.model = model
        self.mode = mode
        self.maxDraft = maxDraft
        self.host = host
        self.apiKey = apiKey
        self.port = port
        self.portIsExplicit = portIsExplicit
        self.modelDirs = modelDirs
    }

    /// Serve on every interface (`0.0.0.0`) or loopback only.
    public var servesLAN: Bool { host == "0.0.0.0" || host == "::" }

    /// Where the APP talks to its own engine. A wildcard is a listener address, never a client
    /// destination — the app's control traffic stays on loopback even while other devices
    /// reach the same port over the LAN.
    public var clientHost: String { servesLAN ? "127.0.0.1" : host }

    /// The env var name the engine reads (`load.MODEL_DIRS_ENV`).
    public static let modelDirsEnvironmentKey = "MLX_DSPARK_MODEL_DIRS"

    /// `modelDirs` as the engine expects it: `:`-joined, blanks dropped, `~` expanded, order
    /// kept, duplicates removed. Nil when there is nothing to pass (the variable is then left
    /// alone, so a user who exported it in their shell keeps it).
    public var modelDirsEnvironmentValue: String? {
        var seen: [String] = []
        for raw in modelDirs {
            let path = (raw.trimmingCharacters(in: .whitespacesAndNewlines) as NSString)
                .expandingTildeInPath
            if !path.isEmpty, !seen.contains(path) { seen.append(path) }
        }
        return seen.isEmpty ? nil : seen.joined(separator: ":")
    }
}

/// Owns the `mlx-dspark serve` child process.
///
/// The app never links the engine — it drives the same CLI a terminal user would, over HTTP.
/// That keeps every feature reachable from both, and means a bug reproduces without the app.
public actor ServerSupervisor {

    public private(set) var state: ServerState = .idle
    public private(set) var port: Int = 0

    private var process: Process?
    private let logStore: LogStore
    private let engine: URL
    private var onStateChange: (@Sendable (ServerState) -> Void)?

    public init(engine: URL, logStore: LogStore) {
        self.engine = engine
        self.logStore = logStore
    }

    public func observeState(_ handler: @escaping @Sendable (ServerState) -> Void) {
        onStateChange = handler
        handler(state)
    }

    /// Start the server and wait until `/health` answers.
    public func start(config: ServerConfig, readyTimeout: TimeInterval = 900) async throws -> Int {
        if case .ready = state { return port }
        await stop()

        let chosenPort: Int
        if config.port > 0 {
            // Fixed port (the app default, or Settings → Local server). A quit-and-relaunch
            // can race the old process still letting go of it, so wait briefly before
            // declaring it taken — then a USER-chosen port fails with a message that names
            // the fix, while the app-DEFAULT port falls back to automatic (a default must
            // never keep the app from launching because some other tool holds it).
            let deadline = Date().addingTimeInterval(5)
            while !Self.portIsFree(config.port, host: config.host), Date() < deadline {
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            if Self.portIsFree(config.port, host: config.host) {
                chosenPort = config.port
            } else if config.portIsExplicit {
                let msg = "Port \(config.port) is already in use — quit whatever holds it, "
                        + "or set the engine port back to automatic in Settings."
                transition(.failed(msg))
                throw ShellError.launchFailed("mlx-dspark serve", underlying: msg)
            } else {
                logStore.note("default port \(config.port) is in use — "
                              + "starting on an automatic port instead")
                chosenPort = try Self.freePort(host: config.host)
            }
        } else {
            chosenPort = try Self.freePort(host: config.host)
        }
        port = chosenPort

        var args = ["serve", "--host", config.host, "--port", String(chosenPort),
                    "--mode", config.mode]
        // No model = a deliberate fast start (`--no-model`): the server is up in seconds and
        // a model loads later via /admin/load. Omitting the flag entirely would silently load
        // the engine's default target instead.
        if let model = config.model { args.append(contentsOf: ["--model", model]) }
        else { args.append("--no-model") }
        if let maxDraft = config.maxDraft { args.append(contentsOf: ["--max-draft", maxDraft]) }
        if let key = config.apiKey, !key.isEmpty { args.append(contentsOf: ["--api-key", key]) }

        transition(.starting(detail: "Launching engine"))

        let proc = Process()
        proc.executableURL = engine
        proc.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"        // otherwise loading progress arrives only at exit
        env.removeValue(forKey: "VIRTUAL_ENV")
        if let dirs = config.modelDirsEnvironmentValue {
            env[ServerConfig.modelDirsEnvironmentKey] = dirs
        }
        proc.environment = env

        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        attach(outPipe, stream: .stdout)
        attach(errPipe, stream: .stderr)

        do {
            try proc.run()
        } catch {
            transition(.failed(error.localizedDescription))
            throw ShellError.launchFailed("mlx-dspark serve", underlying: error.localizedDescription)
        }
        process = proc

        // A crash before /health answers must surface immediately, not after the timeout.
        proc.terminationHandler = { [weak self] p in
            Task { await self?.childExited(code: p.terminationStatus) }
        }

        let client = APIClient(baseURL: URL(string: "http://\(config.clientHost):\(chosenPort)")!,
                               apiKey: config.apiKey)
        let deadline = Date().addingTimeInterval(readyTimeout)
        while Date() < deadline {
            if case .failed = state { throw ShellError.exited(
                command: "mlx-dspark serve", code: proc.terminationStatus,
                tail: "server exited during startup") }
            if let health = try? await client.health() {
                transition(.ready(port: chosenPort, health: health))
                return chosenPort
            }
            try? await Task.sleep(nanoseconds: 400_000_000)
        }
        await stop()
        transition(.failed("The engine did not become ready in time."))
        throw ShellError.exited(command: "mlx-dspark serve", code: -1,
                                tail: "timed out waiting for /health")
    }

    public func stop() async {
        guard let proc = process, proc.isRunning else {
            process = nil
            return
        }
        proc.terminationHandler = nil
        proc.terminate()                                  // SIGTERM
        // Give it a moment to unwind cleanly; the engine holds GPU buffers and a scheduler
        // thread, so an immediate SIGKILL can leave the wired-memory limit raised.
        for _ in 0..<20 {
            if !proc.isRunning { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        process = nil
        transition(.stopped)
    }

    private func childExited(code: Int32) {
        process = nil
        if case .ready = state {
            transition(.failed("The engine stopped unexpectedly (code \(code))."))
        } else if case .starting = state {
            transition(.failed("The engine failed to start (code \(code)). See the log."))
        }
    }

    private func transition(_ new: ServerState) {
        state = new
        onStateChange?(new)
    }

    private nonisolated func attach(_ pipe: Pipe, stream: OutputLine.Stream) {
        let buffer = LineBuffer()
        let store = logStore
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            for line in buffer.consume(data) {
                store.append(OutputLine(stream: stream, text: line))
            }
        }
    }

    /// Can a listener bind this port right now? Advisory (the engine does the real bind
    /// moments later) — its job is a clean, actionable error for a fixed port that is
    /// genuinely taken. Uses SO_REUSEADDR like the engine's own HTTP server, so lingering
    /// TIME_WAIT connections from a previous run don't read as "busy".
    public static func portIsFree(_ portNumber: Int, host: String) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return true }        // can't check -> let the engine's bind decide
        defer { close(fd) }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(portNumber).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bound == 0
    }

    /// Ask the kernel for an unused port.
    ///
    /// Binding a fixed 8080 is how every one of these apps first breaks — it collides with
    /// whatever else the user runs. There is an unavoidable race between closing this socket
    /// and the child binding it; the caller retries. (A *user-chosen* fixed port is different:
    /// it's opt-in, checked with ``portIsFree(_:host:)``, and fails with a named fix.)
    public static func freePort(host: String = "127.0.0.1") throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ShellError.launchFailed("socket", underlying: "unavailable") }
        defer { close(fd) }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0                                  // kernel picks
        // A LAN listener must find a port free on EVERY interface, not just loopback.
        addr.sin_addr.s_addr = (host == "0.0.0.0" || host == "::") ? INADDR_ANY : inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            throw ShellError.launchFailed("bind", underlying: String(cString: strerror(errno)))
        }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        guard named == 0 else {
            throw ShellError.launchFailed("getsockname", underlying: "failed")
        }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}


/// The Mac's own IPv4 addresses, for showing a LAN user what base URL to type on another
/// device. Best-effort and point-in-time (Wi-Fi joins/leaves change it); loopback and
/// link-local excluded; `en*` (Wi-Fi/Ethernet) first, then anything else (Thunderbolt bridge,
/// VPN tunnels — those are still reachable addresses, just less likely to be the one wanted).
public enum LocalNetwork {
    public struct Address: Equatable, Sendable {
        public let interface: String
        public let ip: String
    }

    public static func ipv4Addresses() -> [Address] {
        var out: [Address] = []
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return out }
        defer { freeifaddrs(list) }
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let ifa = cursor {
            defer { cursor = ifa.pointee.ifa_next }
            guard let sa = ifa.pointee.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET),
                  (Int32(ifa.pointee.ifa_flags) & IFF_UP) != 0,
                  (Int32(ifa.pointee.ifa_flags) & IFF_LOOPBACK) == 0 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count),
                              nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let ip = String(cString: host)
            if ip.hasPrefix("169.254.") { continue }
            let name = String(cString: ifa.pointee.ifa_name)
            out.append(Address(interface: name, ip: ip))
        }
        return out.sorted { a, b in
            let ea = a.interface.hasPrefix("en"), eb = b.interface.hasPrefix("en")
            return ea != eb ? ea : a.interface < b.interface
        }
    }

    /// A fresh API key: 32 random bytes, URL-safe base64, no padding — pasteable anywhere.
    public static func generateAPIKey() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return "mdk_" + Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
