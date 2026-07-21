import Darwin
import Foundation

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

    public init(model: String? = nil, mode: String = "auto", maxDraft: String? = "auto",
                host: String = "127.0.0.1", apiKey: String? = nil) {
        self.model = model
        self.mode = mode
        self.maxDraft = maxDraft
        self.host = host
        self.apiKey = apiKey
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

        let chosenPort = try Self.freePort()
        port = chosenPort

        var args = ["serve", "--host", config.host, "--port", String(chosenPort),
                    "--mode", config.mode]
        if let model = config.model { args.append(contentsOf: ["--model", model]) }
        if let maxDraft = config.maxDraft { args.append(contentsOf: ["--max-draft", maxDraft]) }
        if let key = config.apiKey, !key.isEmpty { args.append(contentsOf: ["--api-key", key]) }

        transition(.starting(detail: "Launching engine"))

        let proc = Process()
        proc.executableURL = engine
        proc.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"        // otherwise loading progress arrives only at exit
        env.removeValue(forKey: "VIRTUAL_ENV")
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

        let client = APIClient(baseURL: URL(string: "http://\(config.host):\(chosenPort)")!,
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

    /// Ask the kernel for an unused port.
    ///
    /// Binding a fixed 8080 is how every one of these apps first breaks — it collides with
    /// whatever else the user runs. There is an unavoidable race between closing this socket
    /// and the child binding it; the caller retries.
    public static func freePort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ShellError.launchFailed("socket", underlying: "unavailable") }
        defer { close(fd) }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0                                  // kernel picks
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
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
