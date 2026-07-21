import Foundation

/// A line of output from a child process.
public struct OutputLine: Sendable {
    public enum Stream: Sendable { case stdout, stderr }
    public let stream: Stream
    public let text: String
    public let date: Date

    public init(stream: Stream, text: String, date: Date = Date()) {
        self.stream = stream
        self.text = text
        self.date = date
    }
}

public enum ShellError: LocalizedError {
    case launchFailed(String, underlying: String)
    case exited(command: String, code: Int32, tail: String)

    public var errorDescription: String? {
        switch self {
        case .launchFailed(let cmd, let underlying):
            return "Could not run \(cmd): \(underlying)"
        case .exited(let cmd, let code, let tail):
            let detail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            return detail.isEmpty
                ? "\(cmd) exited with code \(code)."
                : "\(cmd) exited with code \(code):\n\(detail)"
        }
    }
}

/// Runs child processes with line-streamed output.
///
/// Everything the app does to set up its runtime is a subprocess (`uv`, then the engine's own
/// CLI), and every one of them needs its output surfaced live — onboarding shows progress, the
/// Logs screen shows the server. So streaming is the default here, not an extra.
public enum Shell {

    /// Run to completion, streaming each output line to `onLine`.
    /// - Returns: the exit code. Throws only if the process could not be launched.
    @discardableResult
    public static func run(
        _ executable: URL,
        _ arguments: [String],
        environment: [String: String]? = nil,
        currentDirectory: URL? = nil,
        onLine: (@Sendable (OutputLine) -> Void)? = nil
    ) async throws -> Int32 {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment ?? ProcessInfo.processInfo.environment
        if let currentDirectory { process.currentDirectoryURL = currentDirectory }

        let outPipe = Pipe(), errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        // Tail kept for the error message: a failure the user sees should quote what actually
        // went wrong, not just an exit code.
        let tail = LineTail(limit: 40)

        attach(outPipe, stream: .stdout, tail: tail, onLine: onLine)
        attach(errPipe, stream: .stderr, tail: tail, onLine: onLine)

        // The handler MUST be installed before run(). Foundation does not invoke a
        // terminationHandler assigned after the process has already exited, so setting it
        // afterwards deadlocks on any command fast enough to finish first — `uv --version`
        // reliably does. The gate resumes exactly once whichever order the two events land in.
        let gate = TerminationGate()
        process.terminationHandler = { _ in gate.signal() }

        do {
            try process.run()
        } catch {
            throw ShellError.launchFailed(executable.lastPathComponent,
                                          underlying: error.localizedDescription)
        }

        await gate.wait()
        // The readability handlers race the termination handler; give the pipes a moment to
        // drain so the last lines (usually the actual error) are not lost.
        outPipe.fileHandleForReading.readabilityHandler = nil
        errPipe.fileHandleForReading.readabilityHandler = nil
        for pipe in [outPipe, errPipe] {
            if let rest = try? pipe.fileHandleForReading.readToEnd(),
               let text = String(data: rest, encoding: .utf8), !text.isEmpty {
                for line in text.split(separator: "\n", omittingEmptySubsequences: false)
                where !line.isEmpty {
                    let l = OutputLine(stream: .stdout, text: String(line))
                    tail.append(l.text)
                    onLine?(l)
                }
            }
        }
        return process.terminationStatus
    }

    /// Run, and throw with the captured output if the exit code is non-zero.
    public static func check(
        _ executable: URL,
        _ arguments: [String],
        environment: [String: String]? = nil,
        onLine: (@Sendable (OutputLine) -> Void)? = nil
    ) async throws {
        let tail = LineTail(limit: 40)
        let code = try await run(executable, arguments, environment: environment) { line in
            tail.append(line.text)
            onLine?(line)
        }
        guard code == 0 else {
            throw ShellError.exited(command: executable.lastPathComponent,
                                    code: code, tail: tail.joined())
        }
    }

    /// Capture stdout of a short command (version probes and the like).
    public static func capture(
        _ executable: URL, _ arguments: [String],
        environment: [String: String]? = nil
    ) async -> (code: Int32, output: String) {
        let collector = LineTail(limit: 200)
        let code = (try? await run(executable, arguments, environment: environment) { line in
            collector.append(line.text)
        }) ?? -1
        return (code, collector.joined())
    }

    private static func attach(
        _ pipe: Pipe, stream: OutputLine.Stream,
        tail: LineTail, onLine: (@Sendable (OutputLine) -> Void)?
    ) {
        guard onLine != nil || true else { return }
        let buffer = LineBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            for line in buffer.consume(data) {
                tail.append(line)
                onLine?(OutputLine(stream: stream, text: line))
            }
        }
    }
}

/// One-shot rendezvous between a process's termination callback and an awaiting task.
///
/// Exists because the two can fire in either order: a short-lived child may exit before the
/// caller gets to `await`, and a long one exits well after. Resuming twice traps, never
/// resuming hangs — so both paths funnel through one lock-guarded state machine.
final class TerminationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var terminated = false
    private var continuation: CheckedContinuation<Void, Never>?

    func signal() {
        lock.lock()
        terminated = true
        let waiting = continuation
        continuation = nil
        lock.unlock()
        waiting?.resume()
    }

    func wait() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            lock.lock()
            if terminated {
                lock.unlock()
                c.resume()                 // already exited — resume immediately
            } else {
                continuation = c
                lock.unlock()
            }
        }
    }
}

/// Splits a byte stream into lines across chunk boundaries.
final class LineBuffer: @unchecked Sendable {
    private var pending = Data()
    private let lock = NSLock()

    func consume(_ data: Data) -> [String] {
        lock.lock(); defer { lock.unlock() }
        pending.append(data)
        var lines: [String] = []
        while let idx = pending.firstIndex(of: UInt8(ascii: "\n")) {
            let lineData = pending[pending.startIndex..<idx]
            pending.removeSubrange(pending.startIndex...idx)
            if let text = String(data: lineData, encoding: .utf8) {
                lines.append(text)
            }
        }
        return lines
    }
}

/// Bounded ring of the most recent lines.
final class LineTail: @unchecked Sendable {
    private var lines: [String] = []
    private let limit: Int
    private let lock = NSLock()

    init(limit: Int) { self.limit = limit }

    func append(_ line: String) {
        lock.lock(); defer { lock.unlock() }
        lines.append(line)
        if lines.count > limit { lines.removeFirst(lines.count - limit) }
    }

    func joined() -> String {
        lock.lock(); defer { lock.unlock() }
        return lines.joined(separator: "\n")
    }
}
