import Foundation

/// Serialized, coalescing runner for an idempotent async operation — the "instant apply"
/// primitive. Independent per-change `Task`s can complete out of order across network
/// awaits, leaving a server on a stale value; `Coalescer` guarantees latest-wins instead:
/// one run at a time, and a `schedule()` arriving mid-run queues exactly one re-run, so the
/// final run always observes the newest state (the operation reads its inputs at send time).
///
/// The operation is FIXED at init and the trigger takes no argument — deliberately. A
/// schedule-accepts-a-closure API would re-run an *older* closure after coalescing, which is
/// precisely the staleness this type exists to prevent.
///
/// `@MainActor`: it owns mutable task/pending state and its operation typically captures a
/// main-actor model object, so the isolation is part of the contract.
@MainActor
public final class Coalescer {
    private let op: @MainActor @Sendable () async -> Void
    private var running: Task<Void, Never>?
    private var pending = false

    public init(op: @escaping @MainActor @Sendable () async -> Void) {
        self.op = op
    }

    /// Run the operation, serialized: if a run is in flight, remember to run once more when
    /// it finishes (multiple calls coalesce into that single re-run).
    public func schedule() {
        if running != nil {
            pending = true
            return
        }
        running = Task { [weak self] in
            guard let self else { return }
            repeat {
                self.pending = false
                await self.op()
            } while self.pending
            self.running = nil
        }
    }
}
