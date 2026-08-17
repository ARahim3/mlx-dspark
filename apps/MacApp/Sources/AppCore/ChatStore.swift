import Foundation

/// One chat message, as the app renders and persists it.
///
/// `Equatable` is load-bearing for chat performance: it lets SwiftUI skip re-rendering (and
/// re-parsing the markdown of) a message whose value did not change when the god-object
/// `AppModel` invalidates for an unrelated reason — e.g. the always-on round telemetry
/// appends during a chat turn. Without it a long streaming answer re-parses on every
/// telemetry tick as well as every flush.
public struct ChatMessage: Identifiable, Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable { case user, assistant }

    public let id: UUID
    public let role: Role
    public var text: String
    public var stats: SpecInfo?

    public init(id: UUID = UUID(), role: Role, text: String, stats: SpecInfo? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.stats = stats
    }
}

/// A saved conversation.
public struct ChatSession: Identifiable, Codable, Sendable {
    public let id: UUID
    public var title: String
    public var createdAt: Date
    public var updatedAt: Date
    public var messages: [ChatMessage]

    public init(id: UUID = UUID(), title: String = "New chat",
                createdAt: Date = Date(), updatedAt: Date = Date(),
                messages: [ChatMessage] = []) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messages = messages
    }

    /// Titles derive from the first thing the user asked; nobody names chats by hand.
    public mutating func retitleFromContent() {
        guard let first = messages.first(where: { $0.role == .user })?.text else { return }
        let flat = first.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
        title = flat.count > 48 ? String(flat.prefix(48)) + "…" : (flat.isEmpty ? "New chat" : flat)
    }
}

/// Persists conversations as one JSON file per session under Application Support.
///
/// Files, not a database: a handful of small documents the user can see, copy, and delete.
/// Writes are atomic so a crash mid-save never eats a conversation.
public struct ChatStore: Sendable {
    private let dir: URL

    public init(directory: URL? = nil) {
        self.dir = directory ?? Paths.appSupport.appendingPathComponent("chats", isDirectory: true)
    }

    public func list() -> [ChatSession] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                (try? Data(contentsOf: url)).flatMap { try? decoder.decode(ChatSession.self, from: $0) }
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    public func save(_ session: ChatSession) {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(session) else { return }
        try? data.write(to: fileURL(session.id), options: .atomic)
    }

    public func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: fileURL(id))
    }

    private func fileURL(_ id: UUID) -> URL {
        dir.appendingPathComponent("\(id.uuidString).json")
    }
}
