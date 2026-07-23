import Foundation

/// One coding agent and how to point it at the local server.
///
/// The v0.6.0 work made Claude Code (and any OpenAI/Anthropic client) run on a local model;
/// this is that work made visible. The engine already knows every field each agent needs — the
/// app's job is to present it so setup works on the first try, not to invent the config.
public struct Integration: Decodable, Sendable, Identifiable {
    public let id: String
    public let name: String
    /// "anthropic" or "openai" — which wire format this agent will use.
    public let proto: String
    public let summary: String
    public let setup: [SetupStep]
    public let note: String?

    enum CodingKeys: String, CodingKey {
        case id, name, summary, setup, note
        case proto = "protocol"
    }

    /// One step in wiring up an agent. Different agents need different *kinds* of config — an
    /// env block, a file to write, values to type into a settings pane — so the kind decides
    /// how the app renders and offers to apply it.
    public struct SetupStep: Decodable, Sendable, Identifiable {
        public let kind: Kind
        public let label: String
        public let command: String?
        public let path: String?
        public let content: String?
        public let env: [String: String]?
        public let fields: [String: String]?

        public var id: String { label }

        public enum Kind: String, Decodable, Sendable {
            case command, file, env, fields
        }

        /// The text a "Copy" button should place on the clipboard for this step.
        public var copyText: String? {
            switch kind {
            case .command: return command
            case .file:    return content
            case .env:     return env.map(Self.exportLines)
            case .fields:  return fields.map { pairs in
                pairs.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }
                    .joined(separator: "\n")
            }
            }
        }

        private static func exportLines(_ env: [String: String]) -> String {
            env.sorted { $0.key < $1.key }.map { "export \($0.key)=\($0.value)" }
                .joined(separator: "\n")
        }
    }
}

public struct IntegrationList: Decodable, Sendable {
    public let baseURL: String
    public let model: String
    public let integrations: [Integration]

    enum CodingKeys: String, CodingKey {
        case model, integrations
        case baseURL = "base_url"
    }
}

extension APIClient {
    public func integrations() async throws -> IntegrationList {
        let (data, response) = try await session.data(for: request("admin/integrations"))
        try Self.check(response, data)
        return try JSONDecoder().decode(IntegrationList.self, from: data)
    }
}
