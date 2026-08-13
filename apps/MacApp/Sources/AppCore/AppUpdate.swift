import Foundation

/// Checks GitHub for a newer app release.
///
/// The app's releases are tagged `app-vX.Y.Z` (distinct from the engine's `vX.Y.Z` PyPI tags
/// in the same repo). This is deliberately *not* an auto-updater — installing is Homebrew's
/// job (`brew upgrade --cask mlx-dspark`) or a fresh DMG; the app only tells the user a newer
/// version exists. The engine updates itself separately (the bootstrapper tracks PyPI).
public enum AppUpdate {

    public struct Available: Sendable, Equatable {
        public let version: String
        public let url: String
    }

    /// The newest `app-v*` release newer than `current`, or nil (up to date, no releases
    /// yet, or offline — all three mean "say nothing").
    public static func check(current: String,
                             repo: String = AppIdentity.repoSlug,
                             timeout: TimeInterval = 6) async -> Available? {
        struct Release: Decodable {
            let tag_name: String
            let html_url: String
            let draft: Bool
            let prerelease: Bool
        }
        guard let url = URL(string: "https://api.github.com/repos/\(repo)/releases?per_page=20")
        else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let releases = try? JSONDecoder().decode([Release].self, from: data)
        else { return nil }

        let newest = releases
            .filter { !$0.draft && !$0.prerelease && $0.tag_name.hasPrefix("app-v") }
            .map { (version: String($0.tag_name.dropFirst("app-v".count)), url: $0.html_url) }
            .max { isOlder($0.version, than: $1.version) }
        guard let newest, isOlder(current, than: newest.version) else { return nil }
        return Available(version: newest.version, url: newest.url)
    }

    /// Dotted-numeric comparison; anything unparseable compares as 0 (never blocks launch).
    static func isOlder(_ lhs: String, than rhs: String) -> Bool {
        let a = lhs.split(separator: ".").map { Int($0.prefix(while: \.isNumber)) ?? 0 }
        let b = rhs.split(separator: ".").map { Int($0.prefix(while: \.isNumber)) ?? 0 }
        for i in 0..<max(a.count, b.count) {
            let x = i < a.count ? a[i] : 0
            let y = i < b.count ? b[i] : 0
            if x != y { return x < y }
        }
        return false
    }
}
