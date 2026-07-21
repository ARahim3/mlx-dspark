import Foundation

/// Where the app keeps things on disk.
///
/// Design rule: **everything the app installs lives outside the .app bundle**, under
/// Application Support. That is what keeps notarization to "our Swift binary + one signed
/// `uv`" instead of thousands of nested `.dylib`s, and it lets the engine be upgraded without
/// re-shipping the app.
public enum Paths {
    /// `~/Library/Application Support/<bundleID>`
    public static var appSupport: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent(AppIdentity.bundleID, isDirectory: true)
    }

    /// Root of the app-owned Python runtime.
    public static var runtimeDir: URL {
        appSupport.appendingPathComponent("runtime", isDirectory: true)
    }

    public static var venvDir: URL {
        runtimeDir.appendingPathComponent("venv", isDirectory: true)
    }

    /// The engine CLI inside the app-owned venv. This is the only `mlx-dspark` the app ever
    /// executes — a copy on the user's PATH is theirs, with dependency state we don't control,
    /// and must never be adopted as the engine.
    public static var engineExecutable: URL {
        venvDir.appendingPathComponent("bin/mlx-dspark")
    }

    /// Records which engine version+wheel produced the current venv, so an app update can tell
    /// a stale runtime from a current one even when the version string didn't change.
    public static var runtimeFingerprint: URL {
        runtimeDir.appendingPathComponent("fingerprint.json")
    }

    public static var logsDir: URL {
        appSupport.appendingPathComponent("logs", isDirectory: true)
    }

    /// `uv`, preferring the copy vendored in the app bundle.
    ///
    /// Falling back to a `uv` on PATH is a *development* convenience — `swift run` has no
    /// bundle to look in. Shipping builds always find the vendored one first, so the user is
    /// never at the mercy of a Homebrew uv they may not have.
    public static func uvExecutable(bundle: Bundle = .main) -> URL? {
        if let vendored = bundle.url(forResource: "uv", withExtension: nil, subdirectory: "bin"),
           FileManager.default.isExecutableFile(atPath: vendored.path) {
            return vendored
        }
        for candidate in ["/opt/homebrew/bin/uv", "/usr/local/bin/uv",
                          NSHomeDirectory() + "/.local/bin/uv"] {
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return URL(fileURLWithPath: candidate)
            }
        }
        return nil
    }

    public static func ensureDirectories() throws {
        for dir in [appSupport, runtimeDir, logsDir] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }
}
