import Foundation

/// Every branded string in the app, in one place.
///
/// The app's display name is still provisional, so nothing else in the codebase may hard-code
/// it — read it from here. Renaming is this struct plus the product name in `Package.swift`
/// plus `APP_NAME` in `packaging/make_app.sh`.
public enum AppIdentity {
    /// Display name (window title, menu bar, About box).
    public static let displayName = "mlx-dspark"

    /// Reverse-DNS bundle id. Also the Application Support directory name, so changing it
    /// orphans an existing runtime install — bump deliberately.
    public static let bundleID = "com.arahim.mlx-dspark"

    /// The PyPI package this app installs and drives.
    public static let enginePackage = "mlx-dspark"

    /// The engine tracks the **latest PyPI release automatically**: the bootstrapper resolves
    /// the newest version in the background after launch and applies it as an in-place
    /// upgrade on the next launch (offline launches keep whatever is installed). There is
    /// deliberately no pinned version — a pin rots the moment the engine ships a release,
    /// which is how the app once sat on 0.6.1 while 0.8.1 was current and could not load the
    /// newer model families.

    /// There is deliberately no engine version *floor* either: when a cached engine predates
    /// surface the app wants (e.g. `serve --no-model`), the app degrades gracefully (the
    /// supervisor falls back to the classic `--model` start) and the background update check
    /// brings the engine to latest by the next launch anyway.

    /// CPython the runtime venv is built against. uv fetches a managed build if the machine
    /// has none, which is what lets onboarding avoid ever mentioning Homebrew.
    public static let pythonVersion = "3.12"

    /// GitHub repo, for the app-release update check (`app-v*` tags).
    public static let repoSlug = "ARahim3/mlx-dspark"

    /// This build's version, as stamped into Info.plist by `make_app.sh` (`APP_VERSION`).
    /// "dev" when running outside a bundle (swift run, tests).
    public static var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
    }
}
