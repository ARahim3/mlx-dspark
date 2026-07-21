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

    /// Minimum engine version this app's UI expects. The bootstrapper installs exactly this;
    /// a runtime built from a different version is torn down and reinstalled (see
    /// `RuntimeBootstrapper`) — same-version rebuilds are the trap that MTPLX documents.
    public static let engineVersion = "0.6.0"

    /// CPython the runtime venv is built against. uv fetches a managed build if the machine
    /// has none, which is what lets onboarding avoid ever mentioning Homebrew.
    public static let pythonVersion = "3.12"
}
