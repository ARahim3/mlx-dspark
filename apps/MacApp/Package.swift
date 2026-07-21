// swift-tools-version:5.9
//
// The mlx-dspark Mac app.
//
// Module names are deliberately generic (AppCore / AppHost) so the *product* name is the only
// branded string in the build graph — renaming the app is this file's product name plus
// `AppIdentity.swift` plus one variable in packaging/make_app.sh, and nothing else.
//
// TOOLCHAIN: builds with Command Line Tools alone (verified: Swift 6.2.4, macOS SDK 26.2 —
// no Xcode). `swift build` produces a bare executable; packaging/make_app.sh wraps it into
// the .app bundle a SwiftUI app needs to behave like one.
//
// Running `swift test` DOES need full Xcode: SwiftPM packages tests as an .xctest bundle and
// the `xctest` runner ships only with Xcode. The tests use swift-testing (not XCTest, which
// CLT omits entirely), so they run unmodified in CI where Xcode is present.

import PackageDescription
import Foundation

/// A Command Line Tools install puts Testing.framework outside the SDK, where SwiftPM does not
/// look, so the compiler has to be pointed at it explicitly.
///
/// Both conditions below are tested *directly* rather than inferred from whether Xcode is
/// installed — that proxy is wrong on this very machine: Xcode.app is present but
/// `xcode-select` points at Command Line Tools, so the CLT toolchain is what actually builds.
/// An extra `-F` is additive and harmless under any toolchain; the overlay flag is gated on
/// the real defect it works around.
let testingFrameworkSettings: (swift: [SwiftSetting], linker: [LinkerSetting]) = {
    let fm = FileManager.default
    let searchPaths = [
        "/Library/Developer/CommandLineTools/Library/Developer/Frameworks",
        "/Applications/Xcode.app/Contents/Developer/Library/Frameworks",
    ]
    guard let dir = searchPaths.first(where: {
        fm.fileExists(atPath: $0 + "/Testing.framework")
    }) else {
        return ([], [])                       // toolchain provides it; nothing to do
    }

    var swiftFlags = ["-F", dir]
    // CLT ships _Testing_Foundation as a binary with NO .swiftmodule, so the cross-import
    // overlay triggered by `import Foundation` next to `import Testing` cannot resolve.
    let overlayModules = dir + "/_Testing_Foundation.framework/Versions/A/Modules"
    if !fm.fileExists(atPath: overlayModules) {
        swiftFlags += ["-Xfrontend", "-disable-cross-import-overlays"]
    }
    return (
        [.unsafeFlags(swiftFlags)],
        // `-rpath` is a linker argument, so each token needs its own -Xlinker; passing it
        // bare reaches the driver instead and fails with "unknown argument: '-rpath'".
        [.unsafeFlags(["-F", dir, "-Xlinker", "-rpath", "-Xlinker", dir])]
    )
}()

let package = Package(
    name: "MlxDspark",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MlxDspark", targets: ["AppHost"]),
        .library(name: "AppCore", targets: ["AppCore"]),
    ],
    targets: [
        // Logic only — no SwiftUI import, so it stays testable without a running app and the
        // rule "every feature is a server endpoint first" has somewhere to live.
        .target(name: "AppCore"),
        .executableTarget(name: "AppHost", dependencies: ["AppCore"]),
        .testTarget(
            name: "AppCoreTests",
            dependencies: ["AppCore"],
            swiftSettings: testingFrameworkSettings.swift,
            linkerSettings: testingFrameworkSettings.linker
        ),
    ]
)
