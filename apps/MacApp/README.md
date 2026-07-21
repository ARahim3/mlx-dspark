# mlx-dspark — Mac app

A native SwiftUI app for the mlx-dspark engine. It does **not** link Python: it installs an
app-owned runtime, launches `mlx-dspark serve`, and drives it over HTTP + SSE — the same CLI a
terminal user would run.

> Status: **bootstrap spike**. The runtime → server → stream spine is working and verified;
> the real screens (Lab, Models, Agents) are not built yet.

## Build

Needs only **Command Line Tools** — no Xcode.

```bash
swift build                       # or: swift build -c release
./packaging/make_app.sh --debug   # assemble + ad-hoc sign build/mlx-dspark.app
```

Run against a local working tree instead of PyPI:

```bash
open -n --env MLXDSPARK_ENGINE_SOURCE=/path/to/dspark ./build/mlx-dspark.app
```

`open --env` is required — a Finder-launched app inherits no shell environment, and running
`Contents/MacOS/MlxDspark` directly does not register with LaunchServices (no window appears).

### Tests

`swift test` needs **full Xcode**: SwiftPM packages tests as an `.xctest` bundle and the
`xctest` runner ships only with Xcode. They compile under CLT (`swift build --build-tests`) and
run in CI. Tests use **swift-testing**, not XCTest — CLT omits XCTest entirely.

## Layout

```
Sources/AppCore/   logic, no SwiftUI — unit-testable, no running app required
  AppIdentity      every branded string (renaming the app is 3 strings, see below)
  Paths            install locations — everything lives OUTSIDE the .app bundle
  Shell            line-streamed subprocesses
  RuntimeBootstrapper   vendored uv → managed CPython → app-owned venv → engine
  ServerSupervisor      free-port preflight, spawn, /health readiness, shutdown
  APIClient        /health, /v1/models, /metrics, SSE chat + the x_mlx_dspark stats block
  LogStore         bounded ring, mirrored to logs/app.log
Sources/AppHost/   SwiftUI views
packaging/         make_app.sh
```

There is **no `.xcodeproj`** on purpose: `swift build` is reproducible and an xcodeproj is a
merge-conflict generator. Generate one with xcodegen if a Mac App Store build is ever needed.

## Renaming the app

The display name is provisional. It is centralised in exactly three places:

1. `Sources/AppCore/AppIdentity.swift` — `displayName`, `bundleID`
2. `Package.swift` — the executable product name
3. `packaging/make_app.sh` — `APP_NAME`, `BUNDLE_ID`

Changing `bundleID` orphans an existing runtime install under Application Support, so bump it
deliberately.

## Where things live at runtime

```
~/Library/Application Support/com.arahim.mlx-dspark/
  runtime/venv/          app-owned Python venv (engine installed here)
  runtime/fingerprint.json
  logs/app.log
```

Delete `runtime/` to force a clean re-bootstrap.

## Distribution

Ad-hoc signing (`codesign -s -`) is what `make_app.sh` does — enough to run locally. Shipping a
DMG that opens without right-click→Open needs an Apple Developer Program membership, a
Developer ID Application certificate, and notarization. Not set up yet.
