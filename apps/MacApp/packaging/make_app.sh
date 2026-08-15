#!/usr/bin/env bash
#
# Assemble the .app bundle.
#
# SwiftPM emits a bare executable; a SwiftUI app needs a real bundle to get a menu bar, window
# focus and a Dock icon. There is no .xcodeproj in this repo on purpose (it is a merge-conflict
# generator, and `swift build` is reproducible), so the bundle is assembled here instead.
#
#   ./packaging/make_app.sh            release build
#   ./packaging/make_app.sh --debug    faster build, for iterating
#
# Signing: ad-hoc (`codesign -s -`) — enough to run locally. Distribution needs a Developer ID
# certificate and notarization; see packaging/README.md.

set -euo pipefail

cd "$(dirname "$0")/.."

APP_NAME="mlx-dspark"                 # display name — see AppIdentity.swift
BUNDLE_ID="com.arahim.mlx-dspark"
EXECUTABLE="MlxDspark"                # SwiftPM product name
VERSION="${APP_VERSION:-0.3.0}"
MIN_MACOS="14.0"

CONFIG="release"
[[ "${1:-}" == "--debug" ]] && CONFIG="debug"

echo "==> Building ($CONFIG)"
swift build -c "$CONFIG"
BUILT="$(swift build -c "$CONFIG" --show-bin-path)/$EXECUTABLE"
[[ -f "$BUILT" ]] || { echo "error: no executable at $BUILT" >&2; exit 1; }

APP="build/${APP_NAME}.app"
echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin"

cp "$BUILT" "$APP/Contents/MacOS/$EXECUTABLE"

# App icon. Regenerate the .icns from the committed master with Resources/makeicon.swift.
ICON="Resources/AppIcon.icns"
if [[ -f "$ICON" ]]; then
    cp "$ICON" "$APP/Contents/Resources/AppIcon.icns"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                  <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>           <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>            <string>${BUNDLE_ID}</string>
    <key>CFBundleExecutable</key>            <string>${EXECUTABLE}</string>
    <key>CFBundleIconFile</key>              <string>AppIcon</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>${VERSION}</string>
    <key>CFBundleVersion</key>               <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>        <string>${MIN_MACOS}</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <key>LSApplicationCategoryType</key>     <string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

# --- vendor uv ---------------------------------------------------------------------------
#
# This one binary is what lets onboarding avoid ever saying "install Homebrew": uv fetches its
# own CPython, builds the venv, and installs the engine. Everything it installs lands OUTSIDE
# this bundle (Application Support), which is why notarization only ever sees two binaries.
UV_SRC="${UV_BINARY:-$(command -v uv || true)}"
if [[ -z "$UV_SRC" ]]; then
    echo "warn: no uv found to vendor. The app will fall back to a uv on PATH at runtime," >&2
    echo "      which is fine for development but must be fixed before shipping." >&2
    echo "      Set UV_BINARY=/path/to/uv, or install from https://astral.sh/uv" >&2
else
    cp "$UV_SRC" "$APP/Contents/Resources/bin/uv"
    chmod +x "$APP/Contents/Resources/bin/uv"
    echo "==> Vendored uv from $UV_SRC ($("$UV_SRC" --version))"
fi

# --- sign --------------------------------------------------------------------------------
# Nested code first, then the bundle: signing outer-first invalidates the outer signature.
echo "==> Signing (ad-hoc)"
[[ -f "$APP/Contents/Resources/bin/uv" ]] && codesign --force -s - "$APP/Contents/Resources/bin/uv"
codesign --force --deep -s - "$APP"

echo
echo "Built $APP"
echo "  open $APP"
echo "  # develop against this working tree instead of PyPI:"
echo "  MLXDSPARK_ENGINE_SOURCE=$(cd ../.. && pwd) open -n $APP"
