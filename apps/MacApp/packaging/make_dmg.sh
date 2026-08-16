#!/usr/bin/env bash
#
# Build a distributable .dmg from the assembled .app.
#
#   ./packaging/make_dmg.sh            build/mlx-dspark-<version>.dmg
#   APP_VERSION=0.2.0 ./packaging/make_dmg.sh
#
# Produces a compressed disk image with the app and a drag-to-Applications shortcut, and prints
# the sha256 the Homebrew cask needs.
#
# NOTE ON SIGNING: make_app.sh ad-hoc signs the bundle, which is enough to *run*, but the DMG is
# NOT notarized (that needs a paid Apple Developer ID). On macOS 15 a downloaded, un-notarized
# app can't be opened from Finder with a right-click any more — the user must go to System
# Settings › Privacy & Security › "Open Anyway". The Homebrew cask sidesteps this entirely with
# `--no-quarantine` (see Casks/mlx-dspark.rb), which is why the cask is the recommended install.

set -euo pipefail

cd "$(dirname "$0")/.."

APP_NAME="mlx-dspark"
VERSION="${APP_VERSION:-0.4.0}"
APP="build/${APP_NAME}.app"
DMG="build/${APP_NAME}-${VERSION}.dmg"
VOLNAME="${APP_NAME} ${VERSION}"

# 1. Build the app (release) unless one is already staged.
if [[ "${SKIP_BUILD:-}" != "1" || ! -d "$APP" ]]; then
    APP_VERSION="$VERSION" ./packaging/make_app.sh
fi
[[ -d "$APP" ]] || { echo "error: no app at $APP" >&2; exit 1; }

# 2. Stage exactly what the DMG should contain: the app plus an /Applications shortcut, so the
#    window reads "drag this there".
STAGE="build/dmg-stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

# 3. Build the compressed image. UDZO = zlib-compressed, the standard read-only distribution
#    format. A generous size cap avoids "no space" on the intermediate rw image; hdiutil shrinks
#    it on compress.
echo "==> Building $DMG"
rm -f "$DMG"
hdiutil create \
    -volname "$VOLNAME" \
    -srcfolder "$STAGE" \
    -fs HFS+ \
    -format UDZO \
    -imagekey zlib-level=9 \
    -ov \
    "$DMG" >/dev/null

rm -rf "$STAGE"

# 4. The sha256 the cask pins. Printed prominently because it changes every build and a stale
#    one makes `brew install` fail with a checksum mismatch.
SHA=$(shasum -a 256 "$DMG" | awk '{print $1}')
SIZE=$(du -h "$DMG" | awk '{print $1}')

echo
echo "Built $DMG  ($SIZE)"
echo "  sha256: $SHA"
echo
# App releases use an `app-v` tag prefix so they don't collide with the engine's own vX.Y.Z
# PyPI tags in the same repo (the cask's livecheck matches only `app-v*`).
echo "Next, for a release:"
echo "  1. gh release create app-v$VERSION $DMG --title \"mlx-dspark app v$VERSION\""
echo "  2. update Casks/mlx-dspark.rb — version \"$VERSION\" and sha256 \"$SHA\""
echo "  3. commit the cask"
