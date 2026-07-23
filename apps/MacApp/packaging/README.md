# Packaging & releasing the Mac app

The app and the engine version independently and ship differently:

- **engine** (`mlx-dspark` on PyPI) — released with `vX.Y.Z` git tags, installed by the app at
  runtime via vendored `uv`. Nothing here touches it.
- **app** (`mlx-dspark.app`) — released as a DMG attached to an **`app-vX.Y.Z`** GitHub release,
  and installed via the Homebrew cask. That's what this directory builds.

The `app-v` tag prefix matters: both live in one repo, so without it the cask (at, say, app
0.1.0) would look permanently "outdated" against the engine's 0.6.1 tags. The cask's `livecheck`
only matches `app-v*`.

## Build

```bash
cd apps/MacApp
./packaging/make_app.sh          # assemble + ad-hoc sign build/mlx-dspark.app
./packaging/make_dmg.sh          # build/mlx-dspark-<version>.dmg, prints the sha256
```

Set the version with `APP_VERSION` (defaults to `0.1.0`):

```bash
APP_VERSION=0.2.0 ./packaging/make_dmg.sh
```

`make_dmg.sh` calls `make_app.sh` itself; pass `SKIP_BUILD=1` to reuse a staged `.app`.

## Cut a release

```bash
# 1. build the DMG and note the printed sha256
APP_VERSION=0.1.0 ./packaging/make_dmg.sh

# 2. create the release under the app- tag and attach the DMG
gh release create app-v0.1.0 build/mlx-dspark-0.1.0.dmg --title "mlx-dspark app v0.1.0"

# 3. update Casks/mlx-dspark.rb — the two lines that change each release:
#      version "0.1.0"
#      sha256  "<the printed sha256>"
#    then commit it.
```

Install, from a user's side:

```bash
brew tap ARahim3/mlx-dspark https://github.com/ARahim3/mlx-dspark
brew install --cask --no-quarantine mlx-dspark
```

## Why `--no-quarantine`

The app is **ad-hoc signed but not notarized** — notarization needs a paid Apple Developer
Program membership. On macOS 15 a downloaded, un-notarized app no longer opens with a Finder
right-click → Open; the user has to go to System Settings › Privacy & Security › "Open Anyway".

`brew install --cask --no-quarantine` skips the quarantine attribute, so the app just opens.
This is the reason the cask is the recommended install over a raw DMG download. The `caveats`
block in the cask explains it to anyone who installs without the flag.

Keeping the Python runtime **outside** the bundle (it installs to Application Support at first
launch) is what keeps signing to just "the Swift binary + one vendored `uv`" — a full embedded
Python would mean codesigning thousands of nested `.dylib`s. That decision is what makes an
eventual notarization step small, if a certificate is ever bought.

## Validating the cask

```bash
ruby -c Casks/mlx-dspark.rb                      # syntax
# style needs a real tap layout:
brew tap-new you/tmp --no-git
cp Casks/mlx-dspark.rb "$(brew --repository)/Library/Taps/you/homebrew-tmp/Casks/"
brew style --cask you/tmp/mlx-dspark
brew untap you/tmp
```

`brew audit --new` will also complain until the release actually exists (it tries to download
the DMG) and that the repo is <30 days old — both expected pre-release, not cask defects.

## If a Developer ID is ever bought

Notarization drops the `--no-quarantine` requirement entirely. The steps would be, after
`make_app.sh`:

```bash
codesign --force --deep --options runtime --sign "Developer ID Application: <name> (<team>)" build/mlx-dspark.app
# then, on the DMG:
xcrun notarytool submit build/mlx-dspark-<v>.dmg --keychain-profile <profile> --wait
xcrun stapler staple build/mlx-dspark-<v>.dmg
```

At that point drop `--no-quarantine` from the install instructions and the cask caveats.

## Not done here (deliberately)

- **Fancy DMG window** (background image, positioned icons). Needs Finder AppleScript
  automation, which needs an automation permission this build environment doesn't have. The DMG
  is functional — app + Applications shortcut — just not decorated.
- **Sparkle auto-update.** The plan mentions it; the app currently has no in-app updater, so a
  new version is a `brew upgrade`. Add Sparkle when the release cadence justifies it.
