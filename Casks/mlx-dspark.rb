# Homebrew cask for the mlx-dspark Mac app.
#
# Install (Homebrew 6+):
#   brew tap ARahim3/mlx-dspark https://github.com/ARahim3/mlx-dspark
#   brew trust arahim3/mlx-dspark
#   brew install --cask mlx-dspark
#   xattr -dr com.apple.quarantine /Applications/mlx-dspark.app
#
# Homebrew 6 (verified 2026-08-16) removed `--no-quarantine` entirely (brew/#20755 — "invalid
# option") and requires `brew trust` for third-party taps; installed casks now KEEP the
# quarantine attribute. The app is ad-hoc signed but NOT notarized (that needs a paid Apple
# Developer ID), so a quarantined copy fails Gatekeeper (`spctl: rejected`) and macOS 15 blocks
# the first launch behind System Settings › Privacy & Security › "Open Anyway". The `xattr`
# line clears the attribute once — the exact thing `--no-quarantine` used to do — and the
# caveats below tell the user so at install time. On Homebrew ≤ 5 the old
# `brew install --cask --no-quarantine mlx-dspark` still works. All of this retires when the
# app is notarized (also note Homebrew intends to require Gatekeeper-passing casks from
# Sep 2026 — notarization is the real fix, not a nicety).
#
# Per release, two lines change: `version` and `sha256`. `make_dmg.sh` prints both.
cask "mlx-dspark" do
  version "0.7.0"
  sha256 "af4df84f8521a63dfc0fc6d6a894d223ca6ae26da5ef991d0ea78830046700ae"

  # App releases are tagged `app-vX.Y.Z`, distinct from the engine's own `vX.Y.Z` PyPI tags in
  # the same repo — the app and the engine version independently, and without the prefix a cask
  # at app 0.1.0 would look forever "outdated" against engine 0.6.1.
  url "https://github.com/ARahim3/mlx-dspark/releases/download/app-v#{version}/mlx-dspark-#{version}.dmg"
  name "mlx-dspark"
  desc "Speculative-decoding cockpit for local LLMs on Apple Silicon"
  homepage "https://github.com/ARahim3/mlx-dspark"

  # Only look at app- tags, so engine releases in the same repo don't register as updates.
  livecheck do
    url "https://github.com/ARahim3/mlx-dspark/releases"
    regex(/app-v?(\d+(?:\.\d+)+)/i)
    strategy :page_match
  end

  # The app self-updates its engine, but the app bundle itself is what a new release ships.
  auto_updates false
  depends_on macos: :sonoma # 14+

  app "mlx-dspark.app"

  # The app installs a Python runtime under Application Support on first launch (vendored uv →
  # managed CPython → venv). `zap` removes everything the app created, so an uninstall is clean.
  zap trash: [
    "~/Library/Application Support/com.arahim.mlx-dspark",
    "~/Library/Caches/com.arahim.mlx-dspark",
    "~/Library/Preferences/com.arahim.mlx-dspark.plist",
  ]

  caveats <<~EOS
    mlx-dspark runs entirely on your Mac. On first launch it downloads its engine
    (a Python runtime, a few hundred MB) and then a model you choose — this needs
    a network connection and a few minutes, once.

    Requires Apple Silicon (M1 or newer).

    The app is not notarized yet, so macOS will block the first launch. Either run

        xattr -dr com.apple.quarantine /Applications/mlx-dspark.app

    once, or allow it under System Settings › Privacy & Security › "Open Anyway"
    after the first launch attempt. (Homebrew 6 removed the old --no-quarantine
    flag; on Homebrew 5 and older it still works at install time.)
  EOS
end
