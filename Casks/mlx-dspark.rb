# Homebrew cask for the mlx-dspark Mac app.
#
# Install:
#   brew tap ARahim3/mlx-dspark https://github.com/ARahim3/mlx-dspark
#   brew install --cask --no-quarantine mlx-dspark
#
# The `--no-quarantine` flag is what makes this pleasant. The app is ad-hoc signed but NOT
# notarized (that needs a paid Apple Developer ID), so without it macOS 15 blocks the first
# launch behind System Settings › Privacy & Security › "Open Anyway". `--no-quarantine` skips
# the quarantine attribute entirely, so the app just opens. Drop it once the app is notarized.
#
# Per release, two lines change: `version` and `sha256`. `make_dmg.sh` prints both.
cask "mlx-dspark" do
  version "0.1.0"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

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

    The app is not notarized yet, so it was installed with --no-quarantine. If you
    installed without that flag and macOS blocks it, either reinstall with
    `--no-quarantine` or allow it under System Settings › Privacy & Security.
  EOS
end
