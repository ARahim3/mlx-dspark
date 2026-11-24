#!/usr/bin/env bash
# Wait for the CI run of a pushed commit; exit non-zero if it fails.
#
# The step between `git push origin main` and publishing a release. CI's runner has a GPU no
# local check reproduces (its Metal kernels get smaller threadgroups than a real Mac's), so a
# release is only known good once the run for its commit has passed.
#
#   scripts/wait_ci.sh            # the run for HEAD
#   scripts/wait_ci.sh HEAD~1     # ... for another commit
#
# Needs the GitHub CLI (`gh`), signed in.
set -euo pipefail

SHA="$(git rev-parse "${1:-HEAD}")"
echo "==> Waiting for the CI run of ${SHA:0:7}"
ID=""
for _ in $(seq 60); do                      # a run takes a few seconds to appear after a push
    ID="$(gh run list --workflow CI --commit "$SHA" --limit 1 \
          --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
    [[ -n "$ID" ]] && break
    sleep 5
done
[[ -n "$ID" ]] || { echo "error: no CI run for $SHA (is it pushed to main?)" >&2; exit 1; }
gh run watch "$ID" --exit-status
