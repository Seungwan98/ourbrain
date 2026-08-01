#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
label="com.ourbrain.review-watcher"
domain="gui/$(id -u)"
plist="$HOME/Library/LaunchAgents/$label.plist"
watcher="$project/scripts/mac/watch_review_and_launch_v0_2.sh"
state_dir="$project/artifacts/review_watcher"

usage() {
  printf 'usage: %s [--uninstall|--status]\n' "$0"
}

action="install"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "${1:-}" in
  '') ;;
  --uninstall) action="uninstall" ;;
  --status) action="status" ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "$action" == "status" ]]; then
  launchctl print "$domain/$label"
  exit $?
fi

if [[ "$action" == "uninstall" ]]; then
  launchctl bootout "$domain" "$plist" 2>/dev/null || true
  python3 - "$plist" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).unlink(missing_ok=True)
PY
  printf 'uninstalled %s\n' "$label"
  exit 0
fi

for command in bash git python3 uv; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'required command is missing: %s\n' "$command" >&2
    exit 2
  fi
done
if [[ ! -x "$watcher" ]]; then
  printf 'watcher is missing or not executable: %s\n' "$watcher" >&2
  exit 2
fi

mkdir -p "$(dirname "$plist")" "$state_dir"

path_value="$({
  dirname "$(command -v uv)"
  dirname "$(command -v git)"
  dirname "$(command -v python3)"
  printf '%s\n' /usr/local/bin /usr/bin /bin /usr/sbin /sbin
} | awk '!seen[$0]++' | paste -sd: -)"

python3 - "$plist" "$label" "$project" "$watcher" "$state_dir" "$path_value" <<'PY'
from pathlib import Path
import plistlib
import sys

plist_path = Path(sys.argv[1])
payload = {
    "Label": sys.argv[2],
    "ProgramArguments": ["/bin/bash", sys.argv[4]],
    "WorkingDirectory": sys.argv[3],
    "RunAtLoad": True,
    "KeepAlive": {"SuccessfulExit": False},
    "ThrottleInterval": 30,
    "ProcessType": "Background",
    "EnvironmentVariables": {"PATH": sys.argv[6]},
    "StandardOutPath": str(Path(sys.argv[5]) / "launchd.stdout.log"),
    "StandardErrorPath": str(Path(sys.argv[5]) / "launchd.stderr.log"),
}
with plist_path.open("wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY

plutil -lint "$plist"
launchctl bootout "$domain" "$plist" 2>/dev/null || true
launchctl bootstrap "$domain" "$plist"
launchctl enable "$domain/$label"
launchctl kickstart -k "$domain/$label"
launchctl print "$domain/$label" | sed -n '1,35p'
printf 'installed %s from %s\n' "$label" "$plist"
