#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project"

review_url="${OURBRAIN_REVIEW_URL:-https://ourbrain-tunnel-review.vercel.app}"
env_file="${OURBRAIN_REVIEW_ENV_FILE:-$project/.env.remote-review.local}"
finish_script="${OURBRAIN_FINISH_SCRIPT:-$project/scripts/mac/finish_review_and_launch_v0_2.sh}"
state_dir="${OURBRAIN_REVIEW_WATCH_STATE_DIR:-$project/artifacts/review_watcher}"
interval="${OURBRAIN_REVIEW_POLL_SECONDS:-60}"
once=0

usage() {
  printf 'usage: %s [--once] [--interval SECONDS]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      once=1
      shift
      ;;
    --interval)
      if [[ $# -lt 2 ]]; then
        usage >&2
        exit 2
      fi
      interval="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -lt 10 ]]; then
  printf 'poll interval must be an integer of at least 10 seconds.\n' >&2
  exit 2
fi

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi
if [[ -z "${OURBRAIN_REVIEW_TOKEN:-}" ]]; then
  printf 'OURBRAIN_REVIEW_TOKEN is not configured in %s or the environment.\n' \
    "$env_file" >&2
  exit 2
fi
for command in git uv python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'required command is missing: %s\n' "$command" >&2
    exit 2
  fi
done
if [[ ! -x "$finish_script" ]]; then
  printf 'finish script is missing or not executable: %s\n' "$finish_script" >&2
  exit 2
fi

mkdir -p "$state_dir"
lock_dir="$state_dir/lock"
completion_file="$state_dir/completed.json"
latest_status="$state_dir/latest-status.json"
watch_log="$state_dir/watcher.log"

if [[ -f "$completion_file" ]]; then
  printf 'review watcher already completed: %s\n' "$completion_file"
  exit 0
fi

acquire_lock() {
  if mkdir "$lock_dir" 2>/dev/null; then
    printf '%s\n' "$$" >"$lock_dir/pid"
    return 0
  fi

  local existing_pid=""
  if [[ -f "$lock_dir/pid" ]]; then
    existing_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
  fi
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'review watcher is already running with PID %s.\n' "$existing_pid" >&2
    return 4
  fi

  python3 - "$lock_dir" <<'PY'
from pathlib import Path
import sys

lock = Path(sys.argv[1])
pid_file = lock / "pid"
pid_file.unlink(missing_ok=True)
try:
    lock.rmdir()
except FileNotFoundError:
    pass
PY
  mkdir "$lock_dir"
  printf '%s\n' "$$" >"$lock_dir/pid"
}

acquire_lock
cleanup() {
  python3 - "$lock_dir" "$$" <<'PY'
from pathlib import Path
import sys

lock = Path(sys.argv[1])
expected_pid = sys.argv[2]
pid_file = lock / "pid"
if pid_file.exists() and pid_file.read_text().strip() == expected_pid:
    pid_file.unlink()
    lock.rmdir()
PY
}
trap cleanup EXIT

log_line() {
  local message="$1"
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$message" | tee -a "$watch_log"
}

while true; do
  temporary_status="$state_dir/latest-status.tmp.$$"
  if uv run ourbrain-cv remote-review-status \
    --url "$review_url" \
    --summary-only >"$temporary_status"; then
    mv "$temporary_status" "$latest_status"
  else
    python3 - "$temporary_status" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).unlink(missing_ok=True)
PY
    log_line 'remote review status request failed; retrying'
    if [[ $once -eq 1 ]]; then
      exit 1
    fi
    sleep "$interval"
    continue
  fi

  IFS=$'\t' read -r reviewed total unreviewed conflicts negative_train negative_val negative_test eligible < <(
    python3 - "$latest_status" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)["summary"]
negative_counts = [summary["bySplit"][name]["negative"] for name in ("train", "val", "test")]
eligible = (
    summary["complete"]
    and summary["reviewed"] == summary["total"] == 200
    and summary["unreviewed"] == 0
    and summary["conflicts"] == 0
    and all(count > 0 for count in negative_counts)
)
print(
    summary["reviewed"],
    summary["total"],
    summary["unreviewed"],
    summary["conflicts"],
    *negative_counts,
    int(eligible),
    sep="\t",
)
PY
  )

  log_line "reviewed=$reviewed/$total unreviewed=$unreviewed conflicts=$conflicts negatives=$negative_train/$negative_val/$negative_test"

  if [[ "$eligible" == "1" ]]; then
    launch_log="$state_dir/finish-and-launch-$(date -u '+%Y%m%dT%H%M%SZ').log"
    log_line "strict gate open; running $finish_script --launch-training"
    if "$finish_script" --launch-training 2>&1 | tee "$launch_log"; then
      python3 - "$latest_status" "$completion_file" "$(git rev-parse HEAD)" "$launch_log" <<'PY'
import datetime as dt
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    status = json.load(handle)
payload = {
    "schema_version": 1,
    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_commit": sys.argv[3],
    "launch_log": sys.argv[4],
    "review_status": status,
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
      log_line "strict import and Windows A/B launch completed: $completion_file"
      exit 0
    fi
    log_line 'finish/import/launch failed; retaining logs and retrying'
  fi

  if [[ $once -eq 1 ]]; then
    exit 3
  fi
  sleep "$interval"
done
