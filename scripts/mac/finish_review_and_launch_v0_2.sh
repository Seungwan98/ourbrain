#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project"

review_url="https://ourbrain-tunnel-review.vercel.app"
env_file="$project/.env.remote-review.local"
review_csv="$project/data/negative_review/negative_review_reviewed.csv"
local_manifest="$project/artifacts/manifest_with_negatives.csv"
remote_alias="${OURBRAIN_GPU_SSH_ALIAS:-ourbrain-gpu}"
launch_training=0

if [[ "${1:-}" == "--launch-training" ]]; then
  launch_training=1
elif [[ $# -gt 0 ]]; then
  printf 'usage: %s [--launch-training]\n' "$0" >&2
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

for command in uv ssh scp python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'required command is missing: %s\n' "$command" >&2
    exit 2
  fi
done

status_file="$(mktemp "${TMPDIR:-/tmp}/ourbrain-review-status.XXXXXX.json")"
cleanup() {
  python3 - "$status_file" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).unlink(missing_ok=True)
PY
}
trap cleanup EXIT

uv run ourbrain-cv remote-review-status \
  --url "$review_url" \
  --summary-only >"$status_file"

if ! python3 - "$status_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)["summary"]
print(
    "review status: "
    f"{summary['reviewed']}/{summary['total']} reviewed, "
    f"{summary['unreviewed']} unreviewed, "
    f"{summary['conflicts']} conflicts"
)
raise SystemExit(
    0
    if summary["complete"]
    and summary["reviewed"] == summary["total"] == 200
    and summary["unreviewed"] == 0
    and summary["conflicts"] == 0
    and all(
        summary["bySplit"][split_name]["negative"] > 0
        for split_name in ("train", "val", "test")
    )
    else 3
)
PY
then
  printf 'Strict gate is closed; no CSV, manifest, or training task was changed.\n' >&2
  exit 3
fi

uv run ourbrain-cv remote-review-download \
  --url "$review_url" \
  --output "$review_csv"

uv run ourbrain-cv import-negatives \
  --review "$review_csv" \
  --manifest "$project/artifacts/manifest.csv" \
  --output "$local_manifest"

uv run ourbrain-cv training-preflight \
  --config "$project/configs/v0_2_a_baseline_with_negatives.yaml" \
  --manifest "$local_manifest" \
  --require-local-checkpoint
uv run ourbrain-cv training-preflight \
  --config "$project/configs/v0_2_b_recall_with_negatives.yaml" \
  --manifest "$local_manifest" \
  --require-local-checkpoint

candidate_files=()
while IFS= read -r -d '' candidate_file; do
  candidate_files[${#candidate_files[@]}]="$candidate_file"
done < <(
  find "$project/data/negative_review" -maxdepth 1 \
    -type f -name '*_neg_*.png' -print0
)
if [[ ${#candidate_files[@]} -ne 200 ]]; then
  printf 'expected 200 candidate PNG files, found %s\n' \
    "${#candidate_files[@]}" >&2
  exit 2
fi

ssh "$remote_alias" \
  'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path D:\ourbrain\data\negative_review | Out-Null"'
scp "${candidate_files[@]}" \
  "$remote_alias:/D:/ourbrain/data/negative_review/"
scp "$review_csv" \
  "$remote_alias:/D:/ourbrain/data/negative_review/negative_review_reviewed.csv"
source_files=("$project"/src/ourbrain_cv/*.py)
windows_scripts=("$project"/scripts/windows/*.ps1)
scp "${source_files[@]}" \
  "$remote_alias:/D:/ourbrain/src/ourbrain_cv/"
scp "${windows_scripts[@]}" \
  "$remote_alias:/D:/ourbrain/scripts/windows/"
scp \
  "$project/configs/v0_2_a_baseline_with_negatives_cuda.yaml" \
  "$project/configs/v0_2_b_recall_with_negatives_cuda.yaml" \
  "$remote_alias:/D:/ourbrain/configs/"

if [[ $launch_training -eq 1 ]]; then
  ssh "$remote_alias" \
    'powershell -NoProfile -ExecutionPolicy Bypass -File D:\ourbrain\scripts\windows\import_v0_2_negatives.ps1 -LaunchTraining'
else
  ssh "$remote_alias" \
    'powershell -NoProfile -ExecutionPolicy Bypass -File D:\ourbrain\scripts\windows\import_v0_2_negatives.ps1'
fi

if [[ $launch_training -eq 1 ]]; then
  printf 'Strict import passed on both machines and the Windows A/B task started.\n'
else
  printf 'Strict import passed on both machines. Re-run with --launch-training to start A/B.\n'
fi
