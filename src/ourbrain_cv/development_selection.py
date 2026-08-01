"""Validation-only model comparison with group-aware paired bootstrap gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import write_json_atomic


def _load_metrics(path: str | Path) -> dict[str, Any]:
    metrics_path = Path(path).expanduser().resolve()
    with metrics_path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if payload.get("evaluation_split") != "val":
        raise ValueError(f"development comparison requires val metrics: {metrics_path}")
    if float(payload.get("threshold", -1.0)) != 0.5:
        raise ValueError(f"development comparison requires threshold 0.5: {metrics_path}")
    rows = payload.get("per_sample")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"development metrics have no per_sample rows: {metrics_path}")
    return payload


def _paired_group_differences(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> np.ndarray:
    baseline_rows = baseline["per_sample"]
    candidate_rows = candidate["per_sample"]
    baseline_by_key = {
        (row.get("image_path"), row.get("mask_path")): row for row in baseline_rows
    }
    candidate_by_key = {
        (row.get("image_path"), row.get("mask_path")): row for row in candidate_rows
    }
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("candidate and baseline validation samples do not match")

    differences_by_group: dict[str, list[float]] = {}
    for key, baseline_row in baseline_by_key.items():
        candidate_row = candidate_by_key[key]
        baseline_group = str(baseline_row.get("group_id") or "")
        candidate_group = str(candidate_row.get("group_id") or "")
        if not baseline_group or candidate_group != baseline_group:
            raise ValueError(f"candidate and baseline group_id mismatch for sample: {key}")
        difference = float(candidate_row["crack_dice"]) - float(
            baseline_row["crack_dice"]
        )
        differences_by_group.setdefault(baseline_group, []).append(difference)

    return np.asarray(
        [np.mean(differences_by_group[group]) for group in sorted(differences_by_group)],
        dtype=np.float64,
    )


def paired_group_bootstrap(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired source-group Dice differences with replacement."""

    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    differences = _paired_group_differences(baseline, candidate)
    if differences.size == 0:
        raise ValueError("development comparison requires at least one source group")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        differences,
        size=(iterations, differences.size),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "groups": int(differences.size),
        "mean_group_dice_difference": float(differences.mean()),
        "paired_group_bootstrap_95_ci": [float(lower), float(upper)],
        "iterations": iterations,
        "seed": seed,
    }


def compare_development_models(
    baseline_path: str | Path,
    candidates: dict[str, str | Path],
    *,
    output_json: str | Path | None = None,
    maximum_recall_drop: float = 0.02,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply the v0.3 development gate without opening the held-out test."""

    if not candidates:
        raise ValueError("at least one development candidate is required")
    if not 0 <= maximum_recall_drop <= 1:
        raise ValueError("maximum_recall_drop must be between 0 and 1")

    baseline_file = Path(baseline_path).expanduser().resolve()
    baseline = _load_metrics(baseline_file)
    records: list[dict[str, Any]] = []
    for candidate_id, candidate_path in candidates.items():
        candidate_file = Path(candidate_path).expanduser().resolve()
        candidate = _load_metrics(candidate_file)
        bootstrap = paired_group_bootstrap(
            baseline,
            candidate,
            iterations=iterations,
            seed=seed,
        )
        dice_improved = float(candidate["crack_dice"]) > float(baseline["crack_dice"])
        ci_excludes_zero = bootstrap["paired_group_bootstrap_95_ci"][0] > 0.0
        recall_drop = float(baseline["recall"]) - float(candidate["recall"])
        recall_gate = recall_drop <= maximum_recall_drop
        records.append(
            {
                "id": candidate_id,
                "metrics": str(candidate_file),
                "crack_dice": float(candidate["crack_dice"]),
                "recall": float(candidate["recall"]),
                "boundary_f1": float(candidate["boundary_f1"]),
                "samples_per_second": float(candidate["samples_per_second"]),
                "dice_difference": float(candidate["crack_dice"])
                - float(baseline["crack_dice"]),
                "recall_drop": recall_drop,
                **bootstrap,
                "gates": {
                    "dice_improved": dice_improved,
                    "group_bootstrap_ci_excludes_zero": ci_excludes_zero,
                    "recall_drop_within_limit": recall_gate,
                },
                "development_gate_passed": bool(
                    dice_improved and ci_excludes_zero and recall_gate
                ),
            }
        )

    eligible = [record for record in records if record["development_gate_passed"]]
    winner = (
        max(
            eligible,
            key=lambda record: (
                record["crack_dice"],
                record["recall"],
                record["boundary_f1"],
                record["samples_per_second"],
            ),
        )["id"]
        if eligible
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "v0.3-development-selection-complete",
        "development_only": True,
        "production_eligible": False,
        "positive_only": True,
        "held_out_test_opened": False,
        "baseline": {
            "id": "v0-positive-only",
            "metrics": str(baseline_file),
            "crack_dice": float(baseline["crack_dice"]),
            "recall": float(baseline["recall"]),
            "boundary_f1": float(baseline["boundary_f1"]),
        },
        "maximum_recall_drop": maximum_recall_drop,
        "candidates": records,
        "selected_development_candidate": winner,
        "retained_model": winner or "v0-positive-only",
        "selection_policy": (
            "require higher aggregate crack Dice, a positive lower bound for the "
            "paired source-group bootstrap 95% CI, and recall drop <= limit; "
            "then maximize Dice, recall, boundary F1, and throughput"
        ),
        "limitations": [
            "No human-reviewed normal images were used.",
            "Operational false-positive specificity is unavailable.",
            "The held-out test split was not opened.",
            "This result cannot promote a model to production.",
        ],
    }
    if output_json is not None:
        output = write_json_atomic(output_json, result)
        result["output_json"] = str(output)
    return result


def _candidate(value: str) -> tuple[str, str]:
    candidate_id, separator, path = value.partition("=")
    if not separator or not candidate_id or not path:
        raise argparse.ArgumentTypeError("candidate must use ID=METRICS_JSON")
    return candidate_id, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-recall-drop", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = compare_development_models(
        args.baseline,
        dict(args.candidate),
        output_json=args.output,
        maximum_recall_drop=args.maximum_recall_drop,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
