from __future__ import annotations

import json
from pathlib import Path

import pytest

from ourbrain_cv.model_selection import select_calibrated_model
from ourbrain_cv.provenance import checkpoint_sha256, sha256_file


def _candidate(
    root: Path,
    candidate_id: str,
    manifest: Path,
    *,
    recall: float,
    specificity: float,
    dice: float,
    threshold: float,
    boundary_score: float = 0.8,
    boundary_tolerance: int = 2,
    constraint_met: bool = True,
) -> dict[str, str]:
    config = root / f"{candidate_id}.yaml"
    config.write_text(f"id: {candidate_id}\n", encoding="utf-8")
    checkpoint = root / f"{candidate_id}-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(candidate_id.encode())
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    review_audit = manifest.with_suffix(".review.json")
    if not review_audit.exists():
        review_audit.write_text('{"complete": true}', encoding="utf-8")
    calibration = root / f"{candidate_id}-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "minimum_image_recall": 0.95,
                "recall_constraint_met": constraint_met,
                "selected_threshold": threshold,
                "boundary_tolerance": boundary_tolerance,
                "selected_metrics": {
                    "threshold": threshold,
                    "image_level_recall": recall,
                    "image_level_specificity": specificity,
                    "crack_dice": dice,
                    "boundary_f1": boundary_score,
                },
                "provenance": {
                    "selection_split": "val",
                    "config": str(config.resolve()),
                    "config_sha256": sha256_file(config),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha256(checkpoint),
                    "manifest": str(manifest.resolve()),
                    "manifest_sha256": sha256_file(manifest),
                    "review_audit": str(review_audit.resolve()),
                    "review_audit_sha256": sha256_file(review_audit),
                    "positive_only_override": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "id": candidate_id,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "calibration": str(calibration),
    }


def _descriptor(path: Path, candidates: list[dict[str, str]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "candidates": candidates}),
        encoding="utf-8",
    )
    return path


def test_select_calibrated_model_uses_recall_then_specificity_policy(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    baseline = _candidate(
        tmp_path,
        "baseline",
        manifest,
        recall=0.98,
        specificity=0.80,
        dice=0.30,
        threshold=0.5,
    )
    recall_model = _candidate(
        tmp_path,
        "recall",
        manifest,
        recall=0.95,
        specificity=0.90,
        dice=0.25,
        threshold=0.6,
    )
    candidates = _descriptor(
        tmp_path / "candidates.json",
        [baseline, recall_model],
    )
    output = tmp_path / "selection.json"

    result = select_calibrated_model(candidates, output)

    assert result["winner"]["id"] == "recall"
    assert result["selection_split"] == "val"
    assert result["manifest_sha256"] == sha256_file(manifest)
    assert result["review_audit_sha256"] == sha256_file(
        manifest.with_suffix(".review.json")
    )
    assert result["boundary_tolerance"] == 2
    assert result["candidate_descriptor_sha256"] == sha256_file(candidates)
    assert json.loads(output.read_text(encoding="utf-8"))["winner"]["id"] == "recall"
    assert not (tmp_path / "selection.json.tmp").exists()


def test_select_calibrated_model_excludes_unmet_recall_constraint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    eligible = _candidate(
        tmp_path,
        "eligible",
        manifest,
        recall=0.95,
        specificity=0.5,
        dice=0.2,
        threshold=0.4,
    )
    ineligible = _candidate(
        tmp_path,
        "ineligible",
        manifest,
        recall=0.99,
        specificity=1.0,
        dice=0.9,
        threshold=0.9,
        constraint_met=False,
    )

    result = select_calibrated_model(
        _descriptor(tmp_path / "candidates.json", [eligible, ineligible]),
        tmp_path / "selection.json",
    )

    assert result["winner"]["id"] == "eligible"
    assert result["candidates"][1]["recall_constraint_met"] is False


def test_select_calibrated_model_rejects_checkpoint_tampering(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    candidate = _candidate(
        tmp_path,
        "candidate",
        manifest,
        recall=0.95,
        specificity=0.8,
        dice=0.3,
        threshold=0.5,
    )
    (Path(candidate["checkpoint"]) / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        select_calibrated_model(
            _descriptor(tmp_path / "candidates.json", [candidate]),
            tmp_path / "selection.json",
        )


def test_select_calibrated_model_fails_when_no_candidate_is_eligible(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    candidate = _candidate(
        tmp_path,
        "candidate",
        manifest,
        recall=0.94,
        specificity=0.9,
        dice=0.3,
        threshold=0.5,
    )

    with pytest.raises(RuntimeError, match="no candidate met"):
        select_calibrated_model(
            _descriptor(tmp_path / "candidates.json", [candidate]),
            tmp_path / "selection.json",
        )


def test_select_calibrated_model_uses_boundary_f1_after_dice(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    lower_boundary = _candidate(
        tmp_path,
        "lower-boundary",
        manifest,
        recall=0.95,
        specificity=0.9,
        dice=0.4,
        boundary_score=0.7,
        threshold=0.7,
    )
    higher_boundary = _candidate(
        tmp_path,
        "higher-boundary",
        manifest,
        recall=0.95,
        specificity=0.9,
        dice=0.4,
        boundary_score=0.8,
        threshold=0.5,
    )

    result = select_calibrated_model(
        _descriptor(
            tmp_path / "candidates.json",
            [lower_boundary, higher_boundary],
        ),
        tmp_path / "selection.json",
    )

    assert result["winner"]["id"] == "higher-boundary"


def test_select_calibrated_model_rejects_mixed_boundary_tolerances(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    first = _candidate(
        tmp_path,
        "first",
        manifest,
        recall=0.95,
        specificity=0.8,
        dice=0.3,
        threshold=0.5,
        boundary_tolerance=2,
    )
    second = _candidate(
        tmp_path,
        "second",
        manifest,
        recall=0.95,
        specificity=0.8,
        dice=0.3,
        threshold=0.5,
        boundary_tolerance=3,
    )

    with pytest.raises(ValueError, match="same boundary_tolerance"):
        select_calibrated_model(
            _descriptor(tmp_path / "candidates.json", [first, second]),
            tmp_path / "selection.json",
        )
