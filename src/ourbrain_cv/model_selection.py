"""Validation-only selection of a calibrated segmentation checkpoint."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ourbrain_cv.provenance import (
    checkpoint_sha256,
    sha256_file,
    write_json_atomic,
)


def _finite_metric(payload: dict[str, Any], field: str, candidate_id: str) -> float:
    try:
        value = float(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate {candidate_id} has invalid calibration metric: {field}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"candidate {candidate_id} has non-finite calibration metric: {field}"
        )
    return value


def _unit_metric(payload: dict[str, Any], field: str, candidate_id: str) -> float:
    value = _finite_metric(payload, field, candidate_id)
    if not 0 <= value <= 1:
        raise ValueError(
            f"candidate {candidate_id} calibration metric is outside [0, 1]: {field}"
        )
    return value


def _resolved_existing(path_value: Any, kind: str, candidate_id: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"candidate {candidate_id} has no {kind} path")
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"candidate {candidate_id} {kind} does not exist: {path}")
    return path


def select_calibrated_model(
    candidates_json: str | Path,
    output_json: str | Path,
    *,
    minimum_image_recall: float = 0.95,
) -> dict[str, Any]:
    """Select by recall eligibility, specificity, Dice, boundary F1, then threshold."""

    if not 0 <= minimum_image_recall <= 1:
        raise ValueError("minimum_image_recall must be between 0 and 1")
    descriptor_path = Path(candidates_json).expanduser().resolve()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != 1:
        raise ValueError("candidate descriptor has unsupported schema_version")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate descriptor must contain a non-empty candidates list")

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    common_manifest: tuple[str, str] | None = None
    common_review_audit: tuple[str, str] | None = None
    common_boundary_tolerance: int | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate descriptor entries must be objects")
        candidate_id = str(candidate.get("id", "")).strip()
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError(f"candidate id is blank or duplicated: {candidate_id!r}")
        seen_ids.add(candidate_id)
        config = _resolved_existing(candidate.get("config"), "config", candidate_id)
        checkpoint = _resolved_existing(
            candidate.get("checkpoint"), "checkpoint", candidate_id
        )
        calibration_path = _resolved_existing(
            candidate.get("calibration"), "calibration", candidate_id
        )
        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8-sig")
        )
        if calibration.get("schema_version") != 1:
            raise ValueError(
                f"candidate {candidate_id} calibration has unsupported schema_version"
            )
        provenance = calibration.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"candidate {candidate_id} has no calibration provenance")
        if provenance.get("selection_split") != "val":
            raise ValueError(
                f"candidate {candidate_id} threshold was not selected on val"
            )
        if Path(str(provenance.get("config", ""))).resolve() != config:
            raise ValueError(f"candidate {candidate_id} calibration config mismatch")
        if provenance.get("config_sha256") != sha256_file(config):
            raise ValueError(f"candidate {candidate_id} config hash mismatch")
        if Path(str(provenance.get("checkpoint", ""))).resolve() != checkpoint:
            raise ValueError(f"candidate {candidate_id} calibration checkpoint mismatch")
        if provenance.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
            raise ValueError(f"candidate {candidate_id} checkpoint hash mismatch")
        manifest = _resolved_existing(
            provenance.get("manifest"), "manifest", candidate_id
        )
        manifest_hash = sha256_file(manifest)
        if provenance.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"candidate {candidate_id} manifest hash mismatch")
        current_manifest = (str(manifest), manifest_hash)
        if common_manifest is None:
            common_manifest = current_manifest
        elif common_manifest != current_manifest:
            raise ValueError("candidate calibrations do not use the same manifest")
        if provenance.get("positive_only_override") is True:
            raise ValueError(
                f"candidate {candidate_id} used the positive-only calibration override"
            )
        review_audit = _resolved_existing(
            provenance.get("review_audit"), "review audit", candidate_id
        )
        review_audit_hash = sha256_file(review_audit)
        if provenance.get("review_audit_sha256") != review_audit_hash:
            raise ValueError(f"candidate {candidate_id} review audit hash mismatch")
        current_review_audit = (str(review_audit), review_audit_hash)
        if common_review_audit is None:
            common_review_audit = current_review_audit
        elif common_review_audit != current_review_audit:
            raise ValueError("candidate calibrations do not use the same review audit")
        raw_boundary_tolerance = calibration.get("boundary_tolerance")
        if (
            isinstance(raw_boundary_tolerance, bool)
            or not isinstance(raw_boundary_tolerance, int)
            or raw_boundary_tolerance < 0
        ):
            raise ValueError(
                f"candidate {candidate_id} has invalid boundary_tolerance"
            )
        if common_boundary_tolerance is None:
            common_boundary_tolerance = raw_boundary_tolerance
        elif common_boundary_tolerance != raw_boundary_tolerance:
            raise ValueError(
                "candidate calibrations do not use the same boundary_tolerance"
            )

        calibrated_recall_floor = _unit_metric(
            calibration, "minimum_image_recall", candidate_id
        )
        if calibrated_recall_floor != minimum_image_recall:
            raise ValueError(
                f"candidate {candidate_id} recall floor "
                f"{calibrated_recall_floor} != required {minimum_image_recall}"
            )
        metrics = calibration.get("selected_metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"candidate {candidate_id} has no selected_metrics")
        image_recall = _unit_metric(
            metrics, "image_level_recall", candidate_id
        )
        image_specificity = _unit_metric(
            metrics, "image_level_specificity", candidate_id
        )
        crack_dice = _unit_metric(metrics, "crack_dice", candidate_id)
        boundary_score = _unit_metric(metrics, "boundary_f1", candidate_id)
        threshold = _unit_metric(calibration, "selected_threshold", candidate_id)
        metric_threshold = _unit_metric(metrics, "threshold", candidate_id)
        if threshold != metric_threshold:
            raise ValueError(
                f"candidate {candidate_id} selected threshold does not match its metrics"
            )
        constraint_met = (
            calibration.get("recall_constraint_met") is True
            and image_recall >= minimum_image_recall
        )
        records.append(
            {
                "id": candidate_id,
                "config": str(config),
                "checkpoint": str(checkpoint),
                "calibration": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "boundary_tolerance": raw_boundary_tolerance,
                "recall_constraint_met": constraint_met,
                "threshold": threshold,
                "metrics": metrics,
                "_rank": (
                    image_specificity,
                    crack_dice,
                    boundary_score,
                    threshold,
                ),
            }
        )

    eligible = [record for record in records if record["recall_constraint_met"]]
    if not eligible:
        raise RuntimeError(
            f"no candidate met validation image recall >= {minimum_image_recall}"
        )
    winner = max(eligible, key=lambda record: record["_rank"])
    public_records = [
        {key: value for key, value in record.items() if key != "_rank"}
        for record in records
    ]
    public_winner = {key: value for key, value in winner.items() if key != "_rank"}
    result = {
        "schema_version": 1,
        "selection_split": "val",
        "minimum_image_recall": minimum_image_recall,
        "policy": (
            "eligible recall constraint, then image specificity, crack Dice, "
            "boundary F1, and higher threshold"
        ),
        "candidate_descriptor": str(descriptor_path),
        "candidate_descriptor_sha256": sha256_file(descriptor_path),
        "manifest": common_manifest[0] if common_manifest else None,
        "manifest_sha256": common_manifest[1] if common_manifest else None,
        "review_audit": common_review_audit[0] if common_review_audit else None,
        "review_audit_sha256": (
            common_review_audit[1] if common_review_audit else None
        ),
        "boundary_tolerance": common_boundary_tolerance,
        "winner": public_winner,
        "candidates": public_records,
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(output_json, result)
    return result


__all__ = ["select_calibrated_model"]
