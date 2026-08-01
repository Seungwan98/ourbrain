import json
from pathlib import Path

import pytest

from ourbrain_cv.development_selection import compare_development_models


def _metrics(path: Path, *, dice_shift: float, recall: float = 0.6) -> Path:
    rows = []
    for index, (group, dice) in enumerate(
        (("a", 0.2), ("a", 0.3), ("b", 0.4), ("b", 0.5))
    ):
        rows.append(
            {
                "sample_index": index,
                "image_path": f"image-{index}.png",
                "mask_path": f"mask-{index}.png",
                "group_id": group,
                "crack_dice": dice + dice_shift,
            }
        )
    payload = {
        "evaluation_split": "val",
        "threshold": 0.5,
        "crack_dice": 0.35 + dice_shift,
        "recall": recall,
        "boundary_f1": 0.7 + dice_shift,
        "samples_per_second": 5.0,
        "per_sample": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_development_selection_promotes_only_significant_recall_safe_candidate(
    tmp_path: Path,
) -> None:
    baseline = _metrics(tmp_path / "baseline.json", dice_shift=0.0)
    winning = _metrics(tmp_path / "winning.json", dice_shift=0.1, recall=0.59)
    recall_regression = _metrics(
        tmp_path / "recall-regression.json", dice_shift=0.2, recall=0.55
    )

    result = compare_development_models(
        baseline,
        {"winning": winning, "recall-regression": recall_regression},
        iterations=500,
        output_json=tmp_path / "selection.json",
    )

    assert result["selected_development_candidate"] == "winning"
    assert result["retained_model"] == "winning"
    assert result["production_eligible"] is False
    assert result["held_out_test_opened"] is False
    assert (tmp_path / "selection.json").is_file()


def test_development_selection_retains_baseline_when_ci_crosses_zero(
    tmp_path: Path,
) -> None:
    baseline = _metrics(tmp_path / "baseline.json", dice_shift=0.0)
    tied = _metrics(tmp_path / "tied.json", dice_shift=0.0)

    result = compare_development_models(
        baseline,
        {"tied": tied},
        iterations=100,
    )

    assert result["selected_development_candidate"] is None
    assert result["retained_model"] == "v0-positive-only"


def test_development_selection_rejects_sample_mismatch(tmp_path: Path) -> None:
    baseline = _metrics(tmp_path / "baseline.json", dice_shift=0.0)
    candidate = _metrics(tmp_path / "candidate.json", dice_shift=0.1)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["per_sample"][0]["image_path"] = "different.png"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="samples do not match"):
        compare_development_models(baseline, {"candidate": candidate})
