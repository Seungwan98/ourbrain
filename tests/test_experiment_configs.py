from pathlib import Path

from ourbrain_cv.config import load_config

ROOT = Path(__file__).resolve().parents[1]
A_CONFIG = ROOT / "configs" / "v0_2_a_baseline_with_negatives.yaml"
B_CONFIG = ROOT / "configs" / "v0_2_b_recall_with_negatives.yaml"
WINDOWS_A_CONFIG = (
    ROOT / "configs" / "v0_2_a_baseline_with_negatives_cuda.yaml"
)
WINDOWS_B_CONFIG = (
    ROOT / "configs" / "v0_2_b_recall_with_negatives_cuda.yaml"
)
DEV_A_CONFIG = (
    ROOT / "configs" / "v0_2_dev_a_augmentation_positive_only.yaml"
)
DEV_B_CONFIG = (
    ROOT / "configs" / "v0_2_dev_b_augmentation_recall_positive_only.yaml"
)
WINDOWS_DEV_A_CONFIG = (
    ROOT / "configs" / "v0_2_dev_a_augmentation_positive_only_cuda.yaml"
)
WINDOWS_DEV_B_CONFIG = (
    ROOT / "configs" / "v0_2_dev_b_augmentation_recall_positive_only_cuda.yaml"
)
V03_CONFIGS = (
    ROOT / "configs" / "v0_3_a_upernet_swin_tiny_positive_only.yaml",
    ROOT / "configs" / "v0_3_b_segformer_b1_positive_only.yaml",
    ROOT / "configs" / "v0_3_c_segformer_b2_positive_only.yaml",
)
WINDOWS_V03_CONFIGS = (
    ROOT / "configs" / "v0_3_a_upernet_swin_tiny_positive_only_cuda.yaml",
    ROOT / "configs" / "v0_3_b_segformer_b1_positive_only_cuda.yaml",
    ROOT / "configs" / "v0_3_c_segformer_b2_positive_only_cuda.yaml",
)


def test_v0_2_configs_share_controlled_training_budget() -> None:
    baseline = load_config(A_CONFIG)
    recall = load_config(B_CONFIG)

    assert baseline["seed"] == recall["seed"] == 42
    assert baseline["model"] == recall["model"]
    assert baseline["data"]["manifest"] == recall["data"]["manifest"]
    assert baseline["data"]["augmentation"] == recall["data"]["augmentation"]

    controlled_data_fields = (
        "image_size",
        "mask_threshold",
        "synthetic_negative_probability",
        "synthetic_negative_crop_size",
        "crack_centered_crop_size",
    )
    for field in controlled_data_fields:
        assert baseline["data"][field] == recall["data"][field]

    controlled_training_fields = (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "mixed_precision",
        "freeze_batch_norm",
        "freeze_backbone_epochs",
        "early_stopping_patience",
        "lr_scheduler",
        "warmup_ratio",
        "minimum_learning_rate_ratio",
        "save_last_checkpoint",
    )
    for field in controlled_training_fields:
        assert baseline["training"][field] == recall["training"][field]

    assert baseline["inference"] == recall["inference"]
    assert baseline["training"]["output_dir"] != recall["training"]["output_dir"]


def test_v0_2_configs_use_controlled_thin_crack_augmentation() -> None:
    for path in (A_CONFIG, B_CONFIG):
        augmentation = load_config(path)["data"]["augmentation"]
        assert augmentation == {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.25,
            "brightness_jitter": 0.2,
            "contrast_jitter": 0.2,
            "rotation_degrees": 8.0,
            "gamma_jitter": 0.15,
            "gaussian_blur_probability": 0.15,
            "gaussian_blur_radius": 1.0,
            "gaussian_noise_probability": 0.2,
            "gaussian_noise_std": 0.015,
        }


def test_v0_2_configs_only_enable_recall_experiment_features_in_b() -> None:
    baseline = load_config(A_CONFIG)
    recall = load_config(B_CONFIG)

    assert baseline["data"]["crack_centered_probability"] == 0.0
    assert recall["data"]["crack_centered_probability"] == 0.5

    assert baseline["training"]["focal_weight"] == 1.0
    assert baseline["training"]["dice_weight"] == 1.0
    assert baseline["training"]["tversky_weight"] == 0.0
    assert baseline["training"]["cldice_weight"] == 0.0

    assert recall["training"]["focal_weight"] == 0.5
    assert recall["training"]["dice_weight"] == 0.0
    assert recall["training"]["tversky_weight"] == 1.0
    assert recall["training"]["cldice_weight"] == 0.5


def test_v0_2_configs_require_reviewed_negative_manifest() -> None:
    for path in (A_CONFIG, B_CONFIG):
        config = load_config(path)
        assert config["data"]["manifest"] == "artifacts/manifest_with_negatives.csv"
        assert config["training"]["save_last_checkpoint"] is True


def test_windows_v0_2_configs_only_materialize_platform_paths() -> None:
    path_fields = (
        ("model", "checkpoint"),
        ("data", "manifest"),
        ("training", "output_dir"),
    )
    for portable_path, windows_path in (
        (A_CONFIG, WINDOWS_A_CONFIG),
        (B_CONFIG, WINDOWS_B_CONFIG),
    ):
        portable = load_config(portable_path)
        windows = load_config(windows_path)
        for section, field in path_fields:
            portable[section].pop(field)
            windows[section].pop(field)
        assert windows == portable

    windows_a = load_config(WINDOWS_A_CONFIG)
    windows_b = load_config(WINDOWS_B_CONFIG)
    assert windows_a["model"]["checkpoint"] == (
        r"D:\ourbrain\runs\v0-positive-only\checkpoint"
    )
    assert windows_a["data"]["manifest"] == (
        r"D:\ourbrain\artifacts\manifest_with_negatives.csv"
    )
    assert windows_a["training"]["output_dir"].endswith(
        r"v0.2-a-baseline-with-negatives\checkpoint"
    )
    assert windows_b["training"]["output_dir"].endswith(
        r"v0.2-b-recall-with-negatives\checkpoint"
    )


def test_v0_2_dev_configs_are_positive_only_and_isolated_from_final_runs() -> None:
    final_a = load_config(A_CONFIG)
    final_b = load_config(B_CONFIG)
    dev_a = load_config(DEV_A_CONFIG)
    dev_b = load_config(DEV_B_CONFIG)

    assert dev_a["data"]["manifest"] == dev_b["data"]["manifest"] == (
        "artifacts/manifest.csv"
    )
    assert dev_a["data"]["augmentation"] == final_a["data"]["augmentation"]
    assert dev_b["data"]["augmentation"] == final_b["data"]["augmentation"]
    assert dev_a["data"]["crack_centered_probability"] == 0.0
    assert dev_b["data"]["crack_centered_probability"] == 0.5
    assert dev_a["training"] == {
        **final_a["training"],
        "output_dir": "checkpoints/v0.2-dev-a-augmentation-positive-only",
    }
    assert dev_b["training"] == {
        **final_b["training"],
        "output_dir": "checkpoints/v0.2-dev-b-augmentation-recall-positive-only",
    }
    assert {
        dev_a["training"]["output_dir"],
        dev_b["training"]["output_dir"],
    }.isdisjoint(
        {
            final_a["training"]["output_dir"],
            final_b["training"]["output_dir"],
        }
    )


def test_windows_v0_2_dev_configs_only_materialize_platform_paths() -> None:
    path_fields = (
        ("model", "checkpoint"),
        ("data", "manifest"),
        ("training", "output_dir"),
    )
    for portable_path, windows_path in (
        (DEV_A_CONFIG, WINDOWS_DEV_A_CONFIG),
        (DEV_B_CONFIG, WINDOWS_DEV_B_CONFIG),
    ):
        portable = load_config(portable_path)
        windows = load_config(windows_path)
        for section, field in path_fields:
            portable[section].pop(field)
            windows[section].pop(field)
        assert windows == portable

    windows_a = load_config(WINDOWS_DEV_A_CONFIG)
    windows_b = load_config(WINDOWS_DEV_B_CONFIG)
    assert windows_a["data"]["manifest"] == r"D:\ourbrain\artifacts\manifest.csv"
    assert windows_a["training"]["output_dir"].endswith(
        r"v0.2-dev-a-augmentation-positive-only\checkpoint"
    )
    assert windows_b["training"]["output_dir"].endswith(
        r"v0.2-dev-b-augmentation-recall-positive-only\checkpoint"
    )


def test_v0_3_configs_hold_every_non_model_variable_constant() -> None:
    configs = [load_config(path) for path in V03_CONFIGS]
    reference = configs[0]
    for candidate in configs[1:]:
        assert candidate["seed"] == reference["seed"] == 42
        assert candidate["data"] == reference["data"]
        assert candidate["inference"] == reference["inference"]
        reference_training = {
            key: value
            for key, value in reference["training"].items()
            if key != "output_dir"
        }
        candidate_training = {
            key: value
            for key, value in candidate["training"].items()
            if key != "output_dir"
        }
        assert candidate_training == reference_training

    assert [config["model"]["architecture"] for config in configs] == [
        "upernet",
        "segformer",
        "segformer",
    ]
    assert len({config["model"]["checkpoint"] for config in configs}) == 3
    assert len({config["training"]["output_dir"] for config in configs}) == 3


def test_v0_3_configs_use_recall_corrected_topology_loss() -> None:
    for path in V03_CONFIGS:
        config = load_config(path)
        assert config["data"]["manifest"] == "artifacts/manifest.csv"
        assert config["data"]["crack_centered_probability"] == 0.5
        assert config["training"]["epochs"] == 30
        assert config["training"]["focal_weight"] == 1.0
        assert config["training"]["dice_weight"] == 1.0
        assert config["training"]["boundary_weight"] == 0.25
        assert config["training"]["tversky_weight"] == 0.25
        assert config["training"]["tversky_alpha"] == 0.3
        assert config["training"]["tversky_beta"] == 0.7
        assert config["training"]["cldice_weight"] == 0.15


def test_windows_v0_3_configs_only_materialize_platform_paths() -> None:
    for portable_path, windows_path in zip(
        V03_CONFIGS, WINDOWS_V03_CONFIGS, strict=True
    ):
        portable = load_config(portable_path)
        windows = load_config(windows_path)
        assert windows["data"].pop("manifest") == (
            r"D:\ourbrain\artifacts\manifest_windows.csv"
        )
        assert windows["training"].pop("output_dir").startswith(r"D:\ourbrain\runs\v0.3-")
        portable["data"].pop("manifest")
        portable["training"].pop("output_dir")
        assert windows == portable
