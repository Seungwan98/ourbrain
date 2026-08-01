import pytest
import torch

from ourbrain_cv.losses import (
    CrackSegmentationLoss,
    LossWeights,
    binary_focal_loss,
    boundary_loss,
    crack_probabilities,
    soft_cldice_loss,
    soft_dice_loss,
    soft_tversky_loss,
)


def test_crack_probabilities_supports_two_class_logits():
    logits = torch.tensor([[[[2.0]], [[4.0]]]])
    probs = crack_probabilities(logits)
    assert probs.shape == (1, 1, 1)
    assert torch.allclose(probs, torch.softmax(logits, dim=1)[:, 1])


def test_combined_loss_is_finite_and_backpropagates_for_sparse_line():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[0, 3, 1:7] = 1

    loss = CrackSegmentationLoss()(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_losses_are_finite_for_empty_masks():
    logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)

    losses = [
        binary_focal_loss(logits, labels),
        soft_dice_loss(logits, labels),
        soft_tversky_loss(logits, labels),
        soft_cldice_loss(logits, labels),
        boundary_loss(logits, labels),
    ]
    assert all(torch.isfinite(x) for x in losses)


def test_tversky_and_cldice_prefer_aligned_thin_cracks():
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 3, 1:7] = 1
    aligned = torch.full((1, 2, 8, 8), -8.0)
    aligned[:, 0] = 8.0
    aligned[:, 0, 3, 1:7] = -8.0
    aligned[:, 1, 3, 1:7] = 8.0
    missed = aligned.flip(-2)

    assert soft_tversky_loss(aligned, labels) < soft_tversky_loss(missed, labels)
    assert soft_cldice_loss(aligned, labels) < soft_cldice_loss(missed, labels)


def test_tversky_beta_increases_false_negative_cost() -> None:
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    labels[:, 0, 0] = 1
    missed_positive = torch.tensor(
        [[[[8.0, 8.0], [8.0, 8.0]], [[-8.0, -8.0], [-8.0, -8.0]]]]
    )

    low_fn_cost = soft_tversky_loss(
        missed_positive, labels, alpha=0.3, beta=0.3
    )
    high_fn_cost = soft_tversky_loss(
        missed_positive, labels, alpha=0.3, beta=0.7
    )

    assert high_fn_cost > low_fn_cost


def test_combined_loss_includes_configured_tversky_and_cldice():
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 3, 1:7] = 1
    criterion = CrackSegmentationLoss(
        LossWeights(dice=0.0, tversky=1.0, cldice=0.5, cldice_iterations=2)
    )

    components = criterion.components(logits, labels)
    loss = criterion(logits, labels)
    loss.backward()

    assert {"tversky", "cldice"} <= components.keys()
    assert torch.isfinite(loss)
    assert logits.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression")
def test_focal_loss_supports_cuda_autocast():
    logits = torch.randn(1, 2, 8, 8, device="cuda", requires_grad=True)
    labels = torch.zeros(1, 8, 8, dtype=torch.long, device="cuda")
    labels[:, 3, 1:7] = 1

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = binary_focal_loss(logits, labels)

    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
