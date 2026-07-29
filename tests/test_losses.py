import pytest
import torch

from ourbrain_cv.losses import (
    CrackSegmentationLoss,
    binary_focal_loss,
    boundary_loss,
    crack_probabilities,
    soft_dice_loss,
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
        boundary_loss(logits, labels),
    ]
    assert all(torch.isfinite(x) for x in losses)


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
