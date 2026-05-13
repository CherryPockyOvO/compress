import torch

from train import highlight_aware_loss, rgb_to_luma


def test_rgb_to_luma_uses_bt601_coefficients() -> None:
    x = torch.tensor([[[[1.0]], [[0.5]], [[0.25]]]])

    expected = 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.25

    assert torch.allclose(rgb_to_luma(x), torch.tensor([[[[expected]]]]))


def test_highlight_aware_loss_focuses_target_highlights() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0

    missed_highlight = target.clone()
    missed_highlight[:, :, 4:8, 4:8] = 0.0

    false_dark_region_detail = target.clone()
    false_dark_region_detail[:, :, 10:14, 10:14] = 1.0

    missed_loss = highlight_aware_loss(missed_highlight, target)
    dark_region_loss = highlight_aware_loss(false_dark_region_detail, target)

    assert missed_loss > dark_region_loss * 20.0
    assert highlight_aware_loss(target, target).item() == 0.0
