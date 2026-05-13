import torch

from train import (
    highlight_aware_loss,
    highlight_quality_metrics,
    highlight_texture_loss,
    rgb_to_luma,
)


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


def test_highlight_peak_under_weight_strengthens_missed_peak_penalty() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0

    missed_highlight = target.clone()
    missed_highlight[:, :, 4:8, 4:8] = 0.0

    default_loss = highlight_aware_loss(missed_highlight, target, peak_under_weight=0.5)
    stronger_loss = highlight_aware_loss(missed_highlight, target, peak_under_weight=1.5)

    assert stronger_loss > default_loss


def test_highlight_lap_weight_strengthens_edge_penalty() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0

    blurred = torch.zeros_like(target)
    blurred[:, :, 5:7, 5:7] = 1.0

    default_loss = highlight_aware_loss(blurred, target, lap_weight=0.5)
    stronger_loss = highlight_aware_loss(blurred, target, lap_weight=1.0)

    assert stronger_loss > default_loss


def test_highlight_texture_weights_strengthen_detail_terms() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0
    smoothed = torch.zeros_like(target)
    smoothed[:, :, 5:7, 5:7] = 0.75

    weaker_loss = highlight_texture_loss(
        smoothed,
        target,
        texture_lap_weight=0.8,
        texture_contrast_weight=0.3,
    )
    stronger_loss = highlight_texture_loss(
        smoothed,
        target,
        texture_lap_weight=1.2,
        texture_contrast_weight=0.5,
    )

    assert stronger_loss > weaker_loss


def test_highlight_quality_metrics_track_peak_under_reconstruction() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0

    missed_highlight = target.clone()
    missed_highlight[:, :, 4:8, 4:8] = 0.25

    perfect_peak_under, perfect_lap, perfect_contrast = highlight_quality_metrics(target, target)
    missed_peak_under, missed_lap, missed_contrast = highlight_quality_metrics(
        missed_highlight,
        target,
    )

    assert perfect_peak_under.item() == 0.0
    assert missed_peak_under > perfect_peak_under
    assert missed_lap > perfect_lap
    assert missed_contrast > perfect_contrast
