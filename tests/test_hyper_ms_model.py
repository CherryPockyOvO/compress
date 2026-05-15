from pathlib import Path

import torch

from compressai_nano import (
    MODEL_VARIANT_HYPER_MS_Q_NANO,
    get_model,
    infer_model_variant_from_checkpoint,
)
from compressai_nano.models import GaussianConditionalEntropy


def test_hyper_ms_forward_shapes() -> None:
    model = get_model(model_variant=MODEL_VARIANT_HYPER_MS_Q_NANO).eval()
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        output = model(x)
        analysis = model.analysis_transform(x)

    assert output["x_hat"].shape == x.shape
    assert len(analysis) == 4
    assert output["scales_y"].shape == output["y"].shape
    assert output["means_y"].shape == output["y"].shape
    assert output["symbols"]["y"].shape == output["y"].shape


def test_gaussian_conditional_centered_quantization() -> None:
    entropy = GaussianConditionalEntropy(quant_step=0.5)
    y = torch.tensor([[[[1.10, -0.10]]]])
    means = torch.tensor([[[[1.00, -0.25]]]])

    symbols = entropy.quantize(y, means)
    y_hat = entropy.dequantize(symbols, means)

    assert torch.equal(symbols, torch.tensor([[[[0, 0]]]], dtype=torch.int32))
    assert torch.allclose(y_hat, means)


def test_hyper_ms_checkpoint_variant_roundtrip(tmp_path: Path) -> None:
    model = get_model(model_variant=MODEL_VARIANT_HYPER_MS_Q_NANO)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_variant": MODEL_VARIANT_HYPER_MS_Q_NANO,
        "model_config": model.model_config_dict(),
    }
    path = tmp_path / "hyper_ms.pt"
    torch.save(checkpoint, path)

    raw = torch.load(path, map_location="cpu")
    assert infer_model_variant_from_checkpoint(raw) == MODEL_VARIANT_HYPER_MS_Q_NANO
