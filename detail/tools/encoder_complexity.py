from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import MODEL_CONFIGS, MODEL_VARIANT_HYPER_MS_Q, MODEL_VARIANT_NANO, get_model


@dataclass
class OpStats:
    name: str
    module_type: str
    output_shape: tuple[int, ...]
    params: int
    macs: int

    @property
    def flops(self) -> int:
        return 2 * self.macs


class AnalysisWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not hasattr(self.model, "analysis_transform"):
            y = self.model.encoder(x)
            zeros = y.new_zeros(1)
            return y, zeros, zeros
        return self.model.analysis_transform(x)


def count_module_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_channels, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels // module.groups
    return int(batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w)


def conv_transpose2d_macs(module: nn.ConvTranspose2d, input_tensor: torch.Tensor) -> int:
    batch, in_channels, in_h, in_w = input_tensor.shape
    kernel_h, kernel_w = module.kernel_size
    out_channels = module.out_channels // module.groups
    return int(batch * in_channels * in_h * in_w * out_channels * kernel_h * kernel_w)


def profile_module(module: nn.Module, dummy: torch.Tensor) -> tuple[list[OpStats], Any]:
    stats: list[OpStats] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(layer: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output):
                return
            params = count_module_params(layer)
            if isinstance(layer, nn.Conv2d):
                macs = conv2d_macs(layer, output)
            elif isinstance(layer, nn.ConvTranspose2d):
                macs = conv_transpose2d_macs(layer, inputs[0])
            else:
                return
            stats.append(
                OpStats(
                    name=name,
                    module_type=layer.__class__.__name__,
                    output_shape=tuple(int(value) for value in output.shape),
                    params=params,
                    macs=macs,
                )
            )

        return hook

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(layer.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        output = module(dummy)

    for handle in handles:
        handle.remove()

    return stats, output


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}"
    return f"{value:,}"


def print_summary(name: str, stats: list[OpStats], output: Any, input_shape: tuple[int, ...]) -> None:
    params = sum(item.params for item in stats)
    macs = sum(item.macs for item in stats)
    flops = 2 * macs
    fp16_bytes = params * 2
    int8_bytes = params
    print(f"[{name}]")
    print(f"input_shape: {input_shape}")
    if torch.is_tensor(output):
        print(f"output_shape: {tuple(int(value) for value in output.shape)}")
    elif isinstance(output, tuple):
        output_shapes = [
            tuple(int(value) for value in item.shape) if torch.is_tensor(item) else str(type(item))
            for item in output
        ]
        print(f"output_shapes: {output_shapes}")
    print(f"conv_params: {format_number(params)}")
    print(f"fp16_param_size_mib: {fp16_bytes / 1024 / 1024:.3f}")
    print(f"int8_param_size_mib: {int8_bytes / 1024 / 1024:.3f}")
    print(f"macs: {format_number(macs)}")
    print(f"flops: {format_number(flops)}")
    print(f"gmacs: {macs / 1e9:.3f}")
    print(f"gflops: {flops / 1e9:.3f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count Conv/Deconv params and FLOPs for encoder-side networks. "
            "FLOPs are reported as 2 * MACs."
        )
    )
    parser.add_argument(
        "--model-variant",
        choices=tuple(MODEL_CONFIGS.keys()),
        default=MODEL_VARIANT_HYPER_MS_Q,
    )
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("encoder", "analysis", "both"),
        default="both",
        help="encoder counts image->y. analysis counts image->(y,z,scales_y[,means_y]).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-layer Conv/Deconv stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = get_model(model_variant=args.model_variant).eval()
    dummy = torch.randn(args.batch_size, 3, args.height, args.width)
    input_shape = tuple(int(value) for value in dummy.shape)

    modules: list[tuple[str, nn.Module]] = []
    if args.mode in {"encoder", "both"}:
        modules.append(("encoder_y", model.encoder))
    if args.mode in {"analysis", "both"}:
        config = MODEL_CONFIGS[args.model_variant]
        analysis_name = (
            "analysis_y_z_scales_means"
            if config.model_type == "mean_scale_hyperprior"
            else "analysis_y_z_scales"
        )
        if args.model_variant == MODEL_VARIANT_NANO:
            analysis_name = "analysis_y_only"
        modules.append((analysis_name, AnalysisWrapper(model)))

    for name, module in modules:
        stats, output = profile_module(module, dummy)
        print_summary(name, stats, output, input_shape)
        if args.details:
            for item in stats:
                print(
                    f"{item.name:36s} {item.module_type:15s} "
                    f"out={item.output_shape!s:22s} "
                    f"params={item.params:10d} macs={item.macs:16d}"
                )
            print()


if __name__ == "__main__":
    main()
