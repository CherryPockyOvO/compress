from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import get_model, get_model_config, infer_model_variant_from_checkpoint


class AnalysisExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not hasattr(self.model, "analysis_transform"):
            y = self.model.encoder(x)
            zeros = y.new_zeros(1)
            return y, zeros, zeros
        if all(
            hasattr(self.model, name)
            for name in ("encoder", "hyper_encoder", "entropy_bottleneck_z", "hyper_decoder")
        ):
            y = self.model.encoder(x)
            z = self.model.hyper_encoder(y)
            z_symbols = self.model.entropy_bottleneck_z.quantize(z)
            z_hat = self.model.entropy_bottleneck_z.dequantize(
                z_symbols,
                dtype=z.dtype,
                device=z.device,
            )
            hyper = self.model.hyper_decoder(z_hat)
            if isinstance(hyper, tuple):
                return (y, z, *hyper)
            return y, z, hyper
        return self.model.analysis_transform(x)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    if missing:
        print("\n".join(f"  {key}" for key in missing[:20]))
    print(f"unexpected_keys: {len(unexpected)}")
    if unexpected:
        print("\n".join(f"  {key}" for key in unexpected[:20]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export encoder or analysis transform to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--export-mode",
        choices=("encoder", "analysis"),
        default="encoder",
        help="encoder exports image->y. analysis exports image->(y,z,scales_y[,means_y]) when supported.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    config = get_model_config(model_variant)
    model = get_model(model_variant=model_variant).eval()
    load_checkpoint(model, args.checkpoint)
    dummy = torch.randn(1, 3, args.height, args.width)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "latent": {0: "batch", 2: "latent_height", 3: "latent_width"},
        }
        if args.export_mode == "analysis":
            dynamic_axes["hyper_latent"] = {0: "batch", 2: "hyper_height", 3: "hyper_width"}
            dynamic_axes["scales_y"] = {0: "batch", 2: "latent_height", 3: "latent_width"}
            if config.model_type == "mean_scale_hyperprior":
                dynamic_axes["means_y"] = {0: "batch", 2: "latent_height", 3: "latent_width"}
    export_module: nn.Module = model.encoder
    output_names = ["latent"]
    if args.export_mode == "analysis":
        export_module = AnalysisExportWrapper(model)
        output_names = ["latent", "hyper_latent", "scales_y"]
        if config.model_type == "mean_scale_hyperprior":
            output_names.append("means_y")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_module,
        dummy,
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    with torch.no_grad():
        exported = export_module(dummy)
        latent = exported[0] if isinstance(exported, tuple) else exported
    print(f"encoder input : {tuple(dummy.shape)}")
    print(f"encoder output: {tuple(latent.shape)}")
    print(f"model_variant : {model_variant}")
    print(f"export_mode   : {args.export_mode}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
