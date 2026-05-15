from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import get_model, get_model_config, infer_model_variant_from_checkpoint


def load_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    if missing:
        print("\n".join(f"  {key}" for key in missing[:20]))
    print(f"unexpected_keys: {len(unexpected)}")
    if unexpected:
        print("\n".join(f"  {key}" for key in unexpected[:20]))
    return raw if isinstance(raw, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export entropy parameters for C++ CNZ encoding.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("entropy_params.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).eval()
    load_checkpoint(model, args.checkpoint)
    config = get_model_config(model_variant)
    if getattr(model, "supports_cnz_v4", False):
        entropy = model.entropy_bottleneck
        medians = entropy.medians.detach().cpu().to(torch.float32).tolist()
        payload = {
            "format": "compressai-nano-entropy-params-v1",
            "model_variant": model_variant,
            "channels": int(model.M),
            "quant_step": float(entropy.quant_step.detach().cpu()),
            "downsampling_factor": int(model.downsampling_factor),
            "model_config_name": config.name,
            "medians": medians,
        }
    else:
        z_entropy = model.entropy_bottleneck_z
        payload = {
            "format": "compressai-nano-hyper-entropy-params-v1",
            "model_variant": model_variant,
            "cnz4_supported": False,
            "note": (
                "Training/export parameters only. CNZ4 does not support hyperprior; "
                "a CNZ5 bitstream must carry z stream, y stream, hyperprior shape, "
                "and model_variant."
            ),
            "channels_y": int(model.M),
            "channels_z": int(model.Z),
            "model_type": config.model_type,
            "has_means_y": bool(config.model_type == "mean_scale_hyperprior"),
            "quant_step_y": float(model.conditional_entropy_y.quant_step.detach().cpu()),
            "quant_step_z": float(z_entropy.quant_step.detach().cpu()),
            "downsampling_factor": int(model.downsampling_factor),
            "model_config_name": config.name,
            "scale_min": float(model.scale_min),
            "scale_max": float(model.scale_max),
            "z_medians": z_entropy.medians.detach().cpu().to(torch.float32).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
