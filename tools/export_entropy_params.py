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

from compressai_nano import FactorizedPriorNano, get_model_config


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
    model = FactorizedPriorNano().eval()
    load_checkpoint(model, args.checkpoint)
    config = get_model_config()
    entropy = model.entropy_bottleneck
    medians = entropy.medians.detach().cpu().to(torch.float32).tolist()
    payload = {
        "format": "compressai-nano-entropy-params-v1",
        "channels": int(model.M),
        "quant_step": float(entropy.quant_step.detach().cpu()),
        "downsampling_factor": int(model.downsampling_factor),
        "model_config_name": config.name,
        "medians": medians,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
