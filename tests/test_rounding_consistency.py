from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cpp_round_to_even_reference(x: float) -> int:
    lower = math.floor(x)
    frac = x - lower
    if frac > 0.5:
        return lower + 1
    if frac < 0.5:
        return lower
    return lower if lower % 2 == 0 else lower + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check round-to-even behavior against torch.round.")
    parser.add_argument("--random-count", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    half_cases = torch.tensor(
        [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        dtype=torch.float32,
    )
    random_cases = torch.randn(args.random_count, dtype=torch.float32) * 100.0
    values = torch.cat([half_cases, random_cases])
    torch_values = torch.round(values).to(torch.int64).tolist()
    ref_values = [cpp_round_to_even_reference(float(value)) for value in values.tolist()]
    mismatches = [
        (float(values[index]), torch_values[index], ref_values[index])
        for index in range(len(ref_values))
        if torch_values[index] != ref_values[index]
    ]
    print(f"checked={len(values)}")
    print(f"mismatches={len(mismatches)}")
    if mismatches:
        print("first mismatches:")
        for item in mismatches[:10]:
            print(item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
