#!/usr/bin/env python
"""
Download MobileNetV2 ImageNet1K weights and save to a local .pth file.
Usage:
  python scripts/download_mobilenet_v2_weights.py --out mobilenet_v2-imagenet1k-v1.pth
Options:
  --out PATH    Output file path (default: mobilenet_v2-imagenet1k-v1.pth)
  --force       Overwrite if file exists
"""
import argparse
import os
import sys
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MobileNetV2 weights")
    parser.add_argument(
        "--out",
        default="mobilenet_v2-imagenet1k-v1.pth",
        help="Output path for the .pth file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = os.path.abspath(args.out)

    if os.path.exists(out_path) and not args.force:
        print(f"File already exists: {out_path}. Use --force to overwrite.")
        sys.exit(0)

    print("Downloading MobileNetV2 weights (ImageNet1K_V1)...")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    state_dict = model.state_dict()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(state_dict, out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
