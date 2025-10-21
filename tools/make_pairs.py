#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise to an RGB image (uint8 in [0,255])."""
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_directory(
    src: str,
    dst: str,
    sigma: float,
    seed: int = 0,
    max_images: Optional[int] = None,
) -> None:
    os.makedirs(dst, exist_ok=True)
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if max_images is not None:
        files = files[:max_images]

    rng = np.random.default_rng(seed)

    for fname in tqdm(files, desc="Generating degraded (Gaussian noise)"):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)

        img = Image.open(src_path).convert("RGB")
        arr = np.array(img)
        out = add_gaussian_noise(arr, sigma=sigma, rng=rng)
        Image.fromarray(out).save(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Make paired degraded dataset (Gaussian noise only)")
    parser.add_argument("--src", required=True, help="Source folder of clean/GT images (RGB)")
    parser.add_argument("--dst", required=True, help="Destination folder for degraded images (same filenames)")
    parser.add_argument("--sigma", type=float, default=15.0, help="Gaussian noise sigma in pixel units (0-255 scale)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise")
    parser.add_argument("--max", type=int, default=None, help="Optional cap on number of images processed")
    args = parser.parse_args()

    process_directory(args.src, args.dst, sigma=args.sigma, seed=args.seed, max_images=args.max)
    print(f"Done. Degraded images written to: {args.dst}")


if __name__ == "__main__":
    main()

