import argparse
import os
import sys
from typing import Optional, Tuple

import torch

# Ensure repo root is on path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def resolve_checkpoint_pair(
    ckpt_dir: str,
    which: str = "best_psnr",
    use_ema: bool = False,
    iteration: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Resolve (init_path, denoiser_path) under a checkpoint directory.

    which: 'best_psnr' | 'latest' | 'iter'
    use_ema: when True, prefer EMA weights if available
    iteration: required when which == 'iter'
    """
    def p(name: str) -> str:
        return os.path.join(ckpt_dir, name)

    prefix_i = "model_init"
    prefix_d = "model_denoiser"
    if use_ema:
        prefix_i += "_ema"
        prefix_d += "_ema"

    if which == "best_psnr":
        ip = p(f"{prefix_i}_best_psnr.pth")
        dp = p(f"{prefix_d}_best_psnr.pth")
    elif which == "latest":
        ip = p(f"{prefix_i}_latest.pth")
        dp = p(f"{prefix_d}_latest.pth")
    elif which == "iter":
        if iteration is None:
            raise ValueError("--iter must be provided when --which=iter")
        ip = p(f"{prefix_i}_{iteration}.pth")
        dp = p(f"{prefix_d}_{iteration}.pth")
    else:
        raise ValueError(f"Unknown which='{which}'")

    if not os.path.isfile(ip):
        raise FileNotFoundError(f"Init checkpoint not found: {ip}")
    if not os.path.isfile(dp):
        raise FileNotFoundError(f"Denoiser checkpoint not found: {dp}")
    return ip, dp


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocDiff checkpoints on validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (e.g., conf_coco.yml)")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--which", type=str, default="best_psnr", choices=["best_psnr", "latest", "iter"], help="Which checkpoint to evaluate")
    parser.add_argument("--iter", type=int, default=None, help="Iteration number when --which=iter")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--max-samples", type=int, default=16, help="Max validation samples to evaluate")
    parser.add_argument("--native", action="store_true", help="Evaluate at native resolution with tiling (overrides config)")
    args = parser.parse_args()

    # Load config and override minimal fields to avoid test dataset construction issues
    config = load_config(args.config)
    # Force training mode to avoid building test dataloader with empty TEST_* paths
    config._dict["MODE"] = 1
    # Optionally override native resolution behavior
    if args.native:
        config._dict["NATIVE_RESOLUTION"] = "True"

    # Instantiate trainer (this builds validation loader based on VAL_* paths)
    trainer = Trainer(config)

    # Resolve and load checkpoints
    init_path, denoiser_path = resolve_checkpoint_pair(args.ckpt_dir, args.which, args.ema, args.iter)
    print(f"[LOAD] init='{init_path}' | denoiser='{denoiser_path}' (EMA={args.ema})")
    trainer.network.init_predictor.load_state_dict(torch.load(init_path, map_location=trainer.device))
    trainer.network.denoiser.load_state_dict(torch.load(denoiser_path, map_location=trainer.device))
    trainer.network.to(trainer.device).eval()

    # Evaluate
    psnr, ssim, inv_mse, inv_cos, n = trainer.validate(max_samples=args.max_samples)
    print(f"[RESULT] which={args.which}{' (EMA)' if args.ema else ''} | N={n} | PSNR={psnr:.4f} SSIM={ssim:.4f} | INV_MSE={inv_mse:.6f} INV_COS={inv_cos:.6f}")

    # Persist results next to checkpoints
    try:
        out_csv = os.path.join(args.ckpt_dir, "eval_metrics.csv")
        header_needed = not os.path.exists(out_csv)
        import csv as _csv
        with open(out_csv, "a", newline="") as f:
            w = _csv.writer(f)
            if header_needed:
                w.writerow(["which", "ema", "iter", "max_samples", "N", "psnr", "ssim", "inv_mse", "inv_cos"])
            w.writerow([args.which, int(args.ema), args.iter if args.iter is not None else "", args.max_samples, n, psnr, ssim, inv_mse, inv_cos])
        print(f"[WRITE] Appended metrics to {out_csv}")
    except Exception as e:
        print(f"[WARN] Failed to write eval_metrics.csv: {e}")


if __name__ == "__main__":
    main()

