#!/usr/bin/env python3
"""
Plot validation metrics across training iterations for one or more experiments.

Parses lines like:
  VAL iter=5000: PSNR=30.0810 SSIM=0.8303 | INV_MSE=2.764753 INV_COS=-0.481219 (N=16)

Features:
- Accepts multiple log files (paths and/or globs) with optional labels.
- Plots PSNR, SSIM, INV_MSE, INV_COS vs. iteration on 2x2 subplots.
- Optional smoothing (moving average or EMA).
- Saves figure to --out and optional long-form CSV via --csv.

Usage examples:
  python scripts/plot_val_metrics.py logs/*.log --labels expA expB --out results/val_metrics.png
  python scripts/plot_val_metrics.py 1026_inv1_02cocotrain.log 1027step75train.log \
      --labels inv1 step75 --csv results/val_metrics.csv --ema 0.2

If matplotlib is not available, the script will still parse and write CSV (if --csv is set).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VAL_PATTERN = re.compile(
    r"^\s*VAL\s*iter=(?P<iter>\d+)\s*:\s*"
    r"PSNR=(?P<psnr>[-+]?\d+(?:\.\d+)?)\s+"
    r"SSIM=(?P<ssim>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"INV_MSE=(?P<inv_mse>[-+]?\d+(?:\.\d+)?)\s+"
    r"INV_COS=(?P<inv_cos>[-+]?\d+(?:\.\d+)?)\s*"
    r"\(N=(?P<N>\d+)\)\s*$"
)


@dataclass
class Series:
    iteration: List[int]
    psnr: List[float]
    ssim: List[float]
    inv_mse: List[float]
    inv_cos: List[float]
    n: List[int]


@dataclass
class Experiment:
    label: str
    path: str
    series: Series


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    acc = 0.0
    from collections import deque

    q: deque[float] = deque()
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def _ema(values: Sequence[float], alpha: float) -> List[float]:
    if alpha <= 0.0 or alpha >= 1.0 or not values:
        return list(values)
    out: List[float] = []
    m: Optional[float] = None
    for v in values:
        m = v if m is None else (alpha * v + (1.0 - alpha) * m)
        out.append(m)
    return out


def parse_val_lines(path: str) -> Series:
    iters: List[int] = []
    psnr: List[float] = []
    ssim: List[float] = []
    inv_mse: List[float] = []
    inv_cos: List[float] = []
    n_vals: List[int] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "VAL" not in line:
                continue
            m = VAL_PATTERN.match(line)
            if not m:
                # Be tolerant to minor spacing/format deviations
                # Try a looser parse fallback
                try:
                    # Example fallback: split by spaces and symbols
                    if "iter=" in line and "PSNR=" in line and "SSIM=" in line and "INV_MSE=" in line and "INV_COS=" in line:
                        # Extract with simpler regex pieces
                        iter_m = re.search(r"iter=(\d+)", line)
                        psnr_m = re.search(r"PSNR=([\-\+]?\d+(?:\.\d+)?)", line)
                        ssim_m = re.search(r"SSIM=([\-\+]?\d+(?:\.\d+)?)", line)
                        inv_mse_m = re.search(r"INV_MSE=([\-\+]?\d+(?:\.\d+)?)", line)
                        inv_cos_m = re.search(r"INV_COS=([\-\+]?\d+(?:\.\d+)?)", line)
                        n_m = re.search(r"\(N=(\d+)\)", line)
                        if all([iter_m, psnr_m, ssim_m, inv_mse_m, inv_cos_m, n_m]):
                            iters.append(int(iter_m.group(1)))
                            psnr.append(float(psnr_m.group(1)))
                            ssim.append(float(ssim_m.group(1)))
                            inv_mse.append(float(inv_mse_m.group(1)))
                            inv_cos.append(float(inv_cos_m.group(1)))
                            n_vals.append(int(n_m.group(1)))
                            continue
                except Exception:
                    pass
                continue

            iters.append(int(m.group("iter")))
            psnr.append(float(m.group("psnr")))
            ssim.append(float(m.group("ssim")))
            inv_mse.append(float(m.group("inv_mse")))
            inv_cos.append(float(m.group("inv_cos")))
            n_vals.append(int(m.group("N")))

    # Sort by iteration in case lines are out of order
    order = sorted(range(len(iters)), key=lambda i: iters[i])
    iters = [iters[i] for i in order]
    psnr = [psnr[i] for i in order]
    ssim = [ssim[i] for i in order]
    inv_mse = [inv_mse[i] for i in order]
    inv_cos = [inv_cos[i] for i in order]
    n_vals = [n_vals[i] for i in order]

    return Series(iters, psnr, ssim, inv_mse, inv_cos, n_vals)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_long_csv(path: str, experiments: List[Experiment]) -> None:
    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "path", "iteration", "PSNR", "SSIM", "INV_MSE", "INV_COS", "N"])
        for exp in experiments:
            s = exp.series
            for i in range(len(s.iteration)):
                w.writerow(
                    [
                        exp.label,
                        exp.path,
                        s.iteration[i],
                        s.psnr[i],
                        s.ssim[i],
                        s.inv_mse[i],
                        s.inv_cos[i],
                        s.n[i],
                    ]
                )


def find_best(values: Sequence[float], iters: Sequence[int], mode: str = "max") -> Tuple[int, float]:
    if not values:
        return (-1, float("nan"))
    if mode == "max":
        idx = max(range(len(values)), key=lambda i: values[i])
    else:
        idx = min(range(len(values)), key=lambda i: values[i])
    return iters[idx], values[idx]


def smooth_series(series: Series, window: int = 1, ema_alpha: float = 0.0) -> Series:
    # Respect iteration and N as-is; smooth the metric curves only
    if ema_alpha > 0.0 and 0.0 < ema_alpha < 1.0:
        smoother = lambda v: _ema(v, ema_alpha)
    else:
        window = max(1, int(window))
        smoother = lambda v: _moving_average(v, window)

    return Series(
        iteration=list(series.iteration),
        psnr=smoother(series.psnr),
        ssim=smoother(series.ssim),
        inv_mse=smoother(series.inv_mse),
        inv_cos=smoother(series.inv_cos),
        n=list(series.n),
    )


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:
        sys.stderr.write(f"[warn] matplotlib not available: {e}\n")
        return None


def plot_experiments(
    experiments: List[Experiment],
    out_path: Optional[str],
    title: Optional[str] = None,
    log_mse: bool = True,
    dpi: int = 150,
    show: bool = False,
    xlim: Optional[Tuple[int, int]] = None,
    ylim_psnr: Optional[Tuple[float, float]] = None,
) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        if out_path or show:
            sys.stderr.write("[warn] Skipping plot because matplotlib is missing.\n")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_psnr, ax_ssim, ax_mse, ax_cos = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for exp in experiments:
        s = exp.series
        label = exp.label
        ax_psnr.plot(s.iteration, s.psnr, label=label)
        ax_ssim.plot(s.iteration, s.ssim, label=label)
        if log_mse:
            ax_mse.semilogy(s.iteration, s.inv_mse, label=label)
        else:
            ax_mse.plot(s.iteration, s.inv_mse, label=label)
        ax_cos.plot(s.iteration, s.inv_cos, label=label)

    ax_psnr.set_title("PSNR")
    ax_psnr.set_xlabel("Iteration")
    ax_psnr.set_ylabel("dB")
    ax_psnr.grid(True, alpha=0.3)
    if ylim_psnr is not None:
        ax_psnr.set_ylim(*ylim_psnr)
    if xlim is not None:
        ax_psnr.set_xlim(*xlim)

    ax_ssim.set_title("SSIM")
    ax_ssim.set_xlabel("Iteration")
    ax_ssim.grid(True, alpha=0.3)
    if xlim is not None:
        ax_ssim.set_xlim(*xlim)

    ax_mse.set_title("INV_MSE")
    ax_mse.set_xlabel("Iteration")
    ax_mse.grid(True, which="both", alpha=0.3)
    if xlim is not None:
        ax_mse.set_xlim(*xlim)

    ax_cos.set_title("INV_COS")
    ax_cos.set_xlabel("Iteration")
    ax_cos.grid(True, alpha=0.3)
    if xlim is not None:
        ax_cos.set_xlim(*xlim)

    # Shared legend
    handles, labels = ax_psnr.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
        fig.subplots_adjust(bottom=0.15)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=dpi)
        print(f"[info] Saved figure to: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot VAL metrics from DocDiff logs.")
    p.add_argument(
        "logs",
        nargs="+",
        help="One or more log paths and/or globs (e.g. logs/*.log)",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each log; defaults to filename stem.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (e.g., results/val_metrics.png). If omitted, only shows if --show.",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional long-form CSV path to save parsed metrics.",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window (>=1). Ignored if --ema > 0.",
    )
    p.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="EMA smoothing alpha in (0,1). Overrides --smooth when > 0.",
    )
    p.add_argument(
        "--no-log-mse",
        action="store_true",
        help="Disable log scale for INV_MSE subplot.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving.")
    p.add_argument("--show", action="store_true", help="Display the plot interactively.")
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    p.add_argument(
        "--xlim",
        type=str,
        default=None,
        help="Optional x-axis limits for all subplots, format: min,max",
    )
    p.add_argument(
        "--ylim-psnr",
        type=str,
        default=None,
        help="Optional y-axis limits for PSNR, format: min,max",
    )

    args = p.parse_args(argv)
    return args


def expand_logs(paths_or_globs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths_or_globs:
        matches = glob(p)
        if matches:
            out.extend(sorted(matches))
        else:
            out.append(p)
    # De-duplicate while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for p in out:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(p)
    return uniq


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log_paths = expand_logs(args.logs)
    if not log_paths:
        print("No logs found.")
        return 1

    # Prepare labels
    if args.labels is not None and len(args.labels) > 0:
        if len(args.labels) != len(log_paths):
            print("[error] Number of --labels must match number of logs.")
            return 2
        labels = list(args.labels)
    else:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in log_paths]

    # Default output path if not provided but not showing
    out_path = args.out
    if out_path is None and not args.show:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("results", f"val_metrics_{ts}.png")

    experiments: List[Experiment] = []
    for path, label in zip(log_paths, labels):
        if not os.path.exists(path):
            print(f"[warn] Missing log: {path}")
            continue
        series = parse_val_lines(path)
        if not series.iteration:
            print(f"[warn] No VAL entries in: {path}")
            continue
        # Apply smoothing if requested
        if args.ema > 0.0:
            series = smooth_series(series, ema_alpha=float(args.ema))
        elif args.smooth and args.smooth > 1:
            series = smooth_series(series, window=int(args.smooth))

        experiments.append(Experiment(label=label, path=os.path.abspath(path), series=series))

    if not experiments:
        print("[error] No data parsed from provided logs.")
        return 3

    # Write CSV if requested
    if args.csv:
        write_long_csv(args.csv, experiments)
        print(f"[info] Wrote CSV to: {args.csv}")

    # Print quick summary
    print("\nSummary (best by metric):")
    for exp in experiments:
        it_psnr, best_psnr = find_best(exp.series.psnr, exp.series.iteration, mode="max")
        it_ssim, best_ssim = find_best(exp.series.ssim, exp.series.iteration, mode="max")
        it_mse, best_mse = find_best(exp.series.inv_mse, exp.series.iteration, mode="min")
        it_cos, best_cos = find_best(exp.series.inv_cos, exp.series.iteration, mode="max")
        print(
            f"- {exp.label}: PSNR={best_psnr:.4f}@{it_psnr}, "
            f"SSIM={best_ssim:.4f}@{it_ssim}, INV_MSE={best_mse:.6f}@{it_mse}, INV_COS={best_cos:.6f}@{it_cos}"
        )

    # Parse axis limits
    xlim = None
    if args.xlim:
        try:
            xmin, xmax = [int(x.strip()) for x in args.xlim.split(",", 1)]
            xlim = (xmin, xmax)
        except Exception:
            print(f"[warn] Bad --xlim '{args.xlim}', expected 'min,max'. Ignoring.")

    ylim_psnr = None
    if args.ylim_psnr:
        try:
            ymin, ymax = [float(x.strip()) for x in (args.ylim_psnr or "").split(",", 1)]
            ylim_psnr = (ymin, ymax)
        except Exception:
            print(f"[warn] Bad --ylim-psnr '{args.ylim_psnr}', expected 'min,max'. Ignoring.")

    plot_experiments(
        experiments=experiments,
        out_path=out_path,
        title=args.title,
        log_mse=not args.no_log_mse,
        dpi=args.dpi,
        show=args.show,
        xlim=xlim,
        ylim_psnr=ylim_psnr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
