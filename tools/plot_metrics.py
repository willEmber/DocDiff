#!/usr/bin/env python3
import argparse
import os
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_metrics(csv_path: str) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                if v == '' or v is None:
                    data[k].append(float('nan'))
                else:
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        data[k].append(float('nan'))
    return data


def moving_average(xs: List[float], w: int) -> List[float]:
    if w <= 1:
        return xs
    out: List[float] = []
    acc = 0.0
    cnt = 0
    q: List[float] = []
    for x in xs:
        if not (x != x):  # not NaN
            q.append(x)
            acc += x
            cnt += 1
        else:
            q.append(0.0)
        if len(q) > w:
            acc -= q.pop(0)
            cnt -= 1 if cnt > 0 else 0
        out.append(acc / max(cnt, 1))
    return out


def plot_curves(iters: List[float], curves: List[Tuple[str, List[float]]], title: str, out_path: str):
    plt.figure(figsize=(10, 6))
    for name, ys in curves:
        plt.plot(iters, ys, label=name)
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description='Plot training metrics from metrics.csv')
    ap.add_argument('--run_dir', required=True, help='Training run directory (contains metrics.csv)')
    ap.add_argument('--smooth', type=int, default=10, help='Moving average window (iterations)')
    ap.add_argument('--out_dir', default=None, help='Output dir for plots (default: run_dir)')
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")

    data = read_metrics(csv_path)
    iters = [int(x) for x in data['iteration']]

    # Prepare smoothed curves
    def sm(name: str) -> List[float]:
        return moving_average(data[name], args.smooth) if name in data else []

    # 1) Core losses
    plot_curves(
        iters,
        [
            ('loss', sm('loss')),
            ('ddpm_loss', sm('ddpm_loss')),
            ('pixel_total', sm('pixel_total')),
            ('pixel_plain', sm('pixel_plain')),
        ],
        'Core Losses',
        os.path.join(out_dir, 'core_losses.png')
    )

    # 2) Frequency losses (if present) and invertibility
    curves = [('L_inv1', sm('L_inv1'))]
    if 'low_freq_pixel_loss' in data:
        curves.insert(0, ('low_freq_pixel_loss', sm('low_freq_pixel_loss')))
    plot_curves(
        iters,
        curves,
        'Frequency/Invertibility',
        os.path.join(out_dir, 'freq_inv.png')
    )

    # 3) Step time and LR
    plot_curves(
        iters,
        [
            ('step_time_sec', sm('step_time_sec')),
            ('lr', sm('lr')),
        ],
        'Train Speed / LR',
        os.path.join(out_dir, 'speed_lr.png')
    )


if __name__ == '__main__':
    main()

