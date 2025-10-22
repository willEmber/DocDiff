#!/usr/bin/env python3
import argparse
import os
import csv
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_log_for_val_metrics(log_path: str) -> List[Tuple[int, float, float, float, float]]:
    # Parse lines like: VAL iter=12345: PSNR=.. SSIM=.. | INV_MSE=.. INV_COS=.. (N=..)
    pat = re.compile(r"VAL iter=(\d+): PSNR=([\d\.]+) SSIM=([\d\.]+) \| INV_MSE=([\d\.eE\-]+) INV_COS=([\d\.eE\-]+)")
    rows = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                it = int(m.group(1))
                psnr = float(m.group(2))
                ssim = float(m.group(3))
                inv_mse = float(m.group(4))
                inv_cos = float(m.group(5))
                rows.append((it, psnr, ssim, inv_mse, inv_cos))
    return rows


def parse_val_csv(csv_path: str) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((
                int(r['iteration']),
                float(r['psnr']),
                float(r['ssim']),
                float(r['inv_mse']),
                float(r['inv_cossim']),
            ))
    return rows


def plot(rows: List[Tuple[int, float, float, float, float]], out_dir: str):
    if not rows:
        print('No validation metrics found.')
        return
    it = [r[0] for r in rows]
    psnr = [r[1] for r in rows]
    ssim = [r[2] for r in rows]
    inv_mse = [r[3] for r in rows]
    inv_cos = [r[4] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(it, psnr, label='PSNR (dB)')
    plt.plot(it, ssim, label='SSIM')
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title('Image Quality Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, 'val_quality.png')
    plt.tight_layout(); plt.savefig(p1); print('Saved:', p1)

    plt.figure(figsize=(10, 6))
    plt.plot(it, inv_mse, label='EDICT INV MSE')
    plt.plot(it, inv_cos, label='EDICT INV CosSim')
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title('EDICT Inversion Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    p2 = os.path.join(out_dir, 'val_inversion.png')
    plt.tight_layout(); plt.savefig(p2); print('Saved:', p2)


def main():
    ap = argparse.ArgumentParser(description='Plot validation metrics from log or CSV')
    ap.add_argument('--log', help='Training log file to parse (print-based)')
    ap.add_argument('--csv', help='val_metrics.csv produced during training')
    ap.add_argument('--out_dir', required=True, help='Where to save plots')
    args = ap.parse_args()

    rows: List[Tuple[int, float, float, float, float]] = []
    if args.csv and os.path.exists(args.csv):
        rows = parse_val_csv(args.csv)
    elif args.log and os.path.exists(args.log):
        rows = parse_log_for_val_metrics(args.log)
    else:
        raise FileNotFoundError('Provide either --csv or --log with existing file')

    rows = sorted(rows, key=lambda x: x[0])
    plot(rows, args.out_dir)


if __name__ == '__main__':
    main()

