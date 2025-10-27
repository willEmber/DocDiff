#!/usr/bin/env python3
"""
Lightweight local sweep runner for DocDiff.

Usage:
  python scripts/run_sweep.py --sweep scripts/sweep.example.yml [--dry]

Sweep file schema (YAML):
  base: conf.yml
  name: "cocoEDICT_step{TIMESTEPS}_inv{LAMBDA_INV1}"
  grid:
    TIMESTEPS: [50, 75]
    LAMBDA_INV1: [0.2, 0.5]
  set: ["EDICT=True", "EDICT_FP64=True"]  # optional extra fixed overrides

Notes:
  - Runs sequentially and writes logs to Training/<name>/train.log
  - Requires main.py to support --name and --set KEY=VALUE (added in this repo)
"""

import argparse
import itertools
import os
import subprocess
import sys
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(cmd: List[str], log_path: str, dry: bool = False) -> int:
    if dry:
        print("DRY:", ' '.join(cmd))
        return 0
    # Stream to both console and file
    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True, help='Path to sweep YAML file')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable to use')
    parser.add_argument('--dry', action='store_true', help='Print commands without running')
    args = parser.parse_args()

    cfg = load_yaml(args.sweep)
    base = cfg.get('base', 'conf.yml')
    name_tpl = cfg.get('name', 'run_{idx}')
    grid: Dict[str, List[Any]] = cfg.get('grid', {})
    fixed_set: List[str] = cfg.get('set', [])

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values)) if keys else [()]

    for idx, combo in enumerate(combos, start=1):
        kv = {k: combo[i] for i, k in enumerate(keys)}
        # Name with formatting; include index fallback
        kv_fmt = dict(idx=idx, **kv)
        try:
            name = str(name_tpl).format(**kv_fmt)
        except Exception:
            name = f"run_{idx}"
        # Build CLI
        cmd = [
            args.python, '-u', 'main.py', '--config', base, '--name', name,
        ]
        for k, v in kv.items():
            cmd.extend(['--set', f"{k}={v}"])
        for ov in fixed_set:
            cmd.extend(['--set', ov])

        run_dir = os.path.join('Training', name)
        ensure_dir(run_dir)
        code = run(cmd, log_path=os.path.join(run_dir, 'train.log'), dry=args.dry)
        if code != 0:
            print(f"Run '{name}' failed with exit code {code}")
            break


if __name__ == '__main__':
    main()

