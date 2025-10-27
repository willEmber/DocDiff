from src.config import load_config
from src.train import train, test
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    parser.add_argument('--name', type=str, default=None, help='override EXPERIMENT_NAME for this run')
    parser.add_argument(
        '--set', dest='set_kv', action='append', default=[],
        help='Override config key by KEY=VALUE; repeatable. Example: --set TIMESTEPS=75 --set LR=1e-4'
    )
    args = parser.parse_args()

    # Load config and apply CLI overrides
    config = load_config(args.config, overrides=args.set_kv)
    # Optional: override experiment name from CLI
    if args.name:
        # Keep consistent style: Config stores raw dict; ensure set as provided
        try:
            config.set('EXPERIMENT_NAME', args.name)
        except Exception:
            # Fallback to attribute assignment if available
            setattr(config, 'EXPERIMENT_NAME', args.name)
    print('Config loaded')
    mode = config.MODE
    if mode == 1:
        train(config)
    else:
        test(config)
if __name__ == "__main__":
    main()
