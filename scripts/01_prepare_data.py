from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import prepare_dataset
from src.utils import load_config, print_config_summary, set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare the shared text dataset used by all models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run dataset preparation
    """
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    print_config_summary(config)
    output_path = prepare_dataset(config)

    print("Dataset preparation complete")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()