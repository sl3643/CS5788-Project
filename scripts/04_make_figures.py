from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.visualize import make_all_figures


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Generate all result figures")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run figure generation
    """
    args = parse_args()
    config = load_config(args.config)

    output_paths = make_all_figures(config)

    print("Figure generation complete")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()