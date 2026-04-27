"""
Run the full project pipeline from data preparation to figure generation
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data import prepare_dataset
from src.extract_representations import extract_all_representations
from src.utils import load_config, print_config_summary, set_seed
from src.visualize import make_all_figures


def load_compute_all_metrics() -> Callable[[dict], pd.DataFrame]:
    """
    Dynamically load compute_all_metrics from scripts/03_compute_metrics.py
    """
    project_root = Path(__file__).resolve().parents[1]
    metrics_script = project_root / "scripts" / "03_compute_metrics.py"

    spec = importlib.util.spec_from_file_location("compute_metrics_script", metrics_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric script from {metrics_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_all_metrics


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run the full representation analysis pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip model forward passes and reuse existing representations",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the full pipeline
    """
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    print_config_summary(config)

    print("\nStep 1/4: preparing dataset")
    prepare_dataset(config)

    if args.skip_extraction:
        print("\nStep 2/4: skipping representation extraction")
    else:
        print("\nStep 2/4: extracting representations")
        extract_all_representations(config)

    print("\nStep 3/4: computing metrics")
    compute_all_metrics = load_compute_all_metrics()
    compute_all_metrics(config)

    print("\nStep 4/4: generating figures")
    output_paths = make_all_figures(config)

    print("\nFull pipeline complete")
    print("Generated figures:")
    for path in output_paths:
        print(f"  {Path(path)}")


if __name__ == "__main__":
    main()