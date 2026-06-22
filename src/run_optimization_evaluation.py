from __future__ import annotations

import argparse
import json

try:
    from src.evaluation import evaluate_all_models
    from src.optimization import run_all_optimizations
except ModuleNotFoundError:
    from evaluation import evaluate_all_models  # type: ignore
    from optimization import run_all_optimizations  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimization and evaluation for mushroom classifiers.")
    parser.add_argument(
        "--dataset",
        choices=["all"],
        default="all",
        help="Only 'all' is supported because final evaluation compares all datasets and models together.",
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    optimization_summary = run_all_optimizations()
    evaluation_summary = evaluate_all_models(optimization_summary["datasets"])
    print(
        json.dumps(
            {
                "optimization": optimization_summary,
                "evaluation": evaluation_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
