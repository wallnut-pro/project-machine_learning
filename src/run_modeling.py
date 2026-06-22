from __future__ import annotations

import argparse
import json

try:
    from src.modeling import (
        MODEL_CONFIGS,
        run_all_modeling,
        run_modeling_sanity_checks,
        train_all_models_for_dataset,
    )
except ModuleNotFoundError:
    from modeling import (  # type: ignore
        MODEL_CONFIGS,
        run_all_modeling,
        run_modeling_sanity_checks,
        train_all_models_for_dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline model building for mushroom datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "uci", "secondary"],
        default="all",
        help="Dataset to model. Use 'all' for both datasets.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip methodological sanity checks after baseline training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "all":
        summary = run_all_modeling()
        if not args.skip_sanity_check:
            summary["sanity_checks"] = run_modeling_sanity_checks()
    else:
        summary = {args.dataset: train_all_models_for_dataset(args.dataset)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
