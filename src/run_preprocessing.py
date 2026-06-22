from __future__ import annotations

import argparse
import json

try:
    from src.preprocessing import DATASET_CONFIGS, preprocess_all_datasets, preprocess_dataset
except ModuleNotFoundError:
    from preprocessing import DATASET_CONFIGS, preprocess_all_datasets, preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing for mushroom classification datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_CONFIGS)],
        default="all",
        help="Dataset to preprocess. Use 'all' to preprocess every dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "all":
        summaries = preprocess_all_datasets()
    else:
        summaries = {args.dataset: preprocess_dataset(args.dataset)}

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
