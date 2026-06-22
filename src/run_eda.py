from __future__ import annotations

import json

try:
    from src.eda import run_all_eda
except ModuleNotFoundError:
    from eda import run_all_eda


def main() -> None:
    summaries = run_all_eda()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
