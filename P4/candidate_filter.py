"""Shortlist found-item IDs for the semantic matcher using soft metadata scores."""

import argparse
import csv
from datetime import date, datetime
import json
import math
from pathlib import Path

from .prepare_data import normalize


def read_date(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def select_candidates(
    inventory, category_probabilities=None, *, query_date=None,
    query_location=None, max_candidates=100, date_window_days=14,
):
    """Return up to max_candidates item IDs, ordered by metadata score.

    Pass inventory rows for the split being evaluated. Probabilities are a
    mapping from category ID (integer or JSON string) to nonnegative score.
    Missing signals get a neutral score. All thresholds are soft.
    """
    if max_candidates < 0 or date_window_days <= 0:
        raise ValueError("max_candidates must be nonnegative and date_window_days positive")
    probabilities = {
        int(label): float(value) for label, value in (category_probabilities or {}).items()
    }
    if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
        raise ValueError("Category scores must be finite and nonnegative")
    peak = max(probabilities.values(), default=0.0)
    query_day = read_date(query_date)
    location = normalize(query_location)

    def score(row):
        category_score = 0.5
        if peak and row.get("category_id") not in (None, ""):
            category_score = probabilities.get(int(row["category_id"]), 0.0) / peak

        found_day = read_date(row.get("receiving_date"))
        date_score = 0.5
        if query_day and found_day:
            gap = (found_day - query_day).days
            # Earlier receiving dates are less plausible but can survive.
            date_score = max(0.0, 1 - abs(gap) / date_window_days)
            if gap < 0:
                date_score *= 0.5

        station = normalize(row.get("station_name"))
        location_score = float(station == location) if station and location else 0.5
        return 0.50 * category_score + 0.30 * date_score + 0.20 * location_score

    ranked = sorted(inventory, key=lambda row: (-score(row), row["item_id"]))
    return [row["item_id"] for row in ranked[:max_candidates]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path(__file__).with_name("data") / "inventory.csv")
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--date", help="Approximate query date in YYYY-MM-DD format")
    parser.add_argument("--location", help="Approximate station name")
    parser.add_argument("--probabilities", type=json.loads, default={},
                        help='JSON mapping of label IDs to scores, e.g. {"2": 0.8}')
    parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8", newline="") as file:
        inventory = [row for row in csv.DictReader(file) if not args.split or row["split"] == args.split]
    ids = select_candidates(
        inventory, args.probabilities, query_date=args.date, query_location=args.location,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()
