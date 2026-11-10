"""Convert the LostFound Delhi Metro inventory into the P4 handoff."""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen


SOURCE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/drishyatomar/lostfound/"
    "Delhi%20Metro%20Dataset%2Fdelhimetrorail.csv"
)
SOURCE_FIELDS = (
    "item_name", "description", "item_quantity", "station_name",
    "receiving_date", "receiving_time",
)
SPLITS = {"train": 0.70, "dev": 0.15, "test": 0.15}
FIELDS = (
    "item_id", "group_id", "split", "item_name", "description", "model_text",
    "text_source", "item_quantity", "station_name", "receiving_date",
    "receiving_time", "category_id", "category",
)
PAIR_FIELDS = (
    "query_id", "split", "query_text", "query_date", "query_location",
    "relevant_item_id", "is_synthetic",
)
QUERY_TYPES = {
    "earbud case": (r"ear\s*buds?\s+case", "wireless earphone case"),
    "earbuds": (r"ear\s*buds?", "wireless earphones"),
    "water bottle": (r"(?:water\s+)?bottle", "drink bottle"),
    "lunch box": (r"lunch\s*box", "food container"),
    "handbag": (r"hand\s*bag", "handheld purse"),
    "bag": (r"bag", "carry bag"),
    "phone": (r"(?:basic\s+)?mobile(?:\s+phone)?", "cell phone"),
    "watch": (r"(?:wrist\s+)?watch", "wristwatch"),
    "helmet": (r"helmet", "protective helmet"),
    "umbrella": (r"umbrella", "rain umbrella"),
    "jacket": (r"jacket", "outerwear jacket"),
    "glasses": (r"spectacles?", "eyeglasses"),
    "purse": (r"purse", "small purse"),
}
COLOR = re.compile(r"\b(black|blue|red|green|white|pink|brown|grey|gray|yellow|silver|purple|orange)\b")


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def stable_id(prefix, *fields):
    value = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def load_categories():
    config = json.loads(Path(__file__).with_name("category_map.json").read_text(encoding="utf-8"))
    return [
        (entry["id"], entry["name"], [re.compile(pattern) for pattern in entry["patterns"]])
        for entry in config["categories"]
    ]


def classify(name, categories):
    # The first recognized object in the item name usually denotes the item itself.
    # E.g. "bag with mobile" is a bag. Preserve multi-item names for later review.
    name = re.sub(r"[^\w\s]", " ", name)
    matches = [
        (match.start(), category_id, category)
        for category_id, category, patterns in categories
        for pattern in patterns
        if (match := pattern.search(name))
    ]
    if matches:
        _, category_id, category = min(matches)
        return category_id, category
    category_id, category, _ = categories[-1]
    return category_id, category


def standardize(value, fmt, kind, issues):
    if not value:
        issues[f"missing_{kind}"] += 1
        return ""
    try:
        parsed = datetime.strptime(value, fmt)
        return parsed.date().isoformat() if kind == "receiving_date" else parsed.time().isoformat()
    except ValueError:
        issues[f"invalid_{kind}"] += 1
        return ""


def clean(raw, categories):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
        raise ValueError(f"Unexpected raw header: {reader.fieldnames}")
    rows, seen, issues = [], set(), Counter()
    raw_rows = 0
    for record in reader:
        raw_rows += 1
        name = normalize(record["item_name"])
        description = normalize(record["description"])
        station = normalize(record["station_name"])
        day = standardize(record["receiving_date"], "%d/%m/%Y", "receiving_date", issues)
        time = standardize(record["receiving_time"], "%H:%M:%S", "receiving_time", issues)
        quantity = normalize(record["item_quantity"])
        if quantity.isdecimal():
            quantity = str(int(quantity))
        elif quantity:
            issues["invalid_item_quantity"] += 1
            quantity = ""
        text = description or name
        if not text:
            issues["empty_text_rows"] += 1
            continue
        key = (name, description, quantity, station, day, time)
        if key in seen:
            issues["duplicate_rows_removed"] += 1
            continue
        seen.add(key)
        category_id, category = classify(name, categories)
        rows.append({
            "item_id": stable_id("item_", *key),
            "group_id": stable_id("group_", text),
            "item_name": name,
            "description": description,
            "model_text": text,
            "text_source": "description" if description else "item_name_fallback",
            "item_quantity": quantity,
            "station_name": station,
            "receiving_date": day,
            "receiving_time": time,
            "category_id": category_id,
            "category": category,
        })
    return rows, raw_rows, issues


def assign_splits(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    totals = Counter(row["category"] for row in rows)
    split_sizes = Counter()
    by_category = {split: Counter() for split in SPLITS}
    # Assign whole text groups while approximating both overall and class proportions.
    for _, group in sorted(groups.items(), key=lambda entry: (-len(entry[1]), entry[0])):
        categories = Counter(row["category"] for row in group)

        def cost(split):
            target = len(rows) * SPLITS[split]
            before = split_sizes[split] - target
            score = ((before + len(group)) ** 2 - before ** 2) / target
            for category, count in categories.items():
                target = totals[category] * SPLITS[split]
                before = by_category[split][category] - target
                score += ((before + count) ** 2 - before ** 2) / target
            return score

        split = min(SPLITS, key=cost)
        split_sizes[split] += len(group)
        by_category[split].update(categories)
        for row in group:
            row["split"] = split


def build_pairs(rows, per_split=12):
    pairs = []
    used_phrases = set()
    for split in SPLITS:
        pools = defaultdict(list)
        for row in rows:
            if row["split"] != split or not row["description"] or not row["receiving_date"]:
                continue
            color = COLOR.search(row["description"])
            if not color:
                continue
            for kind, (pattern, alias) in QUERY_TYPES.items():
                if re.fullmatch(pattern, row["item_name"]):
                    pools[kind].append((row, color.group(), alias))
                    break
        for pool in pools.values():
            pool.sort(key=lambda sample: sample[0]["item_id"])
        used_groups = set()
        uses_by_kind = Counter()
        templates = ("I misplaced a {color} {alias}.", "Looking for my {color} {alias}.",
                     "Has anyone found a {color} {alias}?")
        for index in range(per_split):
            for kind, pool in pools.items():
                pools[kind] = [
                    (row, color, alias) for row, color, alias in pool
                    if row["group_id"] not in used_groups and (color, alias) not in used_phrases
                ]
            eligible = [kind for kind, pool in pools.items() if pool]
            if not eligible:
                raise ValueError(f"Only {index} distinct synthetic queries available for {split}")
            # Favor underrepresented kinds and pick the most constrained pool first.
            kind = min(eligible, key=lambda key: (uses_by_kind[key], len(pools[key]), key))
            row, color, alias = pools[kind].pop(0)
            uses_by_kind[kind] += 1
            used_groups.add(row["group_id"])
            used_phrases.add((color, alias))
            date = datetime.fromisoformat(row["receiving_date"]).date() - timedelta(days=index % 2)
            pairs.append({
                "query_id": stable_id("query_", row["item_id"]),
                "split": split,
                "query_text": templates[index % len(templates)].format(color=color, alias=alias),
                "query_date": date.isoformat(),
                "query_location": row["station_name"],
                "relevant_item_id": row["item_id"],
                "is_synthetic": "true",
            })
    return pairs


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Raw delhimetrorail.csv (download from Kaggle if omitted)")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data"))
    args = parser.parse_args()
    if args.input:
        raw = args.input.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=45) as response:
            raw = response.read()
    rows, raw_rows, issues = clean(raw, load_categories())
    if not rows:
        raise ValueError("Input produced no inventory rows")
    assign_splits(rows)
    pairs = build_pairs(rows)
    rows.sort(key=lambda row: row["item_id"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "inventory.csv", FIELDS, rows)
    write_csv(args.output / "matching_pairs.csv", PAIR_FIELDS, pairs)
    summary = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_rows": raw_rows,
        "clean_rows": len(rows),
        "issues": dict(issues),
        "text_sources": dict(Counter(row["text_source"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_category_counts": {
            split: dict(Counter(row["category"] for row in rows if row["split"] == split))
            for split in SPLITS
        },
        "pairs_per_split": dict(Counter(pair["split"] for pair in pairs)),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows):,} inventory records and {len(pairs)} synthetic pairs to {args.output}")


if __name__ == "__main__":
    main()
