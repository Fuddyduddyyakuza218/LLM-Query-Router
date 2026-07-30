import json
from pathlib import Path
from collections import Counter

INPUT_PATH  = Path("data/labels.jsonl")
OUTPUT_PATH = Path("data/labels.jsonl")  # overwrite in-place

# ── relabeling rules ──────────────────────────────────────────────────────────
#
# Logic:
#   GSM8K = expensive by definition (multi-step math)
#   ARC / MMLU = cheap IF cheap model got it right
#   ARC / MMLU = expensive IF cheap model got it wrong but exp model did
#
# This preserves model-correctness signal for MCQ tasks while ensuring
# GSM8K contributes "expensive" labels regardless of extraction success.

def relabel(row: dict) -> dict | None:
    source       = row["source"]
    cheap_correct = row["cheap_correct"]
    exp_correct   = row["exp_correct"]

    if source == "gsm8k":
        # all GSM8K = expensive (multi-step reasoning required by definition)
        tier = "expensive"
    elif source in ("arc", "mmlu"):
        if cheap_correct:
            tier = "cheap"
        elif exp_correct:
            tier = "expensive"
        else:
            return None  # both wrong, discard
    else:
        return None

    row["tier"] = tier
    return row

def main():
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows.")

    relabeled  = []
    discarded  = 0

    for row in rows:
        result = relabel(row)
        if result is None:
            discarded += 1
        else:
            relabeled.append(result)

    tiers   = Counter(r["tier"]   for r in relabeled)
    sources = Counter(r["source"] for r in relabeled)

    print(f"Relabeled : {len(relabeled)}")
    print(f"Discarded : {discarded}")
    print(f"Tiers     : {dict(tiers)}")
    print(f"Sources   : {dict(sources)}")

    ratio = tiers.get("cheap", 0) / max(tiers.get("expensive", 1), 1)
    print(f"Imbalance ratio (cheap:expensive) : {ratio:.1f}:1")

    if ratio > 4:
        print("\nWARNING: imbalance ratio > 4:1 -- will apply class_weight='balanced' in classifiers.")

    with open(OUTPUT_PATH, "w") as f:
        for row in relabeled:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()