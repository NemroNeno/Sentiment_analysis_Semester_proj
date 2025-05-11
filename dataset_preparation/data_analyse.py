import json
import random
from collections import defaultdict

# 1) Configuration
INPUT_PATH  = "dataset_preparation\combined_final.json"                 # replace with your actual file
OUTPUT_PATH = "final_combined_balanced.json"
TARGET_COUNT = { -1: 15000,  1: 15000,  0: None }  # None = keep all

# 2) Load full dataset
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)   # expects a list of dicts

# 3) Group by class
buckets = defaultdict(list)
for item in data:
    cls = item["polarity"]   # replace with your actual key if different
    buckets[cls].append(item)

# 4) Downsample / keep
balanced = []
for cls, items in buckets.items():
    target = TARGET_COUNT.get(cls)
    if target is None or target >= len(items):
        # keep all
        balanced.extend(items)
    else:
        # random.sample without replacement
        balanced.extend(random.sample(items, target))

# 5) Shuffle the final list
random.shuffle(balanced)

# 6) Write out balanced dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

print(f"Original counts:")
for cls, items in buckets.items():
    print(f"  Class {cls!r}: {len(items)}")
print(f"\nBalanced counts:")
counts = { cls: sum(1 for x in balanced if x["polarity"] == cls) for cls in buckets }
for cls, cnt in counts.items():
    print(f"  Class {cls!r}: {cnt}")
print(f"\nWrote {len(balanced)} total examples to {OUTPUT_PATH}")
