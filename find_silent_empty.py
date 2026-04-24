import json
from pathlib import Path

CKPT = Path("checkpoints/dedup_caption_features_v2.json")

with CKPT.open(encoding="utf-8") as f:
    data = json.load(f)

# data is {caption_text: [int, int, ...]}  (one int per feature in all_feature_names order)
all_unknown_substantive = []
all_unknown_trivial = []

for caption, feats in data.items():
    if all(v == 2 for v in feats):  # <-- was feats.values(); feats is a list
        if len((caption or "").strip()) >= 15:
            all_unknown_substantive.append(caption)
        else:
            all_unknown_trivial.append(caption)

print(f"Total all-unknown captions: "
      f"{len(all_unknown_substantive) + len(all_unknown_trivial):,}")
print(f"  - trivially short (<15 chars, expected): "
      f"{len(all_unknown_trivial):,}")
print(f"  - substantive (>=15 chars, the SILENT failure(s)): "
      f"{len(all_unknown_substantive):,}")
print()
print("Substantive all-unknown captions (these are the silent failures):")
for i, cap in enumerate(all_unknown_substantive, 1):
    print(f"\n[{i}] (len={len(cap)} chars)")
    print(f"    {cap[:500]}{'...' if len(cap) > 500 else ''}")