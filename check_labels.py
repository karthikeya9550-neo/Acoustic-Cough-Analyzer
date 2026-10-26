
import json
import glob
from collections import Counter

files = glob.glob(
    "../SPRSound-main/BioCAS2022/train2022_json/*.json"
)

records = Counter()
events = Counter()

for file in files:
    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    records.update([data.get("record_annotation", "Unknown")])

    for event in data.get("event_annotation", []):
        events.update([event.get("type", "Unknown")])

print("Files found:", len(files))
print("\nRECORD LABELS:")
print(records.most_common())

print("\nEVENT LABELS:")
print(events.most_common())
