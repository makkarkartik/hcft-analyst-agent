"""Quick sanity checks for the migrated chunks collection."""

from pymongo import MongoClient

coll = MongoClient("mongodb://127.0.0.1:27017")["hcft"]["chunks"]

print(f"total docs: {coll.count_documents({}):,}")
for split in ["train", "val", "test"]:
    print(f"  {split}: {coll.count_documents({'split': split}):,}")

doc = coll.find_one({"_id": "0f76aeca2faf78b3_p27_c0"})
print("hydration sample:", doc["hospital"], "|", doc["text"][:60])

top_states = list(
    coll.aggregate([
        {"$group": {"_id": "$state", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
        {"$limit": 5},
    ])
)
print("top states by chunk count:", top_states)

print("indexes:", sorted(coll.index_information().keys()))
