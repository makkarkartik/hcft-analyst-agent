"""One-shot migration: HCFT chunk JSONL files -> local MongoDB.

Loads data/chunks/{train,val,test}.jsonl from the SLM_Fine_Tuning repo into a
single `chunks` collection. The collection serves two consumers in the agent:
  1. Retriever tool: hydrate chunk text by chunk_id after Pinecone + rerank.
  2. Analytics agent: aggregations over state/hospital/year/document_type.

Usage:
    .venv\\Scripts\\python.exe scripts\\migrate_chunks.py [--drop]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pymongo import ASCENDING, MongoClient
from pymongo.errors import BulkWriteError
from tqdm import tqdm

CHUNKS_DIR = Path(r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\chunks")
SPLITS = ["train", "val", "test"]
MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME = "hcft"
COLLECTION = "chunks"
BATCH_SIZE = 2000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop existing collection first")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[DB_NAME][COLLECTION]

    if args.drop:
        coll.drop()
        print(f"dropped {DB_NAME}.{COLLECTION}")

    total_inserted = 0
    t0 = time.time()
    for split in SPLITS:
        path = CHUNKS_DIR / f"{split}.jsonl"
        if not path.exists():
            sys.exit(f"missing input file: {path}")
        batch = []
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=split, unit=" rows"):
                doc = json.loads(line)
                doc["_id"] = doc["chunk_id"]  # natural key; makes reruns idempotent-ish
                batch.append(doc)
                if len(batch) >= BATCH_SIZE:
                    total_inserted += insert_batch(coll, batch)
                    batch = []
        if batch:
            total_inserted += insert_batch(coll, batch)

    print("creating indexes...")
    coll.create_index([("doc_id", ASCENDING)])
    coll.create_index([("split", ASCENDING)])
    coll.create_index([("state", ASCENDING), ("hospital", ASCENDING), ("year", ASCENDING)])
    coll.create_index([("document_type", ASCENDING)])

    print(f"inserted {total_inserted:,} docs in {time.time() - t0:,.0f}s")
    print(f"collection count: {coll.count_documents({}):,}")


def insert_batch(coll, batch) -> int:
    try:
        result = coll.insert_many(batch, ordered=False)
        return len(result.inserted_ids)
    except BulkWriteError as e:
        # duplicate _id on rerun without --drop: skip dupes, count the rest
        n_dupes = len(e.details.get("writeErrors", []))
        return len(batch) - n_dupes


if __name__ == "__main__":
    main()
