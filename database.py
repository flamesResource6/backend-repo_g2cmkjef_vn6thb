"""
Database Helper Functions with resilient MongoDB and local JSON fallback.

- If DATABASE_URL and DATABASE_NAME are set and pymongo is available, use MongoDB.
- Otherwise, transparently fall back to a simple JSONL store under data/local_db.

This ensures the API never crashes in environments without MongoDB while still
providing persistence within the sandbox.
"""
from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Union, List, Dict, Any, Optional

# Try to load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Try to import pymongo, but don't crash if unavailable
try:
    from pymongo import MongoClient  # type: ignore
    HAVE_PYMONGO = True
except Exception:
    MongoClient = None  # type: ignore
    HAVE_PYMONGO = False

_client = None
db = None

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if HAVE_PYMONGO and DATABASE_URL and DATABASE_NAME:
    try:
        _client = MongoClient(DATABASE_URL)
        db = _client[DATABASE_NAME]
    except Exception:
        db = None

# Local JSONL fallback directory
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_DB_DIR = os.path.join(DATA_DIR, "local_db")
os.makedirs(LOCAL_DB_DIR, exist_ok=True)


def _now_iso() -> datetime:
    return datetime.now(timezone.utc)


def _local_path(collection_name: str) -> str:
    safe = "".join(ch for ch in collection_name if ch.isalnum() or ch in ("_", "-"))
    return os.path.join(LOCAL_DB_DIR, f"{safe}.jsonl")


def _local_write(collection_name: str, doc: Dict[str, Any]) -> str:
    path = _local_path(collection_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "_id" not in doc:
        doc["_id"] = str(uuid.uuid4())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, default=str) + "\n")
    return str(doc["_id"])


def _local_read_all(collection_name: str) -> List[Dict[str, Any]]:
    path = _local_path(collection_name)
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def create_document(collection_name: str, data: Union["BaseModel", Dict[str, Any]]):
    """Insert a single document with timestamps.

    Works with MongoDB when available; otherwise uses a JSONL fallback per collection.
    Returns the inserted document id as a string.
    """
    # Convert Pydantic model to dict if needed (lazy import to avoid hard dependency)
    try:
        from pydantic import BaseModel  # type: ignore
    except Exception:
        BaseModel = object  # type: ignore

    if isinstance(data, BaseModel):  # type: ignore
        data_dict = data.model_dump()  # type: ignore
    else:
        data_dict = dict(data)

    data_dict['created_at'] = _now_iso()
    data_dict['updated_at'] = _now_iso()

    # Prefer MongoDB if available
    if db is not None:
        try:
            result = db[collection_name].insert_one(data_dict)
            return str(result.inserted_id)
        except Exception:
            # fall back silently to local store
            return _local_write(collection_name, data_dict)
    # Local fallback
    return _local_write(collection_name, data_dict)


def get_documents(collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
    """Get documents from collection.

    Supports basic equality filters. Works with MongoDB or local JSONL fallback.
    """
    filter_dict = filter_dict or {}

    # Try MongoDB first
    if db is not None:
        try:
            cursor = db[collection_name].find(filter_dict)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception:
            pass

    # Local fallback: filter in Python
    items = _local_read_all(collection_name)
    def _match(doc: Dict[str, Any]) -> bool:
        for k, v in filter_dict.items():
            if doc.get(k) != v:
                return False
        return True

    results = [d for d in items if _match(d)]
    if limit:
        results = results[:limit]
    return results
