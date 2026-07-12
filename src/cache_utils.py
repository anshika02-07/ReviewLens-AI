"""
ReviewLens AI
Cache utilities — avoid recomputing expensive operations
(embeddings, FAISS index, sentiment) on every Streamlit rerun.
"""

import hashlib
import os
import pickle

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_file_hash(uploaded_file):
    """
    Returns a stable hash for an uploaded file's contents.
    Used to key all caches so re-uploading the SAME file reuses
    previous work, but a DIFFERENT file always gets fresh results.
    """
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()


def _cache_path(file_hash, name):
    return os.path.join(CACHE_DIR, f"{file_hash}_{name}.pkl")


def save_cache(file_hash, name, obj):
    with open(_cache_path(file_hash, name), "wb") as f:
        pickle.dump(obj, f)


def load_cache(file_hash, name):
    path = _cache_path(file_hash, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def cache_exists(file_hash, name):
    return os.path.exists(_cache_path(file_hash, name))