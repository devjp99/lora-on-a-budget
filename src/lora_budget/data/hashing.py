"""Hashing helpers shared by the manifest scripts, loaders, and tests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace before hashing."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def text_hash(text: str) -> str:
    """SHA-256 of the normalized text, used for exact-duplicate checks."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
