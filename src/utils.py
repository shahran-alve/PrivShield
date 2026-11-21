"""Shared helpers: derived seeds, hashing and canonical JSON."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable component seed from the global seed plus some labels.

    Used so that, say, the 25k sample and the 50k sample get different but
    reproducible seeds without anyone calling an unseeded generator.
    """
    joined = canonical_json([base_seed, *parts])
    digest = sha256_text(joined)
    # 8 hex characters give a 32-bit integer, the seed size numpy and scikit-learn accept.
    return int(digest[:8], 16)
