"""Build and persist dataset manifests in one call."""
from __future__ import annotations

from .manifest import build_action_manifest, save_manifest


def prepare_action_manifest(root: str, out: str) -> int:
    entries = build_action_manifest(root)
    save_manifest(entries, out)
    return len(entries)
