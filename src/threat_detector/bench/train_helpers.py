"""Pure training helpers, CPU-testable and free of torch."""
from __future__ import annotations

from ..data.manifest import ClipEntry
from ..fusion.bayes_fuser import ACTIONS


def build_head_dataset(entries: list[ClipEntry], split: str) -> list[tuple[str, int]]:
    return [(e.path, ACTIONS.index(e.label)) for e in entries if e.split == split]
