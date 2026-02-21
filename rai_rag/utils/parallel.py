from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ParallelConfig:
    mode: str = "thread"   # thread | process | none
    max_workers: int = 8
    chunksize: int = 1


def parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    cfg: Optional[ParallelConfig] = None,
) -> List[Any]:
    """
    Minimal, safe parallel map utility.
    - thread mode: good for IO-bound tasks
    - process mode: good for CPU-bound tasks (must be picklable)
    """
    cfg = cfg or ParallelConfig()
    mode = (cfg.mode or "thread").lower()
    if mode == "none" or len(items) == 0 or cfg.max_workers <= 1:
        return [fn(x) for x in items]

    if mode == "thread":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(cfg.max_workers)) as ex:
            return list(ex.map(fn, items))

    if mode == "process":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(cfg.max_workers)) as ex:
            # chunksize not supported in ex.map here across all versions; keep simple
            return list(ex.map(fn, items))

    raise ValueError(f"Unknown parallel mode: {cfg.mode}")

