from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set random seeds for:
      - python random
      - numpy (if installed)
      - torch (if installed)

    deterministic_torch=True can reduce speed but improves reproducibility.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # numpy
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    # torch
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

