"""Reproducible seeding.

The original 2024 implementation (git tag ``legacy-2024``) used
``tf.random.set_seed(7)``. This project uses PyTorch, so exact numeric parity is
impossible — the goal is architecture + hyperparameter parity.
``derive_fold_seed`` gives each (dataset, target, model, fold) run an independent,
order-independent seed so results are reproducible regardless of execution order.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np


def set_seed(seed: int = 7) -> None:
    """Seed ``random``, ``numpy`` and (if available) ``torch`` on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def derive_fold_seed(base_seed: int, *parts) -> int:
    """Deterministic per-run seed derived from ``base_seed`` plus string parts."""
    key = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
