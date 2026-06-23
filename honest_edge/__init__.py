"""honest_edge: the shared library behind the course notebooks.

Each notebook in `notebooks/` imports from here so the heavy, reusable logic
(data loading, indicators, the primary signal, labeling, evaluation) lives in one
tested place and the notebooks stay readable. Install once with `pip install -e .`
from the repo root, then `from honest_edge import data` works from any notebook.

The whole library obeys one rule, the no-look-ahead contract (see data.py):
a decision made at bar t may use data only up to bar t's close.
"""

from honest_edge import data, indicators, signal, labeling, features, evaluation

__all__ = ["data", "indicators", "signal", "labeling", "features", "evaluation"]
