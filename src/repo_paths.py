"""Repository path anchors, resolved from this installed module's location.

With the editable install this file lives at ``<repo>/src/repo_paths.py``, so the repo root is two
parents up. This gives notebooks and scripts a stable, cwd-independent way to find run outputs under
``runs/`` (and docs/notebooks) without sys.path hacks.

    from repo_paths import RUNS
    CURVE = RUNS / "bravyi" / "bb12" / "bb144_curve_hi"
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs"
DOCS = REPO_ROOT / "docs"
NOTEBOOKS = REPO_ROOT / "notebooks"


def run_dir(*parts) -> Path:
    """Path under runs/, e.g. run_dir('bravyi', 'bb12', 'bb144_curve_hi')."""
    return RUNS.joinpath(*parts)
