"""Recover independently executable experiment cells from the Colab export.

The source ranges below refer to the unmodified archival export.  The
standalone files preserve every line of experiment code; only the Colab
markdown marker between cells is excluded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive" / "claps.py"
EXPECTED_SHA256 = "7522c43a5ecef32c59b75b8d7a1b3f21361648f703422a7e1691f389bfbbcc60"

# Inclusive, one-based line ranges in archive/claps.py.
RANGES = {
    "01_weak_support.py": (1, 1647),
    "02_weighted_geometry.py": (1649, 2605),
    "03_epistemic_regimes.py": (2607, 4064),
    "04_real_data.py": (4066, 5935),
    "05_prior_precision.py": (5937, 7514),
    "06_representation_dimension.py": (7516, 9382),
    "07_variance_floor.py": (9384, 11378),
    "08_ood_stress.py": (11380, 13513),
}


def extract(destination: Path = ROOT / "experiments") -> None:
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Source hash changed: {digest}")

    lines = raw.splitlines(keepends=True)
    if len(lines) != 13513:
        raise RuntimeError(f"Unexpected source line count: {len(lines)}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, (first, last) in RANGES.items():
        content = b"".join(lines[first - 1 : last])
        compile(content, filename, "exec")
        (destination / filename).write_bytes(content)


if __name__ == "__main__":
    extract()
