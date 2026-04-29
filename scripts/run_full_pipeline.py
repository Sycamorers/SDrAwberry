#!/usr/bin/env python3
"""CLI wrapper for the config-driven SDrAwberry full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdrawberry.pipeline import main


if __name__ == "__main__":
    main()
