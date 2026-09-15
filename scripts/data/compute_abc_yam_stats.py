#!/usr/bin/env python3
"""Run numeric-only ABC statistics without importing DreamZero's model stack."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                       "groot/vla/data/dataset/abc_yam_stats.py"), run_name="__main__")
