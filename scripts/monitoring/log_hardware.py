#!/usr/bin/env python3
"""
log_hardware.py
===============
Standalone script to log the full hardware & software environment to the
console and to a JSON manifest file.  Called automatically by
``run_pipeline.ps1`` at the start of every pipeline run, providing
thesis-grade documentation of the execution environment.

Usage
-----
    python log_hardware.py                        # print to stdout
    python log_hardware.py --log-dir logs         # also save JSON + log file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.thesis_logger import init_thesis_logging, log_hardware_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log hardware & software environment for thesis documentation."
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory to save hardware_manifest.json and log file.",
    )
    args = parser.parse_args()

    if args.log_dir:
        # Full thesis logging (file + console + hardware JSON)
        init_thesis_logging(log_dir=args.log_dir)
    else:
        # Console-only
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_hardware_summary()


if __name__ == "__main__":
    main()
