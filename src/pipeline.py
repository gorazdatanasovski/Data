"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Master orchestration script. Runs the complete data lake pipeline in the
correct dependency order:

  Stage 1 — ingest.py         : Daily OHLCV from inception for all tickers
  Stage 2 — aggregate.py      : Monthly and yearly aggregation from daily
  Stage 3 — ingest_intraday.py: Intraday bars (1min → 4h) for all intervals
  Stage 4 — ingest_tick.py    : Raw tick data (sub-minute) for all tickers
  Stage 5 — validator.py      : Full structural and statistical validation

Each stage is run as a separate subprocess so that a failure in one stage
does not silently corrupt a later stage. Exit codes are checked — if a
stage fails, the pipeline halts and reports the failure.

Usage:
    python pipeline.py             # Run all stages
    python pipeline.py --skip-tick # Skip tick ingestion (large files)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / 'pipeline.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('pipeline')

# ── Script paths ──────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parent

STAGES = {
    "daily_ingest":    SRC_DIR / "ingest.py",
    "aggregate":       SRC_DIR / "aggregate.py",
    "intraday_ingest": SRC_DIR / "ingest_intraday.py",
    "tick_ingest":     SRC_DIR / "ingest_tick.py",
    "validate":        SRC_DIR / "validator.py",
}


def _run_stage(name: str, script_path: Path) -> bool:
    """
    Run a pipeline stage as a subprocess using the current Python interpreter.

    Parameters
    ----------
    name        : Human-readable stage name for logging
    script_path : Absolute path to the Python script to execute

    Returns
    -------
    True if the script exits with code 0, False otherwise.
    """
    log.info(f"{'─' * 60}")
    log.info(f"Starting stage: {name}")
    log.info(f"Script        : {script_path}")
    log.info(f"{'─' * 60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
    )

    if result.returncode == 0:
        log.info(f"Stage '{name}' completed successfully. Exit code: 0")
        return True
    else:
        log.error(
            f"Stage '{name}' FAILED. Exit code: {result.returncode}. "
            f"Pipeline halted. Inspect logs in {config.LOG_DIR}."
        )
        return False


def main(skip_tick: bool = False) -> None:
    """
    Run the full pipeline in dependency order.

    Parameters
    ----------
    skip_tick : If True, skip the tick ingestion stage (Stage 4).
                Tick files are large and slow. Use this flag when you only
                need bar data updated.
    """
    log.info("=" * 72)
    log.info("Bloomberg Data Lake Pipeline — Full Run")
    log.info(f"Skip tick ingestion: {skip_tick}")
    log.info("=" * 72)

    # ── Stage 1: Daily ingestion ──────────────────────────────────────────────
    if not _run_stage("Daily Ingestion", STAGES["daily_ingest"]):
        sys.exit(1)

    # ── Stage 2: Aggregation ──────────────────────────────────────────────────
    if not _run_stage("Monthly & Yearly Aggregation", STAGES["aggregate"]):
        sys.exit(1)

    # ── Stage 3: Intraday bar ingestion ───────────────────────────────────────
    if not _run_stage("Intraday Bar Ingestion", STAGES["intraday_ingest"]):
        sys.exit(1)

    # ── Stage 4: Tick ingestion (optional) ───────────────────────────────────
    if not skip_tick:
        if not _run_stage("Raw Tick Ingestion", STAGES["tick_ingest"]):
            sys.exit(1)
    else:
        log.info("Stage 4 (Tick Ingestion) skipped via --skip-tick flag.")

    # ── Stage 5: Full validation ──────────────────────────────────────────────
    if not _run_stage("Full Validation", STAGES["validate"]):
        sys.exit(1)

    log.info("=" * 72)
    log.info("Pipeline complete. All stages passed.")
    log.info("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bloomberg Data Lake Pipeline — Master Orchestrator"
    )
    parser.add_argument(
        "--skip-tick",
        action="store_true",
        help="Skip Stage 4 (raw tick ingestion). Use when tick files are not required.",
    )
    args = parser.parse_args()
    main(skip_tick=args.skip_tick)