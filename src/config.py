"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration and registry for the Bloomberg data lake pipeline.

Design principles:
  1. Single source of truth. Every ticker, field, path, and constant is
     defined here and imported everywhere else. No hardcoded values in
     any other module.

  2. Registry-driven. Adding a new instrument requires a single dict entry
     in REGISTRY. All downstream modules (ingest, aggregate, intraday,
     validator) iterate the registry programmatically — no module-level
     changes required.

  3. Path generation is deterministic and derived exclusively from the
     registry. File names follow the strict convention:
         {prefix_lower}_raw_inception_{frequency}.parquet
     where frequency ∈ {daily, monthly, yearly, 1min, 2min, 3min, 5min,
     10min, 15min, 30min, 1h, 2h, 4h, tick}.

  4. Bloomberg constraints are constants, not magic numbers.
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

# ── Directory structure ──────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
LOG_DIR     = BASE_DIR / "logs"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Bloomberg temporal constants ─────────────────────────────────────────────

# Earliest date Bloomberg will accept for daily history requests.
START_DATE = '1900-01-01'

# Bloomberg IntradayBarRequest hard window.
# The terminal enforces a maximum of 140 calendar days.
# We use 139 as the operational constant to guarantee we never hit the
# boundary regardless of timezone offset between the client and Bloomberg.
BLOOMBERG_INTRADAY_WINDOW_DAYS = 139

# ── Intraday interval registry ───────────────────────────────────────────────
#
# Maps bar interval in minutes (int) → filename label (str).
# Ordered from finest to coarsest resolution.
# Bloomberg's IntradayBarRequest accepts any positive integer number of
# minutes. The set below covers every standard quantitative finance interval.
#
# Sub-minute resolution is architecturally impossible via IntradayBarRequest.
# It requires IntradayTickRequest (bdtick), handled in ingest_tick.py.

INTRADAY_INTERVALS: dict[int, str] = {
    1:   '1min',
    2:   '2min',
    3:   '3min',
    5:   '5min',
    10:  '10min',
    15:  '15min',
    30:  '30min',
    60:  '1h',
    120: '2h',
    240: '4h',
}

# ── Ticker registry ──────────────────────────────────────────────────────────
#
# Each dict entry defines the complete data contract for one instrument.
#
# Keys:
#   ticker        Bloomberg ticker string passed to blp.bdh() / blp.bdib()
#   prefix        Column prefix used in all parquet schemas (e.g. "SPX")
#   daily_fields  List of Bloomberg field mnemonics for daily bdh() pull
#   volume        True  → instrument carries traded volume (ETFs, futures)
#                 False → synthetic index, no traded volume (VIX, NDX, SPX, SPXT)
#   intraday      True  → Bloomberg serves IntradayBarRequest for this ticker
#                 False → Bloomberg does not compute intraday for this ticker
#                         (applies to total return indices: SPXT)
#   tick_types    List of Bloomberg tick event types for bdtick() pull.
#                 Traded ETFs: ['TRADE', 'BID', 'ASK']
#                 Synthetic indices: ['BID', 'ASK'] only — no TRADE events
#                 None → tick ingestion disabled for this ticker (SPXT)
#
# Aggregation rules are derived automatically by aggregate.py from the
# daily_fields list and the volume flag. No per-ticker aggregation
# configuration is required here.

REGISTRY: list[dict] = [
    {
        "ticker":       "SPX Index",
        "prefix":       "SPX",
        "daily_fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"],
        "volume":       False,   # SPX is a price-return index, not a traded instrument.
                                 # PX_VOLUME from Bloomberg for SPX is inferred from
                                 # constituent activity and is structurally sparse.
                                 # We retain the field for completeness but do not treat
                                 # it as a traded volume series.
        "intraday":     True,
        "tick_types":   ["BID", "ASK"],
    },
    {
        "ticker":       "SPXT Index",
        "prefix":       "SPXT",
        "daily_fields": ["PX_LAST"],
        "volume":       False,   # Total return index — no traded volume.
        "intraday":     False,   # Bloomberg does not compute intraday bars for
                                 # total return indices. IntradayBarRequest returns
                                 # empty for SPXT.
        "tick_types":   None,    # Tick ingestion disabled for total return indices.
    },
    {
        "ticker":       "VIX Index",
        "prefix":       "VIX",
        "daily_fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"],
        "volume":       False,   # VIX is a synthetic volatility index. No traded volume.
        "intraday":     True,
        "tick_types":   ["BID", "ASK"],
    },
    {
        "ticker":       "NDX Index",
        "prefix":       "NDX",
        "daily_fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"],
        "volume":       False,   # NDX is a synthetic index. No traded volume.
        "intraday":     True,
        "tick_types":   ["BID", "ASK"],
    },
    {
        "ticker":       "SPY US Equity",
        "prefix":       "SPY",
        "daily_fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"],
        "volume":       True,    # SPY is a traded ETF — volume is real and complete.
        "intraday":     True,
        "tick_types":   ["TRADE", "BID", "ASK"],
    },
    {
        "ticker":       "QQQ US Equity",
        "prefix":       "QQQ",
        "daily_fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"],
        "volume":       True,    # QQQ is a traded ETF — volume is real and complete.
        "intraday":     True,
        "tick_types":   ["TRADE", "BID", "ASK"],
    },
]

# ── Path generation functions ────────────────────────────────────────────────
#
# All parquet paths in the entire pipeline are generated exclusively through
# these four functions. No path string is constructed anywhere else.
# This guarantees naming consistency across all modules.

def daily_path(prefix: str) -> Path:
    """Return the canonical path for a ticker's daily parquet file."""
    return STORAGE_DIR / f"{prefix.lower()}_raw_inception_daily.parquet"


def monthly_path(prefix: str) -> Path:
    """Return the canonical path for a ticker's monthly parquet file."""
    return STORAGE_DIR / f"{prefix.lower()}_raw_inception_monthly.parquet"


def yearly_path(prefix: str) -> Path:
    """Return the canonical path for a ticker's yearly parquet file."""
    return STORAGE_DIR / f"{prefix.lower()}_raw_inception_yearly.parquet"


def intraday_path(prefix: str, interval_min: int) -> Path:
    """
    Return the canonical path for a ticker's intraday bar parquet file.

    Parameters
    ----------
    prefix       : Instrument prefix string (e.g. 'SPX', 'VIX')
    interval_min : Bar interval in minutes. Must be a key in INTRADAY_INTERVALS.
    """
    if interval_min not in INTRADAY_INTERVALS:
        raise ValueError(
            f"interval_min={interval_min} is not in INTRADAY_INTERVALS. "
            f"Valid values: {list(INTRADAY_INTERVALS.keys())}"
        )
    label = INTRADAY_INTERVALS[interval_min]
    return STORAGE_DIR / f"{prefix.lower()}_raw_inception_{label}.parquet"


def tick_path(prefix: str) -> Path:
    """Return the canonical path for a ticker's raw tick parquet file."""
    return STORAGE_DIR / f"{prefix.lower()}_raw_inception_tick.parquet"