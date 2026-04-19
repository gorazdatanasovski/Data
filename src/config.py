from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '1900-01-01'

# ── SPX Index (Price Return) ────────────────────────────────────────────────
SPX_TICKER  = 'SPX Index'
SPX_FIELDS  = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']

SPX_DAILY   = STORAGE_DIR / 'spx_raw_inception_daily.parquet'
SPX_MONTHLY = STORAGE_DIR / 'spx_raw_inception_monthly.parquet'
SPX_YEARLY  = STORAGE_DIR / 'spx_raw_inception_yearly.parquet'
SPX_1MIN    = STORAGE_DIR / 'spx_raw_inception_1min.parquet'
SPX_5MIN    = STORAGE_DIR / 'spx_raw_inception_5min.parquet'
SPX_1H      = STORAGE_DIR / 'spx_raw_inception_1h.parquet'

# ── SPXT Index (Total Return) ───────────────────────────────────────────────
# Note: SPXT has no intraday bars — Bloomberg does not compute them.
SPXT_TICKER  = 'SPXT Index'
SPXT_FIELDS  = ['PX_LAST']

SPXT_DAILY   = STORAGE_DIR / 'spxt_raw_inception_daily.parquet'
SPXT_MONTHLY = STORAGE_DIR / 'spxt_raw_inception_monthly.parquet'
SPXT_YEARLY  = STORAGE_DIR / 'spxt_raw_inception_yearly.parquet'

# ── VIX Index ───────────────────────────────────────────────────────────────
# Note: VIX volume omitted — synthetic index, no traded volume.
VIX_TICKER  = 'VIX Index'
VIX_FIELDS  = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']

VIX_DAILY   = STORAGE_DIR / 'vix_raw_inception_daily.parquet'
VIX_MONTHLY = STORAGE_DIR / 'vix_raw_inception_monthly.parquet'
VIX_YEARLY  = STORAGE_DIR / 'vix_raw_inception_yearly.parquet'
VIX_1MIN    = STORAGE_DIR / 'vix_raw_inception_1min.parquet'
VIX_5MIN    = STORAGE_DIR / 'vix_raw_inception_5min.parquet'
VIX_1H      = STORAGE_DIR / 'vix_raw_inception_1h.parquet'