"""
fetch_equity.py
---------------
Fetches daily OHLCV equity data from Yahoo Finance via yfinance
and lands it raw into DuckDB as the Bronze layer.
Idempotent — safe to re-run at any time.

Fix applied: uses yf.download() to batch all tickers in ONE request
instead of hitting Yahoo Finance 8 times sequentially (which triggers
rate limiting).

Run:
    python fetch_equity.py
"""

import os
import time
import logging
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, UTC
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

DB_PATH     = "../data/warehouse.duckdb"
FETCH_START = "2000-01-01"
FETCH_END   = datetime.today().strftime("%Y-%m-%d")

TICKERS = {
    "^GSPC": "S&P 500 index",
    "^VIX":  "CBOE Volatility Index — market fear gauge",
    "SPY":   "S&P 500 ETF (liquid benchmark)",
    "QQQ":   "Nasdaq-100 ETF (growth/tech exposure)",
    "IEF":   "7-10Y Treasury ETF (bond market behaviour)",
    "GLD":   "Gold ETF (safe haven signal)",
    "XLF":   "Financials sector ETF (rate-cycle sensitive)",
    "XLE":   "Energy sector ETF (inflation hedge proxy)",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fetch_equity.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Database helpers ──────────────────────────────────────────────────────────

def get_connection() -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)


def create_bronze_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bronze.equity_prices (
            ticker          VARCHAR NOT NULL,
            ticker_desc     VARCHAR,
            price_date      DATE NOT NULL,
            open            DOUBLE,
            high            DOUBLE,
            low             DOUBLE,
            close           DOUBLE,
            adj_close       DOUBLE,
            volume          BIGINT,
            ingested_at     TIMESTAMP NOT NULL
        )
    """)
    log.info("Bronze equity_prices table ready.")


def upsert_ticker(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    ticker_desc: str,
    df: pd.DataFrame,
) -> int:
    conn.execute(
        "DELETE FROM bronze.equity_prices WHERE ticker = ?",
        [ticker],
    )

    df_to_insert = df.copy()
    df_to_insert["ticker"]      = ticker
    df_to_insert["ticker_desc"] = ticker_desc
    df_to_insert["ingested_at"] = datetime.now(UTC)

    conn.execute("""
        INSERT INTO bronze.equity_prices
            (ticker, ticker_desc, price_date, open, high, low,
             close, adj_close, volume, ingested_at)
        SELECT ticker, ticker_desc, price_date, open, high, low,
               close, adj_close, volume, ingested_at
        FROM df_to_insert
    """)
    return len(df_to_insert)


# ── Fetch logic ───────────────────────────────────────────────────────────────

def fetch_all_tickers() -> dict:
    """
    Download all tickers in ONE batched request using yf.download().

    Why this instead of yf.Ticker() in a loop?
    Calling yf.Ticker().history() 8 times in quick succession hits Yahoo's
    rate limiter. yf.download() sends a single request for all tickers,
    which is both faster and avoids rate limiting.

    auto_adjust=False: we want both raw Close and Adj Close stored in Bronze.
    group_by='ticker': returns a multi-level DataFrame — ticker on top level,
    OHLCV columns underneath.
    """
    ticker_list = list(TICKERS.keys())
    log.info(f"Downloading {len(ticker_list)} tickers in one batch request...")

    raw = yf.download(
        tickers=ticker_list,
        start=FETCH_START,
        end=FETCH_END,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("yf.download returned empty DataFrame — possible rate limit or network issue.")

    results = {}

    for ticker in ticker_list:
        try:
            df_ticker = raw[ticker].copy().reset_index()

            # Flatten any multi-level columns yfinance sometimes returns
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = [c[0].lower().replace(" ", "_") for c in df_ticker.columns]
            else:
                df_ticker.columns = [c.lower().replace(" ", "_") for c in df_ticker.columns]

            # Strip timezone from date column
            date_col = "date" if "date" in df_ticker.columns else df_ticker.columns[0]
            df_ticker["price_date"] = pd.to_datetime(df_ticker[date_col]).dt.tz_localize(None).dt.date

            # Map column names defensively — yfinance naming is inconsistent
            def get_col(df, *names):
                for n in names:
                    if n in df.columns:
                        return pd.to_numeric(df[n], errors="coerce")
                return pd.Series([None] * len(df))

            df_clean = pd.DataFrame({
                "price_date": df_ticker["price_date"],
                "open":       get_col(df_ticker, "open"),
                "high":       get_col(df_ticker, "high"),
                "low":        get_col(df_ticker, "low"),
                "close":      get_col(df_ticker, "close"),
                "adj_close":  get_col(df_ticker, "adj_close", "close"),
                "volume":     get_col(df_ticker, "volume").fillna(0).astype(int),
            })

            df_clean = df_clean.dropna(subset=["open", "high", "low", "close"], how="all")

            results[ticker] = df_clean
            log.info(f"  Parsed {ticker}: {len(df_clean)} rows "
                     f"({df_clean['price_date'].min()} to {df_clean['price_date'].max()})")

        except Exception as exc:
            log.error(f"  Could not parse {ticker}: {exc}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = get_connection()
    create_bronze_table(conn)

    log.info("Waiting 3 seconds before fetching (rate-limit buffer)...")
    time.sleep(3)

    ticker_data = fetch_all_tickers()

    total_rows = 0
    for ticker, desc in TICKERS.items():
        if ticker not in ticker_data:
            log.warning(f"  Skipping {ticker} — no data parsed.")
            continue
        try:
            rows = upsert_ticker(conn, ticker, desc, ticker_data[ticker])
            total_rows += rows
            log.info(f"  ✓ {ticker}: {rows} rows written to bronze.")
        except Exception as exc:
            log.error(f"  ✗ {ticker} failed on DB write: {exc}")

    log.info(f"Done. Total rows in bronze.equity_prices: {total_rows}")
    conn.close()


if __name__ == "__main__":
    main()

