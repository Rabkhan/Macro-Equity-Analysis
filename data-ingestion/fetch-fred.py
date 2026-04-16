"""
fetch_fred.py
-------------
Fetches macro time-series from the FRED API and lands them raw
into DuckDB as the Bronze layer. Designed to be idempotent —
safe to run multiple times without creating duplicate rows.

"""

import os
import logging
import duckdb
import pandas as pd
from datetime import datetime, UTC
from dotenv import load_dotenv
from fredapi import Fred

# ____Configuration______

load_dotenv()

DB_PATH = "../data/warehouse.duckdb"

# Start date for all series — covers 3 full rate cycles - the base interest rate 
FETCH_START = "2000-01-01"

SERIES = {
    "FEDFUNDS":       "Fed funds effective rate (monthly)",
    "T10Y2Y":         "10Y minus 2Y Treasury spread — yield curve (daily)",
    "CPIAUCSL":       "CPI all items — inflation (monthly)",
    "UNRATE":         "US unemployment rate (monthly)",
    "T10YIE":         "10Y breakeven inflation rate (daily)",
    "BAMLH0A0HYM2":   "High yield credit spread OAS (daily)",
}

# ____Logging setup____
# Logs go to both terminal AND a file so we have a record of every run.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                        # prints to terminal
        logging.FileHandler("fetch_fred.log"),          # writes to file
    ],
)
log = logging.getLogger(__name__)


# ____Database helpers____

def get_connection() -> duckdb.DuckDBPyConnection:
    #Open (or create) the DuckDB warehouse file.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)


def create_bronze_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create the bronze schema and raw FRED table if they don't exist.

    Why store as raw columns instead of a JSON blob?
    Because DuckDB is columnar — storing typed columns lets dbt
    filter and aggregate without parsing JSON on every query.
    The 'ingested_at' column is our audit trail: we always know
    exactly when a row was written.
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bronze.fred_observations (
            series_id       VARCHAR NOT NULL,   -- e.g. 'FEDFUNDS'
            series_desc     VARCHAR,            -- human-readable label
            observation_date DATE NOT NULL,     -- the date the value refers to
            value           DOUBLE,             -- NULL if FRED reports '.'
            ingested_at     TIMESTAMP NOT NULL  -- when we wrote this row
        )
    """)
    log.info("Bronze schema and fred_observations table ready.")


def upsert_series(
    conn: duckdb.DuckDBPyConnection,
    series_id: str,
    series_desc: str,
    df: pd.DataFrame,
) -> int:
    """
    Idempotent write: delete existing rows for this series, then insert fresh.

    Why delete-then-insert instead of INSERT OR IGNORE?
    FRED revises historical values. If CPI for March 2023 gets revised,
    we want the new value, not the stale cached one. So we always replace
    the full series. For daily series with 6000+ rows this takes ~50ms.

    Returns the number of rows inserted.
    """
    conn.execute(
        "DELETE FROM bronze.fred_observations WHERE series_id = ?",
        [series_id],
    )

    df_to_insert = df.copy()
    df_to_insert["series_id"]   = series_id
    df_to_insert["series_desc"] = series_desc
    df_to_insert["ingested_at"] = datetime.now(UTC)

    conn.execute("""
        INSERT INTO bronze.fred_observations
            (series_id, series_desc, observation_date, value, ingested_at)
        SELECT series_id, series_desc, observation_date, value, ingested_at
        FROM df_to_insert
    """)

    return len(df_to_insert)


# ── Fetch logic ───────────────────────────────────────────────────────────────

def fetch_series(fred: Fred, series_id: str) -> pd.DataFrame:
    """
    Pull one FRED series from the API.
    Returns a clean DataFrame with columns: observation_date, value.

    fredapi returns a pandas Series indexed by date.
    We convert it to a DataFrame and handle FRED's missing-value
    convention: FRED uses '.' for missing — fredapi maps these to NaN,
    which we keep as NULL in DuckDB. Don't drop them — Silver layer
    decides how to handle gaps (forward-fill, interpolate, flag).
    """
    raw: pd.Series = fred.get_series(series_id, observation_start=FETCH_START)

    df = raw.reset_index()
    df.columns = ["observation_date", "value"]
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        log.error("FRED_API_KEY not found in environment. Check your .env file.")
        raise SystemExit(1)

    fred = Fred(api_key=api_key)
    log.info("FRED client initialised.")

    conn = get_connection()
    create_bronze_table(conn)

    total_rows = 0

    for series_id, series_desc in SERIES.items():
        log.info(f"Fetching {series_id} — {series_desc}")
        try:
            df = fetch_series(fred, series_id)
            rows = upsert_series(conn, series_id, series_desc, df)
            total_rows += rows
            log.info(f"  ✓ {series_id}: {rows} rows written to bronze.")

        except Exception as exc:
            # Log the error but keep going — one bad series
            # shouldn't abort the whole run.
            log.error(f"  ✗ {series_id} failed: {exc}")

    log.info(f"Done. Total rows in bronze.fred_observations: {total_rows}")
    conn.close()


if __name__ == "__main__":
    main()