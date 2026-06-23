"""Loading SPY price data, cleanly and the same way every time.

This is the foundation every later notebook stands on. The job here is small but
easy to get subtly wrong: read an OHLCV CSV into a tidy, time-sorted table that
later code can trust. Four invariants make a price table trustworthy:

    1. parse the timestamp into a real datetime (not a string),
    2. sort ascending in time (never assume the file is in order),
    3. drop duplicate timestamps (exports sometimes repeat a row),
    4. put the datetime on the index (so time-based slicing just works).

We ship two SPY files inside the package, in honest_edge/data/:
    spy_daily.csv   one bar per trading day, 2006 -> 2026  (~5000 bars)
    spy_hourly.csv  one bar per market hour, 2023 -> 2026  (~5000 bars)

Both have columns: datetime, symbol, open, high, low, close, volume.
Two quirks we normalize away: the symbol carries an exchange prefix ("AMEX:SPY")
and the daily bars are stamped at 08:30 (a data-export artifact), not midnight.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Resolve the data folder relative to THIS file, not the caller's working
# directory. That way a notebook loads the same data no matter where Jupyter was
# launched from. The CSVs ship inside the package (honest_edge/data/), so this
# also resolves correctly after a plain `pip install .`, not just an editable one.
DATA_DIR = Path(__file__).resolve().parent / "data"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(path: str | Path, *, daily: bool = False) -> pd.DataFrame:
    """Load one OHLCV CSV into a clean, time-indexed DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV (columns: datetime, symbol, open, high, low, close, volume).
    daily : bool
        If True, snap the index to the calendar date. Our daily bars are stamped
        08:30 (an export quirk); normalizing makes "one row per day" behave when
        we join or resample later. Leave False for intraday (hourly) data.

    Returns
    -------
    DataFrame indexed by datetime, columns: symbol, open, high, low, close, volume.
    """
    df = pd.read_csv(path)

    # Lower-case the headers so we never fight over "Close" vs "close".
    df.columns = df.columns.str.lower()

    # (1) Parse the timestamp into a real datetime.
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Strip the exchange prefix: "AMEX:SPY" -> "spy". (.str works on pandas 3.0's
    # native string dtype just as it did on the old object dtype.)
    df["symbol"] = df["symbol"].str.split(":").str[-1].str.lower()

    # (2) sort ascending, (3) drop duplicate timestamps, (4) index by datetime.
    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime", keep="first")
        .set_index("datetime")
    )

    if daily:
        # 08:30 stamp -> the date itself (midnight). Pure relabeling of the index.
        df.index = df.index.normalize()

    # Return columns in a predictable order; keep symbol for sanity checks.
    return df[["symbol", *OHLCV_COLUMNS]]


def load_spy_daily() -> pd.DataFrame:
    """SPY daily bars, 2006 -> 2026. The main dataset for the course."""
    return load_ohlcv(DATA_DIR / "spy_daily.csv", daily=True)


def load_spy_hourly() -> pd.DataFrame:
    """SPY hourly bars, 2023 -> 2026. Reserved for the Section 8 confirmation lesson."""
    return load_ohlcv(DATA_DIR / "spy_hourly.csv", daily=False)
