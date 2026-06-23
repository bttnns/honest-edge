"""Features that describe each dip (Section 5).

Section 4 built the answer key (the meta-label y, plus a sample weight) for each of
the 168 Connors RSI-2 dips. This module builds the OTHER half of the training set:
the feature matrix X, one row per dip, describing the *setup* the model sees before
it decides take-or-skip.

Every feature obeys the no-look-ahead contract (Section 0): its value at a dip's entry
bar t is computed from data only up to and including t's close. We compute each feature
as a causal (trailing) series over the whole history, then read off the values at the
entry timestamps, so a feature can never see past the bar it describes.

The features are deliberately grouped into a few economic THEMES, because within a
theme they are near-substitutes (three volatility gauges all measure roughly the same
thing). That redundancy is the point of Section 5's importance lesson: collinear
features confuse single-feature importance, so we read importance by cluster, not by
individual column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honest_edge import indicators as ind


# Which theme each feature belongs to. Used by the notebook to colour the correlation
# heatmap and to read importance by cluster rather than by individual (collinear) column.
FEATURE_GROUPS = {
    "dist200": "trend",
    "ret_5": "momentum", "ret_21": "momentum", "ret_63": "momentum", "ret_252": "momentum",
    "rv_21": "volatility", "atr_pct": "volatility", "bb_width": "volatility",
    "pullback_63": "pullback", "days_since_high_63": "pullback",
    "rsi2": "oversold", "gap5": "oversold",
    "overnight_21": "overnight", "on_minus_id_21": "overnight",
}


def _feature_frame(df):
    """All feature series over the full history, each strictly causal (trailing).

    Returns a DataFrame aligned to df.index; the notebook reads the rows at the dip
    entry timestamps. No look-ahead: every column at bar t uses data only through t.
    """
    close, high, low = df["close"], df["high"], df["low"]
    open_ = df["open"]
    logret = np.log(close).diff()

    f = pd.DataFrame(index=df.index)

    # --- Trend regime: how far above its 200-day average the price sits. ---
    f["dist200"] = close / ind.sma(close, 200) - 1.0

    # --- Momentum / past returns across horizons. Short = reversal, long = momentum. ---
    for k in (5, 21, 63, 252):
        f[f"ret_{k}"] = close.pct_change(k)

    # --- Volatility regime (three proxies for the same thing, on purpose). ---
    f["rv_21"] = logret.rolling(21).std() * np.sqrt(252)       # realized vol, annualized
    f["atr_pct"] = ind.atr(high, low, close, 14) / close       # ATR as a fraction of price
    f["bb_width"] = ind.bollinger_width(close, 20, 2.0)        # (upper - lower) / mid, shared band def

    # --- Pullback shape: depth below a recent high, and how long the slide has run. ---
    roll_max = close.rolling(63).max()
    f["pullback_63"] = close / roll_max - 1.0                  # <= 0; how far below the 63-day high
    # Bars since the MOST RECENT 63-day high: argmax on the reversed window finds the last
    # occurrence of the max, so a revisited / flat high reads as recent, not stale.
    f["days_since_high_63"] = close.rolling(63).apply(
        lambda a: int(np.argmax(a[::-1])), raw=True)

    # --- Depth of the oversold trigger itself. ---
    f["rsi2"] = ind.rsi(close, 2)                              # lower = more oversold
    f["gap5"] = close / ind.sma(close, 5) - 1.0               # stretch below the 5-day exit average

    # --- Overnight vs intraday character of the recent tape (the documented split). ---
    overnight = open_ / close.shift(1) - 1.0                   # close-to-open gap, known at t's close
    intraday = close / open_ - 1.0                            # open-to-close move, known at t's close
    f["overnight_21"] = overnight.rolling(21).mean()
    f["on_minus_id_21"] = (overnight - intraday).rolling(21).mean()  # the "tug of war" signal

    return f


def build_features(df, events):
    """Feature matrix X for the dips: one row per entry timestamp, columns per feature.

    Parameters
    ----------
    df : DataFrame      OHLC(V) with a DatetimeIndex
    events : index-like entry timestamps (the distinct dips from Section 4)

    Returns
    -------
    DataFrame indexed by entry date, one column per feature in FEATURE_GROUPS order.
    Rows still carrying an indicator warm-up NaN (e.g. an early dip without a full
    252-day history) are the caller's to drop; we leave them visible rather than
    silently imputing, so the no-look-ahead story stays clean.
    """
    f = _feature_frame(df)
    X = f.reindex(pd.Index(events))
    return X[list(FEATURE_GROUPS)]
