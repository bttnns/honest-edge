"""Technical indicators, computed with no look-ahead.

Every indicator here reads only past and present bars: the value at bar t uses
data up to and including t's close, never the future. These are the building
blocks for the Connors RSI-2 signal (Section 3) and the features (Section 5),
so they are defined once, here, and reused everywhere.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100.

    High RSI = recent moves were mostly up (overbought); low = mostly down
    (oversold). We use Wilder's smoothing, an exponential moving average with
    alpha = 1/length, which is the standard definition. This is the same `rsi`
    used in Section 1; Section 3 just calls it with length=2.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # When there are no down moves, RSI is defined as 100 (pure uptrend). When
    # there is no movement at all, fall back to the neutral 50 (never happens on
    # real prices, but keeps the function well-defined).
    flat = (avg_gain == 0) & (avg_loss == 0)
    out = out.where(avg_loss != 0, 100.0)
    return out.where(~flat, 50.0)


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple moving average: the rolling mean over `length` bars."""
    return close.rolling(length).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range (Wilder): a volatility gauge in price units.

    True Range is the largest of: today's high-low, |high - prev close|, and
    |low - prev close|. ATR is Wilder's smoothing of that. Reserved for the
    triple-barrier labels in Section 4 (barriers scaled to volatility).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def bollinger_pct(close: pd.Series, length: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger %B: where price sits inside its bands (0 = lower, 1 = upper).

    Bands are the `length`-bar average plus/minus `n_std` standard deviations.
    %B can go below 0 or above 1 when price pierces a band. Reserved as a
    feature in Section 5.
    """
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return (close - lower) / (upper - lower)


def bollinger_width(close: pd.Series, length: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger band width as a fraction of the mid: (upper - lower) / mid = 2*n_std*sd/mid.

    A volatility gauge: wide bands mean a turbulent regime, narrow bands a calm one.
    Shares its band definition (same `length`, same `n_std`, population std) with
    `bollinger_pct`, so '2 sigma' means one thing everywhere. Used as a Section 5 feature.
    """
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    return (2.0 * n_std * sd) / mid
