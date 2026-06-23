"""The primary signal: Connors RSI-2 mean-reversion on SPY.

This is the strategy the course puts on trial. It is a real, named, decades-old
rule (Connors & Alvarez, "Short Term Trading Strategies That Work", 2008), not an
ML guess. The idea: in an uptrend, buy short-term panic and sell the bounce.

    enter long when:  close > SMA(200)   (only buy dips in an uptrend)
                and   RSI(2) < rsi_buy   (price is sharply oversold)
    exit when:        close > SMA(5)     (the bounce has arrived)
    stop:             none               (Connors found stops hurt mean-reversion)

Everything is decided at a bar's close from data up to that close. The backtest
(`evaluation.backtest_positions`) applies the one-bar execution lag, so we never
shift here: doing so would lag the trade twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honest_edge import indicators as ind


def connors_rsi2_signals(df, rsi_len=2, rsi_buy=10, sma_trend=200, sma_exit=5):
    """Return the raw (entry, exit) boolean Series for the Connors RSI-2 rule.

    `entry` is True on EVERY bar where the entry condition holds (uptrend +
    oversold), so a single dip that stays oversold for two or three bars shows up
    as a run of consecutive True values, not one event. `exit` is True on bars
    where an open long should be closed (price back above its short average).
    Section 4 builds its candidate-trade list from the DISTINCT dips (the first
    bar of each oversold run), not from the raw per-bar `entry` mask, so
    overlapping near-duplicate labels do not double-count the same trade. Note
    those 168 dips are NOT the 150 round-trips `connors_rsi2` actually trades: 18
    dips fire while a prior trade is still open and are never entered. See NOTES.md.
    """
    close = df["close"]
    rsi2 = ind.rsi(close, rsi_len)
    trend = ind.sma(close, sma_trend)
    short_ma = ind.sma(close, sma_exit)

    entry = (close > trend) & (rsi2 < rsi_buy)   # NaN during SMA warm-up -> False (no signal yet)
    exit_ = close > short_ma
    return entry.fillna(False), exit_.fillna(False)


def _walk_positions(n, exit_mask, should_enter, size_at):
    """Shared long/flat state machine for both signals below.

    When flat, `should_enter(i)` opening a position of size `size_at(i)` on bar i;
    while in a trade, the position is held at that size until `exit_mask[i]` fires,
    then it goes flat. `connors_rsi2` and `filtered_positions` differ ONLY in their
    entry rule (a live oversold condition vs a pre-approved dip list), so both route
    their bar loop through here and the hold/exit mechanics stay identical by
    construction. Returns the held position per bar as a NumPy array.
    """
    pos = np.zeros(n)
    in_trade = False
    cur = 0.0
    for i in range(n):
        if in_trade:
            if exit_mask[i]:          # exit: flat from this close onward
                in_trade = False
                cur = 0.0
            else:
                pos[i] = cur          # still in the trade, at its size
        elif should_enter(i):         # flat and an entry fires: open at its size
            in_trade = True
            cur = size_at(i)
            pos[i] = cur
    return pos


def connors_rsi2(df, rsi_len=2, rsi_buy=10, sma_trend=200, sma_exit=5):
    """Connors RSI-2 long-only positions: a 0/1 Series decided at each close.

    Walks the bars with a small state machine: when flat, a live entry signal
    opens a long; while long, the position stays 1 until an exit signal closes
    it. Holding a minimum of one bar avoids flip-flopping when an entry and exit
    condition land on the same bar. The position is what we HELD as of each
    close; the backtest lags it one bar to get tradeable returns.
    """
    entry, exit_ = connors_rsi2_signals(df, rsi_len, rsi_buy, sma_trend, sma_exit)
    e = entry.to_numpy()
    x = exit_.to_numpy()
    pos = _walk_positions(len(df), x, lambda i: bool(e[i]), lambda i: 1.0)
    return pd.Series(pos, index=df.index, name="position")


def filtered_positions(df, sizes, sma_exit=5):
    """The same RSI-2 state machine, but only enter the dips a meta-model approved.

    This is how Section 6 trades the meta-labeling decision. `sizes` is a Series
    indexed by dip entry date, whose value is the position SIZE for that dip: 0 means
    "skip this dip", a positive value (1.0 for a plain take, or a fraction for a
    confidence-sized bet) means "enter long at this size". When flat and a fresh,
    approved dip arrives we enter at its size and hold until the usual exit (close
    back above the `sma_exit`-day average), exactly like `connors_rsi2`. Dips that
    are not in `sizes` (or have size 0) are passed over.

    With every dip approved at size 1.0 this reproduces `connors_rsi2` bar for bar.
    The position is what we HELD at each close; the backtest adds the one-bar lag.
    """
    close = df["close"]
    exit_mask = (close > ind.sma(close, sma_exit)).fillna(False).to_numpy()
    idx = df.index
    # Map approved dip dates to integer bar positions. get_indexer returns -1 for any
    # date not on this calendar (a holiday, a differently-trimmed frame), which we skip
    # rather than crash, the same robustness triple_barrier_labels uses for its events.
    locs = idx.get_indexer(pd.Index(sizes.index))
    size_by_pos = {loc: float(s) for loc, s in zip(locs, sizes.to_numpy())
                   if loc >= 0 and float(s) > 0}

    pos = _walk_positions(len(df), exit_mask,
                          lambda i: i in size_by_pos, lambda i: size_by_pos[i])
    return pd.Series(pos, index=idx, name="position")
