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
    bar of each run, or equivalently the 0 -> 1 transitions of `connors_rsi2`),
    not from the raw per-bar `entry` mask, so overlapping near-duplicate labels do
    not double-count the same trade.
    """
    close = df["close"]
    rsi2 = ind.rsi(close, rsi_len)
    trend = ind.sma(close, sma_trend)
    short_ma = ind.sma(close, sma_exit)

    entry = (close > trend) & (rsi2 < rsi_buy)   # NaN during SMA warm-up -> False (no signal yet)
    exit_ = close > short_ma
    return entry.fillna(False), exit_.fillna(False)


def connors_rsi2(df, rsi_len=2, rsi_buy=10, sma_trend=200, sma_exit=5):
    """Connors RSI-2 long-only positions: a 0/1 Series decided at each close.

    Walks the bars with a small state machine: when flat, a fresh entry signal
    opens a long; while long, the position stays 1 until an exit signal closes
    it. Holding a minimum of one bar avoids flip-flopping when an entry and exit
    condition land on the same bar. The position is what we HELD as of each
    close; the backtest lags it one bar to get tradeable returns.
    """
    entry, exit_ = connors_rsi2_signals(df, rsi_len, rsi_buy, sma_trend, sma_exit)
    e = entry.to_numpy()
    x = exit_.to_numpy()

    pos = np.zeros(len(df))
    in_trade = False
    for i in range(len(df)):
        if in_trade:
            if x[i]:                  # exit signal: flat from this close onward
                in_trade = False
            else:
                pos[i] = 1.0          # still long
        elif e[i]:                    # flat and a fresh oversold dip: go long
            in_trade = True
            pos[i] = 1.0
    return pd.Series(pos, index=df.index, name="position")
