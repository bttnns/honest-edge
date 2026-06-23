"""Honest evaluation: time-respecting splits, a cost-aware backtest, and metrics.

This module is the course's truth serum. Section 2 builds it; every later section
reuses it. It exists to answer one question without lying: *if we had actually
traded this rule, paying real costs and never peeking at the future, what would
have happened, and was it better than just holding SPY?*

Everything here obeys the no-look-ahead contract from Section 0: a position
decided at bar t's close is applied to the t -> t+1 return, and costs are charged
when the position actually changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Time-respecting cross-validation
# --------------------------------------------------------------------------
def purged_walk_forward(n, n_splits=5, embargo_frac=0.02, label_horizon=1):
    """Expanding-window walk-forward split with purging and an embargo.

    Train on the past, test on the next contiguous block, roll forward. Two
    safeguards stop overlapping labels from leaking the future into training:

      - PURGE: drop the last `label_horizon` training rows, whose forward-looking
        label would otherwise reach into the test block.
      - EMBARGO: drop a small extra buffer (embargo_frac of all rows) just before
        each test block, to kill leakage through autocorrelation.

    Parameters
    ----------
    n : int                length of the dataset (number of rows)
    n_splits : int         number of test folds
    embargo_frac : float   embargo size as a fraction of n (0.02 of ~5000 daily bars is ~100 bars, a few months)
    label_horizon : int    how many bars forward the label looks (1 for next-bar)

    Returns
    -------
    list of (train_idx, test_idx) integer-position arrays, with
    max(train_idx) strictly less than min(test_idx) by the purge + embargo gap.
    Fewer than n_splits folds may be returned if an early fold leaves no room to
    train (only happens on very short series).
    """
    embargo = int(embargo_frac * n)
    # Split all rows into n_splits+1 contiguous blocks; the first is train-only,
    # the remaining n_splits blocks are each used once as a test set.
    blocks = np.array_split(np.arange(n), n_splits + 1)
    folds = []
    for i in range(1, n_splits + 1):
        test_idx = blocks[i]
        # Exclusive upper bound on train. Dropping rows [train_hi, test_lo) removes
        # exactly label_horizon (purge) + embargo rows whose labels or market mood
        # would otherwise bleed into the test block.
        train_hi = test_idx[0] - label_horizon - embargo
        if train_hi <= 0:
            continue
        train_idx = np.arange(0, train_hi)
        folds.append((train_idx, test_idx))
    return folds


def shuffled_kfold(n, n_splits=5, seed=0):
    """THE CHEATING BASELINE. A plain shuffled K-fold split.

    This deliberately ignores time order, scattering near-duplicate neighboring
    rows across train and test. We include it only to MEASURE how much that
    cheating inflates a score. Never validate a real strategy with this.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = []
    for test_idx in np.array_split(perm, n_splits):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=True)
        folds.append((np.sort(train_idx), np.sort(test_idx)))
    return folds


# --------------------------------------------------------------------------
# Cost-aware backtest
# --------------------------------------------------------------------------
def backtest_positions(positions, close, cost_bps=2.0):
    """Turn a series of positions into net per-bar strategy returns.

    Enforces the two honesty rules:
      - one-bar execution lag: the position decided at close t is held over the
        t -> t+1 bar (positions.shift(1)), so we never trade on a price we could
        not have known.
      - costs on turnover: we pay `cost_bps` (basis points, 1 bp = 0.01%) each
        time the held position changes, i.e. on every actual trade.

    Parameters
    ----------
    positions : Series   target position per bar (e.g. 1=long, 0=flat), decided at that bar's close
    close : Series       close prices, same index
    cost_bps : float     round-trip-ish cost per unit of turnover, in basis points

    Returns
    -------
    Series of net strategy returns per bar.
    """
    ret = close.pct_change().fillna(0.0)
    held = positions.reindex(close.index).shift(1).fillna(0.0)   # position actually held over each bar
    trade = held.diff().abs()
    trade.iloc[0] = 0.0                                          # bar 0 is flat (shift), so no diff and no cost
    cost = (cost_bps / 1e4) * trade
    return held * ret - cost


def buy_and_hold(close):
    """Always-long per-bar returns: the headline baseline every strategy must beat."""
    return close.pct_change().fillna(0.0)


def random_positions(index, seed=0, long_only=True):
    """Coin-flip positions (the 'monkey' baseline). Same machinery, no skill."""
    rng = np.random.default_rng(seed)
    if long_only:
        vals = rng.integers(0, 2, len(index))          # 0 or 1
    else:
        vals = rng.integers(-1, 2, len(index))         # -1, 0, or 1
    return pd.Series(vals, index=index, dtype=float)


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------
def perf_metrics(returns, periods_per_year=252):
    """Summarize a return stream. Always read risk-adjusted AND total return together.

    Returns a dict: total_return, ann_return, ann_vol, sharpe, sortino,
    max_drawdown, calmar, hit_rate. Daily data -> periods_per_year=252.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return {k: np.nan for k in
                ["total_return", "ann_return", "ann_vol", "sharpe", "sortino",
                 "max_drawdown", "calmar", "hit_rate"]}

    total_return = (1 + r).prod() - 1
    ann_return = (1 + total_return) ** (periods_per_year / n) - 1
    std = r.std(ddof=1)
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = (r.mean() / std) * np.sqrt(periods_per_year) if std > 0 else np.nan

    # Downside deviation: root-mean-square of the negative returns, averaged over
    # ALL periods (target return = 0). The common full-sample convention.
    downside = np.sqrt((r.clip(upper=0.0) ** 2).mean())
    sortino = (r.mean() / downside) * np.sqrt(periods_per_year) if downside > 0 else np.nan

    equity = (1 + r).cumprod()
    max_drawdown = (equity / equity.cummax() - 1).min()
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    # Hit rate over ACTIVE bars only (those with non-zero P&L). A long/flat strategy
    # sits flat most of the time; counting those zero-return days as non-wins would
    # bury the real number (e.g. a rule in the market 10% of the time would show a
    # ~5% "hit rate"). For an always-invested series like buy-and-hold, nearly every
    # bar is active, so this matches the naive share-of-up-days. Caveat: this is bar
    # level and cost-aware, so a bar that only pays an exit cost counts as a loss;
    # for a per-trade win rate, measure the trades directly (see Section 3).
    active = r != 0
    hit_rate = (r[active] > 0).mean() if active.any() else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": hit_rate,
    }


# --------------------------------------------------------------------------
# Small display helpers (used by the notebooks to render tables and curves)
# --------------------------------------------------------------------------
def equity_curve(returns):
    """Growth of $1: the cumulative product of (1 + per-bar returns)."""
    return (1 + pd.Series(returns)).cumprod()


def summary_row(returns):
    """A compact metrics dict for side-by-side tables.

    Projects `perf_metrics` down to the four numbers the course compares most:
    Sharpe, Sortino, max drawdown, and total return. Defined here (not in each
    notebook) so every section renders metrics exactly one way.
    """
    m = perf_metrics(returns)
    return {"Sharpe": m["sharpe"], "Sortino": m["sortino"],
            "max DD": m["max_drawdown"], "total return": m["total_return"]}
