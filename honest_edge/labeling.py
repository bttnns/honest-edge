"""Triple-barrier labels and overlap-aware sample weights (Section 4).

Section 3 left us with a candidate generator: the Connors RSI-2 rule raises its
hand on 168 distinct oversold dips, but takes every one blindly. To teach a model
*which* dips are worth taking (the meta-labeling of Sections 5-6), we first need an
honest answer key: for each dip, was it a win or a loss? This module builds that
answer key the right way.

Two ideas from Lopez de Prado's *Advances in Financial Machine Learning* (2018):

  - The TRIPLE-BARRIER METHOD (Ch. 3). Put three exit barriers around each trade:
    a profit-take above, a stop-loss below, and a time limit to the right. The
    label is whichever barrier the price touches FIRST. The horizontal barriers are
    scaled to volatility (here ATR), because a fixed 2% target is trivial to hit in
    a panic and nearly impossible in a calm month; scaling to ATR makes a "win"
    mean the same thing in every regime.

  - SAMPLE WEIGHTS by uniqueness (Ch. 4). Because dips cluster, their holding
    windows overlap in time, so the labels are NOT independent. A burst of
    near-simultaneous trades should not count as many independent votes. We weight
    each label by how UNIQUE its window is (and by the return it earned), so the
    model and our cross-validation do not over-trust a redundant cluster.

The no-look-ahead contract (Section 0) is enforced throughout: a dip's barriers are
sized from ATR as of the ENTRY bar's close (data up to t0 only), and the
first-touch search then walks strictly FORWARD, from t0+1 onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honest_edge import indicators as ind


# --------------------------------------------------------------------------
# The triple-barrier method
# --------------------------------------------------------------------------
def triple_barrier_labels(df, events, side=1, atr_mult=2.0, max_hold=5, atr_len=14):
    """Label each event by which of three barriers price touches first.

    For every entry timestamp in `events`, we draw three barriers around the entry
    close and walk forward until one is touched:

      - UPPER  (profit-take): entry + atr_mult * ATR(entry)
      - LOWER  (stop-loss)  : entry - atr_mult * ATR(entry)
      - VERTICAL (time limit): max_hold bars after entry

    Touches are read on the CLOSE path (the course's whole engine is close-to-close,
    market-on-close), so a single close is one number and can never breach the upper
    and lower barrier at once: there is no intrabar tie to break. If neither
    horizontal barrier is hit within `max_hold` bars, the vertical barrier resolves
    the trade at that bar's close.

    Because the primary signal already fixes the SIDE of the bet (here `side=1`,
    always long the dip), the label is the binary META-LABEL of Ch. 3:

        label = 1  if the side-adjusted return at the first touch is positive
                0  otherwise   (a stop-out, or a time-out that ended underwater)

    A profit-take is always a 1 and a stop-loss is always a 0; only the vertical
    barrier is decided by the sign of the realized return. 1 means "this dip would
    have paid, take it"; 0 means "skip it". That is exactly the target Section 6's
    model learns.

    No look-ahead: ATR(entry) uses data only through the entry close, and the
    first-touch search starts at the NEXT bar (t0+1). Trades whose full `max_hold`
    window would run past the end of the data are RIGHT-CENSORED and dropped, rather
    than scored on a truncated path (which would quietly bias the most recent labels).

    Parameters
    ----------
    df : DataFrame      OHLC(V) with a DatetimeIndex (e.g. from data.load_spy_daily)
    events : index-like entry timestamps, one per distinct dip (NOT the raw per-bar
             mask; see signal.connors_rsi2_signals)
    side : int          +1 long, -1 short. The primary signal's direction.
    atr_mult : float    barrier half-width in ATRs (symmetric: same up and down)
    max_hold : int      vertical barrier, in bars after entry
    atr_len : int       ATR lookback (Wilder)

    Returns
    -------
    DataFrame indexed by entry date (t0), with columns:
        t1       touch date (entry to exit interval is [t0, t1])
        entry, upper, lower   the price and its two horizontal barriers
        ret      side-adjusted simple return from entry close to t1 close
        outcome  'pt' | 'sl' | 'time'  (which barrier was touched first)
        hold     bars held (t1 - t0, in index positions)
        label    the binary meta-label, 1 (win/take) or 0 (loss/skip)
    """
    close = df["close"]
    atr = ind.atr(df["high"], df["low"], close, atr_len)
    closes = close.to_numpy()
    atrs = atr.to_numpy()
    idx = df.index
    n = len(df)

    locs = idx.get_indexer(pd.Index(events))   # integer positions of the entry bars
    rows = []
    for t0 in locs:
        if t0 < 0:
            continue
        a = atrs[t0]
        if not np.isfinite(a):                 # ATR still warming up: no defined barrier
            continue
        if t0 + max_hold > n - 1:              # right-censored near the end of data: drop
            continue

        entry = closes[t0]
        upper = entry + atr_mult * a
        lower = entry - atr_mult * a

        # Which horizontal barrier MEANS profit vs stop depends on the side: for a long
        # the upper is the profit-take and the lower is the stop; for a short they flip.
        # Tag the outcome by its meaning, not by direction, so 'pt'/'sl' stay honest.
        lower_outcome = "sl" if side == 1 else "pt"
        upper_outcome = "pt" if side == 1 else "sl"

        t1 = t0 + max_hold                     # default: the vertical (time) barrier
        outcome = "time"
        for t in range(t0 + 1, t0 + max_hold + 1):   # walk FORWARD from the next bar
            c = closes[t]
            if c <= lower:                     # lower barrier touched first
                t1, outcome = t, lower_outcome
                break
            if c >= upper:                     # upper barrier touched first
                t1, outcome = t, upper_outcome
                break

        ret = side * (closes[t1] / entry - 1.0)
        rows.append({
            "t0": idx[t0], "t1": idx[t1], "entry": entry, "upper": upper, "lower": lower,
            "ret": ret, "outcome": outcome, "hold": t1 - t0,
            "label": int(ret > 0),
        })

    return pd.DataFrame(rows).set_index("t0")


# --------------------------------------------------------------------------
# Overlap-aware sample weights (Ch. 4)
# --------------------------------------------------------------------------
def count_concurrent(index, labels):
    """How many labels are 'alive' on each bar: the concurrency series c_t.

    A label spans the inclusive interval [t0, t1] (entry to first touch). c_t is the
    number of those intervals that cover bar t. Where c_t > 1, trades overlap, which
    is the whole reason the next two functions exist. Bars no label touches are 0.
    """
    c = pd.Series(0, index=index, dtype="int64")
    for t0, t1 in labels["t1"].items():
        c.loc[t0:t1] += 1
    return c


def average_uniqueness(labels, concurrency):
    """Average uniqueness of each label: the mean of 1/c_t over its own lifespan.

    A label that never overlaps anything has uniqueness 1.0. One that shares all of
    its bars with a second label averages around 0.5, and so on. The SUM of these
    per-label uniqueness values is the honest 'effective number of independent
    samples': with heavy overlap it can sit well below the raw label count.
    """
    u = {}
    for t0, t1 in labels["t1"].items():
        seg = concurrency.loc[t0:t1]
        u[t0] = (1.0 / seg).mean()
    return pd.Series(u, name="uniqueness")


def return_attribution_weights(close, labels, concurrency):
    """Sample weights by absolute, concurrency-deflated return (de Prado 4.10).

    Each bar's log return is split across the c_t labels alive on it (r_t / c_t),
    summed over a label's lifespan, and taken in absolute value. This rewards labels
    driven by large, decisive moves and, via the 1/c_t split, stops one big day from
    being counted once for every overlapping trade. Weights are normalized to sum to
    the number of labels (so the mean weight is 1, and the training loss keeps its
    usual scale). Pass these to a classifier as `sample_weight` in Section 6.
    """
    logret = np.log(close).diff()
    w = {}
    for t0, t1 in labels["t1"].items():
        # NOTE: the slice [t0, t1] is inclusive of the entry bar, so it folds in the
        # return earned INTO t0 (the pre-entry move), not just the trade's own path. We
        # keep this deliberately: it reproduces de Prado AFML Snippet 4.10 (p. 69) exactly,
        # and matches the [t0, t1]-inclusive convention `count_concurrent` uses above.
        seg_r = logret.loc[t0:t1]
        seg_c = concurrency.loc[t0:t1]
        w[t0] = (seg_r / seg_c).sum()
    w = pd.Series(w, name="weight").abs()
    if w.sum() > 0:
        w *= len(w) / w.sum()
    return w
