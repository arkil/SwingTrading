"""
Breakout Screener — 6 Research-Backed Strategies
=================================================
Sources: Minervini SEPA, Darvas Box, NR7 (Toby Crabel), Bollinger Squeeze,
         IBD/CAN SLIM, QuantifiedStrategies backtests.

Strategies implemented:
  1. 52-Week High Breakout   — 68-76% WR in bull markets (volume-confirmed)
  2. Volume Surge Breakout   — Price breaks resistance with 150%+ avg volume
  3. NR7 Breakout            — Narrowest range in 7 bars → volatility expansion
  4. Bollinger Band Squeeze  — BB bandwidth at 6-month low then close beyond band
  5. Inside Bar Breakout     — Tight inside bar → close outside mother bar
  6. MA Reclaim Breakout     — Price reclaims / breaks 50-day EMA with volume
"""

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from livermore_pivotal_screener import get_universe, DEFAULT_TICKERS


# ── Shared indicators ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

def _rs(close: pd.Series, spy_close: pd.Series, period: int = 52) -> pd.Series:
    """Simple relative strength vs SPY over `period` bars."""
    stock_ret = close / close.shift(period) - 1
    spy_ret   = spy_close / spy_close.shift(period) - 1
    return stock_ret - spy_ret


def _bars_since(flag: pd.Series) -> pd.Series:
    """
    Returns how many bars ago `flag` was last True (0 = current bar is True).
    Returns a large number (999) where flag has never been True.
    Fully vectorised — no ambiguous truth-value operations.
    """
    idx   = np.arange(len(flag))
    last  = pd.Series(np.where(flag, idx, np.nan), index=flag.index)
    last  = last.ffill().fillna(-999)
    return pd.Series(idx - last.values, index=flag.index, dtype=int)


def _consec_true(s: pd.Series) -> pd.Series:
    """
    Count consecutive True bars (resets to 0 on False).
    Vectorised: avoids groupby on boolean Series.
    """
    # Each "run" gets a unique id by counting leading-edge changes
    runs   = (s != s.shift(1)).cumsum()
    counts = s.groupby(runs).cumsum()        # cumulative count within each run
    return counts.where(s, other=0).astype(int)


def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"]    = _ema(df["Close"], 50)
    df["ema150"]   = _ema(df["Close"], 150)
    df["ema200"]   = _ema(df["Close"], 200)
    df["sma50"]    = _sma(df["Close"], 50)
    df["atr14"]    = _atr(df, 14)
    df["vol20"]    = _sma(df["Volume"], 20)
    df["high52w"]  = df["High"].rolling(252).max()
    df["low52w"]   = df["Low"].rolling(252).min()
    # Bollinger Bands (20, 2)
    mid             = _sma(df["Close"], 20)
    std             = df["Close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_mid"]   = mid
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid   # normalized bandwidth
    return df


# ── Strategy 1: 52-Week High Breakout ────────────────────────────────────────

def scan_52w_high(df: pd.DataFrame, vol_mult: float = 1.5) -> pd.Series:
    """
    Bullish: Close > prior 52W high (new all-time 1-year high) + volume surge.
    Bearish: Close < prior 52W low + volume surge.
    Uses the 252-bar rolling max/min shifted by 1 to avoid look-ahead.
    """
    prev_high52 = df["high52w"].shift(1)
    prev_low52  = df["low52w"].shift(1)
    vol_ok      = df["Volume"] > df["vol20"] * vol_mult
    trend_up    = df["ema50"] > df["ema200"]
    trend_down  = df["ema50"] < df["ema200"]

    bull = (df["Close"] > prev_high52) & vol_ok & trend_up
    bear = (df["Close"] < prev_low52)  & vol_ok & trend_down

    sig = pd.Series("", index=df.index)
    sig[bull] = "BULL_52W_HIGH"
    sig[bear] = "BEAR_52W_LOW"
    return sig


# ── Strategy 2: Volume Surge Breakout ─────────────────────────────────────────

def scan_volume_surge(
    df: pd.DataFrame,
    lookback: int = 20,
    vol_mult: float = 2.0,
    breakout_pct: float = 0.5,
) -> pd.Series:
    """
    Finds resistance/support as rolling max/min over `lookback` bars (shifted),
    then flags closes that break out with vol_mult × avg volume.
    breakout_pct: close must exceed resistance by at least this % to reduce fakeouts.
    """
    resistance = df["High"].rolling(lookback).max().shift(1)
    support    = df["Low"].rolling(lookback).min().shift(1)
    vol_ok     = df["Volume"] > df["vol20"] * vol_mult
    # Close must be a real break, not just touching the level
    bull = (df["Close"] > resistance * (1 + breakout_pct / 100)) & vol_ok
    bear = (df["Close"] < support  * (1 - breakout_pct / 100)) & vol_ok
    # Close near high/low of breakout candle (conviction)
    bar_range     = df["High"] - df["Low"]
    close_near_hi = (df["High"] - df["Close"]) < bar_range * 0.35
    close_near_lo = (df["Close"] - df["Low"])  < bar_range * 0.35

    sig = pd.Series("", index=df.index)
    sig[bull & close_near_hi] = "BULL_VOL_SURGE"
    sig[bear & close_near_lo] = "BEAR_VOL_SURGE"
    return sig


# ── Strategy 3: NR7 Breakout ──────────────────────────────────────────────────

def scan_nr7(df: pd.DataFrame) -> pd.Series:
    """
    NR7: day whose High-Low range is the smallest of the last 7 bars.
    Signal fires on the NEXT bar's close breaking above/below the NR7 bar's H/L.
    """
    bar_range = df["High"] - df["Low"]
    # True if today's range is min of last 7 bars (inclusive)
    is_nr7    = bar_range == bar_range.rolling(7).min()

    nr7_high  = df["High"].where(is_nr7).ffill()
    nr7_low   = df["Low"].where(is_nr7).ffill()

    # Only fire if the NR7 bar was within the last 3 bars
    bars_since_nr7 = _bars_since(is_nr7)

    bull = (df["Close"] > nr7_high.shift(1)) & (bars_since_nr7 <= 3)
    bear = (df["Close"] < nr7_low.shift(1))  & (bars_since_nr7 <= 3)

    # Trend context: above/below 50 EMA
    bull = bull & (df["Close"] > df["ema50"])
    bear = bear & (df["Close"] < df["ema50"])

    sig = pd.Series("", index=df.index)
    sig[bull] = "BULL_NR7"
    sig[bear] = "BEAR_NR7"
    return sig


# ── Strategy 4: Bollinger Band Squeeze Breakout ───────────────────────────────

def scan_bb_squeeze(
    df: pd.DataFrame,
    squeeze_lookback: int = 126,   # ~6 months
    vol_mult: float = 1.5,
) -> pd.Series:
    """
    Squeeze: BB width is at its lowest point in `squeeze_lookback` bars.
    Breakout: next close beyond upper or lower band with volume expansion.
    """
    # Squeeze: bandwidth at N-bar low
    bb_min = df["bb_width"].rolling(squeeze_lookback).min()
    in_squeeze    = df["bb_width"] == bb_min          # tightest point
    # Allow up to 5 bars after squeeze to fire
    post_squeeze = _bars_since(in_squeeze) <= 5

    vol_ok = df["Volume"] > df["vol20"] * vol_mult

    bull = (df["Close"] > df["bb_upper"]) & post_squeeze & vol_ok
    bear = (df["Close"] < df["bb_lower"]) & post_squeeze & vol_ok

    sig = pd.Series("", index=df.index)
    sig[bull] = "BULL_BB_SQUEEZE"
    sig[bear] = "BEAR_BB_SQUEEZE"
    return sig


# ── Strategy 5: Inside Bar Breakout ──────────────────────────────────────────

def scan_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Inside bar (IB): today's high < yesterday's high AND low > yesterday's low.
    Signal: next close outside the mother bar's range.
    Trend confirmation: price on correct side of 50 EMA.
    """
    prev_high = df["High"].shift(1)
    prev_low  = df["Low"].shift(1)

    is_inside  = (df["High"] < prev_high) & (df["Low"] > prev_low)
    mother_hi  = prev_high.where(is_inside).ffill()
    mother_lo  = prev_low.where(is_inside).ffill()
    recent_ib  = _bars_since(is_inside) <= 3

    bull = (df["Close"] > mother_hi.shift(1)) & recent_ib & (df["Close"] > df["ema50"])
    bear = (df["Close"] < mother_lo.shift(1)) & recent_ib & (df["Close"] < df["ema50"])

    sig = pd.Series("", index=df.index)
    sig[bull] = "BULL_INSIDE_BAR"
    sig[bear] = "BEAR_INSIDE_BAR"
    return sig


# ── Strategy 6: MA Reclaim / Break ────────────────────────────────────────────

def scan_ma_reclaim(df: pd.DataFrame, vol_mult: float = 1.2) -> pd.Series:
    """
    Bullish: price was BELOW 50 EMA for ≥3 bars, then closes ABOVE it (reclaim).
    Bearish: price was ABOVE 50 EMA for ≥3 bars, then closes BELOW it (break).
    Requires above-average volume on the reclaim/break bar.
    Minervini trend template: for bull signals, 50 EMA must be above 200 EMA.
    """
    above50 = df["Close"] > df["ema50"]
    below50 = df["Close"] < df["ema50"]

    # Count consecutive bars on each side using the safe helper
    consec_below = _consec_true(below50)
    consec_above = _consec_true(above50)

    vol_ok = df["Volume"] > df["vol20"] * vol_mult

    # Reclaim: was below ≥3 bars, now above
    bull = above50 & (consec_below.shift(1) >= 3) & vol_ok & (df["ema50"] > df["ema200"])
    # Break: was above ≥3 bars, now below
    bear = below50 & (consec_above.shift(1) >= 3) & vol_ok & (df["ema50"] < df["ema200"])

    sig = pd.Series("", index=df.index)
    sig[bull] = "BULL_MA_RECLAIM"
    sig[bear] = "BEAR_MA_BREAK"
    return sig


# ── Combine all strategies ────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "BULL_52W_HIGH":    ("52W High Breakout",    "bullish"),
    "BEAR_52W_LOW":     ("52W Low Breakdown",    "bearish"),
    "BULL_VOL_SURGE":   ("Volume Surge Breakout","bullish"),
    "BEAR_VOL_SURGE":   ("Volume Surge Breakdown","bearish"),
    "BULL_NR7":         ("NR7 Breakout",          "bullish"),
    "BEAR_NR7":         ("NR7 Breakdown",         "bearish"),
    "BULL_BB_SQUEEZE":  ("BB Squeeze Breakout",   "bullish"),
    "BEAR_BB_SQUEEZE":  ("BB Squeeze Breakdown",  "bearish"),
    "BULL_INSIDE_BAR":  ("Inside Bar Breakout",   "bullish"),
    "BEAR_INSIDE_BAR":  ("Inside Bar Breakdown",  "bearish"),
    "BULL_MA_RECLAIM":  ("MA Reclaim",            "bullish"),
    "BEAR_MA_BREAK":    ("MA Break",              "bearish"),
}

STRATEGY_GROUPS = {
    "52W Breakout":      ["BULL_52W_HIGH", "BEAR_52W_LOW"],
    "Volume Surge":      ["BULL_VOL_SURGE", "BEAR_VOL_SURGE"],
    "NR7":               ["BULL_NR7", "BEAR_NR7"],
    "BB Squeeze":        ["BULL_BB_SQUEEZE", "BEAR_BB_SQUEEZE"],
    "Inside Bar":        ["BULL_INSIDE_BAR", "BEAR_INSIDE_BAR"],
    "MA Reclaim":        ["BULL_MA_RECLAIM", "BEAR_MA_BREAK"],
}


def run_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 6 strategies to a prepared DataFrame. Returns annotated df."""
    df = add_base_indicators(df)
    sigs = pd.DataFrame(index=df.index)
    sigs["52w"]     = scan_52w_high(df)
    sigs["vol"]     = scan_volume_surge(df)
    sigs["nr7"]     = scan_nr7(df)
    sigs["bb"]      = scan_bb_squeeze(df)
    sigs["ib"]      = scan_inside_bar(df)
    sigs["ma"]      = scan_ma_reclaim(df)
    # Store all signals as separate columns for dashboard
    df["sig_52w"]  = sigs["52w"]
    df["sig_vol"]  = sigs["vol"]
    df["sig_nr7"]  = sigs["nr7"]
    df["sig_bb"]   = sigs["bb"]
    df["sig_ib"]   = sigs["ib"]
    df["sig_ma"]   = sigs["ma"]

    # Combined signal: comma-separated list of all that fired
    # Build vectorised to avoid df.apply returning a DataFrame in pandas 2.x
    # when yfinance leaves duplicate/MultiIndex columns behind.
    _sig_cols = ["sig_52w", "sig_vol", "sig_nr7", "sig_bb", "sig_ib", "sig_ma"]
    _parts = pd.DataFrame(index=df.index)
    for _c in _sig_cols:
        _col = df[_c]
        if isinstance(_col, pd.DataFrame):   # duplicate column edge-case
            _col = _col.iloc[:, 0]
        _parts[_c] = _col.astype(str).where(_col.astype(str).ne(""), "")
    df["all_signals"] = _parts.apply(
        lambda row: ", ".join(v for v in row if v and v != "nan"),
        axis=1,
        result_type="reduce",
    )
    return df


# ── Per-ticker screener ───────────────────────────────────────────────────────

def screen_breakout_ticker(
    ticker: str,
    recent_bars: int = 5,
    period_days: int = 400,
    strategies: Optional[List[str]] = None,   # None = all
) -> Optional[Dict[str, Any]]:

    end   = datetime.today()
    start = end - timedelta(days=period_days)

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return None

    if raw is None or len(raw) < 260:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = run_all_strategies(df)

    recent   = df.tail(recent_bars)
    sig_cols = ["sig_52w", "sig_vol", "sig_nr7", "sig_bb", "sig_ib", "sig_ma"]

    # Filter to user-selected strategies if specified
    if strategies:
        allowed = set()
        for s in strategies:
            allowed.update(STRATEGY_GROUPS.get(s, []))
    else:
        allowed = set(STRATEGY_LABELS.keys())

    # Find most recent bar with any active signal
    # Use iloc (positional) to guarantee a scalar Series even with duplicate dates
    best_row   = None
    best_date  = None
    best_fired = []
    n_recent   = len(recent)
    for i in range(n_recent - 1, -1, -1):
        row   = recent.iloc[i]
        fired = [str(row[c]) for c in sig_cols
                 if str(row.get(c, "")) and str(row.get(c, "")) in allowed]
        if fired:
            best_row   = row
            best_date  = recent.index[i]
            best_fired = fired
            break

    if best_row is None:
        return None

    current   = float(df["Close"].iloc[-1])
    bars_ago  = n_recent - 1 - i     # i is the loop index of the matched bar
    atr_val   = float(best_row["atr14"])
    entry     = best_row["Close"]

    # Direction from first signal
    direction = "bullish" if any(STRATEGY_LABELS.get(s, ("",""))[1] == "bullish" for s in best_fired) else "bearish"
    stop      = round(entry - 1.5 * atr_val, 2) if direction == "bullish" else round(entry + 1.5 * atr_val, 2)
    target    = round(entry + 3.0 * atr_val, 2) if direction == "bullish" else round(entry - 3.0 * atr_val, 2)

    pct_from_52h = (current / best_row["high52w"] - 1) * 100
    pct_from_52l = (current / best_row["low52w"]  - 1) * 100
    vol_ratio    = round(best_row["Volume"] / best_row["vol20"], 2) if best_row["vol20"] > 0 else None

    ema_stack = (
        "BULL" if df["ema50"].iloc[-1] > df["ema200"].iloc[-1] else "BEAR"
    )

    return {
        "Ticker":        ticker,
        "Signals":       ", ".join(best_fired),
        "Direction":     direction.upper(),
        "Signal Date":   best_date.strftime("%Y-%m-%d"),
        "Bars Ago":      bars_ago,
        "Entry":         round(entry, 2),
        "Current":       round(current, 2),
        "P&L %":         round((current / entry - 1) * 100 * (1 if direction == "bullish" else -1), 2),
        "Stop":          stop,
        "Target":        target,
        "R/R":           2.0,
        "Vol / Avg":     vol_ratio,
        "ATR14":         round(atr_val, 2),
        "% from 52W H":  round(pct_from_52h, 2),
        "% from 52W L":  round(pct_from_52l, 2),
        "EMA Trend":     ema_stack,
        "BB Width":      round(best_row["bb_width"] * 100, 2),
    }


# ── Full screener ─────────────────────────────────────────────────────────────

def run_breakout_screener(
    tickers: List[str],
    recent_bars: int = 5,
    strategies: Optional[List[str]] = None,
    direction_filter: str = "ALL",
) -> pd.DataFrame:

    results = []
    total   = len(tickers)
    label   = ", ".join(strategies) if strategies else "ALL strategies"

    print(f"\n{'='*60}")
    print(f"  Breakout Screener  |  {label}")
    print(f"  Scanning {total} tickers  |  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Signal window: last {recent_bars} bars")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>3}/{total}] {ticker:<8}", end="\r")
        result = screen_breakout_ticker(ticker, recent_bars=recent_bars, strategies=strategies)
        if result:
            results.append(result)

    print(" " * 40, end="\r")

    if not results:
        print("  No breakout signals found.\n")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if direction_filter != "ALL":
        df = df[df["Direction"] == direction_filter]

    df = df.sort_values(["Direction", "Bars Ago"], ascending=[True, True]).reset_index(drop=True)
    return df


def print_breakout_results(df: pd.DataFrame):
    if df.empty:
        return
    bull = df[df["Direction"] == "BULLISH"]
    bear = df[df["Direction"] == "BEARISH"]

    def _section(title, sec):
        if sec.empty: return
        cols = ["Ticker", "Signals", "Signal Date", "Bars Ago",
                "Entry", "Current", "P&L %", "Stop", "Target",
                "Vol / Avg", "% from 52W H", "EMA Trend"]
        cols = [c for c in cols if c in sec.columns]
        print(f"\n{'─'*90}\n  {title}  ({len(sec)})\n{'─'*90}")
        print(sec[cols].to_string(index=False))

    _section("BULLISH BREAKOUTS ▲", bull)
    _section("BEARISH BREAKDOWNS ▼", bear)
    print(f"\n{'='*90}")
    print(f"  Total: {len(df)}  |  Bullish: {len(bull)}  |  Bearish: {len(bear)}")
    print(f"{'='*90}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-Strategy Breakout Screener")
    p.add_argument("--tickers",    nargs="+", default=None)
    p.add_argument("--file",       type=str,  default=None)
    p.add_argument("--universe",   choices=["default","sp500","nasdaq100","both","watchlist"], default="default")
    p.add_argument("--strategies", nargs="+", choices=list(STRATEGY_GROUPS.keys()), default=None,
                   help="Limit to specific strategies. Default: all.")
    p.add_argument("--recent-bars",type=int,  default=5)
    p.add_argument("--direction",  choices=["ALL","BULLISH","BEARISH"], default="ALL")
    p.add_argument("--output",     type=str,  default=None)
    args = p.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.file:
        with open(args.file) as fh:
            tickers = [l.strip().upper() for l in fh if l.strip() and not l.startswith("#")]
    else:
        tickers = get_universe(args.universe)

    results = run_breakout_screener(
        tickers=tickers,
        recent_bars=args.recent_bars,
        strategies=args.strategies,
        direction_filter=args.direction,
    )
    print_breakout_results(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"  Saved to {args.output}\n")
