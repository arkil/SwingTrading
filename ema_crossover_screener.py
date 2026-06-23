"""
EMA Crossover Screener — Swing Trading Strategy
================================================
Based on research from QuantifiedStrategies, StockCharts, OpoFinance,
and AlphaExCapital literature on validated EMA crossover systems.

STRATEGY: 9/21 EMA Crossover with 55 EMA trend filter (daily timeframe)
──────────────────────────────────────────────────────────────────────────
ENTRY RULES (bullish):
  1. EMA-9 crosses ABOVE EMA-21 (crossover bar confirmed on close)
  2. Price close above EMA-55 (trend filter — trades in direction of trend)
  3. ADX(14) > 20 (market is trending, not ranging)
  4. Volume > 1.5x 20-bar average (institutional participation)
  5. RSI(14) between 30 and 70 (avoids overbought entries, not deeply oversold)

ENTRY RULES (bearish): mirror of above with EMA-9 crossing BELOW EMA-21

STOPS & TARGETS (ATR-based):
  Stop loss  = entry ± 1.5 × ATR(14)
  Target 1   = entry ± 2.0 × ATR(14)   (1.33:1 R/R — partial exit)
  Target 2   = entry ± 3.0 × ATR(14)   (2.0:1 R/R — full exit)

PULLBACK MODE (optional, higher-quality entries):
  After a crossover, wait for price to pull back and touch/cross the 21 EMA,
  then enter on the next candle that closes back in the trend direction.

ADDITIONAL FILTER (optional):
  200 EMA alignment: only take longs when price > EMA-200, shorts when < EMA-200.

SECONDARY PRESET — Conservative 20/50/200:
  Fast=20, Slow=50, Trend=200, same filter stack.
  Fewer signals, higher win-rate. Best for large-caps / indices.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Import the shared universe helpers from the Livermore screener
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from livermore_pivotal_screener import get_universe, DEFAULT_TICKERS


# ── Indicator helpers ─────────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed ADX."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_val   = atr(df, 1)  # raw TR
    atr_s    = tr_val.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr_s
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()


# ── Core signal detection ─────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, fast: int, slow: int, trend: int) -> pd.DataFrame:
    df = df.copy()
    df[f"ema{fast}"]  = ema(df["Close"], fast)
    df[f"ema{slow}"]  = ema(df["Close"], slow)
    df[f"ema{trend}"] = ema(df["Close"], trend)
    df["ema200"]      = ema(df["Close"], 200)
    df["rsi14"]       = rsi(df["Close"], 14)
    df["atr14"]       = atr(df, 14)
    df["adx14"]       = adx(df, 14)
    df["vol_avg20"]   = df["Volume"].rolling(20).mean()
    return df


def detect_crossovers(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    trend: int,
    adx_threshold: float = 15.0,
    vol_mult: float = 1.0,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    require_200_align: bool = False,
    pullback_mode: bool = False,
    atr_stop_mult: float = 1.5,
    atr_target1_mult: float = 2.0,
    atr_target2_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Detect EMA crossover signals and annotate with ATR stops/targets.
    Returns df with signal columns added.

    Filter logic (all must pass for a signal):
      ADX      — market must be trending; threshold typically 15-25
      Volume   — vol_mult × 20-bar avg; 1.0 = any volume, 1.5 = above-avg
      RSI      — directional: bullish needs RSI < rsi_high (not overbought);
                              bearish needs RSI > rsi_low  (not oversold)
      Trend    — price must be on the correct side of the trend EMA (±3% tolerance)
      200 EMA  — optional stricter alignment with the 200-bar EMA
    """
    df = compute_indicators(df, fast, slow, trend)

    fast_col  = f"ema{fast}"
    slow_col  = f"ema{slow}"
    trend_col = f"ema{trend}"

    prev_fast = df[fast_col].shift(1)
    prev_slow = df[slow_col].shift(1)

    # Raw crossovers
    bullish_cross = (prev_fast <= prev_slow) & (df[fast_col] > df[slow_col])
    bearish_cross = (prev_fast >= prev_slow) & (df[fast_col] < df[slow_col])

    # ── Filter stack ──────────────────────────────────────────────────────────

    # ADX: market must be trending
    trending = df["adx14"] > adx_threshold

    # Volume: only require above-average when vol_mult > 1.0
    if vol_mult > 1.0:
        vol_ok = df["Volume"] > df["vol_avg20"] * vol_mult
    else:
        vol_ok = pd.Series(True, index=df.index)

    # RSI — directional, not symmetric:
    #   Bullish cross: RSI must be below overbought level (room to run up)
    #   Bearish cross: RSI must be above oversold level (room to run down)
    rsi_bull_ok = df["rsi14"] < rsi_high   # not overbought on entry
    rsi_bear_ok = df["rsi14"] > rsi_low    # not oversold on entry

    # Trend EMA filter with ±3% tolerance so near-EMA crosses aren't excluded
    tol = df[trend_col] * 0.03
    above_trend = df["Close"] > (df[trend_col] - tol)
    below_trend = df["Close"] < (df[trend_col] + tol)

    # Optional 200 EMA alignment
    if require_200_align:
        above_200 = df["Close"] > df["ema200"]
        below_200 = df["Close"] < df["ema200"]
    else:
        above_200 = pd.Series(True, index=df.index)
        below_200 = pd.Series(True, index=df.index)

    bullish_signal = bullish_cross & above_trend & trending & vol_ok & rsi_bull_ok & above_200
    bearish_signal = bearish_cross & below_trend & trending & vol_ok & rsi_bear_ok & below_200

    # ── Pullback mode ─────────────────────────────────────────────────────────
    if pullback_mode:
        # After a bullish cross, flag the first bar that pulls back to slow EMA
        # and then closes above it again
        in_bull_setup = False
        in_bear_setup = False
        pb_bull = pd.Series(False, index=df.index)
        pb_bear = pd.Series(False, index=df.index)

        for i in range(1, len(df)):
            if bullish_signal.iloc[i]:
                in_bull_setup = True
                in_bear_setup = False
            if bearish_signal.iloc[i]:
                in_bear_setup = True
                in_bull_setup = False

            if in_bull_setup:
                touched = df["Low"].iloc[i] <= df[slow_col].iloc[i]
                bounced = df["Close"].iloc[i] > df[slow_col].iloc[i]
                if touched and bounced and not bullish_signal.iloc[i]:
                    pb_bull.iloc[i] = True
                    in_bull_setup = False

            if in_bear_setup:
                touched = df["High"].iloc[i] >= df[slow_col].iloc[i]
                bounced = df["Close"].iloc[i] < df[slow_col].iloc[i]
                if touched and bounced and not bearish_signal.iloc[i]:
                    pb_bear.iloc[i] = True
                    in_bear_setup = False

        bullish_signal = pb_bull
        bearish_signal = pb_bear

    # ── Annotate ──────────────────────────────────────────────────────────────
    df["signal"]      = ""
    df["entry_price"] = np.nan
    df["stop_loss"]   = np.nan
    df["target1"]     = np.nan
    df["target2"]     = np.nan
    df["rr_ratio"]    = np.nan

    for idx in df.index[bullish_signal]:
        entry = df.at[idx, "Close"]
        a     = df.at[idx, "atr14"]
        df.at[idx, "signal"]      = "BULLISH_CROSS"
        df.at[idx, "entry_price"] = round(entry, 2)
        df.at[idx, "stop_loss"]   = round(entry - atr_stop_mult   * a, 2)
        df.at[idx, "target1"]     = round(entry + atr_target1_mult * a, 2)
        df.at[idx, "target2"]     = round(entry + atr_target2_mult * a, 2)
        df.at[idx, "rr_ratio"]    = round(atr_target2_mult / atr_stop_mult, 2)

    for idx in df.index[bearish_signal]:
        entry = df.at[idx, "Close"]
        a     = df.at[idx, "atr14"]
        df.at[idx, "signal"]      = "BEARISH_CROSS"
        df.at[idx, "entry_price"] = round(entry, 2)
        df.at[idx, "stop_loss"]   = round(entry + atr_stop_mult   * a, 2)
        df.at[idx, "target1"]     = round(entry - atr_target1_mult * a, 2)
        df.at[idx, "target2"]     = round(entry - atr_target2_mult * a, 2)
        df.at[idx, "rr_ratio"]    = round(atr_target2_mult / atr_stop_mult, 2)

    return df


# ── Per-ticker screener ───────────────────────────────────────────────────────

def screen_ema_ticker(
    ticker: str,
    fast: int = 9,
    slow: int = 21,
    trend: int = 55,
    adx_threshold: float = 15.0,
    vol_mult: float = 1.0,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    require_200_align: bool = False,
    pullback_mode: bool = False,
    recent_bars: int = 10,
    period_days: int = 365,
) -> Optional[Dict[str, Any]]:
    end   = datetime.today()
    start = end - timedelta(days=period_days + 60)

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return None

    if raw is None or len(raw) < max(slow, trend, 200) + 30:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df = detect_crossovers(
        df, fast=fast, slow=slow, trend=trend,
        adx_threshold=adx_threshold, vol_mult=vol_mult,
        rsi_low=rsi_low, rsi_high=rsi_high,
        require_200_align=require_200_align,
        pullback_mode=pullback_mode,
    )

    recent  = df.tail(recent_bars)
    signals = recent[recent["signal"] != ""]

    if signals.empty:
        return None

    last_sig      = signals.iloc[-1]
    bars_ago      = len(df) - 1 - df.index.get_loc(last_sig.name)
    current_price = df["Close"].iloc[-1]

    # Position of current price vs. stop and targets
    pnl_pct = (current_price / last_sig["entry_price"] - 1) * 100
    if last_sig["signal"] == "BEARISH_CROSS":
        pnl_pct = -pnl_pct

    year_high = df["High"].tail(252).max()
    year_low  = df["Low"].tail(252).min()

    # EMA stack snapshot
    ema_stack = (
        "BULL STACK"
        if df[f"ema{fast}"].iloc[-1] > df[f"ema{slow}"].iloc[-1] > df[f"ema{trend}"].iloc[-1]
        else "BEAR STACK"
        if df[f"ema{fast}"].iloc[-1] < df[f"ema{slow}"].iloc[-1] < df[f"ema{trend}"].iloc[-1]
        else "MIXED"
    )

    return {
        "Ticker":         ticker,
        "Signal":         last_sig["signal"],
        "Signal Date":    last_sig.name.strftime("%Y-%m-%d"),
        "Bars Ago":       bars_ago,
        "Entry":          last_sig["entry_price"],
        "Current":        round(current_price, 2),
        "P&L %":          round(pnl_pct, 2),
        "Stop":           last_sig["stop_loss"],
        "Target 1":       last_sig["target1"],
        "Target 2":       last_sig["target2"],
        "R/R":            last_sig["rr_ratio"],
        f"EMA{fast}":     round(df[f"ema{fast}"].iloc[-1], 2),
        f"EMA{slow}":     round(df[f"ema{slow}"].iloc[-1], 2),
        f"EMA{trend}":    round(df[f"ema{trend}"].iloc[-1], 2),
        "ADX":            round(df["adx14"].iloc[-1], 1),
        "RSI":            round(df["rsi14"].iloc[-1], 1),
        "Vol vs Avg":     round(df["Volume"].iloc[-1] / df["vol_avg20"].iloc[-1], 2),
        "EMA Stack":      ema_stack,
        "52W High":       round(year_high, 2),
        "52W Low":        round(year_low, 2),
    }


# ── Full screener run ─────────────────────────────────────────────────────────

PRESETS = {
    "9/21/55  — Aggressive Swing": dict(fast=9,  slow=21, trend=55),
    "20/50/200 — Conservative":    dict(fast=20, slow=50, trend=200),
    "13/48/200 — Momentum":        dict(fast=13, slow=48, trend=200),
}


def run_ema_screener(
    tickers: List[str],
    fast: int = 9,
    slow: int = 21,
    trend: int = 55,
    adx_threshold: float = 15.0,
    vol_mult: float = 1.0,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    require_200_align: bool = False,
    pullback_mode: bool = False,
    recent_bars: int = 10,
    signal_filter: str = "ALL",
) -> pd.DataFrame:

    results = []
    total   = len(tickers)

    print(f"\n{'='*60}")
    print(f"  EMA Crossover Screener  ({fast}/{slow}/{trend})")
    print(f"  Scanning {total} tickers  |  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  ADX>{adx_threshold}  |  Vol>{vol_mult}x  |  RSI {rsi_low}-{rsi_high}")
    print(f"  200 EMA align: {require_200_align}  |  Pullback mode: {pullback_mode}")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>3}/{total}] {ticker:<8}", end="\r")
        result = screen_ema_ticker(
            ticker,
            fast=fast, slow=slow, trend=trend,
            adx_threshold=adx_threshold, vol_mult=vol_mult,
            rsi_low=rsi_low, rsi_high=rsi_high,
            require_200_align=require_200_align,
            pullback_mode=pullback_mode,
            recent_bars=recent_bars,
        )
        if result:
            results.append(result)

    print(" " * 40, end="\r")

    if not results:
        print("  No EMA crossover signals found.\n")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if signal_filter != "ALL":
        df = df[df["Signal"] == signal_filter]

    # Sort: bullish first (by P&L desc), then bearish
    df["_sort"] = df["Signal"].map({"BULLISH_CROSS": 0, "BEARISH_CROSS": 1})
    df = df.sort_values(["_sort", "P&L %"], ascending=[True, False]).drop(columns=["_sort"]).reset_index(drop=True)

    return df


def print_ema_results(df: pd.DataFrame):
    if df.empty:
        return
    bull = df[df["Signal"] == "BULLISH_CROSS"]
    bear = df[df["Signal"] == "BEARISH_CROSS"]

    def _section(title, sec):
        if sec.empty:
            return
        cols = ["Ticker", "Signal Date", "Bars Ago", "Entry", "Current",
                "P&L %", "Stop", "Target 2", "R/R", "ADX", "RSI",
                "Vol vs Avg", "EMA Stack"]
        cols = [c for c in cols if c in sec.columns]
        print(f"\n{'─'*80}\n  {title}  ({len(sec)} signals)\n{'─'*80}")
        print(sec[cols].to_string(index=False))

    _section("BULLISH CROSSOVERS  ▲", bull)
    _section("BEARISH CROSSOVERS  ▼", bear)
    print(f"\n{'='*80}")
    print(f"  Total: {len(df)}  |  Bullish: {len(bull)}  |  Bearish: {len(bear)}")
    print(f"{'='*80}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EMA Crossover Swing Screener")
    p.add_argument("--tickers",  nargs="+", default=None)
    p.add_argument("--file",     type=str,  default=None)
    p.add_argument("--universe", choices=["default", "sp500", "nasdaq100", "both", "watchlist"], default="default")
    p.add_argument("--preset",   choices=list(PRESETS), default=None, help="Use a named preset (overrides fast/slow/trend)")
    p.add_argument("--fast",     type=int,   default=9)
    p.add_argument("--slow",     type=int,   default=21)
    p.add_argument("--trend",    type=int,   default=55)
    p.add_argument("--adx",      type=float, default=20.0, help="Min ADX (default 20)")
    p.add_argument("--vol-mult", type=float, default=1.5,  help="Min volume multiplier vs 20-bar avg (default 1.5)")
    p.add_argument("--rsi-low",  type=float, default=30.0)
    p.add_argument("--rsi-high", type=float, default=70.0)
    p.add_argument("--no-200-filter",  action="store_true", help="Disable 200 EMA alignment requirement")
    p.add_argument("--pullback",       action="store_true", help="Enable pullback entry mode")
    p.add_argument("--recent-bars",    type=int, default=5)
    p.add_argument("--signal",  choices=["ALL", "BULLISH_CROSS", "BEARISH_CROSS"], default="ALL")
    p.add_argument("--output",  type=str, default=None)
    args = p.parse_args()

    if args.preset:
        params = PRESETS[args.preset]
        fast, slow, trend = params["fast"], params["slow"], params["trend"]
    else:
        fast, slow, trend = args.fast, args.slow, args.trend

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.file:
        with open(args.file) as fh:
            tickers = [l.strip().upper() for l in fh if l.strip() and not l.startswith("#")]
    else:
        tickers = get_universe(args.universe)

    results = run_ema_screener(
        tickers=tickers, fast=fast, slow=slow, trend=trend,
        adx_threshold=args.adx, vol_mult=args.vol_mult,
        rsi_low=args.rsi_low, rsi_high=args.rsi_high,
        require_200_align=not args.no_200_filter,
        pullback_mode=args.pullback,
        recent_bars=args.recent_bars,
        signal_filter=args.signal,
    )

    print_ema_results(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"  Saved to {args.output}\n")
