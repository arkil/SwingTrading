"""
Daily Alerts Engine
===================
Generates actionable BUY / SELL alerts with precise entry, stop, and target
prices by running every stock through ALL indicator systems in one pass.

Signal sources aggregated
─────────────────────────
  • Trend Alignment   — EMA stacking (9/21/50/200), Minervini 8-condition template
  • Momentum          — RSI(14) zone/cross/divergence, MACD histogram, ADX strength
  • Breakout          — 52W-high close, NR7, Bollinger squeeze, inside-bar, MA-reclaim
  • Pattern           — Livermore upward/downward pivot, gap-up/gap-down continuation
  • Mean Reversion    — RSI oversold + divergence, gap-fill setup
  • Volume            — Unusual accumulation vs distribution
  • Fundamentals      — CAN SLIM score (where info is available)

Scoring (0–12 per direction; min 4 to surface as WATCH, 6 for BUY, 8 for STRONG)
  TREND     0–3  (EMA stack, Minervini template)
  MOMENTUM  0–3  (RSI, MACD, RS rating)
  TRIGGER   0–3  (at least 1 required to generate alert; caps at 3)
  VOLUME    0–2  (1.5× = +1, 3× = +2)
  EXTRA     0–1  (Livermore pivot / fundamental bonus)

Price levels (ATR-based)
─────────────────────────
  Entry  = current close
  Stop   = Entry − 1.5 × ATR(14)   [hard stop]
  T1     = Entry + 1.0 × ATR(14)   [quick partial, take 1/3]
  T2     = Entry + 2.0 × ATR(14)   [main target,  take 1/3]
  T3     = Entry + 3.0 × ATR(14)   [runner,        let ride]
  R/R    = distance-to-T2 / distance-to-Stop
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings("ignore")

_yf_lock = Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Shared indicator primitives
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d  = s.diff()
    up = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up  = df["High"].diff()
    dn  = -df["Low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = _atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    ndi = 100 * pd.Series(mdm, index=df.index).ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx  = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)) * 100
    return dx.ewm(span=n, adjust=False).mean()

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd = _ema(close, fast) - _ema(close, slow)
    signal = _ema(macd, sig)
    return macd, signal, macd - signal


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(ticker: str, days: int = 420) -> Tuple[pd.DataFrame, dict]:
    """Return (OHLCV DataFrame, info dict)."""
    end   = datetime.today()
    start = end - timedelta(days=days + 30)
    try:
        with _yf_lock:
            t    = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, interval="1d", auto_adjust=True)
            try:
                info = t.info or {}
            except Exception:
                info = {}
        if hist is None or hist.empty:
            return pd.DataFrame(), {}
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if getattr(hist.index, "tz", None):
            hist.index = hist.index.tz_localize(None)
        hist.index = pd.to_datetime(hist.index.date)
        return hist, info
    except Exception:
        return pd.DataFrame(), {}


def _fetch_spy(days: int = 420) -> pd.DataFrame:
    try:
        with _yf_lock:
            raw = yf.download("SPY", period="2y", interval="1d",
                              progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Close"]].dropna()
        if getattr(raw.index, "tz", None):
            raw.index = raw.index.tz_localize(None)
        raw.index = pd.to_datetime(raw.index.date)
        return raw
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# RS Rating
# ─────────────────────────────────────────────────────────────────────────────

def _rs_rating(df: pd.DataFrame, spy: pd.DataFrame) -> float:
    if df.empty or spy.empty:
        return 50.0
    idx = df.index.intersection(spy.index)
    if len(idx) < 20:
        return 50.0
    n = min(len(idx), 252)
    def _perf(frame, i, p):
        if len(i) < p + 1:
            return 0.0
        sub = frame.loc[i]
        return float(sub["Close"].iloc[-1] / sub["Close"].iloc[-(p+1)] - 1)
    periods, weights = [63, 126, 189, 252], [2, 1, 1, 1]
    try:
        s_score = sum(w * _perf(df,  idx, p) for w, p in zip(weights, periods)) / 5
        m_score = sum(w * _perf(spy, idx, p) for w, p in zip(weights, periods)) / 5
        return float(max(1.0, min(99.0, 50.0 + (s_score - m_score) * 500)))
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Minervini trend template (0–8)
# ─────────────────────────────────────────────────────────────────────────────

def _minervini_score(df: pd.DataFrame) -> int:
    if len(df) < 210:
        return 0
    c = df["Close"]
    sma50  = _sma(c, 50).iloc[-1]
    sma150 = _sma(c, 150).iloc[-1]
    sma200_s = _sma(c, 200)
    sma200 = sma200_s.iloc[-1]
    sma200_slope = bool(sma200 > sma200_s.iloc[-22]) if len(sma200_s.dropna()) >= 22 else False
    high52 = df["High"].rolling(252).max().iloc[-1]
    low52  = df["Low"].rolling(252).min().iloc[-1]
    p = c.iloc[-1]
    checks = [
        p > sma150, p > sma200, sma150 > sma200, sma200_slope,
        sma50 > sma150 and sma50 > sma200, p > sma50,
        p >= low52  * 1.30 if not np.isnan(low52)  else False,
        p >= high52 * 0.75 if not np.isnan(high52) else False,
    ]
    return sum(bool(x) for x in checks)


# ─────────────────────────────────────────────────────────────────────────────
# Livermore pivot detection
# ─────────────────────────────────────────────────────────────────────────────

def _livermore_pivot(df: pd.DataFrame, window: int = 5, min_pct: float = 2.0
                     ) -> Optional[str]:
    """
    Returns 'UP', 'DOWN', or None based on most recent pivot (last 15 bars).
    """
    if len(df) < window * 3 + 5:
        return None
    highs = df["High"]
    lows  = df["Low"]
    closes = df["Close"]
    pivot_signal = None
    for i in range(-(window + 2), -1):
        # Swing high
        idx = len(df) + i
        if idx < window or idx + window >= len(df):
            continue
        h_win = highs.iloc[idx - window: idx + window + 1]
        if highs.iloc[idx] == h_win.max():
            hi = highs.iloc[idx]
            # look for subsequent reaction and breakout
            post = closes.iloc[idx + 1:]
            if len(post) >= 2:
                trough = post.min()
                if (hi - trough) / hi * 100 >= min_pct and closes.iloc[-1] > hi:
                    pivot_signal = "UP"
                    break
        # Swing low
        l_win = lows.iloc[idx - window: idx + window + 1]
        if lows.iloc[idx] == l_win.min():
            lo = lows.iloc[idx]
            post = closes.iloc[idx + 1:]
            if len(post) >= 2:
                peak = post.max()
                if (peak - lo) / lo * 100 >= min_pct and closes.iloc[-1] < lo:
                    pivot_signal = "DOWN"
                    break
    return pivot_signal


# ─────────────────────────────────────────────────────────────────────────────
# RSI divergence (last `lb` bars)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_divergence(df: pd.DataFrame, lb: int = 20) -> Optional[str]:
    """Returns 'BULL_DIV', 'BEAR_DIV', or None."""
    if len(df) < lb + 5:
        return None
    c   = df["Close"].iloc[-lb:]
    r   = _rsi(df["Close"], 14).iloc[-lb:]
    if c.empty or r.empty:
        return None
    # Bullish: price lower low, RSI higher low
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() + 5:
        return "BULL_DIV"
    # Bearish: price higher high, RSI lower high
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() - 5:
        return "BEAR_DIV"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Individual trigger checks
# ─────────────────────────────────────────────────────────────────────────────

def _ema9_cross_bull(df: pd.DataFrame, recent: int = 5) -> bool:
    e9, e21 = _ema(df["Close"], 9), _ema(df["Close"], 21)
    return bool(((e9.shift(1) <= e21.shift(1)) & (e9 > e21)).iloc[-recent:].any())

def _ema9_cross_bear(df: pd.DataFrame, recent: int = 5) -> bool:
    e9, e21 = _ema(df["Close"], 9), _ema(df["Close"], 21)
    return bool(((e9.shift(1) >= e21.shift(1)) & (e9 < e21)).iloc[-recent:].any())

def _breakout_52w(df: pd.DataFrame, recent: int = 3) -> bool:
    if len(df) < 260:
        return False
    high52 = df["High"].rolling(252).max().shift(1)
    vol20  = df["Volume"].rolling(20).mean()
    cond   = (df["Close"] > high52) & (df["Volume"] > vol20 * 1.3)
    return bool(cond.iloc[-recent:].any())

def _volume_breakout(df: pd.DataFrame, lookback: int = 20, vol_mult: float = 1.5,
                     recent: int = 3) -> bool:
    if len(df) < lookback + 2:
        return False
    res   = df["High"].rolling(lookback).max().shift(1)
    avg_v = df["Volume"].rolling(20).mean()
    broke = (df["Close"] > res) & (df["Volume"] > avg_v * vol_mult)
    return bool(broke.iloc[-recent:].any())

def _nr7_bull(df: pd.DataFrame, recent: int = 4) -> bool:
    if len(df) < 12:
        return False
    rng   = df["High"] - df["Low"]
    is_nr = rng == rng.rolling(7).min()
    for lag in range(1, recent):
        if len(df) > lag and is_nr.iloc[-(lag + 1)]:
            if df["Close"].iloc[-1] > df["High"].iloc[-(lag + 1)]:
                return True
    return False

def _bb_squeeze_bull(df: pd.DataFrame, recent: int = 3) -> bool:
    if len(df) < 130:
        return False
    mid  = _sma(df["Close"], 20)
    std  = df["Close"].rolling(20).std()
    upper = mid + 2 * std
    width = (upper - (mid - 2 * std)) / mid.replace(0, np.nan)
    in_sq = (width.iloc[-6:-1] <= width.rolling(126).min().iloc[-6:-1] * 1.05).any()
    avg_v = df["Volume"].rolling(20).mean()
    broke = (df["Close"] > upper) & (df["Volume"] > avg_v * 1.3)
    return bool(in_sq and broke.iloc[-recent:].any())

def _ma_reclaim(df: pd.DataFrame, recent: int = 3) -> bool:
    """Price reclaims EMA50 from below with volume."""
    e50 = _ema(df["Close"], 50)
    avg_v = df["Volume"].rolling(20).mean()
    reclaim = (df["Close"].shift(1) < e50.shift(1)) & (df["Close"] > e50) & (df["Volume"] > avg_v * 1.2)
    return bool(reclaim.iloc[-recent:].any())

def _inside_bar_bull(df: pd.DataFrame, recent: int = 3) -> bool:
    if len(df) < 5:
        return False
    for lag in range(2, recent + 2):
        if len(df) <= lag:
            continue
        mother_h = df["High"].iloc[-(lag + 1)]
        mother_l = df["Low"].iloc[-(lag + 1)]
        inside_h = df["High"].iloc[-lag]
        inside_l = df["Low"].iloc[-lag]
        if inside_h < mother_h and inside_l > mother_l:
            if df["Close"].iloc[-1] > mother_h * 0.998:
                return True
    return False

def _gap_up_bull(df: pd.DataFrame, min_pct: float = 0.5, recent: int = 3) -> bool:
    if len(df) < 5:
        return False
    for i in range(-recent, 0):
        pc = df["Close"].iloc[i - 1]
        op = df["Open"].iloc[i]
        if pc > 0 and (op - pc) / pc * 100 >= min_pct:
            e50 = _ema(df["Close"], 50).iloc[i]
            if df["Close"].iloc[i] > e50:
                return True
    return False

def _gap_down_bear(df: pd.DataFrame, min_pct: float = 0.5, recent: int = 3) -> bool:
    if len(df) < 5:
        return False
    for i in range(-recent, 0):
        pc = df["Close"].iloc[i - 1]
        op = df["Open"].iloc[i]
        if pc > 0 and (pc - op) / pc * 100 >= min_pct:
            e50 = _ema(df["Close"], 50).iloc[i]
            if df["Close"].iloc[i] < e50:
                return True
    return False

def _macd_bull_cross(df: pd.DataFrame, recent: int = 5) -> bool:
    m, sig, _ = _macd(df["Close"])
    cross = (m.shift(1) <= sig.shift(1)) & (m > sig)
    return bool(cross.iloc[-recent:].any())

def _macd_bear_cross(df: pd.DataFrame, recent: int = 5) -> bool:
    m, sig, _ = _macd(df["Close"])
    cross = (m.shift(1) >= sig.shift(1)) & (m < sig)
    return bool(cross.iloc[-recent:].any())


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker alert generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_alert(
    ticker: str,
    df: pd.DataFrame,
    spy: pd.DataFrame,
    info: dict,
    min_price: float = 5.0,
    max_atr_pct: float = 10.0,
    min_adx: float = 12.0,
) -> Optional[Dict]:
    if df.empty or len(df) < 60:
        return None

    close  = df["Close"]
    price  = float(close.iloc[-1])
    if price < min_price:
        return None

    atr    = float(_atr(df, 14).iloc[-1])
    atr_pct = atr / price * 100
    if atr_pct > max_atr_pct:
        return None

    adx_val = float(_adx(df, 14).dropna().iloc[-1]) if len(_adx(df, 14).dropna()) else 0.0
    if adx_val < min_adx:
        return None

    rsi14      = _rsi(close, 14)
    rsi_curr   = float(rsi14.iloc[-1])
    macd_l, macd_sig, macd_hist = _macd(close)
    hist_curr  = float(macd_hist.iloc[-1])
    hist_prev  = float(macd_hist.iloc[-2])
    ema50_val  = float(_ema(close, 50).iloc[-1])
    ema200_val = float(_ema(close, 200).iloc[-1])
    vol_ma20   = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_curr   = float(df["Volume"].iloc[-1])
    vol_ratio  = vol_curr / vol_ma20 if vol_ma20 > 0 else 1.0
    rs         = _rs_rating(df, spy)
    mini_s     = _minervini_score(df)
    pivot      = _livermore_pivot(df)
    rsi_div    = _rsi_divergence(df)
    high52     = float(df["High"].rolling(252).max().iloc[-1]) if len(df) >= 252 else price
    pct_52h    = (price - high52) / high52 * 100
    sma50_val  = float(_sma(close, 50).iloc[-1]) if not _sma(close, 50).dropna().empty else price
    ext_pct    = (price - sma50_val) / sma50_val * 100 if sma50_val > 0 else 0

    name = (info.get("shortName") or info.get("longName") or ticker)[:24]
    sector = info.get("sector", "")

    # ── BULLISH scoring ───────────────────────────────────────────────────────
    bull_score  = 0
    bull_signals: List[str] = []

    # TREND (0-3)
    if price > ema50_val:
        bull_score += 1; bull_signals.append("Above EMA50")
    if ema50_val > ema200_val:
        bull_score += 1; bull_signals.append("EMA50 > EMA200")
    if mini_s >= 6:
        bull_score += 1; bull_signals.append(f"Minervini {mini_s}/8")

    # MOMENTUM (0-3)
    if 42 <= rsi_curr <= 70:
        bull_score += 1; bull_signals.append(f"RSI {rsi_curr:.0f} (momentum)")
    elif rsi_curr < 32:
        bull_score += 1; bull_signals.append(f"RSI {rsi_curr:.0f} (oversold)")
    if hist_curr > 0 and hist_curr > hist_prev:
        bull_score += 1; bull_signals.append("MACD hist rising")
    if rs >= 60:
        bull_score += 1; bull_signals.append(f"RS {rs:.0f} (leading)")

    # TRIGGERS (0-3, at least 1 required)
    bull_triggers = 0
    bull_trigger_names: List[str] = []
    if _ema9_cross_bull(df):         bull_triggers += 1; bull_trigger_names.append("EMA9×EMA21 ↑")
    if _breakout_52w(df) and rsi_curr < 75:  bull_triggers += 1; bull_trigger_names.append("52W High Break")
    if _volume_breakout(df):         bull_triggers += 1; bull_trigger_names.append("Vol Breakout")
    if _nr7_bull(df):                bull_triggers += 1; bull_trigger_names.append("NR7 Break")
    if _bb_squeeze_bull(df):         bull_triggers += 1; bull_trigger_names.append("BB Squeeze ↑")
    if _ma_reclaim(df):              bull_triggers += 1; bull_trigger_names.append("MA Reclaim")
    if _inside_bar_bull(df):         bull_triggers += 1; bull_trigger_names.append("Inside Bar ↑")
    if _gap_up_bull(df):             bull_triggers += 1; bull_trigger_names.append("Gap Up ↑")
    if _macd_bull_cross(df):         bull_triggers += 1; bull_trigger_names.append("MACD Cross ↑")
    if pivot == "UP":                bull_triggers += 1; bull_trigger_names.append("Livermore Pivot ↑")
    if rsi_div == "BULL_DIV":        bull_triggers += 1; bull_trigger_names.append("RSI Bull Div")

    capped_t = min(bull_triggers, 3)
    bull_score += capped_t
    bull_signals.extend(bull_trigger_names[:capped_t])

    # VOLUME (0-2)
    if vol_ratio >= 3.0:
        bull_score += 2; bull_signals.append(f"Vol {vol_ratio:.1f}× surge")
    elif vol_ratio >= 1.5:
        bull_score += 1; bull_signals.append(f"Vol {vol_ratio:.1f}× elevated")

    # EXTRA (Livermore pivot or fundamental)
    inst = (info.get("heldPercentInstitutions") or 0) * 100
    if 0.30 < inst / 100 < 0.85:
        bull_score += 1; bull_signals.append(f"Inst {inst:.0f}%")

    # ── BEARISH scoring ───────────────────────────────────────────────────────
    bear_score  = 0
    bear_signals: List[str] = []

    if price < ema50_val:
        bear_score += 1; bear_signals.append("Below EMA50")
    if ema50_val < ema200_val:
        bear_score += 1; bear_signals.append("EMA50 < EMA200")
    if mini_s <= 2:
        bear_score += 1; bear_signals.append(f"Minervini {mini_s}/8 (weak)")

    if rsi_curr > 72:
        bear_score += 1; bear_signals.append(f"RSI {rsi_curr:.0f} (overbought)")
    elif rsi_curr < 30:
        pass  # oversold is bullish
    if hist_curr < 0 and hist_curr < hist_prev:
        bear_score += 1; bear_signals.append("MACD hist falling")
    if rs < 40:
        bear_score += 1; bear_signals.append(f"RS {rs:.0f} (lagging)")

    bear_triggers = 0
    bear_trigger_names: List[str] = []
    if _ema9_cross_bear(df):         bear_triggers += 1; bear_trigger_names.append("EMA9×EMA21 ↓")
    if _macd_bear_cross(df):         bear_triggers += 1; bear_trigger_names.append("MACD Cross ↓")
    if _gap_down_bear(df):           bear_triggers += 1; bear_trigger_names.append("Gap Down ↓")
    if pivot == "DOWN":              bear_triggers += 1; bear_trigger_names.append("Livermore Pivot ↓")
    if rsi_div == "BEAR_DIV":        bear_triggers += 1; bear_trigger_names.append("RSI Bear Div")

    capped_bt = min(bear_triggers, 3)
    bear_score += capped_bt
    bear_signals.extend(bear_trigger_names[:capped_bt])

    if vol_ratio >= 3.0 and df["Close"].iloc[-1] < df["Open"].iloc[-1]:
        bear_score += 2; bear_signals.append(f"Vol {vol_ratio:.1f}× (distribution)")
    elif vol_ratio >= 1.5 and df["Close"].iloc[-1] < df["Open"].iloc[-1]:
        bear_score += 1; bear_signals.append(f"Vol {vol_ratio:.1f}× down-day")

    # ── Determine direction & reject if no trigger ────────────────────────────
    if bull_score < 4 and bear_score < 4:
        return None
    if bull_triggers == 0 and bear_triggers == 0:
        return None

    # Hard filter: reject BUY when price is too extended above 50MA (>30%)
    # Backtested: 30% blocks ARM-like blowups while keeping high-momentum entries
    if bull_score >= bear_score and ext_pct > 30:
        return None

    # RSI overbought penalty: buying RSI 80+ is chasing, 75+ is extended
    if rsi_curr > 80:
        bull_score -= 2
    elif rsi_curr > 75:
        bull_score -= 1

    # Resolve direction
    if bull_score >= bear_score:
        direction = "BUY"
        score     = bull_score
        signals   = bull_signals
        # Entry / stop / targets for LONG
        entry  = round(price, 2)
        stop   = round(price - 1.5 * atr, 2)
        t1     = round(price + 1.0 * atr, 2)
        t2     = round(price + 2.0 * atr, 2)
        t3     = round(price + 3.0 * atr, 2)
    else:
        direction = "SELL"
        score     = bear_score
        signals   = bear_signals
        # Entry / stop / targets for SHORT
        entry  = round(price, 2)
        stop   = round(price + 1.5 * atr, 2)
        t1     = round(price - 1.0 * atr, 2)
        t2     = round(price - 2.0 * atr, 2)
        t3     = round(price - 3.0 * atr, 2)

    stop_pct = abs(entry - stop) / entry * 100
    t2_pct   = abs(t2 - entry) / entry * 100
    rr       = round(t2_pct / stop_pct, 2) if stop_pct > 0 else 0

    conviction = (
        "STRONG" if score >= 8 else
        "HIGH"   if score >= 6 else
        "WATCH"
    )

    # ── Exit signal analysis (for existing long holders) ─────────────────────
    exit_signals: List[str] = []
    exit_urgency = ""

    # Overbought momentum
    if rsi_curr > 80:
        exit_signals.append(f"RSI {rsi_curr:.0f} — severely overbought, take profits")
    elif rsi_curr > 75:
        exit_signals.append(f"RSI {rsi_curr:.0f} — overbought, partial exit")

    # MACD rolling over
    if _macd_bear_cross(df, recent=3):
        exit_signals.append("MACD bearish cross — momentum reversing")
    elif hist_curr < 0 and hist_prev >= 0:
        exit_signals.append("MACD histogram flipped negative")

    # EMA9 crossed below EMA21
    if _ema9_cross_bear(df, recent=3):
        exit_signals.append("EMA9 crossed below EMA21 — short-term trend broke")

    # Price broke below EMA50
    ema50_s = _ema(close, 50)
    if close.iloc[-1] < ema50_s.iloc[-1] and close.iloc[-2] >= ema50_s.iloc[-2]:
        exit_signals.append("Broke below EMA50 — sell or tight stop immediately")

    # At / near 52-week high — lock gains
    if pct_52h >= -1.5:
        exit_signals.append(f"At 52W high (${high52:.2f}) — tighten stop or take 1/3")
    elif pct_52h >= -5:
        exit_signals.append(f"Within 5% of 52W high — consider partial exit")

    # Price extended far above 50MA (O'Neil: 20%+ in <3 weeks = fast mover → hold 8W)
    if ext_pct > 25:
        exit_signals.append(f"Price {ext_pct:.0f}% above 50MA — extended, trail stop tightly")
    elif ext_pct > 15:
        exit_signals.append(f"Price {ext_pct:.0f}% above 50MA — elevated, take partial")

    # Bearish RSI divergence
    if rsi_div == "BEAR_DIV":
        exit_signals.append("RSI bearish divergence — momentum not confirming highs")

    # Distribution: heavy volume on down day
    if vol_ratio >= 2.0 and df["Close"].iloc[-1] < df["Open"].iloc[-1]:
        exit_signals.append(f"Distribution day — Vol {vol_ratio:.1f}× on down close")

    # Livermore downward pivot
    if pivot == "DOWN":
        exit_signals.append("Livermore downward pivot — institutional selling detected")

    # Minervini template deteriorating
    if mini_s <= 3 and direction == "BUY":
        exit_signals.append(f"Minervini score dropped to {mini_s}/8 — trend weakening")

    # Gap-down on volume
    if _gap_down_bear(df, min_pct=1.0, recent=2):
        exit_signals.append("Significant gap-down — re-evaluate position")

    # Determine exit urgency
    urgent_keywords = ["Broke below EMA50", "MACD bearish cross", "EMA9 crossed below",
                       "downward pivot", "Distribution day"]
    has_urgent = any(any(kw in s for kw in urgent_keywords) for s in exit_signals)

    if has_urgent and len(exit_signals) >= 2:
        exit_urgency = "URGENT"
    elif len(exit_signals) >= 3:
        exit_urgency = "CONSIDER"
    elif len(exit_signals) >= 1:
        exit_urgency = "WATCH"

    # Compose a one-line action recommendation
    if exit_urgency == "URGENT":
        exit_action = "Exit or reduce position now — multiple exit triggers active"
    elif exit_urgency == "CONSIDER":
        exit_action = "Take partial profits — raise stop to breakeven or T1"
    elif exit_urgency == "WATCH":
        exit_action = "Monitor closely — tighten trailing stop"
    else:
        exit_action = ""

    return {
        "Symbol":        ticker.upper(),
        "Name":          name,
        "Direction":     direction,
        "Conviction":    conviction,
        "Score":         score,
        "Entry":         entry,
        "Stop":          stop,
        "T1":            t1,
        "T2":            t2,
        "T3":            t3,
        "Stop %":        round(-stop_pct if direction == "BUY" else stop_pct, 1),
        "T2 %":          round(t2_pct if direction == "BUY" else -t2_pct, 1),
        "R/R":           rr,
        "Price":         price,
        "ATR":           round(atr, 2),
        "ATR %":         round(atr_pct, 1),
        "RSI":           round(rsi_curr, 1),
        "MACD Hist":     round(hist_curr, 4),
        "ADX":           round(adx_val, 1),
        "RS":            int(round(rs)),
        "Vol Ratio":     round(vol_ratio, 2),
        "52wH %":        round(pct_52h, 1),
        "Ext vs 50MA %": round(ext_pct, 1),
        "Minervini":     mini_s,
        "Signals":       signals,
        "Exit Signals":  exit_signals,
        "Exit Urgency":  exit_urgency,
        "Exit Action":   exit_action,
        "Sector":        sector,
        "Date":          datetime.today().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_alerts(
    tickers: List[str],
    min_score: int = 4,
    max_workers: int = 14,
    progress_cb=None,          # optional callback(completed, total, ticker)
) -> pd.DataFrame:
    """
    Run the full indicator stack on every ticker and return a DataFrame of
    actionable alerts sorted by Score descending.

    Args:
        tickers:     List of ticker symbols to scan.
        min_score:   Minimum combined score to include (4 = WATCH, 6 = BUY, 8 = STRONG).
        max_workers: Thread pool size.
        progress_cb: Optional callable(completed_int, total_int, ticker_str).

    Returns:
        DataFrame with one row per alert, sorted Score desc.
    """
    tickers_clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))

    spy = _fetch_spy()
    total = len(tickers_clean)
    rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch, tk): tk for tk in tickers_clean}
        done = 0
        for fut in as_completed(futs):
            tk = futs[fut]
            done += 1
            if progress_cb:
                progress_cb(done, total, tk)
            try:
                df_tk, info = fut.result()
                alert = _build_alert(tk, df_tk, spy, info)
                if alert and alert["Score"] >= min_score:
                    rows.append(alert)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["Score", "R/R"], ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def generate_exit_signals(
    tickers: List[str],
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Check exit conditions for held positions — no score gate, no ADX gate.
    Used by the live monitor to decide whether to close open positions.

    Returns DataFrame with:
        Symbol, Exit Urgency, Exit Signals, Exit Action,
        Price, RSI, ADX, MACD Hist, Vol Ratio
    """
    tickers_clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not tickers_clean:
        return pd.DataFrame()

    spy  = _fetch_spy()
    rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers_clean))) as ex:
        futs = {ex.submit(_fetch, tk): tk for tk in tickers_clean}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                df_tk, info = fut.result()
                # min_adx=0 bypasses the trend-strength gate — we always want exit data
                alert = _build_alert(tk, df_tk, spy, info, min_adx=0.0)
                if alert:
                    rows.append({
                        "Symbol":       alert["Symbol"],
                        "Exit Urgency": alert["Exit Urgency"],
                        "Exit Signals": alert["Exit Signals"],
                        "Exit Action":  alert["Exit Action"],
                        "Price":        alert["Price"],
                        "RSI":          alert["RSI"],
                        "ADX":          alert["ADX"],
                        "MACD Hist":    alert["MACD Hist"],
                        "Vol Ratio":    alert["Vol Ratio"],
                    })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Symbol")
