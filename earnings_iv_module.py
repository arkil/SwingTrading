"""
Earnings IV-Crush Strategy Advisor  —  Swing Dashboard module
============================================================
Given a ticker, works out whether the option market is over- or under-pricing
the upcoming earnings move, then picks the structure with the edge:

  SELL premium  (Iron Condor / Credit Spread)  -> market over-paying (IV-crush play)
  BUY  premium  (Debit Spread / Long Strangle) -> market under-pricing vs history
  STAND ASIDE / WAIT                            -> no edge, or earnings too far out

Data: yfinance (free, delayed).  Educational — not advice.  Sizes off cash, no margin.

Research basis
--------------
* IV crush: front-expiry IV collapses right after the print, back months barely
  move.  Term-structure inversion (front IV >> back IV) is the "sell the event"
  signal.  https://support.spotgamma.com/hc/en-us/articles/15249330755859-IV-Crush-Explained
* Options usually OVER-price the earnings move; a long straddle held through
  earnings is on average a loser.  Compare the implied move to the stock's own
  history.  https://orats.com/blog/earnings-options-strategies-backtest
* Long calendars get hurt when the front crushes faster than the back, so the
  default IV-crush structure here is a defined-risk Iron Condor.
  https://optionsamurai.com/blog/iron-condor-earnings/
"""

from __future__ import annotations

import math
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


def _retry(fn, tries: int = 4, base: float = 0.8, label: str = ""):
    """yfinance under load throws transient errors — empty .options, DNS thread
    exhaustion ('getaddrinfo() thread failed to start'), rate limits. Retry with
    backoff before giving up. `fn` should raise or return a falsy value on a
    soft failure; pass validation via the caller."""
    last = None
    for i in range(tries):
        try:
            out = fn()
            if out is not None and not (hasattr(out, "__len__") and len(out) == 0):
                return out
            last = ValueError(f"empty result{(' for ' + label) if label else ''}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(base * (2 ** i))
    raise last if last else RuntimeError("retry failed")


# --------------------------------------------------------------------------- #
# Black-Scholes helpers (no scipy)
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_SQRT2 = math.sqrt(2.0)


def _ncdf_arr(x):
    """Vectorised standard-normal CDF (Abramowitz–Stegun, |err| < 1e-7)."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    d = 0.3989422804014327 * np.exp(-0.5 * x * x)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                 + t * (-1.821255978 + t * 1.330274429))))
    return np.where(x >= 0.0, 1.0 - p, p)


def bs_delta(S, K, T, iv, is_call, r=0.045):
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def bs_price(S, K, T, iv, is_call, r=0.045):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _load_ticker(symbol: str):
    return _load_ticker_impl(symbol)


def _load_ticker_impl(symbol: str):
    from earnings_cache import cached_history, cached_earnings_dates

    hist = cached_history(symbol, retry=_retry)          # parquet disk cache (~4h)
    if hist is None or hist.empty:
        raise ValueError(f"No price history for '{symbol}'.")
    spot = float(hist["Close"].iloc[-1])

    now = pd.Timestamp.now(tz="UTC")
    next_earn, past_earn = None, []
    for ts in cached_earnings_dates(symbol):             # json disk cache (~3d)
        ts = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        if ts >= now - pd.Timedelta(days=2):
            if next_earn is None:
                next_earn = ts
        else:
            past_earn.append(ts)
    if next_earn is None:
        try:
            cal = yf.Ticker(symbol).calendar
            d = None
            if isinstance(cal, dict) and cal.get("Earnings Date"):
                d = cal["Earnings Date"][0]
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                d = cal.loc["Earnings Date"].iloc[0]
            if d is not None:
                next_earn = pd.Timestamp(d).tz_localize("UTC")
        except Exception:
            pass

    expiries = list(_retry(lambda: yf.Ticker(symbol).options, label=f"{symbol} options"))
    return {
        "hist": hist, "spot": spot,
        "next_earn": next_earn, "past_earn": sorted(past_earn)[-12:],
        "expiries": expiries,
    }


def _fresh_spot(symbol: str):
    """Live last price (for the on-demand 're-check' — disk-cached history spot
    can be a few hours stale). Returns None on failure."""
    try:
        fi = yf.Ticker(symbol).fast_info
        for k in ("lastPrice", "last_price"):
            v = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
            if v and float(v) > 0:
                return float(v)
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _chain(symbol: str, expiry: str):
    return _chain_impl(symbol, expiry)


def _chain_impl(symbol: str, expiry: str):
    ch = _retry(lambda: yf.Ticker(symbol).option_chain(expiry), label=f"{symbol} {expiry} chain")
    return ch.calls, ch.puts


def _atm_iv(calls, puts, spot):
    # yfinance intermittently returns near-zero IV on a whole chain (stale feed).
    # Real single-stock ATM IV is essentially never < 5%, so treat that as "no
    # data" and let the caller skip the name rather than score it on garbage.
    out = {}
    for name, df in (("call", calls), ("put", puts)):
        d = df.dropna(subset=["impliedVolatility"])
        d = d[d["impliedVolatility"] > 0.05]
        if d.empty:
            return None, None
        row = d.iloc[(d["strike"] - spot).abs().argmin()]
        out[name] = float(row["impliedVolatility"])
    avg = (out["call"] + out["put"]) / 2.0
    if avg < 0.05:
        return None, None
    return avg, out


def _straddle_pct(calls, puts, spot):
    def mid(df):
        d = df.dropna(subset=["impliedVolatility"])
        if d.empty:
            return None
        r = d.iloc[(d["strike"] - spot).abs().argmin()]
        b, a = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
        return (b + a) / 2.0 if b > 0 and a > 0 else float(r.get("lastPrice", 0) or 0)
    c, p = mid(calls), mid(puts)
    return (c + p) / spot * 100.0 if c and p else None


def _realized_vol(hist, window=20):
    rets = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(rets.tail(window).std() * math.sqrt(252))


def _hist_earnings_moves(hist, past_earn):
    closes = hist["Close"]
    hidx = closes.index
    hidx = hidx.tz_convert("UTC") if hidx.tz is not None else hidx.tz_localize("UTC")
    moves, seen = [], set()
    for ts in past_earn:
        if ts.date() in seen:
            continue
        seen.add(ts.date())
        pos = hidx.searchsorted(ts)
        best = 0.0
        for cand in (pos - 1, pos, pos + 1, pos + 2):
            if 1 <= cand < len(closes):
                best = max(best, abs(closes.iloc[cand] / closes.iloc[cand - 1] - 1.0) * 100.0)
        if 0.5 < best < 60:
            moves.append(best)
    uniq, keys = [], set()
    for mv in moves:
        k = round(mv, 1)
        if k not in keys:
            keys.add(k); uniq.append(mv)
    return uniq[:8]


def _earnings_implied_move(front_iv, back_iv, t_front):
    ev = (front_iv * front_iv - back_iv * back_iv) * t_front
    return math.sqrt(ev) * 100.0 if ev > 0 else None


# --------------------------------------------------------------------------- #
# Strikes + structures
# --------------------------------------------------------------------------- #
def _step(df):
    s = np.diff(sorted(df["strike"].unique()))
    s = s[s > 0]
    return float(np.median(s)) if len(s) else 1.0


def _by_delta(df, spot, iv, T, is_call, target, r=0.045):
    """Strike whose |delta| is closest to `target`. Vectorised over the chain."""
    strikes = np.sort(df["strike"].unique().astype(float))
    if T <= 0 or iv <= 0 or spot <= 0 or strikes.size == 0:
        return None
    with np.errstate(all="ignore"):
        d1 = (np.log(spot / strikes) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        delta = _ncdf_arr(d1) if is_call else _ncdf_arr(d1) - 1.0
        err = np.abs(np.abs(delta) - target)
    err[~np.isfinite(err)] = np.inf
    j = int(np.argmin(err))
    return float(strikes[j]) if np.isfinite(err[j]) else None


def _nearest(df, target):
    s = np.array(sorted(df["strike"].unique()))
    return float(s[np.abs(s - target).argmin()])


def _quote(df, K, spot, iv, T, is_call):
    """Price estimate for strike K + a liquidity read.
    Returns (price, source, liquid, info) where source ∈ {'quote','last','model'}.
    `liquid` = a market you could realistically fill a spread into: two-sided
    quote, spread not insane, real open interest AND evidence it actually trades
    (daily volume, or deep OI as a proxy for a name that trades every day).
    `info` = {'oi': int, 'vol': int} for display."""
    theo = bs_price(spot, K, T, iv, is_call)
    def _f(x):
        try:
            x = float(x)
            return x if math.isfinite(x) else 0.0
        except Exception:
            return 0.0
    row = df[df["strike"] == K]
    if row.empty:
        return theo, "model", False, {"oi": 0, "vol": 0}
    r = row.iloc[0]
    b, a, lp = _f(r.get("bid")), _f(r.get("ask")), _f(r.get("lastPrice"))
    oi, vol = _f(r.get("openInterest")), _f(r.get("volume"))
    info = {"oi": int(oi), "vol": int(vol)}
    if b > 0 and a > 0:
        mid = (b + a) / 2.0
        spread_ok = (a - b) <= max(0.15, 0.5 * mid)          # not a >1.5x-wide market
        traded = vol >= 10 or oi >= 250                       # trades today, or deep book
        liquid = spread_ok and oi >= 50 and traded
        return mid, "quote", liquid, info
    # no two-sided market → not fillable at a known price; anchor to the model,
    # nudged toward lastPrice only if it's in a sane band around theo
    if lp > 0 and 0.3 * theo <= lp <= 3 * theo:
        return (lp + theo) / 2.0, "last", False, info
    return theo, "model", False, info


def _mid(df, K, spot, iv, T, is_call):
    return _quote(df, K, spot, iv, T, is_call)[0]


def _qleg(action, df, K, right, spot, iv, T):
    px, src, liq, info = _quote(df, K, spot, iv, T, right == "C")
    return {"action": action, "strike": float(K), "right": right, "px": round(px, 2),
            "src": src, "liquid": bool(liq), "oi": info["oi"], "vol": info["vol"],
            "label": f"{action} {K:g}{right}"}


def _finish(name, legs, kind, max_loss, breakevens, pop):
    net = abs(sum((l["px"] if l["action"] == "SELL" else -l["px"]) for l in legs))
    # thin = a leg we'd SELL has no real fillable market (the credit is fiction),
    # or, for a debit, a leg we'd BUY is model-only.
    short_thin = any(l["action"] == "SELL" and not l["liquid"] for l in legs)
    modelled = any(l["src"] == "model" for l in legs)
    thin = short_thin or (kind == "debit" and modelled)
    return {"name": name, "legs": [l["label"] for l in legs], "leg_detail": legs,
            "net": net, "net_kind": kind,
            "max_profit": (net if kind == "credit" else max_loss),
            "max_loss": max_loss, "breakevens": breakevens, "pop": pop, "thin": thin}


def _iron_condor(calls, puts, spot, ivc, ivp, T, short_delta=0.16):
    sc = _by_delta(calls, spot, ivc, T, True, short_delta)
    sp = _by_delta(puts, spot, ivp, T, False, short_delta)
    if sc is None or sp is None or sc <= sp:
        return None
    width = max(round((sc - sp) * 0.25, 0), _step(calls))
    lc, lp = _nearest(calls, sc + width), _nearest(puts, sp - width)
    if lc <= sc or lp >= sp:          # chain has no strikes past the shorts → no defined risk
        return None
    legs = [_qleg("SELL", puts, sp, "P", spot, ivp, T), _qleg("BUY", puts, lp, "P", spot, ivp, T),
            _qleg("SELL", calls, sc, "C", spot, ivc, T), _qleg("BUY", calls, lc, "C", spot, ivc, T)]
    net = (legs[2]["px"] - legs[3]["px"]) + (legs[0]["px"] - legs[1]["px"])
    max_loss = max(lc - sc, sp - lp) - net
    pop = 1.0 - abs(bs_delta(spot, sc, T, ivc, True)) - abs(bs_delta(spot, sp, T, ivp, False))
    return _finish("Iron Condor — defined-risk IV-crush play", legs, "credit",
                   max_loss, [sp - net, sc + net], pop)


def _credit_spread(calls, puts, spot, iv, T, bullish, short_delta=0.25,
                   width_frac=0.03, tag="IV-crush"):
    _probe = _by_delta(puts if bullish else calls, spot, iv, T, not bullish, short_delta)
    if _probe is None:
        return _iron_condor(calls, puts, spot, iv, iv, T)
    if bullish:
        s = _by_delta(puts, spot, iv, T, False, short_delta)
        lng = _nearest(puts, s - max(2 * _step(puts), round(spot * width_frac, 0)))
        width, name = s - lng, f"Bull Put Credit Spread — bullish {tag}"
        if width <= 0:      # chain has no strike below the short → no defined risk
            return None
        legs = [_qleg("SELL", puts, s, "P", spot, iv, T), _qleg("BUY", puts, lng, "P", spot, iv, T)]
        pop = 1.0 - abs(bs_delta(spot, s, T, iv, False))
        be_fn = lambda net: [s - net]
    else:
        s = _by_delta(calls, spot, iv, T, True, short_delta)
        lng = _nearest(calls, s + max(2 * _step(calls), round(spot * width_frac, 0)))
        width, name = lng - s, f"Bear Call Credit Spread — bearish {tag}"
        if width <= 0:
            return None
        legs = [_qleg("SELL", calls, s, "C", spot, iv, T), _qleg("BUY", calls, lng, "C", spot, iv, T)]
        pop = 1.0 - abs(bs_delta(spot, s, T, iv, True))
        be_fn = lambda net: [s + net]
    net = legs[0]["px"] - legs[1]["px"]
    if net <= 0:            # short leg priced at/below the wing → not a real credit
        return None
    return _finish(name, legs, "credit", width - net, be_fn(net), pop)


def _long_strangle(calls, puts, spot, ivc, ivp, T):
    kc = _by_delta(calls, spot, ivc, T, True, 0.30) or _nearest(calls, spot * 1.05)
    kp = _by_delta(puts, spot, ivp, T, False, 0.30) or _nearest(puts, spot * 0.95)
    legs = [_qleg("BUY", puts, kp, "P", spot, ivp, T), _qleg("BUY", calls, kc, "C", spot, ivc, T)]
    debit = legs[0]["px"] + legs[1]["px"]
    out = _finish("Long Strangle — market under-pricing the move", legs, "debit",
                  debit, [kp - debit, kc + debit], float("nan"))
    out["max_profit"] = float("inf")
    return out


def _debit_spread(calls, puts, spot, iv, T, bullish):
    if bullish:
        lng = _nearest(calls, spot)
        s = _nearest(calls, lng + max(2 * _step(calls), round(spot * 0.05, 0)))
        width, name = s - lng, "Bull Call Debit Spread — cheap directional"
        legs = [_qleg("BUY", calls, lng, "C", spot, iv, T), _qleg("SELL", calls, s, "C", spot, iv, T)]
        be_fn = lambda deb: [lng + deb]
    else:
        lng = _nearest(puts, spot)
        s = _nearest(puts, lng - max(2 * _step(puts), round(spot * 0.05, 0)))
        width, name = lng - s, "Bear Put Debit Spread — cheap directional"
        legs = [_qleg("BUY", puts, lng, "P", spot, iv, T), _qleg("SELL", puts, s, "P", spot, iv, T)]
        be_fn = lambda deb: [lng - deb]
    debit = legs[0]["px"] - legs[1]["px"]
    if width <= 0 or debit <= 0 or debit >= width:   # degenerate / no edge
        return None
    out = _finish(name, legs, "debit", debit, be_fn(debit), float("nan"))
    out["max_profit"] = width - debit
    return out


def _payoff_fig(strat, spot, lo, hi, dark=False):
    px = np.linspace(lo, hi, 160)
    pnl = np.zeros_like(px)
    for leg in strat["legs"]:
        side, rest = leg.split(" ", 1)
        K = float(rest[:-1]); kind = rest[-1]
        sign = 1 if side == "BUY" else -1
        pnl += sign * (np.maximum(px - K, 0) if kind == "C" else np.maximum(K - px, 0))
    net = strat["net"]
    pnl = (pnl + (net if strat["net_kind"] == "credit" else -net)) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px, y=np.where(pnl >= 0, pnl, np.nan), mode="lines",
                             line=dict(color="#16a34a", width=2), name="profit"))
    fig.add_trace(go.Scatter(x=px, y=np.where(pnl < 0, pnl, np.nan), mode="lines",
                             line=dict(color="#dc2626", width=2), name="loss"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.add_vline(x=spot, line_dash="dash", line_color="#3b82f6",
                  annotation_text=f"spot {spot:.2f}")
    for be in strat["breakevens"]:
        fig.add_vline(x=be, line_dash="dot", line_color="#f59e0b")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False, xaxis_title="stock price at expiry",
                      yaxis_title="P/L ($ / 1 lot)",
                      template="plotly_dark" if dark else "plotly_white")
    return fig


# --------------------------------------------------------------------------- #
# Core analysis — pure, cached, no st.* output.  Shared by single view + scanner.
# --------------------------------------------------------------------------- #
def _analyze_impl(symbol: str, live: bool = False) -> dict:
    """Full IV-crush read for one ticker. Returns a dict with either
    {"error": "..."} or all computed fields + verdict + strat.
    Pure (no st.*) so the nightly prefetch CLI can call it directly.
    live=True bypasses the 1h data cache for an on-demand re-check."""
    _lt = _load_ticker_impl if live else _load_ticker
    _ch = _chain_impl if live else _chain
    symbol = symbol.strip().upper()
    try:
        data = _lt(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    spot, next_earn, expiries = data["spot"], data["next_earn"], data["expiries"]
    if not expiries:
        return {"symbol": symbol, "error": "no listed options"}
    if live:
        spot = _fresh_spot(symbol) or spot

    now_utc = pd.Timestamp.now(tz="UTC")
    dte_earn = (next_earn - now_utc).days if next_earn is not None else None

    exp_ts = [pd.Timestamp(e).tz_localize("UTC") for e in expiries]
    if next_earn is not None:
        after = [e for e, t in zip(expiries, exp_ts) if t >= next_earn]
        front_exp = after[0] if after else expiries[0]
    else:
        front_exp = expiries[0]
    front_t = pd.Timestamp(front_exp).tz_localize("UTC")
    back_cand = [e for e, t in zip(expiries, exp_ts) if t > front_t + pd.Timedelta(days=20)]
    back_exp = back_cand[0] if back_cand else expiries[-1]

    try:
        fc, fp = _ch(symbol, front_exp)
        bc, bp = _ch(symbol, back_exp)
    except Exception as e:
        return {"symbol": symbol, "error": f"chain load failed: {e}"}

    front_iv, det = _atm_iv(fc, fp, spot)
    back_iv, _ = _atm_iv(bc, bp, spot)
    if front_iv is None or back_iv is None:
        return {"symbol": symbol, "error": "no IV data on chain"}

    T_front = max((front_t - now_utc).days, 1) / 365.0
    rv20 = _realized_vol(data["hist"])
    # Sanity: front-month IV (pre-earnings especially) is never far below trailing
    # realized vol. If it is, yfinance's IV feed is degraded — skip, don't score.
    if rv20 and front_iv < 0.6 * rv20:
        return {"symbol": symbol, "error": f"IV feed looks stale (front {front_iv*100:.0f}% vs realized {rv20*100:.0f}%)"}

    straddle_pct = _straddle_pct(fc, fp, spot) or (front_iv * math.sqrt(T_front) * 100.0)
    earn_move = _earnings_implied_move(front_iv, back_iv, T_front)
    hmoves = _hist_earnings_moves(data["hist"], data["past_earn"])
    hist_avg = float(np.mean(hmoves)) if hmoves else None
    hist_max = float(np.max(hmoves)) if hmoves else None

    term_ratio = front_iv / back_iv
    iv_rv = front_iv / rv20 if rv20 else float("nan")
    move_reliable = earn_move is not None and (dte_earn is None or dte_earn <= 21)
    move_ratio = (earn_move / hist_avg) if (move_reliable and hist_avg) else float("nan")

    # ---- scoring ----
    # ── richness score: >0 = market over-pricing the event (sell), <0 = under-pricing (buy)
    rich, reasons, move_risk = 0.0, [], False
    if term_ratio > 1.25:
        rich += 2; reasons.append(f"Term structure strongly inverted ({term_ratio:.2f}) — classic IV-crush setup.")
    elif term_ratio > 1.10:
        rich += 1; reasons.append(f"Front IV elevated vs back ({term_ratio:.2f}).")
    elif term_ratio < 0.97:
        rich -= 1; reasons.append(f"Front IV *below* back ({term_ratio:.2f}) — no event premium priced; leans long-premium.")
    else:
        reasons.append(f"Term structure flat ({term_ratio:.2f}) — little IV-crush juice.")

    if not math.isnan(move_ratio):
        if move_ratio > 1.20:
            rich += 2; reasons.append(f"Implied move ±{earn_move:.1f}% is {move_ratio:.2f}× history (±{hist_avg:.1f}%) — market over-paying.")
        elif move_ratio > 1.05:
            rich += 1; reasons.append(f"Implied move ±{earn_move:.1f}% is {move_ratio:.2f}× history (±{hist_avg:.1f}%) — modestly rich.")
        elif move_ratio < 0.75:
            rich -= 2; reasons.append(f"Implied move ±{earn_move:.1f}% is only {move_ratio:.2f}× history (±{hist_avg:.1f}%) — market badly under-pricing the move.")
        elif move_ratio < 0.90:
            rich -= 1; move_risk = True
            reasons.append(f"Implied move ±{earn_move:.1f}% is {move_ratio:.2f}× history (±{hist_avg:.1f}%) — under-priced; if selling, only wide far-OTM defined-risk.")
        else:
            reasons.append(f"Implied move ≈ historical ({move_ratio:.2f}×) — fairly priced.")
    elif earn_move is None:
        reasons.append("Front expiry carries no event premium yet — earnings too far out to read.")

    if not math.isnan(iv_rv):
        if iv_rv > 1.5:
            rich += 1; reasons.append(f"Front IV is {iv_rv:.2f}× recent realized vol.")
        elif iv_rv < 0.85:
            rich -= 1; reasons.append(f"Front IV is only {iv_rv:.2f}× realized vol — options look cheap vs how the stock is actually moving.")

    # ── directional lean: trend + 25Δ put/call IV skew
    _cl = data["hist"]["Close"]
    ret10 = float(_cl.iloc[-1] / _cl.iloc[-11] - 1.0) * 100 if len(_cl) >= 11 else 0.0
    skew = None
    try:
        p25 = _by_delta(fp, spot, det["put"], T_front, False, 0.25)
        c25 = _by_delta(fc, spot, det["call"], T_front, True, 0.25)
        skew = (float(fp.loc[fp["strike"] == p25, "impliedVolatility"].iloc[0])
                - float(fc.loc[fc["strike"] == c25, "impliedVolatility"].iloc[0]))
    except Exception:
        pass
    dir_score = 0.0
    if ret10 > 3.5: dir_score += 1
    elif ret10 < -3.5: dir_score -= 1
    if skew is not None:
        if skew < -0.02: dir_score += 1      # calls bid = upside demand
        elif skew > 0.04: dir_score -= 1     # puts bid = downside fear
    lean = "bullish" if dir_score >= 1 else "bearish" if dir_score <= -1 else "neutral"

    too_far = next_earn is None or dte_earn is None or dte_earn > 21 or dte_earn < 0
    strat, ivc, ivp = None, det["call"], det["put"]
    score = rich

    # ── Backtest-driven gates (see EARNINGS_IV_BACKTEST.md) ──────────────────
    # 1. The edge IS the vol-risk-premium. With the implied move priced at or
    #    below the stock's own history (VRP <= ~1.0) the iron-condor backtest
    #    LOSES. So SELL requires the implied move to actually be rich, not just
    #    elevated term structure / IV.
    move_ok = (not math.isnan(move_ratio)) and move_ratio >= 1.05
    # 2. Tail guard: if this name's worst historical earnings move dwarfs the
    #    implied move, no condor can be built wide enough — the −43%-of-risk
    #    losers cluster here. Stand aside.
    tail_risk = (hist_max is not None and earn_move is not None
                 and hist_max > 2.2 * earn_move)

    if too_far:
        if next_earn is None:
            verdict = "WAIT — no confirmed earnings date"
        elif dte_earn is not None and dte_earn < 0:
            verdict = "PASSED — already reported; IV has crushed"
        else:
            verdict = f"WAIT — earnings {dte_earn}d out; term structure hasn't loaded the event"
    elif rich >= 2 and not move_ok:
        verdict = "STAND ASIDE — vol elevated but implied move ≈ history; no VRP edge"
        reasons.append("→ Backtest: selling premium here (implied move ≤ historical) is ~breakeven "
                       "to negative. Wait for the implied move to price rich vs this name's history.")
    elif rich >= 2 and tail_risk:
        verdict = "STAND ASIDE — implied move too small vs this name's worst prints"
        reasons.append(f"→ Worst historical earnings move ±{hist_max:.0f}% is >2.2× the implied "
                       f"±{earn_move:.1f}% — a defined-risk condor can't span that. Skip.")
    elif rich >= 2:  # implied move is genuinely rich (move_ok) and no tail_risk
        strength = "strong edge" if rich >= 3 else "moderate — size small"
        # shorts a touch further OTM when the implied move is only modestly rich
        sd = 0.16 if move_ratio >= 1.15 else 0.12
        if lean == "bullish":
            verdict = f"SELL PREMIUM ({strength}) — bullish tilt"
            strat = _credit_spread(fc, fp, spot, ivp, T_front, bullish=True, short_delta=sd + 0.06)
            reasons.append("→ Bull Put credit spread: collect the fat put premium, room below for the drift.")
        elif lean == "bearish":
            verdict = f"SELL PREMIUM ({strength}) — bearish tilt"
            strat = _credit_spread(fc, fp, spot, ivc, T_front, bullish=False, short_delta=sd + 0.06)
            reasons.append("→ Bear Call credit spread: sell the rich calls, room above for the drift.")
        else:
            verdict = f"SELL PREMIUM ({strength}) — neutral"
            strat = _iron_condor(fc, fp, spot, ivc, ivp, T_front, short_delta=sd)
            reasons.append(f"→ Iron Condor ({sd*100:.0f}Δ shorts): no directional read, harvest the crush both sides.")
    elif rich <= -2:
        if lean == "bullish":
            verdict = "BUY PREMIUM — under-priced, bullish"
            strat = _debit_spread(fc, fp, spot, ivc, T_front, bullish=True)
            reasons.append("→ Bull Call debit spread: cheap upside, capped cost into the print.")
        elif lean == "bearish":
            verdict = "BUY PREMIUM — under-priced, bearish"
            strat = _debit_spread(fc, fp, spot, ivp, T_front, bullish=False)
            reasons.append("→ Bear Put debit spread: cheap downside, capped cost into the print.")
        else:
            verdict = "BUY PREMIUM — market under-pricing the move"
            strat = _long_strangle(fc, fp, spot, ivc, ivp, T_front)
            reasons.append("→ Long Strangle: options are cheap vs history and no directional read — buy the move both ways.")
    else:
        verdict = "STAND ASIDE — no clear edge"

    if ("SELL" in verdict or "BUY" in verdict) and strat is None:
        verdict = "STAND ASIDE — can't build a valid structure (no strikes past the shorts / degenerate chain)"
        reasons.append("→ The option chain doesn't have the strikes needed for a defined-risk "
                       "spread here. Skip.")
    elif strat is not None and (strat.get("thin") or not _fin(strat.get("max_loss")) or strat["max_loss"] <= 0):
        verdict = "STAND ASIDE — structure is illiquid / degenerate (no fillable market on the short strikes)"
        reasons.append("→ The chain here has no real two-sided quotes / open interest on the "
                       "strikes we'd trade — any 'credit' is unfillable. Skip until liquidity shows up.")
        strat = None

    return {
        "symbol": symbol, "error": None,
        "spot": spot, "next_earn": next_earn, "dte_earn": dte_earn,
        "front_exp": front_exp, "back_exp": back_exp,
        "front_iv": front_iv, "back_iv": back_iv, "term_ratio": term_ratio,
        "iv_rv": iv_rv, "straddle_pct": straddle_pct, "earn_move": earn_move,
        "hist_avg": hist_avg, "hist_max": hist_max, "hmoves": hmoves,
        "move_ratio": move_ratio, "ret10": ret10, "skew": skew, "lean": lean,
        "score": score, "reasons": reasons, "verdict": verdict, "strat": strat,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_ticker(symbol: str) -> dict:
    """Cached wrapper around _analyze_impl for interactive use."""
    return _analyze_impl(symbol)


def _fin(x) -> bool:
    """True if x is a finite real number. Robust to None (JSON-cleaned nan/inf)."""
    return isinstance(x, (int, float)) and math.isfinite(x)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_scan(kind: str, _mtime: float):
    """Parse the prefetch JSON once per file version — not on every widget
    rerun. `_mtime` in the key invalidates the cache when the file changes."""
    from earnings_prefetch import load_latest, load_premium
    return (load_latest if kind == "earnings" else load_premium)()


def _load_scan(kind: str):
    from earnings_prefetch import _JSON, _PREM_JSON
    path = _JSON if kind == "earnings" else _PREM_JSON
    mt = os.path.getmtime(path) if os.path.exists(path) else 0.0
    return _cached_scan(kind, mt)


def _num(x, nd=None):
    """round() that never raises — returns None for NaN/Inf/None."""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return round(xf, nd) if nd is not None else round(xf)
    except (TypeError, ValueError):
        return None


def _live_recheck(sym: str, kind: str, cached: dict):
    """Button inside a ticket: pull this name's chain NOW (bypass the daily
    cache), re-run the model, show what changed and whether it still qualifies."""
    dark = st.session_state.get("dark_mode", False)
    key = f"live_{kind}_{sym}"
    if st.button(f"🔄 Live price & re-check", key=key,
                 help="Ignores the overnight cache — fetches this option chain right "
                      "now and re-scores it."):
        with st.spinner(f"Pulling {sym}'s chain live…"):
            st.session_state[key + "_res"] = (
                _analyze_impl(sym, live=True) if kind == "earnings"
                else _premium_impl(sym, live=True))
    fresh = st.session_state.get(key + "_res")
    if not fresh:
        return
    if fresh.get("error") or fresh.get("skip"):
        st.warning(f"Live re-check failed: {fresh.get('error') or fresh.get('skip')}")
        return

    fs = fresh.get("strat")
    st.markdown("---")
    st.markdown("**🔴 LIVE** — as of now:")

    if kind == "earnings":
        still = ("SELL" in fresh["verdict"]) or ("BUY" in fresh["verdict"])
        old_c = (cached.get("strat") or {}).get("net")
        new_c = (fs or {}).get("net")
        cols = st.columns(3)
        cols[0].metric("Verdict now", fresh["verdict"].split("—")[0].strip(),
                       help=fresh["verdict"])
        if _fin(old_c) and _fin(new_c):
            cols[1].metric("Credit", f"${new_c:.2f}", f"{new_c-old_c:+.2f} vs cache")
        cols[2].metric("Score", f"{fresh['score']:+.0f}",
                       f"{fresh['score']-cached.get('score',0):+.0f}")
    else:
        still = bool(fresh.get("ok"))
        cols = st.columns(3)
        cols[0].metric("Rated now", "GOOD" if still else "marginal")
        if _fin((cached.get("credit"))) and _fin(fresh.get("credit")):
            cols[1].metric("Credit", f"${fresh['credit']:.2f}",
                           f"{fresh['credit']-cached['credit']:+.2f} vs cache")
        if _fin(fresh.get("pop")):
            cols[2].metric("PoP now", f"{fresh['pop']*100:.0f}%")

    (st.success if still else st.error)(
        "✅ Still qualifies" if still else "✗ No longer qualifies — the edge moved since the overnight scan")

    if fs:
        _md(f"**{fs['name']}** · expiry `{fresh.get('front_exp') or fresh.get('exp')}`")
        _render_legs(fs)
        _md(f"**Net {fs['net_kind']}:** ${fs['net']:.2f}  ·  "
            f"**Max loss:** " + (f"${fs['max_loss']*100:.0f}/lot" if _fin(fs.get("max_loss")) else "—")
            + "  ·  **BE:** " + ", ".join(f"${b:.2f}" for b in fs["breakevens"]))
        rng = max((fresh.get("straddle_pct", 8)) / 100 * fresh["spot"] * 2.2, fresh["spot"] * 0.12)
        st.plotly_chart(_payoff_fig(fs, fresh["spot"], fresh["spot"] - rng, fresh["spot"] + rng, dark),
                        use_container_width=True, key=f"lpf_{kind}_{sym}")
    for reason in fresh.get("reasons", []):
        st.caption("• " + str(reason))


_SRC_TAG = {"quote": "", "last": "  ⚠ last-trade (no live bid)", "model": "  ⚠ model price (no market)"}


def _render_legs(strat: dict):
    """Each leg with its price, price source, and a per-leg cash sign."""
    detail = strat.get("leg_detail")
    if not detail:
        for leg in strat.get("legs", []):
            st.write(f"- {leg}")
        return
    for l in detail:
        px = l.get("px")
        sign = "+" if l["action"] == "SELL" else "−"   # SELL brings cash in
        tag = _SRC_TAG.get(l.get("src", "quote"), "")
        liq = f"  ·  OI {l.get('oi', 0):,} · vol {l.get('vol', 0):,}"
        if l.get("src") == "quote" and not l.get("liquid"):
            liq += "  ⚠ thin"
        if _fin(px):
            _md(f"- **{l['action']} {l['strike']:g}{l['right']}**  ·  ≈ ${px:.2f}  "
                f"→ {sign}${px*100:.0f}/contract{tag}{liq}")
        else:
            _md(f"- **{l['action']} {l['strike']:g}{l['right']}**  ·  _no quote_ (feed gap)")
    if strat.get("thin"):
        st.warning("⚠ **Illiquid** — a leg you'd be selling has no real market to fill into "
                   "(thin open interest / no volume today). The credit above is an estimate you "
                   "likely can't actually get. Info only, not a tradeable ticket.")


def _built_str(built_at) -> str:
    """'Sep 02 13:21 ET (4h ago)' from a UTC ISO string; never raises."""
    try:
        t = pd.Timestamp(built_at)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        try:
            local = t.tz_convert("America/New_York")
            lbl = local.strftime("%b %d %H:%M ET")
        except Exception:
            lbl = t.strftime("%b %d %H:%M UTC")
        age = (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 3600
        return f"{lbl} ({age:.0f}h ago)" if age < 48 else f"{lbl} ({age/24:.0f}d ago)"
    except Exception:
        return "unknown time"


def _md(text: str):
    """st.markdown with $ escaped — Streamlit renders bare $…$ as LaTeX."""
    st.markdown(str(text).replace("$", "\\$"))


def _cap(text: str):
    st.caption(str(text).replace("$", "\\$"))


def _sizing(strat, account, risk_pct):
    risk_budget = account * risk_pct / 100.0
    ml = strat.get("max_loss")
    loss_per_lot = ml * 100 if _fin(ml) else strat["net"] * 100
    lots = max(int(risk_budget // loss_per_lot), 0) if loss_per_lot > 0 else 0
    return risk_budget, loss_per_lot, lots


# --------------------------------------------------------------------------- #
# Render — dispatcher
# --------------------------------------------------------------------------- #
def render_earnings_iv():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>💰</span>"
        "<span class='scanner-title'>Options Premium Advisor</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "Two engines. <b>Earnings IV-crush</b>: is the market over- or under-pricing a "
        "stock's earnings move? Pick the structure with the edge. <b>Premium (no "
        "earnings)</b>: high-probability defined-risk credit spreads on liquid large "
        "caps with no earnings in the trade window. Free delayed data. Educational, not advice."
        "</div>",
        unsafe_allow_html=True,
    )
    mode = st.radio("mode", ["🔍 Earnings — single ticker", "📡 Earnings — scan",
                             "💵 Premium — no earnings"],
                    horizontal=True, label_visibility="collapsed")
    st.divider()
    if mode.startswith("🔍"):
        _render_single()
    elif mode.startswith("📡"):
        _render_scanner()
    else:
        _render_premium()


# --------------------------------------------------------------------------- #
# Render — single ticker
# --------------------------------------------------------------------------- #
def _render_single():
    dark = st.session_state.get("dark_mode", False)
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 0.8])
    symbol = c1.text_input("Ticker", "NVDA").strip().upper()
    account = c2.number_input("Account size ($)", 1000, 100_000_000, 25_000, step=1000)
    risk_pct = c3.slider("Max risk / trade (%)", 0.5, 5.0, 1.5, 0.5)
    run = c4.button("Analyze", type="primary", use_container_width=True)

    if not run:
        st.info("Enter a ticker and press **Analyze**. Works best inside ~2 weeks of a confirmed earnings date.")
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = analyze_ticker(symbol)
    if r.get("error"):
        st.error(f"{symbol}: {r['error']}"); return

    m = st.columns(3)
    m[0].metric("Spot", f"${r['spot']:,.2f}")
    m[1].metric("Next earnings", r["next_earn"].strftime("%Y-%m-%d") if r["next_earn"] is not None else "unknown")
    m[2].metric("Days to earnings", r["dte_earn"] if r["dte_earn"] is not None else "—")

    st.subheader("Volatility read")
    v = st.columns(4)
    v[0].metric("Front ATM IV", f"{r['front_iv']*100:.1f}%", help=f"expiry {r['front_exp']}")
    v[1].metric("Back ATM IV", f"{r['back_iv']*100:.1f}%", help=f"expiry {r['back_exp']}")
    v[2].metric("Term ratio (front/back)", f"{r['term_ratio']:.2f}",
                help=">1.10 elevated, >1.25 strongly inverted → IV-crush edge for sellers")
    v[3].metric("Front IV / realized(20d)", f"{r['iv_rv']:.2f}" if not math.isnan(r['iv_rv']) else "—")

    v2 = st.columns(4)
    v2[0].metric("Implied earnings move", f"±{r['earn_move']:.1f}%" if r['earn_move'] is not None else "not priced yet",
                 help=f"Isolated from the IV term structure. Front straddle prices ±{r['straddle_pct']:.1f}% total.")
    v2[1].metric("Hist. avg earnings move", f"±{r['hist_avg']:.1f}%" if r['hist_avg'] else "—",
                 help=f"{len(r['hmoves'])} prior prints" if r['hmoves'] else "no history")
    v2[2].metric("Hist. max earnings move", f"±{r['hist_max']:.1f}%" if r['hist_max'] else "—")
    v2[3].metric("Implied / historical", f"{r['move_ratio']:.2f}" if not math.isnan(r['move_ratio']) else "—",
                 help=">1.15 → over-paying (sell). <0.90 → under-pricing (buy).")
    if r['hmoves']:
        st.caption("Past earnings-day moves: " + ", ".join(f"{x:.1f}%" for x in r['hmoves']))

    st.subheader("📋 Recommendation")
    for reason in r["reasons"]:
        st.markdown("- " + str(reason).replace("$", "\\$"))
    st.write(f"• 10-day trend **{r['ret10']:+.1f}%** → lean **{r['lean']}**"
             + (f"; 25Δ put–call IV skew {r['skew']:+.2f}" if r['skew'] is not None else ""))

    verdict, strat = r["verdict"], r["strat"]
    (st.success if ("SELL" in verdict or "BUY" in verdict) else st.warning)(f"### → {verdict}")

    if strat is None:
        st.info("No structure recommended. Come back when the term structure inverts or the "
                "implied move diverges from history.")
        _footer(); return

    risk_budget, loss_per_lot, lots = _sizing(strat, account, risk_pct)
    st.markdown(f"**{strat['name']}**  ·  expiry `{r['front_exp']}`")
    L, R = st.columns([1, 1])
    with L:
        st.write("**Legs**")
        for leg in strat["legs"]:
            st.write(f"- {leg}")
        _md(f"**Net {strat['net_kind']}:** ${strat['net']:.2f}  (${strat['net']*100:.0f} / lot)")
        mp = strat["max_profit"]
        _md("**Max profit:** " + ("unlimited" if not math.isfinite(mp) else f"${mp*100:.0f} / lot"))
        _md("**Max loss:** " + ("undefined" if not math.isfinite(strat["max_loss"]) else f"${strat['max_loss']*100:.0f} / lot"))
        _md("**Breakevens:** " + ", ".join(f"${b:.2f}" for b in strat["breakevens"]))
        if not math.isnan(strat.get("pop", float("nan"))):
            st.write(f"**Est. probability of profit:** {strat['pop']*100:.0f}%")
        _md(f"**Suggested size:** {lots} lot(s) → risk ≈ ${lots*loss_per_lot:,.0f} of ${risk_budget:,.0f}")
        if lots == 0 and loss_per_lot > 0:
            _cap(f"⚠ One lot risks ${loss_per_lot:,.0f} = {loss_per_lot/account*100:.1f}% of account — "
                       f"above your {risk_pct:.1f}% cap. Widen risk %, pick a tighter spread, or skip.")
    with R:
        rng = max(r['straddle_pct'] / 100 * r['spot'] * 2.2, r['spot'] * 0.12)
        st.plotly_chart(_payoff_fig(strat, r['spot'], r['spot'] - rng, r['spot'] + rng, dark),
                        use_container_width=True)

    if "SELL" in verdict:
        st.caption("Exit plan: close the day after the print once IV has crushed — don't hold for "
                   "the last few dollars of theta. Risk ≤ 1–2% of account per event.")
    _backtest_note("earnings")
    _footer()


# --------------------------------------------------------------------------- #
# Scanner row builder — shared by the nightly prefetch and the live "refresh now"
# --------------------------------------------------------------------------- #
_VERDICT_ICON = {"SELL": "🟢", "BUY": "🔵", "STAND": "⚪", "WAIT": "🟡", "PASSED": "⚫"}


def scan_row(rep: dict, r: dict) -> dict:
    """One table row from an earnings-calendar entry `rep` and an analysis `r`."""
    strat = r["strat"]
    vkey = r["verdict"].split(" ")[0].split("—")[0].strip()
    return {
        "Ticker": r["symbol"],
        "Earnings": str(rep.get("date", "")),
        "When": rep.get("time", ""),
        "DTE": _num(r.get("dte_earn")),
        "Spot": _num(r.get("spot"), 2),
        "Front IV %": _num((r.get("front_iv") or 0) * 100, 1),
        "Term ratio": _num(r.get("term_ratio"), 2),
        "Impl move %": _num(r.get("earn_move"), 1),
        "Hist avg %": _num(r.get("hist_avg"), 1),
        "Impl/Hist": _num(r.get("move_ratio"), 2),
        "Score": _num(r.get("score"), 1) or 0,
        "Signal": f"{_VERDICT_ICON.get(vkey, '')} {r['verdict']}",
        "Structure": strat["name"].split(" — ")[0] if strat else "—",
        "Credit/Debit": (f"{strat['net_kind'][0].upper()} ${strat['net']:.2f}" if strat else "—"),
        "Max loss $": (_num((strat.get("max_loss") or 0) * 100) if strat else None),
    }


# --------------------------------------------------------------------------- #
# Render — scanner (reads the nightly cache; can refresh on demand)
# --------------------------------------------------------------------------- #
def _render_scanner():
    dark = st.session_state.get("dark_mode", False)
    from earnings_prefetch import build_scan, DEFAULT_UNIVERSE, DEFAULT_DAYS
    from earnings_cache import UNIVERSES

    df, details, meta = _load_scan("earnings")

    top = st.columns([2.4, 1.1, 0.9, 0.9])
    if meta:
        try:
            _bt = pd.Timestamp(meta["built_at"])
            _bt = _bt.tz_localize("UTC") if _bt.tzinfo is None else _bt
            stale = (pd.Timestamp.now(tz="UTC") - _bt).total_seconds() / 3600
        except Exception:
            stale = 999
        badge = "🟢" if stale < 30 else "🟠"
        top[0].caption(
            f"{badge} Last scan **{_built_str(meta.get('built_at'))}** · "
            f"{UNIVERSES.get(meta.get('universe'), meta.get('universe', '?'))} · "
            f"{meta.get('n_reporters', '?')} reporting in {meta.get('days', '?')}d · "
            f"{meta.get('n_ok', 0)} analysed, {meta.get('n_err', 0)} skipped · "
            f"auto-refreshed 10:05 & 13:30 ET on weekdays"
        )
    else:
        top[0].warning("No cached scan yet — the pre-market job hasn't run. Pick a universe and **Refresh now**.")
    ref_uni_label = top[1].selectbox("Refresh universe", list(UNIVERSES.values()),
                                     index=list(UNIVERSES).index("watchlist"))
    refresh = top[2].button("🔄 Refresh", use_container_width=True,
                            help="Re-runs the scan live. Watchlist ≈1 min; S&P 500 can take 10-15 min "
                                 "(the daily launchd job normally does the big one).")
    account = top[3].number_input("Account ($)", 1000, 100_000_000, 25_000, step=1000, key="scan_acct")

    if refresh:
        ref_key = {v: k for k, v in UNIVERSES.items()}[ref_uni_label]
        prog = st.progress(0.0, text="Starting scan…")
        def _cb(done, total, sym):
            prog.progress(done / max(total, 1), text=f"{done}/{total}  ·  {sym}")
        with st.spinner(f"Scanning {ref_uni_label} — hitting yfinance for every reporter…"):
            build_scan(universe=ref_key, days=DEFAULT_DAYS, progress=_cb)
        prog.empty()
        st.cache_data.clear()
        st.rerun()

    if df is None or df.empty:
        n_err = meta.get("n_err", 0) if meta else 0
        if meta and n_err:
            st.warning(f"The last scan analysed **0 names** — {n_err} skipped, mostly because "
                       "yfinance's option IV feed was down at scan time.  Data outage, not a "
                       "market call — hit **Refresh** now, or wait for the 10:05 / 13:30 ET runs.")
            if meta.get("errors"):
                with st.expander(f"the {n_err} skipped names"):
                    st.caption(", ".join(meta["errors"]))
        else:
            st.info("No cached scan yet — **Refresh** lists every upcoming-earnings name with its "
                    "suggested structure.")
        return

    # ── filters over the cached superset ────────────────────────────────────
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    _dte_num = pd.to_numeric(df["DTE"], errors="coerce")
    _mx = _dte_num.max()
    max_dte = int(_mx) if pd.notna(_mx) and _mx >= 2 else max(DEFAULT_DAYS, 2)
    days = f1.slider("Earnings within (days)", 1, max_dte, min(7, max_dte))
    only_act = f2.checkbox("Only SELL / BUY", value=True)
    min_score = f3.slider("Min score", -4.0, 4.0, -4.0, 0.5,
                          help="Richness score: >0 sell premium, <0 buy premium. "
                               "Set to −4 to see BUY-side (long strangle / debit) ideas too.")
    risk_pct = f4.slider("Risk / trade %", 0.5, 5.0, 1.5, 0.5, key="scan_risk")

    def _filtered(d, days_, min_score_, only_act_):
        v = d.copy()
        v["DTE_n"] = pd.to_numeric(v["DTE"], errors="coerce")
        v = v[v["DTE_n"].fillna(999) <= days_]
        v["_act"] = v["Signal"].str.contains("SELL|BUY")
        v["_conv"] = v["Score"].abs()
        v = v[v["Score"] >= min_score_]
        if only_act_:
            v = v[v["_act"]]
        return (v.sort_values(["_act", "_conv"], ascending=[False, False])
                 .drop(columns=["_act", "_conv", "DTE_n"]))

    view = _filtered(df, days, min_score, only_act)

    n_sell = int(df["Signal"].str.contains("SELL").sum())
    n_buy = int(df["Signal"].str.contains("BUY").sum())
    n_err = meta.get("n_err", 0) if meta else 0

    relaxed = None
    if view.empty:
        # auto-relax: widest window, all scores, include STAND ASIDE/WAIT rows too —
        # never leave the page blank when the cache actually has data.
        fallback = _filtered(df, max_dte, -4.0, False)
        if not fallback.empty:
            view, relaxed = fallback, "widened the window and dropped the SELL/BUY-only filter"

    st.success(f"cache: {n_sell} SELL · {n_buy} BUY · {len(df)} names analysed"
              + (f" · {n_err} skipped (data outage — see below)" if n_err else "")
              + f" · showing {len(view)}")

    if relaxed:
        st.warning(f"Nothing cleared your filters, so I {relaxed} to show what's actually in the "
                   f"cache instead of a blank page. Tighten the filters above once there's more to pick from.")
    if view.empty:
        st.info("The cache itself is empty for this window — not a filter problem. Click **Refresh** "
                "or widen the day range.")
        return

    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download CSV", view.to_csv(index=False).encode(),
                       f"earnings_iv_scan_{days}d.csv", "text/csv")

    # ── trade tickets from cached detail (cap to keep the page light) ──────
    st.divider()
    _tix = view["Ticker"].tolist()
    _TICKET_CAP = 15
    st.subheader(f"Trade tickets"
                 + (f" — top {_TICKET_CAP} of {len(_tix)} (filter to see others)"
                    if len(_tix) > _TICKET_CAP else ""))
    for sym in _tix[:_TICKET_CAP]:
        r = details.get(sym)
        if not r:
            continue
        strat = r["strat"]
        with st.expander(f"{sym} · {r['verdict']}"):
            for reason in r["reasons"]:
                st.markdown("- " + str(reason).replace("$", "\\$"))
            st.write(f"• 10-day trend {r['ret10']:+.1f}% → lean {r['lean']}")
            st.caption(f"cached from the {_built_str(meta.get('built_at'))} scan" if meta else "cached scan")
            _live_recheck(sym, "earnings", r)
            if strat is None:
                st.info("No structure in the cached scan — use **Live price & re-check** above to "
                        "see if it qualifies now."); continue
            risk_budget, loss_per_lot, lots = _sizing(strat, account, risk_pct)
            cL, cR = st.columns([1, 1])
            with cL:
                st.markdown(f"**{strat['name']}**  ·  expiry `{r['front_exp']}`")
                _render_legs(strat)
                _md(f"**Net {strat['net_kind']}:** ${strat['net']:.2f} (${strat['net']*100:.0f}/lot)")
                _md("**Max loss:** " + (f"${strat['max_loss']*100:.0f}/lot" if _fin(strat.get("max_loss"))
                                             else "undefined"))
                _md("**Breakevens:** " + ", ".join(f"${b:.2f}" for b in strat["breakevens"]))
                if _fin(strat.get("pop")):
                    st.write(f"**Est. PoP:** {strat['pop']*100:.0f}%")
                _md(f"**Size:** {lots} lot(s) → risk ${lots*loss_per_lot:,.0f} / ${risk_budget:,.0f}")
                if lots == 0 and loss_per_lot > 0:
                    _cap(f"⚠ 1 lot risks ${loss_per_lot:,.0f} ({loss_per_lot/account*100:.1f}% of account) — over your cap.")
            with cR:
                rng = max(r['straddle_pct'] / 100 * r['spot'] * 2.2, r['spot'] * 0.12)
                st.plotly_chart(_payoff_fig(strat, r['spot'], r['spot'] - rng, r['spot'] + rng, dark),
                                use_container_width=True, key=f"pf_{sym}")

    if meta and meta.get("errors"):
        with st.expander(f"Skipped {len(meta['errors'])} names in last scan"):
            st.caption(", ".join(meta["errors"]))
    _backtest_note("earnings")
    _footer()


def _backtest_note(kind: str):
    """Compact backtest headline. `kind` in {'earnings', 'premium'}."""
    with st.expander("📊 Backtest (Black-Scholes simulation — not real option data)"):
        if kind == "earnings":
            st.markdown(
                "**Iron condor through the print · 60 large caps · 2017–2026**\n\n"
                "- Two backtest-driven gates are now live: **(1)** SELL only when implied ÷ "
                "historical move ≥ 1.05 — without it, at VRP 1.00 the strategy *loses* (−34u); "
                "term-structure / IV elevation alone no longer triggers a sell. **(2)** skip names "
                "whose worst recent earnings move > 2.2× the implied move (un-condor-able tail).\n"
                "- With both gates: win **74.5%**, avg **+5.2%** RoR/trade, ann. Sharpe **2.0**, "
                "max DD **7.9u** (down from 17.5u), 2022 the only losing year (−4.1u). "
                "*BS simulation — no slippage/fills; haircut the ~5%/trade edge.*\n"
                "- Fat left tail: avg loser −42% of risk. Size ≤1–2% per event."
            )
        else:
            st.markdown(
                "**35-DTE credit spread, managed at 50% / 21 DTE · 60 large caps · 2017–2026 · 2,497 trades**\n\n"
                "- Win rate **93.2%**, avg **+2.8%** return on risk/trade, ann. Sharpe ≈ **1.9**, "
                "CAGR ≈ +24% at 3%-risk/trade. Worst year 2023: −5.4u.\n"
                "- Robust to the IV/RV assumption (+54u to +94u across 1.00–1.25) — edge comes from "
                "equity drift + management, not just rich vol.\n"
                "- Classic shape: 93% small wins, avg loser **−81%** of risk. Sizing is everything."
            )
        st.caption("No bid/ask slippage, assignment or borrow modelled — haircut the expectancy. "
                   "Full detail: `python earnings_iv_backtest.py --start 2017 --sweep` · "
                   "see EARNINGS_IV_BACKTEST.md")


def _footer():
    st.divider()
    st.caption(
        "Sources: SpotGamma *IV Crush Explained* · ORATS *Earnings Options Strategies Backtest* · "
        "TradeStation *Straddle Opportunities for Earnings* · Options Samurai *Iron Condor Earnings* · "
        "tastylive *Managing winners at 21 DTE / 50% profit*. "
        "Delayed data — verify every strike & price with your broker before trading. No margin; sized off cash."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PREMIUM (NO EARNINGS) — high-probability defined-risk credit spreads on
# liquid large caps, with no earnings inside the trade window.
# ═════════════════════════════════════════════════════════════════════════════
# ETFs / indices — no earnings events, so the "no earnings in window" filter
# must not skip them just because yfinance has no earnings-date data.
_NO_EARNINGS = {"SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "XLV", "XLY",
                "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "GLD", "SLV", "TLT", "HYG",
                "EEM", "EFA", "VXX", "UVXY", "ARKK", "KRE", "XBI", "IBB", "SOXX"}

# "Credible" = mega/large-cap, deep option liquidity, weekly + monthly chains.
CREDIBLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "NFLX", "CRM",
    "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "INTC", "AMAT", "MU", "NOW", "INTU",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "HD", "LOW",
    "XOM", "CVX", "COP", "CAT", "DE", "BA", "HON", "GE", "UPS", "UNP",
    "DIS", "CMCSA", "T", "VZ", "TMUS", "LIN", "RTX", "LMT", "SPY", "QQQ",
    "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "TSLA", "PLTR", "UBER", "COIN",
]


def _premium_impl(symbol: str, dte_target: int = 35, short_delta: float = 0.20,
                  live: bool = False) -> dict:
    """One high-prob credit-spread idea on a name with NO earnings in the window.
    Pure (no st.*). Returns {"error"/"skip": ...} or a full idea dict.
    live=True bypasses the 1h data cache for an on-demand re-check."""
    _lt = _load_ticker_impl if live else _load_ticker
    _ch = _chain_impl if live else _chain
    symbol = symbol.strip().upper()
    try:
        data = _lt(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    spot, expiries, hist = data["spot"], data["expiries"], data["hist"]
    next_earn = data["next_earn"]
    if not expiries:
        return {"symbol": symbol, "error": "no options"}
    if live:
        spot = _fresh_spot(symbol) or spot

    now = pd.Timestamp.now(tz="UTC")
    # Prefer the standard monthly expiry (3rd Friday) in the 20–55 DTE window —
    # that's where option open interest / liquidity concentrates. Fall back to
    # the nearest weekly to the target only if no monthly is in range.
    cand, monthlies = [], []
    for e in expiries:
        et = pd.Timestamp(e)
        d = (et.tz_localize("UTC") - now).days
        if 20 <= d <= 55:
            cand.append((abs(d - dte_target), e, d))
            if et.weekday() == 4 and 15 <= et.day <= 21:      # 3rd Friday
                monthlies.append((abs(d - dte_target), e, d))
    pool = monthlies or cand
    if not pool:
        return {"symbol": symbol, "skip": "no expiry in 20–55 DTE"}
    _, exp, dte = min(pool)
    exp_ts = pd.Timestamp(exp).tz_localize("UTC")

    # hard filter: no earnings between now and expiry + 2 days
    win_lo, win_hi = now - pd.Timedelta(days=1), exp_ts + pd.Timedelta(days=2)
    if symbol not in _NO_EARNINGS:
        if next_earn is not None:
            if win_lo <= next_earn <= win_hi:
                return {"symbol": symbol, "skip": f"earnings {next_earn.date()} inside window"}
        else:
            # yfinance gave no confirmed date — estimate from the last known report
            # (~quarterly) and skip if it could plausibly land in the window.
            past = data.get("past_earn") or []
            if past:
                est = past[-1] + pd.Timedelta(days=91)
                while est < now - pd.Timedelta(days=45):
                    est += pd.Timedelta(days=91)
                if win_lo - pd.Timedelta(days=12) <= est <= win_hi + pd.Timedelta(days=12):
                    return {"symbol": symbol, "skip": f"est. earnings ~{est.date()} may be in window (no confirmed date)"}
            else:
                return {"symbol": symbol, "skip": "no earnings-date data — can't confirm the window is clear"}

    try:
        calls, puts = _ch(symbol, exp)
    except Exception as e:
        return {"symbol": symbol, "error": f"chain: {e}"}
    atm, det = _atm_iv(calls, puts, spot)
    if atm is None:
        return {"symbol": symbol, "error": "no IV"}

    T = max(dte, 1) / 365.0
    rv20 = _realized_vol(hist, 20)
    rv60 = _realized_vol(hist, 60)
    if rv20 and atm < 0.55 * rv20:
        return {"symbol": symbol, "skip": f"IV feed stale ({atm*100:.0f}% vs realized {rv20*100:.0f}%)"}
    iv_rv = atm / rv20 if rv20 else float("nan")

    closes = hist["Close"]
    sma50 = float(closes.tail(50).mean())
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else sma50
    ret63 = float(closes.iloc[-1] / closes.iloc[-64] - 1.0) * 100 if len(closes) > 64 else 0.0
    above50 = spot > sma50
    uptrend = above50 and spot > sma200 and ret63 > -3
    downtrend = (not above50) and spot < sma200 and ret63 < 3

    # structure: trend-following one-sided credit spread; else two-sided condor.
    # Deltas tuned for HIGH probability of profit (this is the "low risk" mode).
    wf = max(0.04, min(0.09, atm * math.sqrt(T) * 0.8))   # wing width ≈ 0.8σ
    if uptrend:
        strat = _credit_spread(calls, puts, spot, det["put"], T, bullish=True,
                               short_delta=0.18, width_frac=wf, tag="trend")
        style = "Bull Put — uptrend"
    elif downtrend:
        strat = _credit_spread(calls, puts, spot, det["call"], T, bullish=False,
                               short_delta=0.18, width_frac=wf, tag="trend")
        style = "Bear Call — downtrend"
    else:
        strat = _iron_condor(calls, puts, spot, det["call"], det["put"], T, short_delta=0.12)
        style = "Iron Condor — rangebound"
    if strat is None or not _fin(strat.get("max_loss")) or strat["max_loss"] <= 0:
        return {"symbol": symbol, "skip": "could not build a clean spread"}
    if strat.get("thin"):
        return {"symbol": symbol, "skip": "illiquid chain — short leg has no live market"}

    credit, max_loss = strat["net"], strat["max_loss"]
    pop = strat.get("pop", float("nan"))
    ror = credit / max_loss if max_loss > 0 else 0.0          # return on risk, per cycle
    ann_ror = ror * (365.0 / dte)                              # gross annualised (theoretical)
    # Quality: PoP-dominated, RoR capped so a wide risky structure can't win on
    # nominal alone, plus a bonus for IV genuinely rich vs realized vol (the edge).
    p = pop if _fin(pop) else 0.6
    quality = p * 100 + min(ror * 100, 25) + max(0.0, iv_rv - 1.0) * 25

    reasons = [
        f"{style}. {dte} DTE, expiry {exp}.",
        f"IV {atm*100:.0f}% vs realized(20d) {rv20*100:.0f}% → IV/RV {iv_rv:.2f}"
        + ("  ✓ premium rich" if iv_rv >= 1.15 else "  ⚠ premium thin" if iv_rv < 1.0 else ""),
        f"Trend: {'up' if uptrend else 'down' if downtrend else 'sideways'}"
        f"  (spot {'>' if above50 else '<'} 50DMA, 3-mo {ret63:+.0f}%).",
        f"Credit ${credit:.2f} on ${max_loss + credit:.0f}-wide → {ror*100:.0f}% return on risk"
        + (f", {pop*100:.0f}% est. PoP" if _fin(pop) else "")
        + f" (~{ann_ror*100:.0f}%/yr gross if repeated, before losing cycles).",
        "→ Manage: close at ~50% of max profit or 21 DTE, whichever comes first (tastylive).",
    ]

    # gates for a genuinely "good premium, low-risk" idea
    ok = (_fin(pop) and pop >= 0.72) and ror >= 0.12 and credit >= 0.10 and iv_rv >= 1.0
    verdict = "GOOD PREMIUM" if ok else "MARGINAL"

    return {
        "symbol": symbol, "error": None, "skip": None,
        "spot": spot, "exp": exp, "dte": dte, "next_earn": next_earn,
        "iv": atm, "rv20": rv20, "rv60": rv60, "iv_rv": iv_rv,
        "ret63": ret63, "trend": "up" if uptrend else "down" if downtrend else "side",
        "style": style, "credit": credit, "max_loss": max_loss,
        "ror": ror, "ann_ror": ann_ror, "pop": pop, "quality": quality,
        "verdict": verdict, "ok": ok, "reasons": reasons, "strat": strat,
        "straddle_pct": atm * math.sqrt(T) * 100,
    }


def premium_row(r: dict) -> dict:
    s = r["strat"]
    return {
        "Ticker": r["symbol"],
        "Verdict": ("🟢 GOOD" if r.get("ok") else "⚪ marginal"),
        "Structure": s["name"].split(" — ")[0],
        "Exp": r.get("exp", ""), "DTE": _num(r.get("dte")), "Spot": _num(r.get("spot"), 2),
        "IV %": _num((r.get("iv") or 0) * 100, 1), "IV/RV": _num(r.get("iv_rv"), 2),
        "Trend": r.get("trend", ""),
        "Credit": _num(r.get("credit"), 2),
        "Max loss $": _num((r.get("max_loss") or 0) * 100),
        "RoR %": _num((r.get("ror") or 0) * 100),
        "Ann. RoR %": _num((r.get("ann_ror") or 0) * 100),
        "PoP %": _num((r.get("pop") or 0) * 100) if _fin(r.get("pop")) else None,
        "Quality": _num(r.get("quality")) or 0,
    }


def _render_premium():
    dark = st.session_state.get("dark_mode", False)
    from earnings_prefetch import build_premium_scan

    df, details, meta = _load_scan("premium")

    top = st.columns([2.6, 0.9, 0.9])
    if meta:
        try:
            _bt = pd.Timestamp(meta["built_at"])
            _bt = _bt.tz_localize("UTC") if _bt.tzinfo is None else _bt
            age = (pd.Timestamp.now(tz="UTC") - _bt).total_seconds() / 3600
        except Exception:
            age = 999
        top[0].caption(f"{'🟢' if age < 30 else '🟠'} Last scan {_built_str(meta.get('built_at'))} · "
                       f"{meta.get('n_names', '?')} credible names · "
                       f"{meta.get('n_ideas', 0)} ideas, {meta.get('n_good', 0)} rated GOOD · "
                       f"no-earnings-in-window filter applied")
    else:
        top[0].warning("No cached premium scan yet — click **Refresh** (or wait for the pre-market job).")
    refresh = top[1].button("🔄 Refresh", use_container_width=True,
                            help="Scans the credible large-cap list live (~1–2 min).")
    account = top[2].number_input("Account ($)", 1000, 100_000_000, 25_000, step=1000, key="prem_acct")

    if refresh:
        prog = st.progress(0.0, text="Starting…")
        build_premium_scan(progress=lambda d, t, s: prog.progress(d / max(t, 1), text=f"{d}/{t} · {s}"))
        prog.empty(); st.cache_data.clear(); st.rerun()

    if df is None or df.empty:
        n_skip = len(meta.get("skipped", [])) if meta else 0
        stale_feed = sum(1 for s in (meta or {}).get("skipped", []) if "stale" in s or "IV feed" in s)
        if meta and n_skip:
            st.warning(f"The last scan found **0 tradeable ideas** — {n_skip} names skipped"
                       + (f", {stale_feed} because yfinance's IV feed was down at scan time"
                          if stale_feed > n_skip / 2 else "")
                       + ".  This is a data outage, not a market call — hit **Refresh** now "
                       "(feed is usually fine once the market's open), or the scheduled 10:05 / "
                       "13:30 ET runs will repopulate it.")
            with st.expander(f"the {n_skip} skipped names"):
                st.caption(", ".join(meta["skipped"]))
        else:
            st.info("No cached scan yet. **Refresh** runs a live scan of the credible large-cap "
                    "list — defined-risk credit spreads on names with liquid options and no "
                    "earnings in the trade window, ranked best-first.")
        return

    f1, f2, f3, f4 = st.columns(4)
    good_only = f1.checkbox("Only GOOD-rated", value=False)
    min_pop = f2.slider("Min PoP %", 50, 90, 60)
    min_ann = f3.slider("Min ann. RoR %", 0, 150, 0, 10)
    risk_pct = f4.slider("Risk / trade %", 0.5, 5.0, 1.5, 0.5, key="prem_risk")

    def _filtered(d, pop_, ann_, good_):
        v = d.copy()
        v = v[pd.to_numeric(v["PoP %"], errors="coerce").fillna(0) >= pop_]
        v = v[pd.to_numeric(v["Ann. RoR %"], errors="coerce").fillna(0) >= ann_]
        if good_:
            v = v[v["Verdict"].str.contains("GOOD")]
        return v.sort_values("Quality", ascending=False)

    view = _filtered(df, min_pop, min_ann, good_only)
    n_good = int(df["Verdict"].str.contains("GOOD").sum())
    n_skip = len(meta.get("skipped", [])) if meta else 0

    relaxed = None
    if view.empty:
        fallback = _filtered(df, 0, 0, False).head(10)
        if not fallback.empty:
            view, relaxed = fallback, "dropped every filter and I'm showing the top 10 by quality anyway"

    st.success(f"{n_good} GOOD in cache · {len(df)} ideas analysed"
              + (f" · {n_skip} skipped (data outage — see below)" if n_skip else "")
              + f" · showing {len(view)}")

    if relaxed:
        st.warning(f"Nothing cleared your filters, so I {relaxed} instead of showing a blank page — "
                   f"these are rated 'marginal', not GOOD, so treat them as ideas to check manually, not "
                   f"a green light. Tighten the filters once the cache has more to offer.")
    if view.empty:
        st.info("The cache itself has no ideas right now — not a filter problem. Click **Refresh** "
                "or check the skipped-names list below (likely a data-feed issue).")
        return

    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download CSV", view.to_csv(index=False).encode(),
                       "premium_no_earnings.csv", "text/csv")

    st.divider()
    _tix = view["Ticker"].tolist()
    _TICKET_CAP = 15
    st.subheader("Trade tickets"
                 + (f" — top {_TICKET_CAP} of {len(_tix)} (filter to see others)"
                    if len(_tix) > _TICKET_CAP else ""))
    for sym in _tix[:_TICKET_CAP]:
        r = details.get(sym)
        if not r:
            continue
        s = r["strat"]
        with st.expander(f"{sym} · {r['style']} · {r['verdict']}"):
            for reason in r["reasons"]:
                st.markdown("- " + str(reason).replace("$", "\\$"))
            st.caption(f"cached from the {_built_str(meta.get('built_at'))} scan" if meta else "cached scan")
            _live_recheck(sym, "premium", r)
            risk_budget, loss_per_lot, lots = _sizing(s, account, risk_pct)
            cL, cR = st.columns(2)
            with cL:
                st.markdown(f"**{s['name']}**  ·  expiry `{r['exp']}`")
                _render_legs(s)
                _md(f"**Net credit:** ${s['net']:.2f} (${s['net']*100:.0f}/lot)")
                _md("**Max loss:** " + (f"${s['max_loss']*100:.0f}/lot" if _fin(s.get("max_loss")) else "undefined"))
                _md("**Breakevens:** " + ", ".join(f"${b:.2f}" for b in s["breakevens"]))
                if _fin(s.get("pop")):
                    st.write(f"**Est. PoP:** {s['pop']*100:.0f}%")
                _md(f"**Size:** {lots} lot(s) → risk ${lots*loss_per_lot:,.0f} / ${risk_budget:,.0f}")
                if lots == 0 and loss_per_lot > 0:
                    _cap(f"⚠ 1 lot risks ${loss_per_lot:,.0f} ({loss_per_lot/account*100:.1f}% of account) — over your cap.")
            with cR:
                rng = max(r["straddle_pct"] / 100 * r["spot"] * 2.4, r["spot"] * 0.14)
                st.plotly_chart(_payoff_fig(s, r["spot"], r["spot"] - rng, r["spot"] + rng, dark),
                                use_container_width=True, key=f"pp_{sym}")
    if meta and meta.get("skipped"):
        with st.expander(f"{len(meta['skipped'])} names skipped (earnings in window / no clean spread)"):
            st.caption(", ".join(meta["skipped"]))
    _backtest_note("premium")
    _footer()
