"""
Stock Analyzer — Multi-Framework Analysis
==========================================
Extracted from stock_analyzer/app.py and integrated into the Swing Dashboard.
Call render_stock_analyzer() to render the full analyzer UI.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import ta

# ── CSS (merged into dashboard global styles) ─────────────────────────────────
SA_CSS = """
.sa-metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}
.sa-signal-tag {
    display: inline-block;
    background: #f3f4f6;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    margin: 4px 4px 4px 0;
    color: #374151;
}
.sa-signal-tag-green { background:#d1fae5; color:#065f46; }
.sa-signal-tag-amber { background:#fef3c7; color:#92400e; }
.sa-signal-tag-red   { background:#fee2e2; color:#991b1b; }
.sa-verdict-box {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.sa-tbl-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 14px;
}
.sa-score-bar-bg {
    background: #f3f4f6;
    border-radius: 4px;
    height: 8px;
    margin-top: 4px;
}
.sa-hold-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: white;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
"""

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

SA_SCANNER_PRESETS = {
    "NASDAQ 100 Growth": [
        "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA",
        "AMD", "NFLX", "COST", "QCOM", "INTU", "AMAT", "KLAC", "LRCX",
        "MRVL", "MU", "PANW", "CRWD", "SNPS", "CDNS", "ADSK", "FTNT",
        "ARM", "APP", "CEG", "VST", "MSTR", "PLTR",
    ],
    "S&P 500 Leaders": [
        "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "LLY", "AVGO",
        "JPM", "V", "MA", "UNH", "XOM", "HD", "PG", "COST", "ABBV",
        "NFLX", "CRM", "ACN", "TXN", "BAC", "WMT", "TMO", "CAT",
        "GE", "RTX", "SPGI", "BLK", "GS", "FICO", "AXON", "DECK",
    ],
    "High-Growth Tech": [
        "NVDA", "AMD", "CRWD", "PANW", "FTNT", "PLTR", "APP", "ARM",
        "AVGO", "MRVL", "SNOW", "DDOG", "ZS", "NET", "HUBS", "TTD",
        "AXON", "FICO", "SMCI", "CELH", "LULU", "WING", "CAVA", "DUOL",
    ],
    "IBD-Style Leaders": [
        "NVDA", "AVGO", "LRCX", "KLAC", "AMAT", "CRWD", "PANW", "AXON",
        "FICO", "DECK", "CELH", "WING", "CAVA", "APP", "PLTR", "GEV",
        "VST", "CEG", "TMUS", "UBER", "META", "ARM", "MSTR", "BROS",
    ],
}

SA_SECTOR_ETF = {
    "Technology": "XLK", "Semiconductors": "SOXX", "Healthcare": "XLV",
    "Financial Services": "XLF", "Financials": "XLF", "Energy": "XLE",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Communication Services": "XLC",
    "Real Estate": "XLRE", "Basic Materials": "XLB", "Materials": "XLB",
    "Utilities": "XLU",
}

SA_SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "INTC", "TSM", "AMAT", "LRCX"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "TSM", "AMAT", "LRCX", "KLAC", "MU"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "C", "USB", "PNC", "V"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "C", "USB", "PNC", "V"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "HAL"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TJX", "LOW", "BKNG", "CMG"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "GIS", "MDLZ"],
    "Industrials": ["GE", "CAT", "HON", "UPS", "BA", "RTX", "LMT", "DE", "MMM", "ETN"],
    "Communication Services": ["META", "GOOGL", "NFLX", "DIS", "T", "VZ", "CMCSA", "SNAP", "PINS", "ROKU"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "WELL", "DLR", "PSA", "EQR"],
    "Basic Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "DOW", "DD"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "DOW", "DD"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ES", "WEC", "AWK"],
}

# ── DATA FETCHING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def sa_get_stock_data(ticker: str):
    stock = yf.Ticker(ticker)
    info = dict(stock.info)
    hist_6m = stock.history(period="6mo")
    hist_1y = stock.history(period="1y")
    hist_2y = stock.history(period="2y")
    return info, hist_6m, hist_1y, hist_2y


@st.cache_data(ttl=600)
def sa_get_financials(ticker: str):
    try:
        return yf.Ticker(ticker).cashflow
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def sa_get_spy_data():
    spy = yf.Ticker("SPY")
    return spy.history(period="1y")


@st.cache_data(ttl=600)
def sa_get_sector_data(etf_ticker: str, peers: list, current_ticker: str):
    results = {}
    etf = yf.Ticker(etf_ticker)
    results["etf_hist"] = etf.history(period="6mo")
    results["etf_info"] = etf.info

    peer_data = {}
    for p in peers:
        if p == current_ticker:
            continue
        try:
            t = yf.Ticker(p)
            h = t.history(period="3mo")
            if not h.empty:
                ret = (h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100
                ma50 = h["Close"].rolling(50).mean().dropna()
                above_ma = h["Close"].iloc[-1] > ma50.iloc[-1] if not ma50.empty else False
                peer_data[p] = {"return_3m": round(ret, 1), "above_ma50": above_ma,
                                 "price": round(h["Close"].iloc[-1], 2)}
        except Exception:
            pass
    results["peers"] = peer_data
    return results


# ── SCORING FUNCTIONS ──────────────────────────────────────────────────────────

def sa_calc_rs(hist_1y, spy_1y) -> float:
    if hist_1y.empty or spy_1y.empty:
        return 50.0
    stock_ret = hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[0] - 1
    spy_ret   = spy_1y["Close"].iloc[-1]  / spy_1y["Close"].iloc[0]  - 1
    rs = (stock_ret - spy_ret) * 100
    return round(max(1, min(99, 50 + rs * 2)), 1)


def sa_oneil_score(info, rs_score, price, ma50, hist_6m):
    score, reasons, max_score = 0, [], 18

    eps_q = info.get("earningsQuarterlyGrowth", 0) or 0
    if eps_q > 0.25:   score += 3; reasons.append(f"EPS Qtr Growth {eps_q*100:.0f}% (need >25%)")
    elif eps_q > 0.15: score += 2
    elif eps_q > 0:    score += 1

    eps_a = info.get("earningsGrowth", 0) or 0
    if eps_a > 0.25:   score += 2; reasons.append(f"Annual EPS Growth {eps_a*100:.0f}%")
    elif eps_a > 0:    score += 1

    rev = info.get("revenueGrowth", 0) or 0
    if rev > 0.20:     score += 2; reasons.append(f"Revenue Growth {rev*100:.0f}%")
    elif rev > 0.10:   score += 1

    if rs_score >= 80:   score += 3; reasons.append(f"RS={rs_score} (Top 20%)")
    elif rs_score >= 60: score += 2
    elif rs_score >= 40: score += 1

    if price > ma50:     score += 2; reasons.append("Price above 50-day MA")

    inst = info.get("heldPercentInstitutions", 0) or 0
    if 0.3 < inst < 0.85: score += 2; reasons.append(f"Institutional {inst*100:.0f}%")
    elif inst > 0:         score += 1

    high52 = info.get("fiftyTwoWeekHigh", price)
    if price >= high52 * 0.95: score += 2; reasons.append("Within 5% of 52w high")

    if not hist_6m.empty and len(hist_6m) > 20:
        h = hist_6m.copy(); h["up"] = h["Close"] > h["Open"]
        if h[h["up"]]["Volume"].mean() > h[~h["up"]]["Volume"].mean():
            score += 1; reasons.append("Accumulation vol pattern")

    verdict = ("STRONG BUY" if score >= 14 else "BUY" if score >= 10
               else "WATCH" if score >= 7 else "PASS")
    return score, max_score, verdict, reasons


def sa_minervini_score(info, price, hist_1y, hist_2y):
    score, reasons, max_score = 0, [], 14

    if hist_1y.empty or len(hist_1y) < 60:
        return 0, max_score, "INSUFFICIENT DATA", []

    close = hist_1y["Close"]
    ma50  = close.rolling(50).mean().iloc[-1]
    ma150 = close.rolling(150).mean().iloc[-1]

    close2 = hist_2y["Close"] if not hist_2y.empty else close
    ma200  = close2.rolling(200).mean().dropna()
    ma200_val = ma200.iloc[-1] if not ma200.empty else None

    if price > ma50:   score += 2; reasons.append("Price > 50-day MA")
    if price > ma150:  score += 2; reasons.append("Price > 150-day MA")
    if ma200_val and price > ma200_val: score += 2; reasons.append("Price > 200-day MA")
    if not np.isnan(ma150) and ma200_val and ma150 > ma200_val:
        score += 1; reasons.append("150MA > 200MA")
    if ma200_val and len(ma200) >= 20:
        if ma200.iloc[-1] > ma200.iloc[-20]: score += 1; reasons.append("200MA trending up")

    low52  = info.get("fiftyTwoWeekLow",  price)
    high52 = info.get("fiftyTwoWeekHigh", price)
    if low52  and price >= low52  * 1.30: score += 2; reasons.append("30%+ above 52w low")
    if high52 and price >= high52 * 0.75: score += 2; reasons.append("Within 25% of 52w high")

    verdict = ("STRONG BUY" if score >= 11 else "BUY" if score >= 8
               else "WATCH" if score >= 5 else "PASS")
    return score, max_score, verdict, reasons


def sa_buffett_score(info, hist_2y):
    score, reasons, max_score = 0, [], 16

    pe  = info.get("trailingPE") or 0
    fpe = info.get("forwardPE")  or 0
    if 0 < pe < 25:   score += 2; reasons.append(f"P/E={pe:.1f} (value zone)")
    elif 0 < pe < 35: score += 1
    if 0 < fpe < 20:  score += 1; reasons.append(f"Fwd P/E={fpe:.1f} cheap")

    roe = (info.get("returnOnEquity") or 0) * 100
    if roe > 20:   score += 3; reasons.append(f"ROE={roe:.1f}% (Buffett loves >15%)")
    elif roe > 15: score += 2
    elif roe > 10: score += 1

    de = info.get("debtToEquity") or 999
    if de < 50:    score += 2; reasons.append(f"Low debt/equity {de:.0f}%")
    elif de < 100: score += 1

    gm = (info.get("grossMargins") or 0) * 100
    if gm > 50:   score += 2; reasons.append(f"Gross margin {gm:.1f}% (wide moat)")
    elif gm > 30: score += 1

    nm = (info.get("profitMargins") or 0) * 100
    if nm > 20:   score += 2; reasons.append(f"Net margin {nm:.1f}%")
    elif nm > 10: score += 1

    fcf = info.get("freeCashflow") or 0
    rev = info.get("totalRevenue") or 1
    fcf_yield = fcf / rev if rev else 0
    if fcf_yield > 0.10: score += 2; reasons.append(f"Strong FCF yield {fcf_yield*100:.1f}%")
    elif fcf_yield > 0:  score += 1

    rev_g = (info.get("revenueGrowth") or 0)
    if rev_g > 0.10:  score += 1; reasons.append("Revenue growing >10%/yr")

    verdict = ("STRONG BUY" if score >= 13 else "BUY" if score >= 9
               else "HOLD/WATCH" if score >= 6 else "PASS")
    return score, max_score, verdict, reasons


def sa_lynch_score(info):
    score, reasons, max_score = 0, [], 12

    peg = info.get("pegRatio") or 0
    if 0 < peg < 1.0:   score += 3; reasons.append(f"PEG={peg:.2f} (undervalued growth, <1 is ideal)")
    elif 0 < peg < 2.0: score += 2; reasons.append(f"PEG={peg:.2f} (fair)")

    rev_g = (info.get("revenueGrowth") or 0) * 100
    if rev_g > 20:   score += 2; reasons.append(f"Revenue growth {rev_g:.0f}%")
    elif rev_g > 10: score += 1

    eps_g = (info.get("earningsGrowth") or 0) * 100
    if eps_g > 20:   score += 2; reasons.append(f"EPS growth {eps_g:.0f}%")
    elif eps_g > 10: score += 1

    pe = info.get("trailingPE") or 999
    if 0 < pe < 20:    score += 2; reasons.append(f"P/E={pe:.1f} reasonable for growth")
    elif 0 < pe < 30:  score += 1

    de = info.get("debtToEquity") or 999
    if de < 50: score += 1; reasons.append("Manageable debt")

    inst = (info.get("heldPercentInstitutions") or 0) * 100
    if inst < 60: score += 1; reasons.append(f"Low institutional coverage {inst:.0f}% (undiscovered?)")

    mkt = info.get("marketCap") or 0
    if mkt > 10e9:   category = "Large grower"
    elif mkt > 2e9:  category = "Mid-cap grower"
    else:            category = "Small/micro-cap (ten-bagger potential)"
    reasons.append(f"Category: {category}")

    verdict = ("STRONG BUY" if score >= 10 else "BUY" if score >= 7
               else "WATCH" if score >= 4 else "PASS")
    return score, max_score, verdict, reasons


def sa_livermore_score(info, price, hist_6m, rs_score):
    score, reasons, max_score = 0, [], 10

    high52 = info.get("fiftyTwoWeekHigh", price)
    if price >= high52 * 0.97:   score += 2; reasons.append("At/near pivotal point (52w high)")
    elif price >= high52 * 0.90: score += 1

    if not hist_6m.empty and len(hist_6m) >= 20:
        avg_vol  = hist_6m["Volume"].iloc[-60:].mean() if len(hist_6m) >= 60 else hist_6m["Volume"].mean()
        last_vol = hist_6m["Volume"].iloc[-1]
        if last_vol > avg_vol * 1.5:   score += 2; reasons.append("Volume surge — smart money moving in")
        elif last_vol > avg_vol: score += 1

    if not hist_6m.empty and len(hist_6m) >= 60:
        ret_3m = (price - hist_6m["Close"].iloc[-60]) / hist_6m["Close"].iloc[-60] * 100
        if ret_3m > 20:   score += 2; reasons.append(f"Strong 3-month momentum +{ret_3m:.1f}%")
        elif ret_3m > 10: score += 1

    if rs_score >= 80:   score += 2; reasons.append("Leading stock in leading market")
    elif rs_score >= 60: score += 1

    if not hist_6m.empty and len(hist_6m) >= 20:
        closes = hist_6m["Close"].iloc[-20:]
        higher_lows = all(closes.iloc[i] >= closes.iloc[i-1] * 0.97 for i in range(1, min(5, len(closes))))
        if higher_lows: score += 1; reasons.append("Series of higher lows — no distribution")
        else: reasons.append("Warning: distribution pattern detected")

    verdict = ("LINE OF LEAST RESISTANCE" if score >= 8 else "EMERGING" if score >= 5
               else "WAIT" if score >= 3 else "NOT YET")
    return score, max_score, verdict, reasons


def sa_weinstein_score(info, price, hist_1y, hist_2y):
    score, reasons, max_score = 0, [], 10

    close_all = hist_2y["Close"] if not hist_2y.empty else hist_1y["Close"]
    ma30w = close_all.rolling(150).mean()

    if ma30w.dropna().empty:
        return 0, max_score, "STAGE UNKNOWN", []

    ma30w_val   = ma30w.iloc[-1]
    ma30w_20ago = ma30w.iloc[-20] if len(ma30w) >= 20 else ma30w.iloc[0]

    above_ma       = price > ma30w_val
    ma_trending_up = ma30w_val > ma30w_20ago

    if above_ma and ma_trending_up:
        stage = 2; stage_label = "Stage 2 (Advancing)"
        score += 4; reasons.append("Stage 2: Price above rising 30-week MA")
    elif above_ma and not ma_trending_up:
        stage = 3; stage_label = "Stage 3 (Topping)"
        score += 1; reasons.append("Stage 3: Price above flattening MA — caution")
    elif not above_ma and ma_trending_up:
        stage = 1; stage_label = "Stage 1 (Basing)"
        score += 2; reasons.append("Stage 1: Basing — wait for Stage 2 breakout")
    else:
        stage = 4; stage_label = "Stage 4 (Declining)"
        reasons.append("Stage 4: Avoid — downtrend in force")

    if stage == 2:
        high52 = info.get("fiftyTwoWeekHigh", price)
        low52  = info.get("fiftyTwoWeekLow",  price)
        if high52 and price >= high52 * 0.95: score += 3; reasons.append("Breakout from base")
        if not hist_1y.empty and len(hist_1y) > 40:
            recent_vol = hist_1y["Volume"].iloc[-5:].mean()
            avg_vol    = hist_1y["Volume"].mean()
            if recent_vol > avg_vol * 1.2: score += 2; reasons.append("Volume expanding on advance")
        if low52 and ma30w_val > low52: score += 1; reasons.append("30-week MA above 52w low")

    verdict = (stage_label + " — BUY"       if stage == 2 else
               stage_label + " — HOLD/SELL" if stage == 3 else
               stage_label + " — WAIT"      if stage == 1 else
               stage_label + " — AVOID")
    return score, max_score, verdict, reasons


def sa_dalio_score(info, price, hist_1y, rs_score):
    score, reasons, max_score = 0, [], 10

    de = info.get("debtToEquity") or 999
    if de < 50:    score += 2; reasons.append(f"Low leverage D/E={de:.0f}% (macro safe)")
    elif de < 100: score += 1

    cr = info.get("currentRatio") or 0
    if cr > 2: score += 1; reasons.append(f"Strong liquidity ratio {cr:.1f}x")

    rev_g = (info.get("revenueGrowth") or 0) * 100
    if rev_g > 10:  score += 2; reasons.append(f"Real growth {rev_g:.0f}% above inflation")
    elif rev_g > 0: score += 1

    nm = (info.get("profitMargins") or 0) * 100
    if nm > 15: score += 2; reasons.append(f"Margin {nm:.0f}% — pricing power in inflationary cycle")
    elif nm > 5: score += 1

    if rs_score >= 60:   score += 2; reasons.append("Outperforming market — macro tailwind")
    elif rs_score >= 40: score += 1

    dy = (info.get("dividendYield") or 0) * 100
    if dy > 0: score += 1; reasons.append(f"Dividend {dy:.1f}% — income component")

    verdict = ("ALL-WEATHER BUY" if score >= 8 else "BALANCED HOLD" if score >= 5
               else "DEFENSIVE PASS" if score >= 3 else "AVOID")
    return score, max_score, verdict, reasons


# ── CHART ──────────────────────────────────────────────────────────────────────

def sa_build_chart(hist_6m, hist_1y, ticker, entry, buy_zone_top, stop_loss, target20, ma50_val):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=hist_6m.index, open=hist_6m["Open"], high=hist_6m["High"],
        low=hist_6m["Low"], close=hist_6m["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        name="Price", showlegend=False,
    ), row=1, col=1)

    ma50  = hist_6m["Close"].rolling(50).mean()
    ma150 = hist_1y["Close"].rolling(150).mean().reindex(hist_6m.index, method="nearest")
    fig.add_trace(go.Scatter(x=hist_6m.index, y=ma50,
                             line=dict(color="#a855f7", width=1.5), name="50-day MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_6m.index, y=ma150,
                             line=dict(color="#f97316", width=1.2, dash="dot"), name="150-day MA"), row=1, col=1)

    x0, x1 = hist_6m.index[0], hist_6m.index[-1]

    def hline(y, color, dash, label):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      line=dict(color=color, width=1.5, dash=dash), row=1, col=1)
        fig.add_annotation(x=x1, y=y, text=f"  {label}  ${y:.2f}",
                            showarrow=False, xanchor="left",
                            font=dict(size=10, color="white"),
                            bgcolor=color, borderpad=3, row=1, col=1)

    hline(entry,        "#3b82f6", "solid", "Entry/Pivot")
    hline(buy_zone_top, "#06b6d4", "dot",   "Buy Zone +3%")
    hline(stop_loss,    "#ef4444", "dash",  "Stop -7%")
    hline(target20,     "#22c55e", "dot",   "Target +20%")

    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(hist_6m["Close"], hist_6m["Open"])]
    fig.add_trace(go.Bar(x=hist_6m.index, y=hist_6m["Volume"],
                         marker_color=colors, showlegend=False, name="Volume"), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — 6-Month Price Action",
        title_font_size=15, xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=130, t=40, b=10), height=500,
        legend=dict(orientation="h", y=-0.12),
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    return fig


def sa_build_sector_chart(etf_hist, etf_ticker):
    fig = go.Figure()
    if etf_hist.empty:
        return fig
    normed = etf_hist["Close"] / etf_hist["Close"].iloc[0] * 100
    fig.add_trace(go.Scatter(x=etf_hist.index, y=normed,
                             line=dict(color="#3b82f6", width=2),
                             fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                             name=etf_ticker))
    ret = round((etf_hist["Close"].iloc[-1] / etf_hist["Close"].iloc[0] - 1) * 100, 1)
    fig.update_layout(
        title=f"{etf_ticker} Sector ETF — 6-Month Performance ({ret:+.1f}%)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10), height=220,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        showlegend=False, font=dict(size=12),
    )
    return fig


# ── HELPERS ────────────────────────────────────────────────────────────────────

def sa_tbl_row(label, value, color=None):
    c = f"color:{color}; font-weight:600;" if color else "font-weight:600;"
    st.markdown(
        f'<div class="sa-tbl-row">'
        f'<span style="color:#6b7280">{label}</span>'
        f'<span style="{c}">{value}</span></div>',
        unsafe_allow_html=True,
    )


def sa_score_bar(score, max_score, colors=("#22c55e", "#f59e0b", "#ef4444")):
    pct   = score / max_score * 100
    color = colors[0] if pct >= 66 else colors[1] if pct >= 40 else colors[2]
    st.markdown(
        f'<div class="sa-score-bar-bg">'
        f'<div style="width:{pct:.0f}%; background:{color}; height:8px; border-radius:4px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sa_verdict_card(name, score, max_score, verdict, reasons, bg, border):
    bullets = "".join(f"<li style='margin:2px 0'>{r}</li>" for r in reasons[:5])
    html = (
        f"<div class='sa-verdict-box' style='background:{bg}; border-left:4px solid {border}'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px'>"
        f"<span style='font-weight:700; color:#111; font-size:15px'>{name}</span>"
        f"<span style='font-weight:700; color:{border}; font-size:13px'>{verdict} &nbsp; {score}/{max_score}</span>"
        f"</div>"
        f"<ul style='margin:4px 0 0 16px; padding:0; color:#374151; font-size:13px; line-height:1.7'>{bullets}</ul>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def sa_calc_hold_time(oneil_v, minervini_v, buffett_v, lynch_v, weinstein_v):
    swing_score = long_score = 0
    if "BUY" in oneil_v:      swing_score += 2
    if "BUY" in minervini_v:  swing_score += 2
    if "Stage 2" in weinstein_v: swing_score += 1
    if "BUY" in buffett_v:    long_score += 3
    if "BUY" in lynch_v:      long_score += 2

    if long_score >= 4 and swing_score >= 2:
        return "1–3 years", "Strong fundamentals + technical setup. Hold through volatility for full thesis to play out. Trim at +25–50%, let a core position run.", "#065f46"
    elif long_score >= 4:
        return "6 months – 2 years", "Value/GARP setup warrants patience. Check earnings every quarter and sell if thesis breaks.", "#1e40af"
    elif swing_score >= 4:
        return "8–26 weeks", "Classic breakout position trade. Sell half at +20%, trail stop on remainder. Re-evaluate weekly.", "#92400e"
    elif swing_score >= 2:
        return "2–8 weeks", "Swing trade only. Keep stop tight at -7%. Take profits at first resistance.", "#6b7280"
    else:
        return "Not recommended", "Score too low across frameworks. Wait for a cleaner setup.", "#991b1b"


# ── DCF HELPERS ───────────────────────────────────────────────────────────────

def _sa_cf_row(df, *names):
    for n in names:
        if n in df.index:
            vals = df.loc[n].dropna()
            if not vals.empty:
                return vals
    return pd.Series(dtype=float)


def _sa_dcf_wacc(info, price):
    shares_out = info.get("sharesOutstanding") or 1
    beta       = max(info.get("beta") or 1.0, 0.5)
    total_debt = info.get("totalDebt") or 0
    mkt_cap    = info.get("marketCap") or (price * shares_out)
    rf, erp    = 0.045, 0.055
    cost_equity = rf + beta * erp
    dw = total_debt / (mkt_cap + total_debt) if (mkt_cap + total_debt) else 0
    wacc = (1 - dw) * cost_equity + dw * (0.05 * 0.79)
    return round(max(0.07, min(0.18, wacc)), 4)


def _sa_smart_fcf(info, cashflow_df):
    shares = info.get("sharesOutstanding") or 1
    cf     = cashflow_df

    op_cf_s = _sa_cf_row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    fcf_s   = _sa_cf_row(cf, "Free Cash Flow")
    da_s    = _sa_cf_row(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion")
    sbc_s   = _sa_cf_row(cf, "Stock Based Compensation")
    capex_s = _sa_cf_row(cf, "Capital Expenditure")

    op_cf = float(op_cf_s.iloc[0]) if not op_cf_s.empty else (info.get("operatingCashflow") or 0)
    da    = float(da_s.iloc[0])    if not da_s.empty    else 0
    sbc   = float(sbc_s.iloc[0])   if not sbc_s.empty   else 0
    capex = abs(float(capex_s.iloc[0])) if not capex_s.empty else 0

    if not fcf_s.empty:
        actual_fcf = float(fcf_s.iloc[0])
        fcf_hist   = fcf_s.dropna().iloc[:3]
        avg3_fcf   = float(fcf_hist.mean()) if len(fcf_hist) >= 2 else actual_fcf
    else:
        actual_fcf = (info.get("freeCashflow") or 0)
        avg3_fcf   = actual_fcf

    if op_cf <= 0 and actual_fcf <= 0:
        return None

    if capex == 0 and op_cf > 0 and actual_fcf > 0:
        capex = op_cf - actual_fcf
    capex_ratio = capex / op_cf if op_cf > 0 else 0

    norm_fcf = op_cf - da if da > 0 else op_cf * 0.65

    if capex_ratio > 0.40:
        if avg3_fcf > 0:
            base = norm_fcf * 0.65 + avg3_fcf * 0.35
            note = (f"Heavy capex cycle — {capex_ratio*100:.0f}% of op CF is capex. "
                    f"Using Damodaran normalization: 65% × (Op CF − D&A) + 35% × 3yr-avg FCF.")
        else:
            base = norm_fcf
            note = (f"Heavy capex cycle ({capex_ratio*100:.0f}%). "
                    f"Normalized FCF = Op CF − D&A.")
        method = "normalized"
    elif avg3_fcf > 0 and actual_fcf > 0 and abs(actual_fcf - avg3_fcf) / max(avg3_fcf, 1) > 0.35:
        base   = avg3_fcf
        note   = f"FCF volatile — using 3-year average (${avg3_fcf/1e9:.1f}B) for stability."
        method = "3yr_avg"
    elif actual_fcf > 0:
        base   = min(actual_fcf, op_cf * 0.95) if op_cf > 0 else actual_fcf
        note   = "Standard reported FCF from cashflow statement."
        method = "reported"
    else:
        base   = op_cf * 0.70
        note   = "Estimated FCF = 70% of operating cash flow."
        method = "estimated"

    if base <= 0:
        return None

    return {
        "fcf_ps":      base / shares,
        "raw_fcf_ps":  actual_fcf / shares if actual_fcf > 0 else 0,
        "norm_fcf_ps": norm_fcf / shares,
        "avg3_fcf_ps": avg3_fcf / shares if avg3_fcf > 0 else 0,
        "op_cf":       op_cf,
        "capex":       capex,
        "da":          da,
        "sbc":         sbc,
        "capex_ratio": capex_ratio,
        "method":      method,
        "note":        note,
    }


def _sa_dcf_defaults(info, capex_ratio=0):
    trailing_eps = info.get("trailingEps") or 0
    forward_eps  = info.get("forwardEps")  or 0
    rev_growth   = info.get("revenueGrowth") or 0
    eps_a        = info.get("earningsGrowth") or 0

    if trailing_eps > 0 and forward_eps > 0:
        fwd_eps_g = (forward_eps / trailing_eps) - 1
        source    = f"Analyst EPS ${trailing_eps:.2f}→${forward_eps:.2f} (+{fwd_eps_g*100:.0f}%)"
    elif trailing_eps < 0 and forward_eps > 0:
        fwd_eps_g = 0.40
        source    = "Turning profitable — 40% growth assumed"
    else:
        fwd_eps_g = max(eps_a, 0)
        source    = f"Trailing EPS growth {eps_a*100:.0f}%"

    if rev_growth > 0.02 and fwd_eps_g > rev_growth * 3:
        fwd_cap = min(fwd_eps_g, rev_growth * 2.5, 0.80)
    else:
        fwd_cap = min(fwd_eps_g, 0.80)

    sustainable = min(max(rev_growth * 1.3, 0.05), 0.25)

    if capex_ratio > 0.40:
        g1_5 = round(min((fwd_cap * 0.5 + sustainable * 0.5), 0.60) * 100)
    else:
        g1_5 = round(((2 * fwd_cap + 3 * sustainable) / 5) * 100)
    g1_5 = max(5, min(g1_5, 60))

    g6_10 = max(5, round(min(sustainable * 0.60, 0.20) * 100))

    return g1_5, g6_10, fwd_eps_g, rev_growth, source


def _sa_run_dcf(fcf_per_share, net_cash_ps, g1, g2, wacc, terminal_g=0.025):
    if wacc <= terminal_g:
        wacc = terminal_g + 0.01
    pv, cf = 0.0, fcf_per_share
    for yr in range(1, 11):
        cf *= (1 + (g1 if yr <= 5 else g2))
        pv += cf / (1 + wacc) ** yr
    tv_pv = (cf * (1 + terminal_g) / (wacc - terminal_g)) / (1 + wacc) ** 10
    return pv + tv_pv + net_cash_ps


def sa_calc_dcf(info, price, cashflow_df=None, g1_pct=None, g2_pct=None, wacc_pct=None):
    cf_df = cashflow_df if cashflow_df is not None else pd.DataFrame()

    fcf_data = _sa_smart_fcf(info, cf_df)
    if fcf_data is None:
        return None

    fcf_ps      = fcf_data["fcf_ps"]
    capex_ratio = fcf_data["capex_ratio"]

    shares_out  = info.get("sharesOutstanding") or 1
    total_debt  = info.get("totalDebt") or 0
    total_cash  = info.get("totalCash") or 0
    net_cash_ps = (total_cash - total_debt) / shares_out

    default_g1, default_g2, fwd_eps_g, rev_growth, source = _sa_dcf_defaults(info, capex_ratio)
    default_wacc = round(_sa_dcf_wacc(info, price) * 100)

    g1   = (g1_pct   / 100) if g1_pct   is not None else (default_g1   / 100)
    g2   = (g2_pct   / 100) if g2_pct   is not None else (default_g2   / 100)
    wacc = (wacc_pct / 100) if wacc_pct is not None else (default_wacc / 100)

    intrinsic = _sa_run_dcf(fcf_ps, net_cash_ps, g1, g2, wacc)
    upside    = round((intrinsic - price) / price * 100, 1)

    implied_g = None
    if fcf_ps > 0:
        lo, hi = 0.0, 3.0
        if _sa_run_dcf(fcf_ps, net_cash_ps, lo, lo * 0.6, wacc) < price:
            for _ in range(60):
                mid = (lo + hi) / 2
                if _sa_run_dcf(fcf_ps, net_cash_ps, mid, mid * 0.6, wacc) < price:
                    lo = mid
                else:
                    hi = mid
            implied_g = round(mid * 100, 1)

    return {
        "intrinsic":     round(intrinsic, 2),
        "mos_20":        round(intrinsic * 0.80, 2),
        "mos_30":        round(intrinsic * 0.70, 2),
        "upside":        upside,
        "wacc":          round(wacc * 100, 1),
        "g1_5":          round(g1 * 100, 1),
        "g6_10":         round(g2 * 100, 1),
        "default_g1":    default_g1,
        "default_g2":    default_g2,
        "default_wacc":  default_wacc,
        "fcf_per_share": round(fcf_ps, 2),
        "raw_fcf_ps":    round(fcf_data["raw_fcf_ps"], 2),
        "norm_fcf_ps":   round(fcf_data["norm_fcf_ps"], 2),
        "avg3_fcf_ps":   round(fcf_data["avg3_fcf_ps"], 2),
        "op_cf":         fcf_data["op_cf"],
        "capex":         fcf_data["capex"],
        "da":            fcf_data["da"],
        "capex_ratio":   capex_ratio,
        "fcf_method":    fcf_data["method"],
        "fcf_note":      fcf_data["note"],
        "fwd_eps_growth": round(fwd_eps_g * 100, 1),
        "rev_growth":    round(rev_growth * 100, 1),
        "growth_source": source,
        "implied_growth": implied_g,
    }


def sa_dcf_sensitivity_table(fcf_ps, net_cash_ps, base_g1, base_g2, base_wacc, price):
    g_offsets   = [-20, -10, 0, +10, +20]
    wacc_deltas = [-3, -2, 0, +2, +3]
    rows = []
    for wd in wacc_deltas:
        w = max(0.06, (base_wacc + wd) / 100)
        cells = []
        for go in g_offsets:
            g1 = max(0.01, (base_g1 + go) / 100)
            g2 = max(0.01, g1 * 0.55)
            iv = _sa_run_dcf(fcf_ps, net_cash_ps, g1, g2, w)
            up = (iv - price) / price * 100
            bg = "#d1fae5" if up > 20 else "#fffbeb" if up > -10 else "#fee2e2"
            cells.append(
                f'<td style="text-align:center;padding:6px 10px;font-size:12px;'
                f'background:{bg};border:1px solid #e5e7eb">'
                f'<b>${iv:.0f}</b><br>'
                f'<span style="color:#6b7280;font-size:10px">{up:+.0f}%</span></td>'
            )
        rows.append(
            f'<tr><td style="padding:6px 10px;font-size:12px;font-weight:600;'
            f'border:1px solid #e5e7eb;background:#f9fafb">'
            f'WACC {base_wacc+wd:.0f}%</td>' + "".join(cells) + "</tr>"
        )

    g_headers = "".join(
        f'<th style="padding:6px 10px;font-size:12px;background:#f3f4f6;border:1px solid #e5e7eb">'
        f'Growth Yr1-5<br>{base_g1+go:.0f}%</th>'
        for go in g_offsets
    )
    header = (f'<tr><th style="padding:6px 10px;font-size:12px;background:#f3f4f6;'
              f'border:1px solid #e5e7eb"></th>{g_headers}</tr>')
    return f'<table style="border-collapse:collapse;width:100%">{header}{"".join(rows)}</table>'


def sa_calc_signals(info, hist_6m, rs_score, price, ma50):
    signals = []
    high52       = info.get("fiftyTwoWeekHigh", price)
    pct_from_high = (price - high52) / high52 * 100

    if pct_from_high >= -5:  signals.append(("Within 5% of 52-week high", "neutral"))
    if rs_score >= 80:       signals.append(("Relative Strength ≥ 80 (Top 20%)", "green"))
    if rs_score < 40:        signals.append(("Relative Strength weak — lagging the market", "red"))
    if price > ma50:         signals.append(("Trading above 50-day MA", "green"))
    else:                    signals.append(("Below 50-day MA — caution", "red"))

    if not hist_6m.empty and len(hist_6m) > 20:
        h  = hist_6m.copy(); h["up"] = h["Close"] > h["Open"]
        uv = h[h["up"]]["Volume"].mean(); dv = h[~h["up"]]["Volume"].mean()
        if uv > dv * 1.1:   signals.append(("Accumulation: up-vol > down-vol", "green"))
        elif dv > uv * 1.1: signals.append(("Distribution: down-vol > up-vol", "red"))

    if not hist_6m.empty and len(hist_6m) > 60:
        q1 = hist_6m["Close"].iloc[:20].mean()
        q2 = hist_6m["Close"].iloc[20:40].mean()
        q3 = hist_6m["Close"].iloc[40:].mean()
        if q1 < q2 < q3: signals.append(("Consistent uptrend across all timeframes", "neutral"))

    inst_pct = info.get("heldPercentInstitutions", 0) * 100
    if inst_pct > 50: signals.append(("Healthy institutional ownership", "neutral"))

    if not hist_6m.empty and len(hist_6m) >= 40:
        p8w = hist_6m["Close"].iloc[-40]
        g8w = (price - p8w) / p8w * 100
        if g8w > 20: signals.append((f"FAST MOVER: +{g8w:.1f}% in 8 weeks — O'Neil: hold for bigger gain", "amber"))

    return signals, pct_from_high


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED TECHNICAL CHART — TradingView-style
# ══════════════════════════════════════════════════════════════════════════════

_TV = {
    "bg":    "#131722", "grid":  "#1e222d", "text":  "#d1d4dc",
    "border":"#2a2e39", "green": "#089981", "red":   "#f23645",
    "blue":  "#2962ff", "orange":"#ff9800", "purple":"#9c27b0",
    "cyan":  "#00bcd4", "yellow":"#ffc107",
    "ema9":  "#f5f542", "ema21": "#ff9800", "ema50": "#2196f3",
    "ema200":"#e91e63", "sma20": "#ab47bc", "sma50": "#26c6da",
    "sma200":"#ef5350", "vwap":  "#ff6d00", "bb":    "#78909c",
    "res":   "#f23645", "sup":   "#089981", "pp":    "#ffc107",
    "fib":   "#9c27b0",
}


@st.cache_data(ttl=300)
def _tv_fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def _tv_supertrend(high, low, close, atr, multiplier=3):
    hl2 = (high + low) / 2
    upper = (hl2 + multiplier * atr).copy()
    lower = (hl2 - multiplier * atr).copy()
    st_line = pd.Series(np.nan, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i - 1] > upper.iloc[i - 1]:
            upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])
        if close.iloc[i - 1] < lower.iloc[i - 1]:
            lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
        if close.iloc[i] > upper.iloc[i]:
            st_line.iloc[i] = lower.iloc[i]
        elif close.iloc[i] < lower.iloc[i]:
            st_line.iloc[i] = upper.iloc[i]
        else:
            st_line.iloc[i] = st_line.iloc[i - 1] if not np.isnan(st_line.iloc[i - 1]) else lower.iloc[i]
    return upper, lower, st_line


def _tv_add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for p in [20, 50, 100, 200]:
        df[f"SMA{p}"] = ta.trend.sma_indicator(c, window=p)
    for p in [9, 13, 21, 34, 50, 89, 200]:
        df[f"EMA{p}"] = ta.trend.ema_indicator(c, window=p)
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    df["BB_pct"] = bb.bollinger_pband()
    tp = (h + l + c) / 3
    df["VWAP"]    = (tp * v).cumsum() / v.cumsum()
    df["RSI"]     = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    df["ATR"]     = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["Stoch_K"], df["Stoch_D"] = stoch.stoch(), stoch.stoch_signal()
    df["Vol_SMA20"] = ta.trend.sma_indicator(v.astype(float), window=20)
    ichi = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df["Ichi_conv"], df["Ichi_base"] = ichi.ichimoku_conversion_line(), ichi.ichimoku_base_line()
    df["Ichi_span_a"], df["Ichi_span_b"] = ichi.ichimoku_a(), ichi.ichimoku_b()
    if "ATR" in df.columns:
        _, _, df["Supertrend"] = _tv_supertrend(h, l, c, df["ATR"], multiplier=3)
    return df


def _tv_find_pivots(df, order=5):
    hi = df["High"].values; lo = df["Low"].values
    ri = argrelextrema(hi, np.greater_equal, order=order)[0]
    si = argrelextrema(lo, np.less_equal,    order=order)[0]
    return (sorted({round(hi[i], 2) for i in ri}, reverse=True),
            sorted({round(lo[i], 2) for i in si}, reverse=True))


def _tv_cluster(levels, tol=0.003):
    if not levels: return []
    out, grp = [], [levels[0]]
    for v in levels[1:]:
        if abs(v - grp[-1]) / grp[-1] < tol:
            grp.append(v)
        else:
            out.append(round(np.mean(grp), 2)); grp = [v]
    out.append(round(np.mean(grp), 2))
    return out


def _tv_pivots(df):
    p = df.iloc[-2]; h, l, c = float(p["High"]), float(p["Low"]), float(p["Close"])
    pp = (h + l + c) / 3
    return {"PP": pp, "R1": 2*pp-l, "R2": pp+(h-l), "R3": h+2*(pp-l),
            "S1": 2*pp-h, "S2": pp-(h-l), "S3": l-2*(h-pp)}


def _tv_fibs(df, lookback=100):
    sub = df.tail(lookback); hi = float(sub["High"].max()); lo = float(sub["Low"].min())
    d = hi - lo
    return {f"Fib {int(f*100)}%": round(hi - d*f, 2)
            for f in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}


def _tv_signal_summary(df):
    last = df.iloc[-1]; close = float(last["Close"]); scores = []; sigs = {}

    def label(s):
        if s >  0.6: return "Strong Bull", "bull"
        if s >  0.2: return "Bull", "bull"
        if s < -0.6: return "Strong Bear", "bear"
        if s < -0.2: return "Bear", "bear"
        return "Neutral", "neutral"

    for p in [9, 21, 50, 200]:
        col = f"EMA{p}"
        if col in df.columns and not pd.isna(last[col]):
            v = float(last[col]); ab = close > v
            sigs[f"EMA {p}"] = (("Above", "bull") if ab else ("Below", "bear"),
                                  f"{close:,.2f} vs {v:,.2f}")
            scores.append(1 if ab else -1)

    for p in [20, 50, 200]:
        col = f"SMA{p}"
        if col in df.columns and not pd.isna(last[col]):
            v = float(last[col]); ab = close > v
            sigs[f"SMA {p}"] = (("Above", "bull") if ab else ("Below", "bear"),
                                  f"{close:,.2f} vs {v:,.2f}")
            scores.append(1 if ab else -1)

    if "RSI" in df.columns and not pd.isna(last["RSI"]):
        rsi = float(last["RSI"])
        if rsi > 70:
            sigs["RSI(14)"] = (("Overbought", "bear"), f"{rsi:.1f}"); scores.append(-1)
        elif rsi < 30:
            sigs["RSI(14)"] = (("Oversold", "bull"),   f"{rsi:.1f}"); scores.append(1)
        else:
            sigs["RSI(14)"] = (("Neutral", "neutral"),  f"{rsi:.1f}"); scores.append(0.5 if rsi > 50 else -0.5)

    if "MACD" in df.columns and not pd.isna(last["MACD"]):
        bull = float(last["MACD"]) > float(last["MACD_signal"])
        sigs["MACD"] = (("Bull Cross", "bull") if bull else ("Bear Cross", "bear"),
                         f"Hist: {float(last['MACD_hist']):.3f}")
        scores.append(1 if bull else -1)

    if "Supertrend" in df.columns and not pd.isna(last["Supertrend"]):
        bull = close > float(last["Supertrend"])
        sigs["Supertrend"] = (("Bullish", "bull") if bull else ("Bearish", "bear"),
                               f"{float(last['Supertrend']):,.2f}")
        scores.append(1 if bull else -1)

    if "BB_pct" in df.columns and not pd.isna(last["BB_pct"]):
        pct = float(last["BB_pct"])
        sigs["BB %B"] = (("Near Upper", "bear") if pct > 0.8 else
                          ("Near Lower", "bull") if pct < 0.2 else
                          ("Mid-Band", "neutral"), f"{pct:.2f}")

    avg = np.mean(scores) if scores else 0
    return {"sigs": sigs, "overall": label(avg), "score": avg}


def _tv_build_chart(df, ticker, opts):
    # Row 1: Price (primary y) + Volume (secondary y via specs)
    # Row 2: Volume histogram (standalone, more readable)
    # Row 3: MACD
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.12, 0.20],
        vertical_spacing=0.020,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    inc_color = [_TV["green"] if float(c) >= float(o) else _TV["red"]
                 for c, o in zip(df["Close"], df["Open"])]

    # ── Candlestick (primary y) ───────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_fillcolor=_TV["green"], decreasing_fillcolor=_TV["red"],
        increasing_line_color=_TV["green"],  decreasing_line_color=_TV["red"],
        line_width=1,
    ), row=1, col=1, secondary_y=False)

    # ── Volume (secondary y — capped to bottom 20% of price panel) ───────────
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=inc_color, opacity=0.25, showlegend=False,
    ), row=1, col=1, secondary_y=True)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    if opts.get("emas"):
        for col, color, w in [
            ("EMA9",  _TV["ema9"],   1.2), ("EMA21",  _TV["ema21"],  1.4),
            ("EMA50", _TV["ema50"],  1.6), ("EMA200", _TV["ema200"], 2.0),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col.replace("EMA", "EMA "),
                    line=dict(color=color, width=w), opacity=0.9,
                ), row=1, col=1, secondary_y=False)

    # ── SMAs ──────────────────────────────────────────────────────────────────
    if opts.get("smas"):
        for col, color, w in [
            ("SMA20", _TV["sma20"], 1.2), ("SMA50",  _TV["sma50"],  1.5),
            ("SMA200",_TV["sma200"],2.0),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col.replace("SMA", "SMA "),
                    line=dict(color=color, width=w, dash="dot"), opacity=0.8,
                ), row=1, col=1, secondary_y=False)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if opts.get("bb") and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color=_TV["bb"], width=1, dash="dash"), opacity=0.65,
        ), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color=_TV["bb"], width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.07)", opacity=0.65,
        ), row=1, col=1, secondary_y=False)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if opts.get("vwap") and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color=_TV["vwap"], width=1.5, dash="dashdot"), opacity=0.85,
        ), row=1, col=1, secondary_y=False)

    # ── Supertrend ────────────────────────────────────────────────────────────
    if opts.get("supertrend") and "Supertrend" in df.columns:
        close_arr = df["Close"].values; st_arr = df["Supertrend"].values; idx_arr = df.index
        bx, by, rx, ry = [], [], [], []
        for i, idx in enumerate(idx_arr):
            if not np.isnan(st_arr[i]):
                if close_arr[i] > st_arr[i]: bx.append(idx); by.append(st_arr[i])
                else:                         rx.append(idx); ry.append(st_arr[i])
        if bx: fig.add_trace(go.Scatter(x=bx, y=by, name="ST Bull",
            line=dict(color=_TV["green"], width=2), mode="lines"),
            row=1, col=1, secondary_y=False)
        if rx: fig.add_trace(go.Scatter(x=rx, y=ry, name="ST Bear",
            line=dict(color=_TV["red"],   width=2), mode="lines"),
            row=1, col=1, secondary_y=False)

    # ── Ichimoku ──────────────────────────────────────────────────────────────
    if opts.get("ichimoku") and "Ichi_conv" in df.columns:
        for nm, col, c in [("Tenkan","Ichi_conv","#26a69a"),("Kijun","Ichi_base","#ef5350")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm,
                line=dict(color=c, width=1.2)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["Ichi_span_a"], name="Span A",
            line=dict(color=_TV["green"], width=0.7)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["Ichi_span_b"], name="Span B",
            line=dict(color=_TV["red"], width=0.7),
            fill="tonexty", fillcolor="rgba(8,153,129,0.05)"),
            row=1, col=1, secondary_y=False)

    # ── S/R Lines ─────────────────────────────────────────────────────────────
    cur = float(df["Close"].iloc[-1]); x0, x1 = df.index[0], df.index[-1]

    if opts.get("sr"):
        for r in [v for v in opts["res"] if v > cur * 0.98][:6]:
            # Solid resistance line
            fig.add_shape(type="line", x0=x0, x1=x1, y0=r, y1=r,
                line=dict(color=_TV["res"], width=1.5),
                row=1, col=1)
            # Faint fill zone above resistance
            fig.add_hrect(y0=r, y1=r * 1.003,
                fillcolor="rgba(242,54,69,0.10)", line_width=0, row=1, col=1)
            fig.add_annotation(
                x=x1, y=r,
                text=f"  ▶ R  {r:,.2f}",
                font=dict(color=_TV["res"], size=11, family="monospace"),
                bgcolor="rgba(30,34,45,0.75)", borderpad=3,
                showarrow=False, xanchor="left", row=1, col=1)
        for s in [v for v in opts["sup"] if v < cur * 1.02][:6]:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=s, y1=s,
                line=dict(color=_TV["sup"], width=1.5),
                row=1, col=1)
            fig.add_hrect(y0=s * 0.997, y1=s,
                fillcolor="rgba(8,153,129,0.10)", line_width=0, row=1, col=1)
            fig.add_annotation(
                x=x1, y=s,
                text=f"  ▶ S  {s:,.2f}",
                font=dict(color=_TV["sup"], size=11, family="monospace"),
                bgcolor="rgba(30,34,45,0.75)", borderpad=3,
                showarrow=False, xanchor="left", row=1, col=1)

    # ── Pivot Points ──────────────────────────────────────────────────────────
    if opts.get("pivots"):
        pc = {"PP":_TV["pp"],"R1":_TV["res"],"R2":_TV["res"],"R3":_TV["res"],
              "S1":_TV["sup"],"S2":_TV["sup"],"S3":_TV["sup"]}
        for nm, val in opts["pivot_vals"].items():
            c = pc.get(nm, _TV["text"])
            fig.add_shape(type="line", x0=x0, x1=x1, y0=val, y1=val,
                line=dict(color=c, width=0.8, dash="dashdot"), row=1, col=1)
            fig.add_annotation(x=x0, y=val, text=f"{nm} {val:,.2f}  ",
                font=dict(color=c, size=9, family="monospace"),
                showarrow=False, xanchor="right", row=1, col=1)

    # ── Fibonacci ─────────────────────────────────────────────────────────────
    if opts.get("fib"):
        for nm, val in opts["fib_vals"].items():
            fig.add_shape(type="line", x0=x0, x1=x1, y0=val, y1=val,
                line=dict(color=_TV["fib"], width=0.6, dash="longdash"), row=1, col=1)
            fig.add_annotation(x=x1, y=val, text=f"  {nm}: {val:,.2f}",
                font=dict(color=_TV["fib"], size=8, family="monospace"),
                showarrow=False, xanchor="left", row=1, col=1)

    # ── Volume row (row 2 — standalone for clarity) ───────────────────────────
    if "Vol_SMA20" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol",
            marker_color=inc_color, opacity=0.5, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Vol_SMA20"], name="Vol MA20",
            line=dict(color=_TV["yellow"], width=1.2), showlegend=False), row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    if "MACD" in df.columns:
        hc = [_TV["green"] if v >= 0 else _TV["red"] for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=hc, opacity=0.65), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=_TV["blue"], width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Sig",
            line=dict(color=_TV["orange"], width=1.5)), row=3, col=1)
        fig.add_hline(y=0, line=dict(color=_TV["border"], width=0.6), row=3, col=1)

    # ── Range-selector buttons (1W / 1M / 3M / 6M / 1Y / ALL) ───────────────
    range_selector = dict(
        buttons=[
            dict(count=7,  label="1W", step="day",   stepmode="backward"),
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="ALL"),
        ],
        bgcolor=_TV["border"],
        activecolor=_TV["blue"],
        bordercolor=_TV["grid"],
        borderwidth=1,
        font=dict(color=_TV["text"], size=10),
        x=0, y=1.0,
    )

    ax = dict(gridcolor=_TV["grid"], zerolinecolor=_TV["border"],
              tickfont=dict(color=_TV["text"], size=10),
              showgrid=True, gridwidth=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_TV["bg"],
        plot_bgcolor=_TV["bg"],
        font=dict(color=_TV["text"], size=11, family="monospace"),
        height=860,
        margin=dict(l=70, r=140, t=60, b=20),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        # Vertical right-side legend — less crowded than horizontal
        legend=dict(
            x=1.01, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(30,34,45,0.85)",
            bordercolor=_TV["border"],
            borderwidth=1,
            font=dict(size=9.5, color=_TV["text"]),
            tracegroupgap=2,
        ),
        # Volume secondary y: capped so bars stay in bottom 18% of price panel
        yaxis2=dict(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[0, float(df["Volume"].max()) * 6],
            fixedrange=True,
        ),
        # Range selector on the top x-axis
        xaxis=dict(rangeselector=range_selector, type="date"),
    )

    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    # Keep volume secondary y clean
    fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True, row=1)

    return fig



# ── MAIN RENDER FUNCTION ──────────────────────────────────────────────────────

def render_stock_analyzer():
    """Render the full Stock Analyzer — Multi-Framework Analysis UI."""

    # Inject CSS
    st.markdown(f"<style>{SA_CSS}</style>", unsafe_allow_html=True)

    st.markdown("## Stock Analyzer — Multi-Framework Analysis")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        ticker_input = st.text_input(
            "", placeholder="Ticker (AAPL · AMD · NVDA · TSLA)",
            label_visibility="collapsed",
            key="sa_ticker_input",
        )
    with c2:
        portfolio_size = st.number_input(
            "Portfolio ($)", value=10000, step=1000, min_value=1000,
            key="sa_portfolio_size",
        )
    with c3:
        risk_pct = st.selectbox(
            "Risk per trade", ["1%", "2%", "3%", "5%"], index=1,
            key="sa_risk_pct",
        )

    if not ticker_input:
        st.info("Enter any US stock ticker above to get a full multi-framework analysis.")
        return

    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching {ticker}..."):
        try:
            info, hist_6m, hist_1y, hist_2y = sa_get_stock_data(ticker)
            spy_1y      = sa_get_spy_data()
            cashflow_df = sa_get_financials(ticker)
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
            return

    if hist_6m.empty:
        st.error("No data found. Check the ticker.")
        return

    # ── CORE VALUES ─────────────────────────────────────────────────────────────
    price    = hist_6m["Close"].iloc[-1]
    ma50s    = hist_6m["Close"].rolling(50).mean()
    ma50_val = ma50s.dropna().iloc[-1] if not ma50s.dropna().empty else price

    entry        = price
    buy_zone_top = round(entry * 1.03, 2)
    stop_loss    = round(entry * 0.93, 2)
    target20     = round(entry * 1.20, 2)
    target25     = round(entry * 1.25, 2)

    risk_per_share   = entry - stop_loss
    reward_per_share = target20 - entry
    rr_ratio         = round(reward_per_share / risk_per_share, 2) if risk_per_share else 0

    risk_dollar = portfolio_size * (float(risk_pct.strip("%")) / 100)
    shares      = max(1, int(risk_dollar / risk_per_share)) if risk_per_share > 0 else 1
    actual_pos  = round(shares * entry, 2)

    rs_score = sa_calc_rs(hist_1y, spy_1y)

    eps_q_g  = (info.get("earningsQuarterlyGrowth") or 0) * 100
    eps_a_g  = (info.get("earningsGrowth")           or 0) * 100
    rev_g    = (info.get("revenueGrowth")             or 0) * 100
    inst_pct = (info.get("heldPercentInstitutions")   or 0) * 100
    float_sh = info.get("floatShares") or 0
    avg_vol  = info.get("averageVolume", 1) or 1
    curr_vol = hist_6m["Volume"].iloc[-1]
    vol_vs_avg = round(curr_vol / avg_vol, 2)
    high52   = info.get("fiftyTwoWeekHigh", price)
    low52    = info.get("fiftyTwoWeekLow",  price)
    pct_52h  = round((price - high52) / high52 * 100, 1)
    sector   = info.get("sector",   "")
    industry = info.get("industry", "")
    company  = info.get("longName", ticker)

    # ── ALL SCORES ──────────────────────────────────────────────────────────────
    oneil_s,  oneil_max,  oneil_v,  oneil_r  = sa_oneil_score(info, rs_score, price, ma50_val, hist_6m)
    mini_s,   mini_max,   mini_v,   mini_r   = sa_minervini_score(info, price, hist_1y, hist_2y)
    buff_s,   buff_max,   buff_v,   buff_r   = sa_buffett_score(info, hist_2y)
    lynch_s,  lynch_max,  lynch_v,  lynch_r  = sa_lynch_score(info)
    liver_s,  liver_max,  liver_v,  liver_r  = sa_livermore_score(info, price, hist_6m, rs_score)
    wein_s,   wein_max,   wein_v,   wein_r   = sa_weinstein_score(info, price, hist_1y, hist_2y)
    dalio_s,  dalio_max,  dalio_v,  dalio_r  = sa_dalio_score(info, price, hist_1y, rs_score)

    signals, _ = sa_calc_signals(info, hist_6m, rs_score, price, ma50_val)

    hold_duration, hold_advice, hold_color = sa_calc_hold_time(
        oneil_v, mini_v, buff_v, lynch_v, wein_v
    )

    buy_votes = sum(
        1 for v in [oneil_v, mini_v, buff_v, lynch_v, liver_v, wein_v]
        if "BUY" in v or "Stage 2" in v or "RESISTANCE" in v
    )
    overall  = "BUY" if buy_votes >= 4 else "WATCH" if buy_votes >= 2 else "PASS"
    ov_color = "#22c55e" if overall == "BUY" else "#f59e0b" if overall == "WATCH" else "#6b7280"

    # ── HEADER ──────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"""
        <div class="sa-metric-card">
            <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap">
                <span style="font-size:30px; font-weight:800; color:#111">{ticker}</span>
                <span style="background:{ov_color}; color:white; padding:4px 16px;
                      border-radius:20px; font-size:14px; font-weight:700">{overall}</span>
                <span style="color:#9ca3af; font-size:13px">{buy_votes}/6 frameworks agree</span>
            </div>
            <div style="color:#374151; font-size:16px; margin-top:6px; font-weight:500">{company}</div>
            <div style="color:#9ca3af; font-size:13px; margin-top:2px">{sector} · {industry}</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        c52 = "#16a34a" if pct_52h >= -5 else "#dc2626"
        st.markdown(f"""
        <div class="sa-metric-card" style="text-align:right">
            <div style="font-size:34px; font-weight:800; color:#111">${price:.2f}</div>
            <div style="color:{c52}; font-size:14px; font-weight:600">{pct_52h:+.1f}% from 52w high</div>
            <div style="color:#9ca3af; font-size:12px; margin-top:4px">52w: ${low52:.2f} – ${high52:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # ── CHART ────────────────────────────────────────────────────────────────────
    fig = sa_build_chart(hist_6m, hist_1y, ticker, entry, buy_zone_top, stop_loss, target20, ma50_val)
    st.plotly_chart(fig, use_container_width=True)

    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns(6)
    lc1.markdown(f"<span style='color:#3b82f6;font-size:13px'>— Entry ${entry:.2f}</span>", unsafe_allow_html=True)
    lc2.markdown(f"<span style='color:#06b6d4;font-size:13px'>— Buy Zone ${buy_zone_top:.2f}</span>", unsafe_allow_html=True)
    lc3.markdown(f"<span style='color:#ef4444;font-size:13px'>— Stop ${stop_loss:.2f}</span>", unsafe_allow_html=True)
    lc4.markdown(f"<span style='color:#22c55e;font-size:13px'>— Target ${target20:.2f}</span>", unsafe_allow_html=True)
    lc5.markdown(f"<span style='color:#a855f7;font-size:13px'>— 50d MA</span>", unsafe_allow_html=True)
    lc6.markdown(f"<span style='color:#f97316;font-size:13px'>··· 150d MA</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── TRADE PLAN + KEY METRICS + FUNDAMENTALS ──────────────────────────────────
    tp_col, km_col, fund_col = st.columns(3)

    with tp_col:
        st.markdown("#### Trade Plan")
        sa_tbl_row("Entry Price",     f"${entry:.2f}")
        sa_tbl_row("Buy Zone Max",    f"${buy_zone_top:.2f}")
        sa_tbl_row("Stop Loss (7%)",  f"${stop_loss:.2f}", "#dc2626")
        sa_tbl_row("Target +20%",     f"${target20:.2f}",  "#16a34a")
        sa_tbl_row("Target +25%",     f"${target25:.2f}",  "#16a34a")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sa_tbl_row("Risk per trade",  risk_pct)
        sa_tbl_row("Shares to Buy",   str(shares))
        sa_tbl_row("Position Size",   f"${actual_pos:,.2f}")
        sa_tbl_row("Risk/Reward",     f"{rr_ratio:.2f}:1",
                   "#16a34a" if rr_ratio >= 2 else "#dc2626")

    with km_col:
        st.markdown("#### Key Metrics")
        pc = lambda v: "#16a34a" if v > 0 else "#dc2626"
        sa_tbl_row("EPS Qtr Growth",    f"{eps_q_g:+.1f}%",  pc(eps_q_g))
        sa_tbl_row("Annual EPS Growth", f"{eps_a_g:+.1f}%",  pc(eps_a_g))
        sa_tbl_row("Revenue Growth",    f"{rev_g:+.1f}%",    pc(rev_g))
        sa_tbl_row("Relative Strength", str(rs_score),       "#16a34a" if rs_score >= 70 else "#dc2626")
        sa_tbl_row("Volume vs Avg",     f"{vol_vs_avg:.2f}x","#16a34a" if vol_vs_avg > 1.2 else "#6b7280")
        sa_tbl_row("Institutional %",   f"{inst_pct:.1f}%",  "#16a34a" if 30 < inst_pct < 85 else "#6b7280")
        sa_tbl_row("Float",             f"{float_sh/1e6:.1f}M" if float_sh else "N/A")
        sa_tbl_row("Avg Volume",        f"{avg_vol/1e6:.1f}M")

    with fund_col:
        st.markdown("#### Long-term Fundamentals")
        pe  = info.get("trailingPE") or 0
        fpe = info.get("forwardPE")  or 0
        peg = info.get("pegRatio")   or 0
        ps  = info.get("priceToSalesTrailingTwelveMonths") or 0
        pb  = info.get("priceToBook")       or 0
        roe = (info.get("returnOnEquity")   or 0) * 100
        roa = (info.get("returnOnAssets")   or 0) * 100
        gm  = (info.get("grossMargins")     or 0) * 100
        om  = (info.get("operatingMargins") or 0) * 100
        nm  = (info.get("profitMargins")    or 0) * 100
        de  = info.get("debtToEquity")      or 0
        fcf = info.get("freeCashflow")      or 0
        dy  = (info.get("dividendYield")    or 0) * 100

        sa_tbl_row("P/E (TTM)",      f"{pe:.1f}"  if pe  else "N/A", "#16a34a" if 0 < pe < 25 else None)
        sa_tbl_row("Forward P/E",    f"{fpe:.1f}" if fpe else "N/A", "#16a34a" if 0 < fpe < 20 else None)
        sa_tbl_row("PEG Ratio",      f"{peg:.2f}" if peg else "N/A",
                   "#16a34a" if 0 < peg < 1 else "#dc2626" if peg > 2 else None)
        sa_tbl_row("Price/Sales",    f"{ps:.1f}"  if ps  else "N/A")
        sa_tbl_row("Price/Book",     f"{pb:.1f}"  if pb  else "N/A")
        sa_tbl_row("ROE",            f"{roe:.1f}%",  "#16a34a" if roe > 15 else "#dc2626")
        sa_tbl_row("Gross Margin",   f"{gm:.1f}%",   "#16a34a" if gm > 40 else None)
        sa_tbl_row("Net Margin",     f"{nm:.1f}%",   "#16a34a" if nm > 10 else "#dc2626" if nm < 0 else None)
        sa_tbl_row("Debt/Equity",    f"{de:.0f}%"  if de else "N/A",
                   "#dc2626" if de > 150 else "#16a34a" if de < 50 else None)
        sa_tbl_row("Free Cash Flow", f"${fcf/1e9:.1f}B" if fcf > 1e9 else f"${fcf/1e6:.0f}M" if fcf else "N/A",
                   "#16a34a" if fcf > 0 else "#dc2626")
        sa_tbl_row("Dividend Yield", f"{dy:.2f}%" if dy else "—")

    st.markdown("---")

    # ── ADVANCED TECHNICAL CHART ─────────────────────────────────────────────────
    with st.expander("📈  Advanced Technical Chart — TradingView Style", expanded=True):
        tv_c1, tv_c2, tv_c3 = st.columns([2, 1, 1])
        with tv_c1:
            tv_period   = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"],
                                        index=3, key=f"tv_period_{ticker}")
        with tv_c2:
            _iv_opts = {"1mo":["1d","1wk"],"3mo":["1d","1wk"],"6mo":["1d","1wk"],
                        "1y":["1d","1wk"],"2y":["1d","1wk"],"5y":["1wk","1mo"]}
            tv_interval = st.selectbox("Interval", _iv_opts.get(tv_period, ["1d"]),
                                        key=f"tv_interval_{ticker}")
        with tv_c3:
            tv_sr_order = st.slider("S/R Sensitivity", 3, 15, 5, key=f"tv_sr_{ticker}")

        ov_c1, ov_c2, ov_c3 = st.columns(3)
        with ov_c1:
            st.markdown("**Overlays**")
            tv_emas  = st.checkbox("EMAs 9/21/50/200",    value=True,  key=f"tv_ema_{ticker}")
            tv_smas  = st.checkbox("SMAs 20/50/200",       value=False, key=f"tv_sma_{ticker}")
            tv_bb    = st.checkbox("Bollinger Bands",      value=True,  key=f"tv_bb_{ticker}")
            tv_vwap  = st.checkbox("VWAP",                 value=True,  key=f"tv_vwap_{ticker}")
        with ov_c2:
            st.markdown("**Advanced**")
            tv_st    = st.checkbox("Supertrend (3×ATR)",   value=True,  key=f"tv_st_{ticker}")
            tv_ichi  = st.checkbox("Ichimoku Cloud",        value=False, key=f"tv_ichi_{ticker}")
        with ov_c3:
            st.markdown("**Levels**")
            tv_sr    = st.checkbox("Support/Resistance",    value=True,  key=f"tv_sr2_{ticker}")
            tv_piv   = st.checkbox("Pivot Points",          value=True,  key=f"tv_piv_{ticker}")
            tv_fib   = st.checkbox("Fibonacci Retracement", value=False, key=f"tv_fib_{ticker}")

        with st.spinner("Building technical chart…"):
            tv_df = _tv_fetch(ticker, tv_period, tv_interval)

        if not tv_df.empty:
            tv_df = _tv_add_indicators(tv_df)

            res_raw, sup_raw = _tv_find_pivots(tv_df, order=tv_sr_order)
            res_levels = _tv_cluster(sorted(res_raw, reverse=True))
            sup_levels = _tv_cluster(sorted(sup_raw, reverse=True))
            pivot_vals = _tv_pivots(tv_df)
            fib_vals   = _tv_fibs(tv_df, lookback=min(100, len(tv_df)))

            tv_opts = dict(
                emas=tv_emas, smas=tv_smas, bb=tv_bb, vwap=tv_vwap,
                supertrend=tv_st, ichimoku=tv_ichi,
                sr=tv_sr, res=res_levels, sup=sup_levels,
                pivots=tv_piv, pivot_vals=pivot_vals,
                fib=tv_fib, fib_vals=fib_vals,
            )
            tv_fig = _tv_build_chart(tv_df, ticker, tv_opts)
            st.plotly_chart(tv_fig, use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": True,
                                    "modeBarButtonsToAdd": ["drawline","drawopenpath","eraseshape"],
                                    "toImageButtonOptions": {"format":"png","scale":2}})

            # ── Key levels table ─────────────────────────────────────────────
            cur_p = float(tv_df["Close"].iloc[-1])
            kl_c1, kl_c2, kl_c3, kl_c4 = st.columns(4)

            with kl_c1:
                st.markdown("**Resistance Levels**")
                for r in [r for r in res_levels if r > cur_p * 0.97][:6]:
                    d = (r - cur_p) / cur_p * 100
                    st.markdown(f"<div style='color:#f23645;font-size:13px;padding:2px 0'>"
                                f"<b>{r:,.2f}</b> <span style='color:#787b86'>+{d:.2f}%</span></div>",
                                unsafe_allow_html=True)

            with kl_c2:
                st.markdown("**Support Levels**")
                for s in [s for s in sup_levels if s < cur_p * 1.03][:6]:
                    d = (s - cur_p) / cur_p * 100
                    st.markdown(f"<div style='color:#089981;font-size:13px;padding:2px 0'>"
                                f"<b>{s:,.2f}</b> <span style='color:#787b86'>{d:.2f}%</span></div>",
                                unsafe_allow_html=True)

            with kl_c3:
                st.markdown("**Moving Averages**")
                for lbl, col in [("EMA 9","EMA9"),("EMA 21","EMA21"),("EMA 50","EMA50"),
                                  ("EMA 200","EMA200"),("SMA 20","SMA20"),("SMA 50","SMA50"),
                                  ("SMA 200","SMA200")]:
                    if col in tv_df.columns and not pd.isna(tv_df[col].iloc[-1]):
                        v = float(tv_df[col].iloc[-1]); above = cur_p > v
                        st.markdown(
                            f"<div style='font-size:12px;padding:2px 0;display:flex;"
                            f"justify-content:space-between'>"
                            f"<span style='color:#787b86'>{lbl}</span>"
                            f"<span style='color:{'#089981' if above else '#f23645'}'>"
                            f"{v:,.2f} {'▲' if above else '▼'}</span></div>",
                            unsafe_allow_html=True)

            with kl_c4:
                st.markdown("**Signal Summary**")
                tv_sum = _tv_signal_summary(tv_df)
                ov_lbl, ov_type = tv_sum["overall"]
                ov_col = "#089981" if ov_type == "bull" else "#f23645" if ov_type == "bear" else "#ffc107"
                st.markdown(f"<div style='background:#1e222d;border-radius:6px;padding:10px;"
                            f"text-align:center;margin-bottom:8px'>"
                            f"<div style='color:#787b86;font-size:11px'>OVERALL</div>"
                            f"<div style='color:{ov_col};font-weight:700;font-size:1.1rem'>{ov_lbl}</div>"
                            f"<div style='color:#787b86;font-size:10px'>Score: {tv_sum['score']:+.2f}</div>"
                            f"</div>", unsafe_allow_html=True)
                for ind, ((lbl, stype), detail) in list(tv_sum["sigs"].items())[:8]:
                    sc = "#089981" if stype == "bull" else "#f23645" if stype == "bear" else "#ffc107"
                    st.markdown(
                        f"<div style='font-size:11px;padding:2px 0;display:flex;"
                        f"justify-content:space-between'>"
                        f"<span style='color:#787b86'>{ind}</span>"
                        f"<span style='color:{sc}'>{lbl}</span></div>",
                        unsafe_allow_html=True)
        else:
            st.warning(f"Could not fetch technical data for {ticker} ({tv_period} / {tv_interval}).")

    st.markdown("---")

    # ── DCF VALUATION ─────────────────────────────────────────────────────────────
    st.markdown("### DCF Intrinsic Value")

    _dcf_probe = sa_calc_dcf(info, price, cashflow_df=cashflow_df)

    if _dcf_probe is None:
        st.warning(
            f"DCF requires positive Free Cash Flow. {ticker} currently has negative/zero FCF — "
            "DCF not applicable. Consider EV/Revenue or P/S multiples instead."
        )
    else:
        with st.expander("⚙️  Adjust DCF Assumptions", expanded=False):
            sl1, sl2, sl3 = st.columns(3)
            with sl1:
                user_g1 = st.slider("Growth Yr 1–5 (%)", 1, 80,
                                     value=int(_dcf_probe["default_g1"]),
                                     key=f"sa_dcf_g1_{ticker}",
                                     help="Avg annual FCF growth for years 1–5.")
            with sl2:
                user_g2 = st.slider("Growth Yr 6–10 (%)", 1, 40,
                                     value=int(_dcf_probe["default_g2"]),
                                     key=f"sa_dcf_g2_{ticker}",
                                     help="Avg annual FCF growth for years 6–10.")
            with sl3:
                user_wacc = st.slider("Discount Rate / WACC (%)", 7, 20,
                                       value=int(_dcf_probe["default_wacc"]),
                                       key=f"sa_dcf_wacc_{ticker}",
                                       help="Weighted average cost of capital.")
            method_labels = {
                "normalized": "Normalized (Op CF − D&A)",
                "3yr_avg":    "3-Year Average",
                "reported":   "Reported FCF",
                "estimated":  "Estimated",
            }
            st.caption(
                f"**FCF method:** {method_labels.get(_dcf_probe['fcf_method'], '')}  |  "
                f"Raw FCF/sh: **${_dcf_probe['raw_fcf_ps']:.2f}**  |  "
                f"Normalized FCF/sh: **${_dcf_probe['norm_fcf_ps']:.2f}**  |  "
                f"3yr-avg FCF/sh: **${_dcf_probe['avg3_fcf_ps']:.2f}**\n\n"
                f"**Growth source:** {_dcf_probe['growth_source']}  |  "
                f"Revenue growth: **{_dcf_probe['rev_growth']:+.1f}%**  |  Terminal: 2.5%"
            )
            if _dcf_probe.get("capex_ratio", 0) > 0.40:
                st.warning(
                    f"⚠️ **Heavy capex cycle detected** ({_dcf_probe['capex_ratio']*100:.0f}% of Op CF). "
                    f"{_dcf_probe['fcf_note']}"
                )

        dcf = sa_calc_dcf(info, price, cashflow_df=cashflow_df,
                          g1_pct=user_g1, g2_pct=user_g2, wacc_pct=user_wacc)

        upside_color    = "#16a34a" if dcf["upside"] > 10 else "#f59e0b" if dcf["upside"] > -10 else "#dc2626"
        valuation_label = ("UNDERVALUED"   if dcf["upside"] > 20 else
                           "FAIRLY VALUED" if dcf["upside"] > -10 else "OVERVALUED")
        val_bg     = "#f0fdf4" if dcf["upside"] > 20 else "#fffbeb" if dcf["upside"] > -10 else "#fef2f2"
        val_border = "#22c55e" if dcf["upside"] > 20 else "#f59e0b" if dcf["upside"] > -10 else "#ef4444"

        d1, d2, d3 = st.columns(3)
        with d1:
            imp_html = (
                f'<div style="font-size:12px; color:#6b7280; margin-top:8px">'
                f'Market implies <b>{dcf["implied_growth"]}%</b> growth priced in</div>'
                if dcf["implied_growth"] is not None else ""
            )
            st.markdown(f"""
            <div style="background:{val_bg}; border:2px solid {val_border}; border-radius:12px; padding:18px 20px; text-align:center">
                <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:1px">DCF Intrinsic Value (10-yr)</div>
                <div style="font-size:34px; font-weight:800; color:#111; margin:6px 0">${dcf['intrinsic']:.2f}</div>
                <div style="font-size:14px; font-weight:700; color:{upside_color}">{dcf['upside']:+.1f}% vs ${price:.2f}</div>
                <div style="font-size:13px; font-weight:700; color:{val_border}; margin-top:4px">{valuation_label}</div>
                {imp_html}
            </div>""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.06)">
                <div style="font-size:13px; font-weight:600; color:#374151; margin-bottom:10px">Margin of Safety</div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:13px">MoS 20% — fair entry</span>
                    <span style="font-weight:700;color:{'#16a34a' if price<=dcf['mos_20'] else '#dc2626'}">${dcf['mos_20']:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:13px">MoS 30% — great entry</span>
                    <span style="font-weight:700;color:{'#16a34a' if price<=dcf['mos_30'] else '#dc2626'}">${dcf['mos_30']:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:13px">Current Price</span>
                    <span style="font-weight:700">${price:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0">
                    <span style="color:#6b7280;font-size:13px">FCF/share (base)</span>
                    <span style="font-weight:600">${dcf['fcf_per_share']:.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with d3:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.06)">
                <div style="font-size:13px; font-weight:600; color:#374151; margin-bottom:10px">Active Assumptions</div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:12px">FCF base used</span>
                    <span style="font-weight:700;color:#1d4ed8;font-size:12px">${dcf['fcf_per_share']:.2f}/sh ({dcf['fcf_method']})</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:12px">Raw reported FCF/sh</span>
                    <span style="font-weight:600;font-size:12px">${dcf['raw_fcf_ps']:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:12px">Growth yr 1–5</span>
                    <span style="font-weight:700;color:#2563eb">{dcf['g1_5']:.0f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:12px">Growth yr 6–10</span>
                    <span style="font-weight:700;color:#7c3aed">{dcf['g6_10']:.0f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6">
                    <span style="color:#6b7280;font-size:12px">Terminal growth</span>
                    <span style="font-weight:600">2.5%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0">
                    <span style="color:#6b7280;font-size:12px">WACC</span>
                    <span style="font-weight:700;color:#dc2626">{dcf['wacc']:.0f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Sensitivity Analysis** — Intrinsic value across growth & discount rate scenarios")
        fcf_ps      = dcf["fcf_per_share"]
        total_debt  = info.get("totalDebt") or 0
        total_cash  = info.get("totalCash") or 0
        shares_out  = info.get("sharesOutstanding") or 1
        net_cash_ps = (total_cash - total_debt) / shares_out
        sens_html = sa_dcf_sensitivity_table(fcf_ps, net_cash_ps, user_g1, user_g2, user_wacc, price)
        st.markdown(f'<div style="overflow-x:auto">{sens_html}</div>', unsafe_allow_html=True)
        st.caption(
            "Green = >20% upside · Yellow = fairly valued (±10%) · Red = overvalued >10%  |  "
            "⚠️ DCF is sensitive to assumptions — use alongside technicals and other frameworks."
        )

    st.markdown("---")

    # ── HOLD TIME ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sa-hold-card">
        <div style="font-size:13px; opacity:0.7; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px">
            Recommended Hold Time
        </div>
        <div style="font-size:26px; font-weight:800; margin-bottom:8px">{hold_duration}</div>
        <div style="font-size:14px; opacity:0.9; line-height:1.6">{hold_advice}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIGNALS ──────────────────────────────────────────────────────────────────
    st.markdown("#### Signals & Warnings")
    tag_html = ""
    for text, kind in signals:
        cls = {"green": "sa-signal-tag-green", "amber": "sa-signal-tag-amber", "red": "sa-signal-tag-red"}.get(kind, "")
        tag_html += f'<span class="sa-signal-tag {cls}">{text}</span>'
    st.markdown(
        f'<div style="margin:6px 0 16px">'
        f'{tag_html or "<span class=sa-signal-tag>No strong signals</span>"}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── TRADER VERDICTS ──────────────────────────────────────────────────────────
    st.markdown("### Expert Trader Verdicts")

    tabs = st.tabs([
        "🏆 O'Neil (CAN SLIM)",
        "🚀 Minervini (SEPA)",
        "💎 Buffett (Value)",
        "📈 Lynch (GARP)",
        "⚡ Livermore (Momentum)",
        "📊 Weinstein (Stage)",
        "🌐 Dalio (Macro)",
    ])

    with tabs[0]:
        sa_score_bar(oneil_s, oneil_max)
        sa_verdict_card("William O'Neil — CAN SLIM", oneil_s, oneil_max, oneil_v, oneil_r,
                        "#f0fdf4", "#22c55e")
        st.markdown(f"""**Philosophy:** Growth stocks breaking out of tight bases with accelerating
        earnings, strong RS, and institutional sponsorship.

        **Hold rule:** Sell when stock drops 7–8% from buy point. Sell at +20–25% unless it's a fast
        mover (rises 20% in under 3 weeks — then hold 8 weeks).

        **{ticker} bottom line:** {
            f"Strong CAN SLIM setup — {oneil_r[0] if oneil_r else 'multiple criteria met'}." if "BUY" in oneil_v
            else "Does not fully meet CAN SLIM criteria. Key missing factors: "
                 + (oneil_r[0] if oneil_r else "earnings acceleration and/or base pattern") + "."
        }""")

    with tabs[1]:
        sa_score_bar(mini_s, mini_max)
        sa_verdict_card("Mark Minervini — SEPA Trend Template", mini_s, mini_max, mini_v, mini_r,
                        "#eff6ff", "#3b82f6")
        st.markdown(f"""**Philosophy:** Specific Entry Point Analysis — only buy stocks in Stage 2
        uptrend with price above all key MAs, 52w high proximity, and expanding RS.

        **Trend template:** Price > 50MA > 150MA > 200MA, 200MA trending up, within 25% of 52w high,
        30%+ above 52w low.

        **{ticker} bottom line:** {
            f"Trend template {('fully' if mini_s >= 11 else 'partially')} satisfied. "
            f"{mini_r[0] if mini_r else ''}." if "BUY" in mini_v
            else f"Trend template not satisfied. Missing: {', '.join(mini_r[:2]) if mini_r else 'multiple criteria'}."
        }""")

    with tabs[2]:
        sa_score_bar(buff_s, buff_max)
        sa_verdict_card("Warren Buffett — Moat & Value", buff_s, buff_max, buff_v, buff_r,
                        "#fffbeb", "#f59e0b")
        pe_v  = info.get("trailingPE")       or 0
        roe_v = (info.get("returnOnEquity")  or 0) * 100
        gm_v  = (info.get("grossMargins")    or 0) * 100
        de_v  = info.get("debtToEquity")     or 0
        st.markdown(f"""**Philosophy:** Buy wonderful companies at fair prices. Look for wide moats
        (high margins, consistent ROE), low debt, strong free cash flow, and a durable competitive advantage.

        **Key metrics:** P/E={pe_v:.1f} | ROE={roe_v:.1f}% | Gross Margin={gm_v:.1f}% | D/E={de_v:.0f}%

        **{ticker} bottom line:** {
            f"Strong business quality. {buff_r[0] if buff_r else ''}. Suitable for long-term hold." if "BUY" in buff_v
            else f"{'Fair price but quality concerns' if buff_s >= 6 else 'Does not meet Buffett quality screen'}. "
                 f"{'P/E too high for value play.' if pe_v > 35 else ''}"
        }""")

    with tabs[3]:
        sa_score_bar(lynch_s, lynch_max)
        sa_verdict_card("Peter Lynch — GARP", lynch_s, lynch_max, lynch_v, lynch_r,
                        "#f0fdf4", "#16a34a")
        peg_v = info.get("pegRatio") or 0
        st.markdown(f"""**Philosophy:** Growth at a Reasonable Price. PEG ratio < 1 is a gift.
        Invest in what you know — understand the business, love simple stories.

        **PEG Ratio:** {f"{peg_v:.2f}" if peg_v else "N/A"} (< 1.0 = undervalued for growth | 1–2 = fair | > 2 = expensive)

        **Lynch categories:** Fast Growers (20%+/yr EPS), Stalwarts (10–20%), Slow Growers (<10%),
        Cyclicals, Turnarounds, Asset Plays.

        **{ticker} bottom line:** {
            f"GARP criteria met. PEG of {peg_v:.2f} suggests growth not fully priced in." if peg_v and peg_v < 1
            else f"PEG={peg_v:.2f} — {'slightly expensive for a Lynch pick' if peg_v and peg_v > 2 else 'fair value range'}."
                 if peg_v else "PEG not available — evaluate manually."
        }""")

    with tabs[4]:
        sa_score_bar(liver_s, liver_max)
        sa_verdict_card("Jesse Livermore — Pivotal Point Momentum", liver_s, liver_max, liver_v, liver_r,
                        "#fdf4ff", "#a855f7")
        st.markdown(f"""**Philosophy:** Trade only the leading stocks in the leading sectors.
        Buy at pivotal points on high volume. Never average down. Let winners run.

        **Key rules:** Only buy breakouts on volume. Stocks near 52w highs are stronger, not more risky.
        Cut losses fast — the first loss is the smallest loss.

        **{ticker} bottom line:** {
            f"Pivotal point setup forming. {liver_r[0] if liver_r else ''}. Watch for volume confirmation." if liver_s >= 5
            else f"Not at a clear pivotal point. {liver_r[-1] if liver_r else 'Wait for better setup'}."
        }""")

    with tabs[5]:
        sa_score_bar(wein_s, wein_max)
        sa_verdict_card("Stan Weinstein — Stage Analysis", wein_s, wein_max, wein_v, wein_r,
                        "#f0f9ff", "#0ea5e9")
        st.markdown(f"""**Philosophy:** Every stock goes through 4 stages. Only buy in Stage 2 (advancing).
        Sell in Stage 3. Never hold Stage 4.

        **Stage guide:**
        - **Stage 1** — Basing: wait patiently
        - **Stage 2** — Advancing: BUY (price > rising 30-week MA)
        - **Stage 3** — Topping: start selling
        - **Stage 4** — Declining: short or avoid

        **{ticker} bottom line:** {wein_v}. {wein_r[0] if wein_r else ''}""")

    with tabs[6]:
        sa_score_bar(dalio_s, dalio_max)
        sa_verdict_card("Ray Dalio — All-Weather / Macro", dalio_s, dalio_max, dalio_v, dalio_r,
                        "#f8fafc", "#64748b")
        de_v = info.get("debtToEquity") or 0
        cr_v = info.get("currentRatio") or 0
        st.markdown(f"""**Philosophy:** Macro-aware, risk-balanced investing. Favor companies with
        low leverage, real earnings growth above inflation, and pricing power. Diversify across
        economic environments (inflationary, deflationary, growth, recession).

        **Debt/Equity:** {de_v:.0f}% | **Current Ratio:** {cr_v:.1f}x

        **{ticker} bottom line:** {
            f"Macro-resilient business. {dalio_r[0] if dalio_r else ''}." if "BUY" in dalio_v
            else f"{'High leverage is a risk in rising rate environment.' if de_v > 150 else 'Limited macro appeal currently.'}  "
                 f"{dalio_r[0] if dalio_r else ''}"
        }""")

    st.markdown("---")

    # ── SECTOR / THEME ANALYSIS ────────────────────────────────────────────────────
    st.markdown("### Sector & Theme Analysis")

    etf_ticker = SA_SECTOR_ETF.get(sector, "SPY")
    peers_list = [p for p in SA_SECTOR_PEERS.get(sector, SA_SECTOR_PEERS.get("Technology", []))
                  if p != ticker][:9]

    with st.spinner(f"Loading {sector} sector data ({etf_ticker})..."):
        sector_data = sa_get_sector_data(etf_ticker, peers_list, ticker)

    etf_hist  = sector_data["etf_hist"]
    peer_data = sector_data["peers"]

    sec_col1, sec_col2 = st.columns([2, 1])

    with sec_col1:
        st.plotly_chart(sa_build_sector_chart(etf_hist, etf_ticker), use_container_width=True)

        total_peers     = len(peer_data)
        above_ma_count  = sum(1 for v in peer_data.values() if v["above_ma50"])
        breadth_pct     = round(above_ma_count / total_peers * 100) if total_peers else 0
        breadth_color   = "#16a34a" if breadth_pct >= 60 else "#f59e0b" if breadth_pct >= 40 else "#dc2626"
        breadth_label   = "Healthy" if breadth_pct >= 60 else "Mixed" if breadth_pct >= 40 else "Weak"

        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:14px 18px; margin-top:8px; box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <span style="font-weight:600; color:#374151">Sector Breadth (% peers above 50-day MA)</span>
                <span style="font-weight:700; color:{breadth_color}; font-size:18px">{breadth_pct}% — {breadth_label}</span>
            </div>
            <div style="background:#f3f4f6; border-radius:4px; height:8px; margin-top:8px">
                <div style="width:{breadth_pct}%; background:{breadth_color}; height:8px; border-radius:4px"></div>
            </div>
            <div style="color:#9ca3af; font-size:12px; margin-top:6px">{above_ma_count} of {total_peers} tracked peers above 50-day MA</div>
        </div>""", unsafe_allow_html=True)

    with sec_col2:
        st.markdown(f"**{sector} Peers — 3-Month Performance**")
        if peer_data:
            sorted_peers = sorted(peer_data.items(), key=lambda x: x[1]["return_3m"], reverse=True)
            peer_rows = ""
            for sym, d in sorted_peers:
                ret    = d["return_3m"]
                col    = "#16a34a" if ret > 0 else "#dc2626"
                ma_dot = "🟢" if d["above_ma50"] else "🔴"
                is_current = "→ " if sym == ticker else ""
                peer_rows += f"""
                <div style="display:flex; justify-content:space-between; padding:5px 0;
                            border-bottom:1px solid #f3f4f6; font-size:13px">
                    <span style="color:#374151; font-weight:{'700' if sym==ticker else '400'}">{is_current}{sym} {ma_dot}</span>
                    <span style="color:{col}; font-weight:600">{ret:+.1f}%</span>
                </div>"""
            st.markdown(
                f'<div style="background:white; border-radius:10px; padding:14px 18px;'
                f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">{peer_rows}'
                f'<div style="color:#9ca3af; font-size:11px; margin-top:8px">🟢 Above 50d MA &nbsp; 🔴 Below 50d MA</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No peer data available.")

    # Sector narrative
    if not etf_hist.empty and peer_data:
        etf_ret_1m   = round((etf_hist["Close"].iloc[-1] / etf_hist["Close"].iloc[-21] - 1) * 100, 1) if len(etf_hist) > 21 else 0
        etf_ret_3m   = round((etf_hist["Close"].iloc[-1] / etf_hist["Close"].iloc[-63] - 1) * 100, 1) if len(etf_hist) > 63 else 0
        etf_ret_6m   = round((etf_hist["Close"].iloc[-1] / etf_hist["Close"].iloc[0]   - 1) * 100, 1)
        momentum     = "strong" if etf_ret_1m > 3 else "moderate" if etf_ret_1m > 0 else "weak"
        theme_health = "bullish" if breadth_pct >= 60 and etf_ret_3m > 0 else "mixed" if breadth_pct >= 40 else "bearish"

        sorted_peers = sorted(peer_data.items(), key=lambda x: x[1]["return_3m"], reverse=True)
        top_peer = sorted_peers[0][0] if peer_data else "N/A"
        top_ret  = sorted_peers[0][1]["return_3m"] if peer_data else 0

        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:16px 20px; margin-top:12px;
                    border-left:4px solid #3b82f6; box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="font-weight:700; color:#1e3a5f; margin-bottom:6px">Sector Theme Analysis — {sector}</div>
            <div style="color:#374151; line-height:1.8; font-size:14px">
                The <strong>{sector}</strong> sector (tracked via <strong>{etf_ticker}</strong>) is showing
                <strong>{momentum} momentum</strong> over the past month (<strong>{etf_ret_1m:+.1f}%</strong>),
                with a 3-month return of <strong>{etf_ret_3m:+.1f}%</strong> and
                6-month return of <strong>{etf_ret_6m:+.1f}%</strong>.
                Sector breadth is <strong>{breadth_label.lower()}</strong> — {breadth_pct}% of peers trade above
                their 50-day MA. The sector theme is overall <strong>{theme_health}</strong>.
                {f"Leading peer: <strong>{top_peer}</strong> (+{top_ret:.1f}% in 3 months)." if top_peer != "N/A" else ""}
                {ticker} {
                    "is outperforming its sector — a sign of relative strength within the theme." if rs_score >= 60
                    else "is underperforming its sector — consider waiting for relative strength to improve."
                }
            </div>
        </div>""", unsafe_allow_html=True)
