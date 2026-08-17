from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from nse_scanner.data import (
    fetch_recent_sessions,
    normalize_cash_frame,
    normalize_futures_oi,
)
from nse_scanner.engine import build_scanner

st.set_page_config(
    page_title="NSE OI + Chaikin Scanner",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 10px 14px;}
    .small-note {opacity:.72; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NSE F&O OI + Chaikin Money Flow Scanner")
st.caption(
    "Find NSE stock-futures underlyings where open interest is building persistently "
    "and cash-market money flow confirms accumulation."
)

with st.expander("How this scanner interprets the signal", expanded=False):
    st.markdown(
        """
- **OI universe:** NSE individual stock futures (`STF`) only. Cash-only NSE stocks do not have futures open interest.
- **3-day OI change:** latest total stock-futures OI versus three trading sessions earlier.
- **Rollover-safe OI:** OI is converted into underlying shares and aggregated across listed expiries.
- **CMF:** 20-session Chaikin Money Flow from cash-market high, low, close and volume.
- **Long buildup:** price ↑ + OI ↑. **Short buildup:** price ↓ + OI ↑.
- The **Score (0–100)** is only a ranking heuristic; it is not a probability of profit.
        """
    )

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    cash_sessions = fetch_recent_sessions("cm", sessions=26, candidate_count=36)
    fo_sessions = fetch_recent_sessions("fo", sessions=5, candidate_count=10)

    if len(cash_sessions) < 20:
        raise RuntimeError(f"Only {len(cash_sessions)} cash sessions were available; need at least 20.")
    if len(fo_sessions) < 4:
        raise RuntimeError(f"Only {len(fo_sessions)} F&O sessions were available; need at least 4.")

    cash = pd.concat(
        [normalize_cash_frame(d, df) for d, df in reversed(cash_sessions)],
        ignore_index=True,
    )
    fo = pd.concat(
        [normalize_futures_oi(d, df) for d, df in reversed(fo_sessions)],
        ignore_index=True,
    )
    scanner, cash_hist, oi_hist = build_scanner(cash, fo, cmf_window=20)
    return scanner, cash_hist, oi_hist, cash_sessions, fo_sessions


top_bar_a, top_bar_b = st.columns([5, 1])
with top_bar_b:
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Loading recent NSE cash and derivatives bhavcopies…"):
        scanner, cash_hist, oi_hist, cash_sessions, fo_sessions = load_data()
except Exception as exc:
    st.error(
        "Could not load enough NSE archive data. This can happen on a holiday, "
        "before the final bhavcopy is published, or when NSE temporarily blocks archive requests."
    )
    st.code(str(exc))
    st.stop()

latest_cash_date = max(d for d, _ in cash_sessions)
latest_fo_date = max(d for d, _ in fo_sessions)

m1, m2, m3, m4 = st.columns(4)
m1.metric("F&O stocks scanned", f"{scanner['symbol'].nunique():,}")
m2.metric("Latest F&O session", latest_fo_date.strftime("%d %b %Y"))
m3.metric("Latest cash session", latest_cash_date.strftime("%d %b %Y"))
m4.metric("Data mode", "NSE EOD")

st.subheader("Filters")

f1, f2, f3, f4 = st.columns(4)
with f1:
    min_oi = st.slider("Minimum 3-day OI increase", 0, 100, 15, 5, format="%d%%")
with f2:
    min_cmf = st.slider("Minimum CMF (20)", -0.25, 0.50, 0.05, 0.01)
with f3:
    min_turnover_cr = st.slider("Minimum avg cash traded value", 0, 500, 20, 10, help="20-session average, ₹ crore")
with f4:
    min_price, max_price = st.slider("Cash price range (₹)", 0, 10000, (0, 5000), 50)

g1, g2, g3, g4 = st.columns(4)
with g1:
    require_monotonic = st.toggle("OI up on all 3 steps", value=True)
with g2:
    require_cmf_rising = st.toggle("CMF rising vs 3 sessions ago", value=True)
with g3:
    bullish_only = st.toggle("Bullish / long buildup only", value=True)
with g4:
    min_vol_ratio = st.slider("Minimum volume / 20D average", 0.0, 3.0, 0.8, 0.1)

flt = scanner.copy()
flt = flt[flt["oi_3d_pct"] >= min_oi]
flt = flt[flt["cmf"] >= min_cmf]
flt = flt[(flt["close"] >= min_price) & (flt["close"] <= max_price)]
flt = flt[(flt["avg_traded_value_20"] / 1e7) >= min_turnover_cr]
flt = flt[flt["volume_ratio"] >= min_vol_ratio]

if require_monotonic:
    flt = flt[flt["oi_monotonic_3d"]]
if require_cmf_rising:
    flt = flt[flt["cmf_change_3d"] > 0]
if bullish_only:
    flt = flt[flt["buildup"].eq("Long buildup")]

flt = flt.sort_values(["score", "oi_3d_pct", "cmf"], ascending=False)

st.divider()

r1, r2, r3 = st.columns(3)
r1.metric("Matches", len(flt))
r2.metric("Median 3D OI change", f"{flt['oi_3d_pct'].median():.1f}%" if len(flt) else "—")
r3.metric("Median CMF", f"{flt['cmf'].median():.3f}" if len(flt) else "—")

if flt.empty:
    st.warning("No stocks match the current thresholds. Loosen OI, CMF, liquidity, or monotonic filters.")
    st.stop()

st.subheader("Top candidates")
top_n = st.slider("Show top N", 5, min(50, len(flt)), min(15, len(flt)))

show = flt.head(top_n).copy()
show["Avg Cash Value ₹Cr"] = show["avg_traded_value_20"] / 1e7
show["OI Latest (Mn shares)"] = show["oi_latest"] / 1e6

display = show[
    [
        "symbol", "score", "buildup", "close", "ret_3d_pct",
        "oi_3d_pct", "oi_1d_pct", "oi_positive_steps",
        "cmf", "cmf_change_3d", "volume_ratio",
        "Avg Cash Value ₹Cr", "OI Latest (Mn shares)"
    ]
].rename(
    columns={
        "symbol": "Symbol",
        "score": "Score",
        "buildup": "Setup",
        "close": "Price ₹",
        "ret_3d_pct": "Price 3D %",
        "oi_3d_pct": "OI 3D %",
        "oi_1d_pct": "OI 1D %",
        "oi_positive_steps": "OI + Steps",
        "cmf": "CMF 20",
        "cmf_change_3d": "CMF Δ3D",
        "volume_ratio": "Vol / 20D",
    }
)

st.dataframe(
    display.style.format(
        {
            "Score": "{:.1f}",
            "Price ₹": "{:.2f}",
            "Price 3D %": "{:.2f}",
            "OI 3D %": "{:.2f}",
            "OI 1D %": "{:.2f}",
            "CMF 20": "{:.3f}",
            "CMF Δ3D": "{:.3f}",
            "Vol / 20D": "{:.2f}x",
            "Avg Cash Value ₹Cr": "{:.1f}",
            "OI Latest (Mn shares)": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
    height=min(630, 38 * (len(display) + 1)),
)

csv = display.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv,
    file_name=f"nse_oi_cmf_scan_{latest_fo_date:%Y%m%d}.csv",
    mime="text/csv",
)

st.subheader("OI vs money-flow confirmation")
fig = px.scatter(
    show,
    x="oi_3d_pct",
    y="cmf",
    size="avg_traded_value_20",
    hover_name="symbol",
    hover_data={
        "score": True,
        "ret_3d_pct": ":.2f",
        "volume_ratio": ":.2f",
        "avg_traded_value_20": False,
    },
    labels={
        "oi_3d_pct": "3-day OI increase (%)",
        "cmf": "Chaikin Money Flow (20)",
        "score": "Score",
        "ret_3d_pct": "3-day price return (%)",
        "volume_ratio": "Volume / 20D",
    },
    height=500,
)
fig.add_hline(y=0, line_dash="dot")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Inspect a stock")
symbol = st.selectbox("Symbol", show["symbol"].tolist())

c = cash_hist[cash_hist["symbol"].eq(symbol)].sort_values("date").tail(26).copy()
o = oi_hist[oi_hist["symbol"].eq(symbol)].sort_values("date").copy()

left, right = st.columns(2)

with left:
    pfig = px.line(
        c,
        x="date",
        y=["close", "cmf_20"],
        markers=True,
        title=f"{symbol}: cash close and CMF",
    )
    st.plotly_chart(pfig, use_container_width=True)

with right:
    ofig = px.line(
        o,
        x="date",
        y="oi_shares",
        markers=True,
        title=f"{symbol}: aggregated stock-futures OI",
    )
    st.plotly_chart(ofig, use_container_width=True)

row = show[show["symbol"].eq(symbol)].iloc[0]
st.info(
    f"{symbol}: {row['buildup']} | 3D OI {row['oi_3d_pct']:.1f}% | "
    f"3D price {row['ret_3d_pct']:.1f}% | CMF {row['cmf']:.3f} | "
    f"Score {row['score']:.1f}/100"
)

st.caption(
    "Source: NSE final UDiFF cash-market and equity-derivatives bhavcopies. "
    "This scanner is for research, not a recommendation to buy or sell."
)
