import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional

try:
    from PIL import Image as PILImage
    _favicon = PILImage.open("static/icon.png")
except Exception:
    _favicon = "📈"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Dashboard — Alpha & Beta",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apple Touch Icon injection (iPhone "Add to Home Screen") ─────────────────
_ICON_URL = (
    "https://raw.githubusercontent.com/dksbluesky/"
    "dk-stock-alpha-beta/main/static/icon.png"
)
st.markdown(
    f"""
    <link rel="apple-touch-icon" sizes="512x512" href="{_ICON_URL}">
    <link rel="apple-touch-icon" sizes="180x180" href="{_ICON_URL}">
    <link rel="shortcut icon" type="image/png"   href="{_ICON_URL}">
    """,
    unsafe_allow_html=True,
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }
  [data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
  }
  .sec-hdr {
    font-size: 1.05rem; font-weight: 700; color: #1a3c5e;
    border-left: 4px solid #3b82f6;
    padding-left: 8px; margin: 1.1rem 0 0.5rem;
  }
  .formula-box {
    background: #f1f5fb; border-radius: 6px;
    padding: 10px 14px; font-size: 0.88rem;
    font-family: monospace; margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

BENCHMARK = "^TWII"
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _download_cached(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices — raises on failure so cache doesn't store None."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].rename(ticker)
    # Handle both tz-aware and tz-naive indices from yfinance
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    return close


def _download(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Wrapper — returns None on failure without caching the failure."""
    try:
        return _download_cached(ticker, start, end)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_0050_yield() -> tuple[float, float, float]:
    """Return (yield_rate, total_12m_dividends, current_price) for 0050.TW."""
    try:
        tk = yf.Ticker("0050.TW")
        divs = tk.dividends
        if divs.empty:
            return 0.017, 0.0, 0.0
        # Normalise timezone before comparison
        divs.index = divs.index.tz_convert(None) if divs.index.tz else divs.index
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
        total_div = float(divs[divs.index >= cutoff].sum())
        hist = tk.history(period="5d")
        current_price = float(hist["Close"].iloc[-1])
        yield_rate = total_div / current_price if current_price else 0.017
        return yield_rate, total_div, current_price
    except Exception:
        return 0.017, 0.0, 0.0


def build_aligned_df(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Strictly align all price series to common trading dates using pd.merge.
    Inner join automatically drops holidays and typhoon closure days that
    affect only some markets.
    """
    items = list(series_dict.values())
    if not items:
        return pd.DataFrame()
    merged = items[0].to_frame()
    for s in items[1:]:
        merged = pd.merge(
            merged, s.to_frame(),
            left_index=True, right_index=True,
            how="inner",
        )
    return merged.dropna()


def period_rf(annual_rf: float, freq: str) -> float:
    """Convert annual Rf to per-period rate (Daily or Weekly)."""
    divisor = 252 if freq == "D" else 52
    return (1 + annual_rf) ** (1 / divisor) - 1


def compute_metrics(
    asset_rets: pd.Series,
    bench_rets: pd.Series,
    rf: float,
) -> dict:
    """
    OLS regression of excess returns.
    Returns alpha (per period), beta, R², p-value, std_err,
    plus the excess-return series used for plotting.
    """
    exc_asset = asset_rets - rf
    exc_bench = bench_rets - rf
    slope, intercept, r, p, se = stats.linregress(
        exc_bench.values, exc_asset.values
    )
    return dict(
        beta=slope,
        alpha_period=intercept,
        r_squared=r ** 2,
        p_value=p,
        std_err=se,
        exc_asset=exc_asset,
        exc_bench=exc_bench,
    )


def annualise_alpha(alpha_period: float, freq: str) -> float:
    mult = 252 if freq == "D" else 52
    return ((1 + alpha_period) ** mult - 1) * 100


def compute_bias(price_series: pd.Series) -> pd.DataFrame:
    """Compute MA5, MA20, Bias_5, Bias_20 from daily close prices."""
    ma5  = price_series.rolling(window=5).mean()
    ma20 = price_series.rolling(window=20).mean()
    return pd.DataFrame({
        "Close":   price_series,
        "MA5":     ma5,
        "MA20":    ma20,
        "Bias_5":  (price_series - ma5)  / ma5  * 100,
        "Bias_20": (price_series - ma20) / ma20 * 100,
    })


def bias_signal(bias: float, window: int) -> str:
    hi = 3.0 if window == 5 else 5.0
    lo = -hi
    if bias > hi:   return "🔴 Overbought"
    if bias > hi/2: return "🟡 Slightly High"
    if bias < lo:   return "🟢 Oversold"
    if bias < lo/2: return "🟡 Slightly Low"
    return "⚪ Neutral"


# ── Plotly builders ───────────────────────────────────────────────────────────

def plot_cumulative(price_df: pd.DataFrame) -> go.Figure:
    cr = (price_df / price_df.iloc[0] - 1) * 100
    fig = go.Figure()
    for i, col in enumerate(cr.columns):
        is_bm = col == BENCHMARK
        fig.add_trace(go.Scatter(
            x=cr.index, y=cr[col], mode="lines", name=col,
            line=dict(
                color="#9E9E9E" if is_bm else COLORS[i % 4],
                width=2.5 if is_bm else 2,
                dash="dot" if is_bm else "solid",
            ),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#ccc")
    fig.update_layout(
        title="Cumulative Returns", xaxis_title="Date",
        yaxis_title="Return (%)", height=460,
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
    )
    return fig


def plot_scatter_grid(results: dict) -> go.Figure:
    tickers = list(results.keys())
    n = len(tickers)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=tickers,
        horizontal_spacing=0.13,
        vertical_spacing=0.20,
    )
    for idx, tk in enumerate(tickers):
        row, col = divmod(idx, 2)
        res = results[tk]
        x = res["exc_bench"].values * 100
        y = res["exc_asset"].values * 100
        c = COLORS[idx % 4]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=c, size=5, opacity=0.55),
            name=tk, showlegend=True,
        ), row=row + 1, col=col + 1)
        xfit = np.linspace(x.min(), x.max(), 200)
        yfit = res["alpha_period"] * 100 + res["beta"] * xfit
        fig.add_trace(go.Scatter(
            x=xfit, y=yfit, mode="lines",
            line=dict(color=c, width=2.5),
            name=f"{tk} fit", showlegend=False,
        ), row=row + 1, col=col + 1)
        fig.update_xaxes(
            title_text=f"{BENCHMARK} excess ret (%)", row=row + 1, col=col + 1
        )
        fig.update_yaxes(
            title_text=f"{tk} excess ret (%)", row=row + 1, col=col + 1
        )
    fig.update_layout(
        height=420 * nrows,
        title_text=f"Excess Return Scatter — vs {BENCHMARK}",
        template="plotly_white",
    )
    return fig


def plot_single_scatter(tk: str, res: dict, color: str) -> go.Figure:
    x = res["exc_bench"].values * 100
    y = res["exc_asset"].values * 100
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = res["alpha_period"] * 100 + res["beta"] * xfit
    fig = go.Figure([
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=color, size=6, opacity=0.6),
            name="Observations",
            hovertemplate=(
                f"{BENCHMARK} excess: %{{x:.2f}}%<br>"
                f"{tk} excess: %{{y:.2f}}%<extra></extra>"
            ),
        ),
        go.Scatter(
            x=xfit, y=yfit, mode="lines",
            line=dict(color="crimson", width=2.5), name="Fit",
        ),
    ])
    fig.add_hline(y=0, line_dash="dot", line_color="#bbb", opacity=0.6)
    fig.add_vline(x=0, line_dash="dot", line_color="#bbb", opacity=0.6)
    fig.update_layout(
        xaxis_title=f"{BENCHMARK} Excess Return (%)",
        yaxis_title=f"{tk} Excess Return (%)",
        height=370, template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_bias_chart(bias_dict: dict, window: int) -> go.Figure:
    key = f"Bias_{window}"
    hi  = 3.0 if window == 5 else 5.0
    fig = go.Figure()
    for i, (tk, df_b) in enumerate(bias_dict.items()):
        series = df_b[key].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series,
            mode="lines", name=tk,
            line=dict(color=COLORS[i % 4], width=2),
        ))
    fig.add_hline(y=hi,  line_dash="dash", line_color="#ef4444",
                  opacity=0.6, annotation_text=f"Overbought +{hi}%",
                  annotation_position="top left")
    fig.add_hline(y=-hi, line_dash="dash", line_color="#22c55e",
                  opacity=0.6, annotation_text=f"Oversold -{hi}%",
                  annotation_position="bottom left")
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", opacity=0.4)
    fig.update_layout(
        title=f"MA{window} BIAS 乖離率 (%)",
        xaxis_title="Date", yaxis_title="BIAS (%)",
        height=360, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
    )
    return fig


def plot_corr_heatmap(ret_df: pd.DataFrame) -> go.Figure:
    cm = ret_df.corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=cm.values,
        x=cm.columns.tolist(),
        y=cm.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=cm.values,
        texttemplate="%{text:.3f}",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title="Return Correlation Heatmap",
        height=380, template="plotly_white",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def build_sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️  Controls")

        # ── Tickers ──
        st.subheader("Tickers  (up to 4)")
        defaults = ["", "", "", ""]
        tickers = []
        for i in range(4):
            val = st.text_input(f"Ticker {i + 1}", value=defaults[i], key=f"tk{i}")
            if val.strip():
                tickers.append(val.strip().upper())

        st.caption(f"Benchmark (fixed): **{BENCHMARK}**")

        # ── Timeframe ──
        st.subheader("Analysis Period")
        tf_choice = st.selectbox(
            "Timeframe",
            ["Short Term (≤ 1Y)", "Medium Term (1 – 2.5Y)", "Long Term (> 2.5Y)"],
        )
        today = datetime.today()
        day_map = {"Short": 365, "Medium": int(365 * 1.75), "Long": int(365 * 3)}
        tf_key = next(k for k in day_map if k in tf_choice)
        start_dt = today - timedelta(days=day_map[tf_key])
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = today.strftime("%Y-%m-%d")
        st.caption(f"{start_str}  →  {end_str}")

        # ── Frequency ──
        st.subheader("Return Frequency")
        default_freq_idx = 0 if tf_key == "Short" else 1
        freq_label = st.radio(
            "Frequency", ["Daily", "Weekly"], index=default_freq_idx
        )
        freq = "D" if freq_label == "Daily" else "W"

        # ── Risk-free rate ──
        st.subheader("Risk-Free Rate  (Rₓ)")
        rf_option = st.radio(
            "Baseline",
            ["Option A: Fixed Rate", "Option B: 0050 Dividend Yield"],
        )
        if "Option A" in rf_option:
            rf_pct = st.slider("Annual Rf (%)", 0.0, 10.0, 1.7, 0.1)
            rf_annual = rf_pct / 100
            rf_label  = f"Fixed {rf_pct:.1f}%"
            rf_meta   = {}
        else:
            with st.spinner("Fetching 0050 dividend data …"):
                yield_rate, total_div, price = fetch_0050_yield()
            rf_annual = yield_rate
            rf_label  = f"0050 Yield {yield_rate * 100:.2f}%"
            rf_meta   = {"total_div": total_div, "price": price}
            if price:
                st.success(
                    f"12-month Dividends: **NT${total_div:.2f}**  "
                    f"|  Price: **NT${price:.2f}**  "
                    f"→  Yield: **{yield_rate * 100:.2f}%**"
                )
            else:
                st.warning("Could not fetch 0050 data — defaulting to 1.70%")

        st.divider()
        run = st.button("🚀  Run Analysis", type="primary", use_container_width=True)

    return dict(
        tickers=tickers,
        start=start_str,
        end=end_str,
        timeframe=tf_choice,
        freq=freq,
        freq_label=freq_label,
        rf_annual=rf_annual,
        rf_label=rf_label,
        rf_meta=rf_meta,
        run=run,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown("## 📈  Financial Dashboard — Alpha & Beta Analysis")
    st.caption(
        "Jensen's Alpha  ·  CAPM Beta  ·  R²  ·  Benchmark: ^TWII  "
        "|  Taiwan market focus"
    )
    st.divider()

    cfg = build_sidebar()

    # Landing screen
    if not cfg["run"] and "state" not in st.session_state:
        st.info("Configure your tickers in the sidebar, then click **Run Analysis**.")
        with st.expander("ℹ️  How it works"):
            st.markdown("""
**Formulas**

```
β  =  Cov(Rₚ − Rf,  Rm − Rf)  /  Var(Rm − Rf)
α  =  intercept of OLS regression of excess returns
Jensen's α (annualised)  =  (1 + α_period)^freq − 1
```

**Date alignment** — `pd.merge` (inner join on index) automatically drops
Taiwan Stock Exchange holidays and typhoon closure days so that all assets
share exactly the same trading dates as the TAIEX benchmark.

**Risk-Free Rate conversion**
```
Rf_period  =  (1 + Rf_annual) ^ (1 / freq)  −  1
freq  =  252  (daily)  |  52  (weekly)
```
            """)
        return

    # ── Run & cache ──
    if cfg["run"]:
        if not cfg["tickers"]:
            st.error("Enter at least one ticker in the sidebar.")
            return

        all_tickers = cfg["tickers"] + [BENCHMARK]
        with st.spinner(f"Downloading data for {', '.join(all_tickers)} …"):
            raw = {tk: _download(tk, cfg["start"], cfg["end"]) for tk in all_tickers}

        failed = [tk for tk, s in raw.items() if s is None]
        if failed:
            st.error(
                f"⚠️ Failed to fetch data for: **{', '.join(failed)}**  \n"
                f"These tickers are excluded. Check the symbol or try again."
            )
        raw = {k: v for k, v in raw.items() if v is not None}

        if BENCHMARK not in raw:
            st.error(f"Benchmark {BENCHMARK} unavailable. Check your connection.")
            return

        # Strictly align dates with pd.merge
        price_df = build_aligned_df(raw)
        if price_df.empty or len(price_df) < 10:
            st.error("Insufficient aligned data — check ticker symbols.")
            return

        # Compute returns
        if cfg["freq"] == "W":
            ret_df = price_df.resample("W-FRI").last().pct_change().dropna()
        else:
            ret_df = price_df.pct_change().dropna()

        rf      = period_rf(cfg["rf_annual"], cfg["freq"])
        f_mult  = 252 if cfg["freq"] == "D" else 52

        asset_tks = [t for t in cfg["tickers"] if t in ret_df.columns]
        results = {}
        for tk in asset_tks:
            res = compute_metrics(ret_df[tk], ret_df[BENCHMARK], rf)
            res["alpha_annual_pct"] = annualise_alpha(res["alpha_period"], cfg["freq"])
            results[tk] = res

        bias_dict = {tk: compute_bias(price_df[tk]) for tk in asset_tks}

        st.session_state["state"] = dict(
            price_df=price_df,
            ret_df=ret_df,
            results=results,
            asset_tks=asset_tks,
            cfg=cfg,
            f_mult=f_mult,
            bias_dict=bias_dict,
        )

    # ── Load from session ──
    s         = st.session_state["state"]
    price_df  = s["price_df"]
    ret_df    = s["ret_df"]
    results   = s["results"]
    asset_tks = s["asset_tks"]
    cfg       = s["cfg"]
    f_mult    = s["f_mult"]
    bias_dict = s.get("bias_dict", {})

    if not asset_tks:
        st.warning("No valid asset tickers found in the aligned data.")
        return

    # ────────────────────────────────────────────────────────────────────
    # TABS
    # ────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📊  Dashboard Summary", "📉  Regression Analysis", "📋  Performance Data"]
    )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — Dashboard Summary
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        # Context bar
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Period:** {cfg['timeframe']}")
        c2.info(f"**Frequency:** {cfg['freq_label']}")
        c3.info(f"**Rf:** {cfg['rf_label']}")

        # ── Compact KPI table (mobile-friendly) ──
        st.markdown('<div class="sec-hdr">Key Performance Indicators</div>',
                    unsafe_allow_html=True)
        kpi_rows = []
        for tk in asset_tks:
            res = results[tk]
            a, b, r2 = res["alpha_annual_pct"], res["beta"], res["r_squared"]
            kpi_rows.append({
                "Ticker":       tk,
                "Ann. Alpha":   f"{a:+.2f}%",
                "Beta (β)":     f"{b:.4f}",
                "R²":           f"{r2:.4f}",
                "Signal":       "✅ Outperforms" if a >= 0 else "❌ Underperforms",
                "Volatility":   "Aggressive" if abs(b) > 1 else "Defensive",
            })
        st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

        # ── Per-ticker detail (collapsed by default — tap to expand) ──
        st.markdown('<div class="sec-hdr">Ticker Detail</div>', unsafe_allow_html=True)
        for i, tk in enumerate(asset_tks):
            res = results[tk]
            a, b, r2 = res["alpha_annual_pct"], res["beta"], res["r_squared"]
            with st.expander(f"{tk}  |  α {a:+.2f}%  ·  β {b:.4f}  ·  R² {r2:.4f}",
                             expanded=False):
                m1, m2, m3 = st.columns(3)
                m1.metric("Ann. Alpha (α)", f"{a:+.2f}%",
                          delta="Outperforms" if a >= 0 else "Underperforms",
                          delta_color="normal" if a >= 0 else "inverse")
                m2.metric("Beta (β)", f"{b:.4f}",
                          delta="Aggressive" if abs(b) > 1 else "Defensive",
                          delta_color="off")
                m3.metric("R²", f"{r2:.4f}",
                          delta=f"{r2*100:.1f}% from {BENCHMARK}",
                          delta_color="off")

        # ── BIAS 乖離率 ──
        st.markdown('<div class="sec-hdr">BIAS 乖離率 — Moving Average Deviation</div>',
                    unsafe_allow_html=True)
        bias_rows = []
        for tk in asset_tks:
            df_b = bias_dict.get(tk, pd.DataFrame()).dropna()
            if df_b.empty:
                continue
            b5  = df_b["Bias_5"].iloc[-1]
            b20 = df_b["Bias_20"].iloc[-1]
            bias_rows.append({
                "Ticker":       tk,
                "Bias_5 (%)":  f"{b5:+.2f}%",
                "5D Signal":   bias_signal(b5,  window=5),
                "Bias_20 (%)": f"{b20:+.2f}%",
                "20D Signal":  bias_signal(b20, window=20),
            })
        if bias_rows:
            st.dataframe(pd.DataFrame(bias_rows), use_container_width=True, hide_index=True)
            st.caption("Thresholds — Bias_5: ±3%  |  Bias_20: ±5%  (Taiwan market convention)")
            with st.expander("📊  BIAS Chart  (tap to expand)", expanded=False):
                b_tab5, b_tab20 = st.tabs(["5-Day BIAS", "20-Day BIAS"])
                with b_tab5:
                    st.plotly_chart(plot_bias_chart(bias_dict, window=5),
                                    use_container_width=True)
                with b_tab20:
                    st.plotly_chart(plot_bias_chart(bias_dict, window=20),
                                    use_container_width=True)

        # ── Cumulative return chart (collapsed by default) ──
        with st.expander("📈  Cumulative Returns  (tap to expand)", expanded=False):
            st.plotly_chart(plot_cumulative(price_df), use_container_width=True)

        # ── Formula reminder ──
        with st.expander("📐  Formula reference"):
            st.markdown("""
<div class="formula-box">
β  =  Cov(Rₚ − Rf, Rm − Rf)  /  Var(Rm − Rf)<br>
α  =  intercept  [OLS regression of excess returns]<br>
Jensen's α (ann.)  =  (1 + α_period)^freq − 1<br>
R²  =  fraction of asset variance explained by benchmark<br><br>
Rf_period  =  (1 + Rf_annual)^(1/freq) − 1
</div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — Regression Analysis
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            f'<div class="sec-hdr">Excess-Return Scatter  '
            f"(Rf = {cfg['rf_label']})</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"X-axis: {BENCHMARK} excess return  |  "
            f"Y-axis: Asset excess return  |  "
            f"Regression line shows CAPM fit"
        )
        st.plotly_chart(plot_scatter_grid(results), use_container_width=True)

        # ── Summary table ──
        st.markdown('<div class="sec-hdr">Regression Summary</div>',
                    unsafe_allow_html=True)
        rows = []
        for tk in asset_tks:
            res = results[tk]
            rows.append({
                "Ticker":          tk,
                "α (per period)":  f"{res['alpha_period'] * 100:.5f}%",
                "α (annualised)":  f"{res['alpha_annual_pct']:+.2f}%",
                "β (Beta)":        f"{res['beta']:.4f}",
                "R²":              f"{res['r_squared']:.4f}",
                "P-value":         f"{res['p_value']:.6f}",
                "Std Error":       f"{res['std_err']:.7f}",
                "N obs":           len(res["exc_asset"]),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Per-asset detail ──
        st.markdown('<div class="sec-hdr">Per-Asset Detail</div>',
                    unsafe_allow_html=True)
        for i, tk in enumerate(asset_tks):
            res = results[tk]
            a   = res["alpha_annual_pct"]
            b   = res["beta"]
            r2  = res["r_squared"]
            with st.expander(f"🔍  {tk}   |   α = {a:+.2f}%   β = {b:.4f}   R² = {r2:.4f}",
                             expanded=(i == 0)):
                left, right = st.columns([3, 2])
                with left:
                    st.plotly_chart(
                        plot_single_scatter(tk, res, COLORS[i % 4]),
                        use_container_width=True,
                    )
                with right:
                    # Interpretation
                    if b > 1.2:    b_txt = f"🔴 Aggressive — {b:.2f}× market sensitivity"
                    elif b > 0.8:  b_txt = f"🟡 Market-like — {b:.2f}× market sensitivity"
                    elif b >= 0:   b_txt = f"🟢 Defensive — {b:.2f}× market sensitivity"
                    else:          b_txt = f"🔵 Inverse — {b:.2f}× market sensitivity"

                    a_txt  = (f"✅ Outperforms by **{a:.2f}%/yr**"
                              if a >= 0 else f"❌ Underperforms by **{abs(a):.2f}%/yr**")
                    r2_cat = ("Strong" if r2 > 0.7 else
                              "Moderate" if r2 > 0.4 else "Weak")
                    r2_txt = f"{r2_cat} — {r2 * 100:.1f}% of variance from {BENCHMARK}"

                    st.markdown(f"""
| Metric | Value |
|--------|-------|
| **α per period** | `{res['alpha_period'] * 100:.5f}%` |
| **Annualised α** | `{a:+.2f}%` |
| **β (Beta)** | `{b:.4f}` |
| **R²** | `{r2:.4f}` |
| **P-value** | `{res['p_value']:.6f}` |
| **Std Error** | `{res['std_err']:.7f}` |
| **Observations** | `{len(res['exc_asset'])}` |
""")
                    st.markdown("---")
                    st.write(f"**β:** {b_txt}")
                    st.write(f"**α:** {a_txt}")
                    st.write(f"**R²:** {r2_txt}")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3 — Performance Data
    # ═══════════════════════════════════════════════════════════════════
    with tab3:
        col_p, col_r = st.columns(2)
        with col_p:
            st.markdown('<div class="sec-hdr">Adjusted Close Prices</div>',
                        unsafe_allow_html=True)
            st.dataframe(price_df.round(2), use_container_width=True, height=320)
        with col_r:
            st.markdown(
                f'<div class="sec-hdr">{cfg["freq_label"]} Returns (%)</div>',
                unsafe_allow_html=True,
            )
            st.dataframe((ret_df * 100).round(4), use_container_width=True, height=320)

        # ── Descriptive statistics ──
        st.markdown('<div class="sec-hdr">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        stat_rows = []
        for tk in ret_df.columns:
            s = ret_df[tk].dropna()
            ann_ret  = ((1 + s.mean()) ** f_mult - 1) * 100
            ann_vol  = s.std() * np.sqrt(f_mult) * 100
            sharpe   = ((ann_ret - cfg["rf_annual"] * 100) / ann_vol
                        if ann_vol else 0.0)
            cum      = (1 + s).cumprod()
            mdd      = (cum / cum.cummax() - 1).min() * 100
            stat_rows.append({
                "Ticker":          tk,
                "Ann. Return":     f"{ann_ret:.2f}%",
                "Ann. Volatility": f"{ann_vol:.2f}%",
                "Sharpe Ratio":    f"{sharpe:.3f}",
                "Max Drawdown":    f"{mdd:.2f}%",
                "Skewness":        f"{s.skew():.3f}",
                "Kurtosis":        f"{s.kurtosis():.3f}",
                "N obs":           len(s),
            })
        st.dataframe(
            pd.DataFrame(stat_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Correlation heatmap ──
        st.markdown('<div class="sec-hdr">Correlation Matrix</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_corr_heatmap(ret_df), use_container_width=True)

        # ── Downloads ──
        st.markdown('<div class="sec-hdr">Export</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "📥  Download Prices (CSV)",
            price_df.to_csv().encode("utf-8"),
            "prices.csv",
            "text/csv",
        )
        dl2.download_button(
            "📥  Download Returns (CSV)",
            ret_df.to_csv().encode("utf-8"),
            "returns.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
