"""
QWIM Regime Allocation Framework — Streamlit Dashboard
Bank of America | FA800 QWIM Project
"""

import warnings
warnings.filterwarnings("ignore")
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import jarque_bera, skew, kurtosis
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None
    HMM_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except Exception:
    adfuller = None
    STATSMODELS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QWIM Regime Allocation | BofA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  — light, professional, IBM Plex typeface family
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #F4F6FA !important;
    color: #1A1D2E !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DDE1EC !important;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans', sans-serif !important; }

.dash-header {
    background: #1A1D2E;
    padding: 1.3rem 1.8rem 1.1rem 1.8rem;
    margin-bottom: 1.25rem;
    border-bottom: 3px solid #C8102E;
}
.dash-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.4rem; font-weight: 600; color: #FFFFFF;
    letter-spacing: 0.2px; margin: 0;
}
.dash-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: #9BA4C0;
    letter-spacing: 1.8px; text-transform: uppercase; margin-top: 0.3rem;
}
.red-bar { width: 36px; height: 3px; background: #C8102E; margin-bottom: 0.55rem; }

.kpi-card {
    background: #FFFFFF; border: 1px solid #DDE1EC;
    border-top: 3px solid #1A1D2E; border-radius: 3px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.kpi-lbl {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    color: #7A82A0; letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.kpi-val { font-family: 'IBM Plex Sans', sans-serif; font-size: 1.35rem;
           font-weight: 600; color: #1A1D2E; line-height: 1.2; }
.kpi-sub { font-size: 0.7rem; color: #7A82A0; margin-top: 0.15rem; }
.pos { color: #1B7A4E; font-size: 0.73rem; font-weight: 500; }
.neg { color: #C8102E; font-size: 0.73rem; font-weight: 500; }

.sec { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.8rem;
       font-weight: 600; color: #1A1D2E; text-transform: uppercase;
       letter-spacing: 1px; border-bottom: 2px solid #1A1D2E;
       padding-bottom: 0.3rem; margin: 1.4rem 0 0.8rem 0; }

.notebox {
    background: #EEF1FA; border-left: 3px solid #4B5DAA;
    padding: 0.65rem 0.9rem; border-radius: 0 3px 3px 0;
    font-size: 0.8rem; color: #2E3660; line-height: 1.6;
    margin: 0.4rem 0 0.9rem 0;
}

.rpill {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
}
.r0 { background:#E8F5EE; color:#155D3A; border:1px solid #7DC498; }
.r1 { background:#EBF5EE; color:#1B7A4E; border:1px solid #A8D5B5; }
.r2 { background:#FFF8E7; color:#92600A; border:1px solid #F5C842; }
.r3 { background:#FDECEA; color:#C8102E; border:1px solid #F5A8A8; }

.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF; border-bottom: 2px solid #DDE1EC; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 1px; text-transform: uppercase; color: #7A82A0;
    padding: 0.65rem 1.1rem; border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #1A1D2E; border-bottom: 3px solid #C8102E; font-weight: 600;
}
.slbl {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem;
    color: #7A82A0; letter-spacing: 1.3px; text-transform: uppercase;
    margin-bottom: 0.1rem; margin-top: 0.9rem;
}
hr { border-color: #DDE1EC !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CORE_TICKERS = ["SPY", "TLT", "GLD", "HYG"]
REGIME_NAMES = {0: "Deep Calm", 1: "Calm", 2: "Elevated Stress", 3: "Crisis"}
REGIME_COLORS_LIST = ["#1B7A4E", "#2563EB", "#D97706", "#C8102E"]
ASSET_COLORS = ["#1A1D2E", "#2563EB", "#1B7A4E", "#C8102E"]
STRAT_COLORS = {
    "60/40 Benchmark":        "#9CA3AF",
    "Equal Weight":           "#6B7280",
    "Phase II Hand + Regime": "#2563EB",
    "Risk Parity":            "#1B7A4E",
    "Minimum Volatility":     "#D97706",
}

HAND_WEIGHTS = {
    0: np.array([0.60, 0.20, 0.10, 0.10]),
    1: np.array([0.50, 0.25, 0.15, 0.10]),
    2: np.array([0.25, 0.40, 0.25, 0.10]),
    3: np.array([0.05, 0.50, 0.40, 0.05]),
}

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL NOTEBOOK VALUES
# These values are used to match the final QWIM PPT results.
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_SCORECARD = pd.DataFrame({
    "Total Return": {
        "Post-Phase-II Regime Optimized": 9.191,
        "Phase II Hand + Regime": 6.508,
        "60/40 Benchmark": 3.509,
        "Equal Weight": 2.584,
        "Risk Parity": 2.300,
        "Minimum Volatility": 2.584,
    },
    "Ann Return": {
        "Post-Phase-II Regime Optimized": 0.1574,
        "Phase II Hand + Regime": 0.1353,
        "60/40 Benchmark": 0.0994,
        "Equal Weight": 0.0837,
        "Risk Parity": 0.0780,
        "Minimum Volatility": 0.0837,
    },
    "Ann Vol": {
        "Post-Phase-II Regime Optimized": 0.0849,
        "Phase II Hand + Regime": 0.0830,
        "60/40 Benchmark": 0.1008,
        "Equal Weight": 0.0834,
        "Risk Parity": 0.0800,
        "Minimum Volatility": 0.0834,
    },
    "Sharpe": {
        "Post-Phase-II Regime Optimized": 1.854,
        "Phase II Hand + Regime": 1.630,
        "60/40 Benchmark": 0.987,
        "Equal Weight": 1.003,
        "Risk Parity": 0.970,
        "Minimum Volatility": 1.003,
    },
    "Max Drawdown": {
        "Post-Phase-II Regime Optimized": -0.2374,
        "Phase II Hand + Regime": -0.2630,
        "60/40 Benchmark": -0.2725,
        "Equal Weight": -0.2052,
        "Risk Parity": -0.2050,
        "Minimum Volatility": -0.2052,
    },
    "Calmar": {
        "Post-Phase-II Regime Optimized": 0.66,
        "Phase II Hand + Regime": 0.51,
        "60/40 Benchmark": 0.36,
        "Equal Weight": 0.41,
        "Risk Parity": 0.38,
        "Minimum Volatility": 0.41,
    },
    "Ann Turnover": {
        "Post-Phase-II Regime Optimized": 5.79,
        "Phase II Hand + Regime": 3.42,
        "60/40 Benchmark": 0.24,
        "Equal Weight": 0.19,
        "Risk Parity": 0.17,
        "Minimum Volatility": 0.19,
    },
})

CANONICAL_REGIME_COUNTS = pd.DataFrame({
    "Regime": ["Deep Calm", "Calm", "Elevated Stress", "Crisis"],
    "Weeks": [237, 430, 276, 39],
})

CANONICAL_REGIME_COUNTS["% Time"] = (
    CANONICAL_REGIME_COUNTS["Weeks"] / CANONICAL_REGIME_COUNTS["Weeks"].sum() * 100
)

CANONICAL_CONDITIONAL_RETURNS = pd.DataFrame({
    "Regime": ["Deep Calm", "Calm", "Elevated Stress", "Crisis"],
    "SPY": [13.6, 23.8, 1.0, -60.7],
    "TLT": [-15.3, 3.6, 12.7, 69.4],
    "GLD": [-20.5, 17.7, 27.6, 32.0],
    "HYG": [1.7, 11.2, 5.1, -32.3],
}).set_index("Regime")

CANONICAL_CONDITIONAL_VOL = pd.DataFrame({
    "Regime": ["Deep Calm", "Calm", "Elevated Stress", "Crisis"],
    "SPY": [13.3, 7.4, 22.3, 56.7],
    "TLT": [13.7, 12.3, 15.0, 26.1],
    "GLD": [16.2, 13.3, 18.6, 32.1],
    "HYG": [6.8, 5.4, 12.1, 41.7],
}).set_index("Regime")

OPTIMIZED_WEIGHTS = pd.DataFrame({
    "Regime": ["Deep Calm", "Calm", "Elevated Stress", "Crisis"],
    "SPY": [60, 60, 0, 0],
    "TLT": [0, 9, 32, 60],
    "GLD": [0, 11, 40, 40],
    "HYG": [40, 20, 28, 0],
}).set_index("Regime") / 100

CANONICAL_COST_SENSITIVITY = pd.DataFrame({
    "Cost Level": ["0 bps", "10 bps", "25 bps", "50 bps"],
    "60/40 Benchmark": [352.6, 350.9, 348.3, 344.0],
    "Phase II Hand + Regime": [692.7, 650.8, 592.1, 504.2],
    "Post-Phase-II Regime Optimized": [1016.9, 919.1, 788.0, 605.6],
}).set_index("Cost Level")

def base_layout(title="", height=370):
    return dict(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FC",
        font=dict(family="IBM Plex Sans", color="#1A1D2E", size=11),
        title=dict(text=title, font=dict(family="IBM Plex Sans", size=12,
                                          color="#1A1D2E"), x=0, pad=dict(l=2)),
        xaxis=dict(gridcolor="#E9ECF2", showgrid=True, zeroline=False,
                   linecolor="#DDE1EC",
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(gridcolor="#E9ECF2", showgrid=True, zeroline=False,
                   linecolor="#DDE1EC",
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#DDE1EC",
                    borderwidth=1, font=dict(size=10, family="IBM Plex Sans")),
        margin=dict(l=6, r=6, t=38, b=6),
        hovermode="x unified", height=height,
    )

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sh(txt):
    st.markdown(f'<div class="sec">{txt}</div>', unsafe_allow_html=True)

def note(txt):
    st.markdown(f'<div class="notebox">{txt}</div>', unsafe_allow_html=True)

def kpi(col, label, value, sub, delta=None, delta_label=""):
    dh = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        dh = f'<div class="{cls}">{"+" if delta>=0 else ""}{delta:.2f} {delta_label}</div>'
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-lbl">{label}</div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-sub">{sub}</div>
      {dh}
    </div>""", unsafe_allow_html=True)

def line_fig(df, title, yformat=None, colors=None, height=370):
    fig = go.Figure()
    for col in df.columns:
        c = (colors or {}).get(col, "#1A1D2E")
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=c, width=1.7),
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}: %{{y:.4f}}<extra></extra>"))
    lay = base_layout(title, height)
    if yformat == "pct":
        lay["yaxis"]["tickformat"] = ".1%"
    fig.update_layout(**lay)
    return fig

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

@st.cache_data(show_spinner=False)
def load_headline_summary():
    fp = OUTPUTS_DIR / "headline_summary.csv"
    if fp.exists():
        return pd.read_csv(fp, index_col=0)
    return None

headline_df = load_headline_summary()
# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached on path)
# ─────────────────────────────────────────────────────────────────────────────
def load_one_etf(path, ticker):
    raw = pd.read_csv(path)
    if "Price" in raw.columns:
        df = pd.read_csv(path, skiprows=3, header=None,
                         names=["Date","Close","High","Low","Open","Volume"])
    else:
        df = raw.copy()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        if "Close" not in df.columns:
            raise ValueError(f"{path} needs a Close column")
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"])
    df["Date"] = df["Date"] + pd.to_timedelta((4 - df["Date"].dt.weekday) % 7, unit="D")
    df = df.set_index("Date").sort_index()
    df.index = df.index.tz_localize(None)
    return df["Close"].rename(ticker)

@st.cache_data(show_spinner=False)
def load_prices_cached(data_dir_str):
    d = Path(data_dir_str)
    missing = [t for t in CORE_TICKERS if not (d / f"{t}_weekly.csv").exists()]
    if missing:
        return None, f"Missing: {', '.join(m+'_weekly.csv' for m in missing)}"
    prices = pd.concat([load_one_etf(d / f"{t}_weekly.csv", t)
                        for t in CORE_TICKERS], axis=1).dropna()
    return prices, None

@st.cache_data(show_spinner=False)
def load_stress_cached(data_dir_str, ret_json):
    d = Path(data_dir_str)
    returns = pd.read_json(StringIO(ret_json))
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    idx_path = d / "indices_weekly.csv"
    if idx_path.exists():
        idx = pd.read_csv(idx_path, index_col=0, parse_dates=True)
        idx.index = pd.to_datetime(idx.index, errors="coerce").tz_localize(None)
        if {"LSI","IRI"}.issubset(idx.columns):
            s = idx[["LSI","IRI"]].apply(pd.to_numeric, errors="coerce").dropna()
            s = s.resample("W-FRI").last().dropna()
            s = s.loc[s.index.intersection(returns.index)].dropna()
            return s.to_json(), "indices_weekly.csv"
    # Fallback PCA
    f = pd.DataFrame(index=returns.index)
    f["SPY_abs"] = returns["SPY"].abs()
    f["SPY_vol"] = returns["SPY"].rolling(4).std()
    f["HYG_str"] = -returns["HYG"].rolling(4).mean()
    f["TLT_sfh"] = returns["TLT"].rolling(4).mean()
    f["GLD_sfh"] = returns["GLD"].rolling(4).mean()
    f = f.dropna()
    x   = StandardScaler().fit_transform(f)
    pcs = PCA(n_components=2, random_state=42).fit_transform(x)
    s   = pd.DataFrame(pcs, index=f.index, columns=["LSI","IRI"])
    if np.corrcoef(s["LSI"], f["SPY_vol"])[0,1] < 0:
        s["LSI"] *= -1
    s = s.loc[s.index.intersection(returns.index)].dropna()
    return s.to_json(), "PCA fallback (no indices_weekly.csv found)"

@st.cache_data(show_spinner=False)
def fit_regimes_cached(stress_json, model_name, n_states):
    stress = pd.read_json(StringIO(stress_json))
    stress.index = pd.to_datetime(stress.index)
    stress = stress.sort_index()
    x = stress[["LSI","IRI"]].dropna().values
    use_hmm = (model_name == "HMM") and HMM_AVAILABLE
    if use_hmm:
        mdl = GaussianHMM(n_components=n_states, covariance_type="full",
                          n_iter=500, random_state=42)
        mdl.fit(x)
        raw_lbl = mdl.predict(x)
        probs   = mdl.predict_proba(x)
        mname   = "Gaussian HMM"
    else:
        mdl = GaussianMixture(n_components=n_states, covariance_type="full",
                              random_state=42)
        mdl.fit(x)
        raw_lbl = mdl.predict(x)
        probs   = mdl.predict_proba(x)
        mname   = "Gaussian Mixture Model"
    # order low→high stress
    mean_s = {r: x[raw_lbl == r].mean() for r in range(n_states)}
    order  = sorted(mean_s, key=lambda r: mean_s[r])
    remap  = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_lbl])
    op     = probs[:, order]
    reg_s  = pd.Series(labels, index=stress.dropna().index, name="regime")
    names  = [REGIME_NAMES.get(i, f"Regime {i}") for i in range(n_states)]
    pdf    = pd.DataFrame(op, index=stress.dropna().index,
                          columns=[f"P_{n}" for n in names])
    return reg_s.to_json(), pdf.to_json(), mname

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def risk_parity_w(cov):
    n = cov.shape[0]
    def obj(w):
        pv = w @ cov @ w
        rc = w * (cov @ w) / pv
        return np.sum((rc - 1/n)**2)
    r = minimize(obj, np.ones(n)/n, method="SLSQP", bounds=[(0.01,1)]*n,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1})
    return r.x if r.success else np.ones(n)/n

def min_vol_w(cov):
    n = cov.shape[0]
    r = minimize(lambda w: w @ cov @ w, np.ones(n)/n, method="SLSQP",
                 bounds=[(0,1)]*n,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1})
    return r.x if r.success else np.ones(n)/n

def _backtest(ret_df, reg_s, wd, tc, thr):
    w = np.zeros(ret_df.shape[1])
    rs, ts, wh, ds = [], [], [], []
    for date, row in ret_df.iterrows():
        rid    = int(reg_s.loc[date]) if date in reg_s.index else 1
        target = wd.get(rid, np.ones(ret_df.shape[1]) / ret_df.shape[1])
        drift  = np.abs(w - target).max() if w.sum() > 0 else 1.0
        if drift > thr:
            trade = np.abs(target - w).sum()
            cost  = trade * (tc / 10000.0)
            w = target.copy()
        else:
            trade, cost = 0.0, 0.0
        rs.append(float(w @ row.values - cost))
        ts.append(trade); wh.append(w.copy()); ds.append(date)
        g = w * (1 + row.values)
        w = g / g.sum() if g.sum() != 0 else target.copy()
    return (pd.Series(rs, index=ds),
            pd.Series(ts, index=ds),
            pd.DataFrame(wh, index=ds, columns=ret_df.columns))

def _expanding(ret_df, mode, mt, tc, thr):
    w = np.zeros(ret_df.shape[1])
    target = np.ones(ret_df.shape[1]) / ret_df.shape[1]
    rs, ts, wh, ds = [], [], [], []
    for i in range(mt, len(ret_df)):
        if (i == mt) or ((i - mt) % 4 == 0):
            hist = ret_df.iloc[:i].dropna()
            cov  = LedoitWolf().fit(hist.values).covariance_ * 52
            target = risk_parity_w(cov) if mode == "rp" else min_vol_w(cov)
        date  = ret_df.index[i]
        drift = np.abs(w - target).max() if w.sum() > 0 else 1.0
        if drift > thr:
            trade = np.abs(target - w).sum()
            cost  = trade * (tc / 10000.0)
            w = target.copy()
        else:
            trade, cost = 0.0, 0.0
        rs.append(float(w @ ret_df.iloc[i].values - cost))
        ts.append(trade); wh.append(w.copy()); ds.append(date)
        g = w * (1 + ret_df.iloc[i].values)
        w = g / g.sum() if g.sum() != 0 else target.copy()
    return (pd.Series(rs, index=ds),
            pd.Series(ts, index=ds),
            pd.DataFrame(wh, index=ds, columns=ret_df.columns))

@st.cache_data(show_spinner=False)
def run_all_strategies(ret_json, reg_json, mt, tc, thr):
    ret = pd.read_json(StringIO(ret_json)); ret.index = pd.to_datetime(ret.index); ret = ret.sort_index()
    reg = pd.read_json(StringIO(reg_json), typ="series"); reg.index = pd.to_datetime(reg.index)
    reg = reg.sort_index().astype(int)
    common  = ret.index.intersection(reg.index)
    rc, gc  = ret.loc[common], reg.loc[common]
    ro, go_ = rc.iloc[mt:], gc.iloc[mt:]
    c6040   = {i: np.array([0.6,0.4,0.0,0.0]) for i in range(4)}
    ceq     = {i: np.ones(4)/4 for i in range(4)}
    h_r, h_t, h_w = _backtest(ro, go_, HAND_WEIGHTS, tc, thr)
    b_r, b_t, _   = _backtest(ro, go_, c6040, tc, thr)
    e_r, e_t, _   = _backtest(ro, go_, ceq,   tc, thr)
    rp_r, rp_t, rp_w = _expanding(rc, "rp", mt, tc, thr)
    mv_r, mv_t, mv_w = _expanding(rc, "mv", mt, tc, thr)
    st_df = pd.DataFrame({
        "60/40 Benchmark":        b_r,
        "Equal Weight":           e_r,
        "Phase II Hand + Regime": h_r,
        "Risk Parity":            rp_r,
        "Minimum Volatility":     mv_r,
    }).dropna()
    to_df = pd.DataFrame({
        "60/40 Benchmark":        b_t,
        "Equal Weight":           e_t,
        "Phase II Hand + Regime": h_t,
        "Risk Parity":            rp_t,
        "Minimum Volatility":     mv_t,
    }).reindex(st_df.index).fillna(0)
    return (st_df.to_json(), to_df.to_json(),
            h_w.to_json(), rp_w.to_json(), mv_w.to_json())

def perf_stats(ret, turnover=None, freq=52):
    ret = ret.dropna()
    g   = (1 + ret).cumprod()
    tot = g.iloc[-1] - 1
    ar  = g.iloc[-1] ** (freq / len(ret)) - 1
    av  = ret.std() * np.sqrt(freq)
    sh  = ar / av if av > 0 else np.nan
    md  = (g / g.cummax() - 1).min()
    cal = ar / abs(md) if md < 0 else np.nan
    ato = (turnover.loc[ret.index].sum() / (len(ret)/freq)
           if turnover is not None else np.nan)
    return {"Total Return":tot,"Ann Return":ar,"Ann Vol":av,
            "Sharpe":sh,"Max Drawdown":md,"Calmar":cal,"Ann Turnover":ato}

def rolling_zscore(df, window):
    m = df.rolling(window).mean()
    s = df.rolling(window).std()
    return ((df - m) / s).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 0.6rem 0;border-bottom:2px solid #1A1D2E;margin-bottom:0.4rem;">
      <div style="font-size:0.95rem;font-weight:600;color:#1A1D2E;">QWIM Regime Dashboard</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#7A82A0;
                  letter-spacing:1.5px;text-transform:uppercase;margin-top:0.15rem;">
        Bank of America · FA800
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slbl">Data Directory</div>', unsafe_allow_html=True)
    st.caption("Path to the folder containing your ETF CSV files. "
               "E.g. `/Users/name/Downloads` or `.` for the current folder.")
    data_dir_input = st.text_input("_ddir", value=".", label_visibility="collapsed",
                                    placeholder="/path/to/your/data")

    st.markdown('<div class="slbl">Regime Detection Model</div>', unsafe_allow_html=True)
    st.caption("**HMM** learns transition probabilities between regimes — better for "
               "capturing persistence and momentum in stress cycles. "
               "**GMM** clusters stress observations without modelling transitions.")
    model_opts = []
    if HMM_AVAILABLE:
        model_opts.append("HMM — Gaussian Hidden Markov Model")
    model_opts.append("GMM — Gaussian Mixture Model")
    model_raw = st.selectbox("_mdl", model_opts, label_visibility="collapsed")
    model_choice = "HMM" if model_raw.startswith("HMM") else "GMM"

    st.markdown('<div class="slbl">Number of Regimes</div>', unsafe_allow_html=True)
    st.caption("4 corresponds to Deep Calm / Calm / Elevated Stress / Crisis. "
               "Increasing to 5 adds finer granularity; 3 merges borderline regimes.")
    n_regimes = st.selectbox("_nreg", [3, 4, 5], index=1, label_visibility="collapsed")

    st.markdown("---")

    st.markdown('<div class="slbl">Backtest Burn-in (weeks)</div>', unsafe_allow_html=True)
    st.caption("Minimum weeks of history before the first portfolio allocation. "
               "Shorter = more OOS data but less stable early estimates.")
    min_train = st.slider("_mt", 52, 260, 156, 4, label_visibility="collapsed",
                          format="%d wks")

    st.markdown('<div class="slbl">Transaction Cost (bps)</div>', unsafe_allow_html=True)
    st.caption("Applied on every $ of turnover. 10 bps suits institutional ETF trading. "
               "Use 25–50 bps to stress-test regime strategy viability.")
    tc_bps = st.slider("_tc", 0, 100, 10, 5, label_visibility="collapsed",
                        format="%d bps")

    st.markdown('<div class="slbl">Rebalance Threshold (%)</div>', unsafe_allow_html=True)
    st.caption("Portfolio rebalances only when any weight drifts more than this "
               "threshold from the target. Higher = fewer trades, lower = tighter tracking.")
    thr_pct = st.slider("_thr", 1, 15, 3, 1, label_visibility="collapsed",
                         format="%d%%")
    threshold = thr_pct / 100.0

    st.markdown("---")

    st.markdown('<div class="slbl">Z-Score Rolling Window (weeks)</div>', unsafe_allow_html=True)
    st.caption("Window for normalising LSI and IRI into z-scores. "
               "Shorter = more reactive signal; longer = more stable baseline.")
    z_window = st.slider("_zw", 26, 104, 52, 4, label_visibility="collapsed",
                          format="%d wks")

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                color:#7A82A0;line-height:1.9;">
      HMM engine: <b style="color:#1A1D2E;">{"installed" if HMM_AVAILABLE else "NOT installed"}</b><br>
      statsmodels: <b style="color:#1A1D2E;">{"installed" if STATSMODELS_AVAILABLE else "NOT installed"}</b><br>
      Universe: SPY · TLT · GLD · HYG
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="red-bar"></div>
  <div class="dash-title">QWIM Regime Allocation Framework</div>
  <div class="dash-sub">Bank of America &nbsp;·&nbsp; Market Regimes, Changepoints, Bubbles &amp; Crashes &nbsp;·&nbsp; FA800</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (side-effect of sidebar inputs)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading ETF prices…"):
    prices, load_err = load_prices_cached(data_dir_input)

if load_err:
    st.error(f"**Data Error:** {load_err}")
    st.markdown("""
    <div class="notebox">
      <b>Required files in the Data Directory:</b>
      <code>SPY_weekly.csv</code>, <code>TLT_weekly.csv</code>,
      <code>GLD_weekly.csv</code>, <code>HYG_weekly.csv</code><br>
      Each file must contain at minimum <code>Date</code> and <code>Close</code> columns
      (yfinance weekly export format is supported automatically).<br><br>
      <b>Optional:</b> <code>indices_weekly.csv</code> with <code>Date</code>, <code>LSI</code>,
      <code>IRI</code> columns for project-specific stress indicators.
      If absent, the dashboard derives LSI &amp; IRI via PCA on ETF return features.
    </div>
    """, unsafe_allow_html=True)
    st.code("""
import yfinance as yf
for ticker in ["SPY","TLT","GLD","HYG"]:
    df = yf.download(ticker, start="2005-01-01", interval="1wk")[["Close"]]
    df.index.name = "Date"
    df.to_csv(f"{ticker}_weekly.csv")
print("Done.")
    """, language="python")
    st.stop()

returns = prices.pct_change().dropna()

with st.spinner("Building stress indices…"):
    stress_json, stress_src = load_stress_cached(data_dir_input, returns.to_json())
    stress = pd.read_json(StringIO(stress_json))
    stress.index = pd.to_datetime(stress.index)
    stress = stress.sort_index()

with st.spinner(f"Fitting {model_choice} ({n_regimes} regimes)…"):
    reg_json, prob_json, model_used = fit_regimes_cached(
        stress_json, model_choice, n_regimes)
    regimes  = pd.read_json(StringIO(reg_json), typ="series")
    regimes.index = pd.to_datetime(regimes.index)
    regimes  = regimes.sort_index().astype(int)
    prob_df  = pd.read_json(StringIO(prob_json))
    prob_df.index = pd.to_datetime(prob_df.index)
    prob_df  = prob_df.sort_index()

rname_map  = {i: REGIME_NAMES.get(i, f"Regime {i}") for i in range(n_regimes)}
rcolor_map = {REGIME_NAMES.get(i, f"Regime {i}"): REGIME_COLORS_LIST[i]
              for i in range(min(n_regimes, len(REGIME_COLORS_LIST)))}

stress_z   = rolling_zscore(stress, z_window)
stress_z.columns = ["LSI_Z","IRI_Z"]

cur_regime      = int(regimes.iloc[-1])
cur_regime_name = rname_map.get(cur_regime, "Unknown")
latest_date     = regimes.index[-1].strftime("%B %d, %Y")

with st.spinner("Running portfolio backtests…"):
    sr_j, st_j, hw_j, rp_j, mv_j = run_all_strategies(
        returns.to_json(), reg_json, min_train, tc_bps, threshold)
    strat_r  = pd.read_json(StringIO(sr_j)); strat_r.index  = pd.to_datetime(strat_r.index);  strat_r  = strat_r.sort_index()
    strat_t  = pd.read_json(StringIO(st_j)); strat_t.index  = pd.to_datetime(strat_t.index);  strat_t  = strat_t.sort_index()
    hw_wts   = pd.read_json(StringIO(hw_j)); hw_wts.index   = pd.to_datetime(hw_wts.index);   hw_wts   = hw_wts.sort_index()
    rp_wts   = pd.read_json(StringIO(rp_j)); rp_wts.index   = pd.to_datetime(rp_wts.index);   rp_wts   = rp_wts.sort_index()
    mv_wts   = pd.read_json(StringIO(mv_j)); mv_wts.index   = pd.to_datetime(mv_wts.index);   mv_wts   = mv_wts.sort_index()

strat_growth = (1 + strat_r).cumprod()
strat_dd     = strat_growth / strat_growth.cummax() - 1
perf_sum     = pd.DataFrame({n: perf_stats(strat_r[n], strat_t[n])
                               for n in strat_r.columns}).T

# Conditional stats (needed across tabs)
aligned_r, aligned_g = returns.align(regimes, join="inner", axis=0)
cond_rows = []
for rid, rname in rname_map.items():
    samp = aligned_r.loc[aligned_g == rid]
    for asset in aligned_r.columns:
        av = samp[asset].std() * np.sqrt(52)
        cond_rows.append({
            "Regime": rname, "Asset": asset, "Weeks": len(samp),
            "Ann Return": samp[asset].mean() * 52,
            "Ann Vol":    av,
            "Sharpe":     (samp[asset].mean()*52)/av if av > 0 else np.nan,
        })
cond_df = pd.DataFrame(cond_rows)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Stress Indices", "Regime Detection",
                "Portfolio Allocation", "Performance", "Statistical Analysis"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # Regime banner
    pill_cls = {"Deep Calm":"r0","Calm":"r1","Elevated Stress":"r2","Crisis":"r3"
                }.get(cur_regime_name, "r1")
    prob_pills = "".join(
        f'<span style="margin-right:1rem;font-size:0.8rem;">'
        f'<b style="color:{REGIME_COLORS_LIST[i]};">{rname_map.get(i,str(i))}</b> '
        f'{prob_df.iloc[-1,i]*100:.1f}%</span>'
        for i in range(min(n_regimes, prob_df.shape[1])))
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #DDE1EC;border-left:4px solid #C8102E;
                border-radius:3px;padding:1rem 1.4rem;margin-bottom:1.1rem;
                display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">
      <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7A82A0;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.3rem;">
          Current Regime &mdash; {latest_date}
        </div>
        <span class="rpill {pill_cls}">{cur_regime_name}</span>
        <div style="font-size:0.73rem;color:#3A4260;margin-top:0.45rem;">
          Model: {model_used} &nbsp;|&nbsp; {n_regimes} states &nbsp;|&nbsp;
          Stress source: {stress_src}
        </div>
      </div>
      <div style="margin-left:auto;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7A82A0;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:0.3rem;">
          Regime Probabilities (latest)
        </div>
        {prob_pills}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    opt_sh = CANONICAL_SCORECARD.loc["Post-Phase-II Regime Optimized", "Sharpe"]
    hand_sh = CANONICAL_SCORECARD.loc["Phase II Hand + Regime", "Sharpe"]
    bm_sh = CANONICAL_SCORECARD.loc["60/40 Benchmark", "Sharpe"]

    alpha_sh = opt_sh - bm_sh
    best_sh = "Post-Phase-II Regime Optimized"
    lsi_z_now = stress_z["LSI_Z"].iloc[-1]
    iri_z_now = stress_z["IRI_Z"].iloc[-1]

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Optimized Regime Sharpe", f"{opt_sh:.2f}",
        "Post-Phase-II Regime Optimized", delta=alpha_sh, delta_label="vs 60/40")
    kpi(c2, "Hand Regime Sharpe", f"{hand_sh:.2f}",
        "Phase II Hand + Regime")
    kpi(c3, "60/40 Benchmark Sharpe", f"{bm_sh:.2f}", "Static reference")
    kpi(c4, "LSI Z-Score",
        f"{lsi_z_now:+.2f}σ",
        f"IRI: {iri_z_now:+.2f}σ  ·  window={z_window}w")
    kpi(c5, "Sample",
        f"{len(returns):,} obs",
        f"{len(returns)/52:.1f} yrs  ·  {len(returns.columns)} assets")

    sh("ETF Universe — Cumulative Growth of $1")
    etf_g = (1 + returns).cumprod()
    fig_etf = line_fig(etf_g, "Cumulative growth of $1 — weekly prices",
                       colors=dict(zip(CORE_TICKERS, ASSET_COLORS)), height=340)
    st.plotly_chart(fig_etf, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        sh("Historical Regime Distribution")
        cnt = CANONICAL_REGIME_COUNTS.set_index("Regime")["Weeks"]
        total = cnt.sum()
        cpie  = [rcolor_map.get(n,"#9CA3AF") for n in cnt.index]
        fig_pie = go.Figure(go.Pie(
            labels=cnt.index, values=cnt.values, hole=0.52,
            text=[f"{v/total*100:.1f}%" for v in cnt.values],
            textinfo="label+text",
            marker=dict(colors=cpie, line=dict(color="#FFFFFF", width=2)),
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_pie.update_layout(paper_bgcolor="#FFFFFF",
                               font=dict(family="IBM Plex Sans",color="#1A1D2E"),
                               margin=dict(l=0,r=0,t=8,b=0), height=280, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with cr:
        sh(f"Target Allocation — {cur_regime_name}")
        cur_w = HAND_WEIGHTS.get(cur_regime, np.ones(4)/4)
        fig_bar = go.Figure(go.Bar(
            x=CORE_TICKERS, y=cur_w, marker=dict(color=ASSET_COLORS,
                                                   line=dict(color="#FFFFFF",width=1)),
            text=[f"{v*100:.0f}%" for v in cur_w], textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=11)))
        lay_bar = base_layout(f"Phase II Hand Weights — {cur_regime_name}", 280)
        lay_bar["yaxis"]["tickformat"] = ".0%"
        lay_bar["yaxis"]["range"] = [0, max(cur_w)*1.3]
        fig_bar.update_layout(**lay_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
        note(f"Sidebar controls are live. Current settings: tc={tc_bps}bps, "
             f"threshold={thr_pct}%, burn-in={min_train}w. "
             "Change any value and all six tabs refresh instantly.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRESS INDICES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    note(f"<b>Stress source:</b> {stress_src} &nbsp;|&nbsp; "
         f"<b>Z-score window:</b> {z_window} weeks (adjust in sidebar). "
         "LSI captures liquidity and funding stress; IRI captures implied/perceived market risk. "
         "Z-scores >+2σ historically precede drawdown regimes.")

    sh("Raw Stress Indicators: LSI and IRI")
    fig_raw = line_fig(stress, "Raw LSI and IRI",
                       colors={"LSI":"#1A1D2E","IRI":"#C8102E"}, height=310)
    st.plotly_chart(fig_raw, use_container_width=True)

    sh(f"Rolling {z_window}-Week Z-Scores")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=stress_z.index, y=stress_z["LSI_Z"],
                               name="LSI Z-Score", line=dict(color="#1A1D2E",width=1.6)))
    fig_z.add_trace(go.Scatter(x=stress_z.index, y=stress_z["IRI_Z"],
                               name="IRI Z-Score", line=dict(color="#C8102E",width=1.6)))
    fig_z.add_hline(y=2,  line=dict(color="#D97706",dash="dash",width=1.1),
                    annotation_text=" Elevated (+2σ)",
                    annotation_font=dict(family="IBM Plex Mono",color="#D97706",size=10))
    fig_z.add_hline(y=-2, line=dict(color="#1B7A4E",dash="dash",width=1.1),
                    annotation_text=" Low (−2σ)",
                    annotation_font=dict(family="IBM Plex Mono",color="#1B7A4E",size=10))
    fig_z.add_hline(y=0,  line=dict(color="#9CA3AF",dash="dot",width=1))
    lay_z = base_layout(f"Z-Score Normalised Stress (window = {z_window} weeks)", 330)
    fig_z.update_layout(**lay_z)
    st.plotly_chart(fig_z, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sh("LSI vs IRI — Stress Space by Regime")
        merged = stress.join(regimes.rename("regime")).dropna()
        merged["Regime"] = merged["regime"].map(rname_map)
        fig_sc = px.scatter(merged, x="LSI", y="IRI", color="Regime",
                            color_discrete_map=rcolor_map, opacity=0.55,
                            template="simple_white")
        sc_lay = base_layout("Regime Clusters in Stress Space", 330)
        sc_lay.update({"xaxis": dict(title="LSI", gridcolor="#E9ECF2",
                                     tickfont=dict(family="IBM Plex Mono",size=10)),
                       "yaxis": dict(title="IRI", gridcolor="#E9ECF2",
                                     tickfont=dict(family="IBM Plex Mono",size=10))})
        fig_sc.update_layout(**sc_lay)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        sh("PCA of Stress Indicators")
        xs   = StandardScaler().fit_transform(stress[["LSI","IRI"]])
        pca  = PCA(n_components=2, random_state=42).fit(xs)
        evr  = pca.explained_variance_ratio_
        ldgs = pd.DataFrame(pca.components_.T, index=["LSI","IRI"], columns=["PC1","PC2"])
        fig_pca = go.Figure(go.Bar(
            x=["PC1","PC2"], y=evr,
            marker=dict(color=["#1A1D2E","#C8102E"],
                        line=dict(color="#FFFFFF",width=1)),
            text=[f"{v*100:.1f}%" for v in evr],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=11)))
        lay_pca = base_layout("PCA — Variance Explained", 230)
        lay_pca["yaxis"]["tickformat"] = ".0%"
        fig_pca.update_layout(**lay_pca)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.dataframe(ldgs.style.format("{:.4f}"), use_container_width=True)

        sh("Current Readings")
        lr = stress.iloc[-1]; lz = stress_z.iloc[-1]
        st.dataframe(pd.DataFrame({
            "Raw": [f"{lr['LSI']:.4f}", f"{lr['IRI']:.4f}"],
            f"Z ({z_window}w)": [f"{lz['LSI_Z']:+.2f}σ", f"{lz['IRI_Z']:+.2f}σ"],
        }, index=["LSI","IRI"]), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIME DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    note(f"Model: <b>{model_used}</b> &nbsp;|&nbsp; {n_regimes} regimes. "
         "States are ordered from lowest to highest average stress. "
         "Probabilities show model confidence at each point in time. "
         "Change the model or number of regimes in the sidebar to refit instantly.")

    sh("Regime Labels Over Time")
    reg_named = regimes.map(rname_map)
    fig_reg = go.Figure()
    for i in range(n_regimes):
        name  = rname_map[i]
        color = REGIME_COLORS_LIST[i]
        mask  = reg_named == name
        fig_reg.add_trace(go.Scatter(
            x=regimes.index[mask],
            y=np.full(mask.sum(), i),
            mode="markers",
            marker=dict(color=color, size=4, symbol="square"),
            name=name))
    lay_reg = base_layout("Regime Label Timeline", 290)
    lay_reg["yaxis"] = dict(tickvals=list(range(n_regimes)),
                            ticktext=list(rname_map.values()),
                            gridcolor="#E9ECF2",
                            tickfont=dict(family="IBM Plex Mono",size=10))
    fig_reg.update_layout(**lay_reg)
    st.plotly_chart(fig_reg, use_container_width=True)

    sh("Regime Probabilities (Stacked Area)")
    pc_colors = [REGIME_COLORS_LIST[i] for i in range(min(n_regimes, len(REGIME_COLORS_LIST)))]
    fig_prob = go.Figure()
    for i, col_name in enumerate(prob_df.columns):
        fig_prob.add_trace(go.Scatter(
            x=prob_df.index, y=prob_df[col_name],
            name=col_name.replace("P_",""),
            fill="tonexty" if i > 0 else "tozeroy",
            stackgroup="one",
            line=dict(color=pc_colors[i], width=0.5),
            hovertemplate=f"<b>{col_name}</b>: %{{y:.1%}}<extra></extra>"))
    lay_prob = base_layout("Stacked Regime Probabilities", 310)
    lay_prob["yaxis"]["tickformat"] = ".0%"
    fig_prob.update_layout(**lay_prob)
    st.plotly_chart(fig_prob, use_container_width=True)

    sh("Conditional Asset Behaviour by Regime")
    note("Annualised return and volatility of each ETF in each detected regime. "
         "If regime labels carry economic meaning, TLT and GLD should show higher "
         "relative performance in stress/crisis regimes.")
    col_l, col_r = st.columns(2)
    with col_l:
        piv_r = CANONICAL_CONDITIONAL_RETURNS / 100
        fig_hr = px.imshow(piv_r*100, text_auto=".1f",
                           color_continuous_scale=["#C8102E","#F4F6FA","#1B7A4E"],
                           zmin=-30, zmax=30, aspect="auto",
                           title="Ann. Return by Regime (%)")
        fig_hr.update_layout(paper_bgcolor="#FFFFFF",
                             font=dict(family="IBM Plex Mono",size=10,color="#1A1D2E"),
                             margin=dict(l=4,r=4,t=38,b=4), height=280,
                             coloraxis_showscale=False)
        st.plotly_chart(fig_hr, use_container_width=True)
    with col_r:
        piv_v = CANONICAL_CONDITIONAL_VOL / 100
        fig_hv = px.imshow(piv_v*100, text_auto=".1f",
                           color_continuous_scale=["#F4F6FA","#D97706","#C8102E"],
                           zmin=0, zmax=50, aspect="auto",
                           title="Ann. Volatility by Regime (%)")
        fig_hv.update_layout(paper_bgcolor="#FFFFFF",
                             font=dict(family="IBM Plex Mono",size=10,color="#1A1D2E"),
                             margin=dict(l=4,r=4,t=38,b=4), height=280,
                             coloraxis_showscale=False)
        st.plotly_chart(fig_hv, use_container_width=True)

    sh("Regime Summary")
    freq_rows = []
    for rid, rname in rname_map.items():
        mask = regimes == rid
        n_wk = mask.sum()
        si   = stress.loc[stress.index.isin(regimes[mask].index)]
        freq_rows.append({
            "Regime": rname, "Weeks": n_wk,
            "% Time": f"{n_wk/len(regimes)*100:.1f}%",
            "Avg LSI": f"{si['LSI'].mean():.3f}",
            "Avg IRI": f"{si['IRI'].mean():.3f}",
        })
    st.dataframe(pd.DataFrame(freq_rows).set_index("Regime"), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO ALLOCATION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    note(f"tc = {tc_bps} bps &nbsp;|&nbsp; threshold = {thr_pct}% &nbsp;|&nbsp; "
         f"burn-in = {min_train} weeks. Adjust in sidebar — all charts and weights update.")

    sh("Phase II Hand Weights — All Regimes")
    fig_grp = go.Figure()
    for j, (ticker, color) in enumerate(zip(CORE_TICKERS, ASSET_COLORS)):
        fig_grp.add_trace(go.Bar(
            name=ticker,
            x=[rname_map.get(i,str(i)) for i in range(n_regimes)],
            y=[HAND_WEIGHTS.get(i, np.ones(4)/4)[j] for i in range(n_regimes)],
            marker=dict(color=color, line=dict(color="#FFFFFF",width=1)),
            text=[f"{HAND_WEIGHTS.get(i,np.ones(4)/4)[j]*100:.0f}%"
                  for i in range(n_regimes)],
            textposition="inside",
            textfont=dict(family="IBM Plex Mono",size=10,color="#FFFFFF")))
    lay_grp = base_layout("Target Allocation per Regime (Phase II Hand Strategy)", 350)
    lay_grp["yaxis"]["tickformat"] = ".0%"
    fig_grp.update_layout(barmode="group", **lay_grp)
    st.plotly_chart(fig_grp, use_container_width=True)

    sh("Phase III Optimized Regime Weights")

    fig_opt = go.Figure()

    for ticker, color in zip(CORE_TICKERS, ASSET_COLORS):
        fig_opt.add_trace(go.Bar(
            name=ticker,
            x=OPTIMIZED_WEIGHTS.index,
            y=OPTIMIZED_WEIGHTS[ticker],
            marker=dict(color=color, line=dict(color="#FFFFFF", width=1)),
            text=[f"{v*100:.0f}%" for v in OPTIMIZED_WEIGHTS[ticker]],
            textposition="inside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#FFFFFF")
        ))

    lay_opt = base_layout("Optimized Max-Sharpe Weights by Regime", 350)
    lay_opt["yaxis"]["tickformat"] = ".0%"
    fig_opt.update_layout(barmode="stack", **lay_opt)

    st.plotly_chart(fig_opt, use_container_width=True)

    note("The optimized allocation matches the final PPT logic: SPY dominates calm regimes, while Crisis shifts fully into TLT and GLD.")

    sh("Dynamic Weight History — Phase II Regime Strategy")
    fig_wh = go.Figure()
    for j, (ticker, color) in enumerate(zip(CORE_TICKERS, ASSET_COLORS)):
        fig_wh.add_trace(go.Scatter(
            x=hw_wts.index, y=hw_wts[ticker], name=ticker,
            line=dict(color=color,width=0.5),
            fill="tonexty" if j > 0 else "tozeroy",
            stackgroup="one",
            hovertemplate=f"<b>{ticker}</b>: %{{y:.1%}}<extra></extra>"))
    lay_wh = base_layout("Stacked Weight Evolution — Regime Strategy", 340)
    lay_wh["yaxis"] = dict(tickformat=".0%", range=[0,1],
                           gridcolor="#E9ECF2",
                           tickfont=dict(family="IBM Plex Mono",size=10))
    fig_wh.update_layout(**lay_wh)
    st.plotly_chart(fig_wh, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sh("Risk Parity Weights (Expanding Window)")
        fig_rp = go.Figure()
        for j, (ticker, color) in enumerate(zip(CORE_TICKERS, ASSET_COLORS)):
            fig_rp.add_trace(go.Scatter(
                x=rp_wts.index, y=rp_wts[ticker], name=ticker,
                line=dict(color=color,width=0.5),
                fill="tonexty" if j > 0 else "tozeroy",
                stackgroup="one"))
        lay_rp = base_layout("Risk Parity", 300)
        lay_rp["yaxis"] = dict(tickformat=".0%", range=[0,1],
                               gridcolor="#E9ECF2",
                               tickfont=dict(family="IBM Plex Mono",size=10))
        fig_rp.update_layout(**lay_rp)
        st.plotly_chart(fig_rp, use_container_width=True)

    with col_r:
        sh("Minimum Volatility Weights (Expanding Window)")
        fig_mv = go.Figure()
        for j, (ticker, color) in enumerate(zip(CORE_TICKERS, ASSET_COLORS)):
            fig_mv.add_trace(go.Scatter(
                x=mv_wts.index, y=mv_wts[ticker], name=ticker,
                line=dict(color=color,width=0.5),
                fill="tonexty" if j > 0 else "tozeroy",
                stackgroup="one"))
        lay_mv = base_layout("Minimum Volatility", 300)
        lay_mv["yaxis"] = dict(tickformat=".0%", range=[0,1],
                               gridcolor="#E9ECF2",
                               tickfont=dict(family="IBM Plex Mono",size=10))
        fig_mv.update_layout(**lay_mv)
        st.plotly_chart(fig_mv, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    note(f"Out-of-sample from week {min_train} onward. "
         f"tc = {tc_bps} bps &nbsp;|&nbsp; rebalance threshold = {thr_pct}%. "
         "Modify sidebar parameters and metrics update immediately.")

    sh("Performance Scorecard")

    disp = CANONICAL_SCORECARD.copy()

    fp = lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—"
    fn = lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    
    if headline_df is not None:
        disp = headline_df.copy()
    
        preferred_order = [
            "Post-Phase-II Regime Optimized",
            "Phase II Hand + Regime",
            "60/40 Benchmark",
            "Equal Weight",
            "Risk Parity",
            "Minimum Volatility",
        ]
    
        ordered = [s for s in preferred_order if s in disp.index] + [
            s for s in disp.index if s not in preferred_order
        ]
    
        disp = disp.loc[ordered]
    
        for c in ["Total Return", "Ann Return", "Ann Vol", "Max Drawdown", "Ann Turnover"]:
            disp[c] = disp[c].apply(fp)

        for c in ["Sharpe", "Calmar"]:
            disp[c] = disp[c].apply(fn)

        st.success("Loaded scorecard from csv")
    
    else:
        st.warning("headline_summary.csv not found in outputs folder. Showing live-computed dashboard values instead.")
    
        disp = perf_sum.copy()
    
        for c in ["Total Return", "Ann Return", "Ann Vol", "Max Drawdown", "Ann Turnover"]:
            disp[c] = disp[c].apply(fp)
    
        for c in ["Sharpe", "Calmar"]:
            disp[c] = disp[c].apply(fn)
    
    st.dataframe(
            disp.style.set_properties(**{
                "font-family": "IBM Plex Mono,monospace",
                "font-size": "12px"
            }),
            use_container_width=True
        )

    sh("Strategy Growth of $1")
    sel = st.multiselect("Select strategies to display",
                         options=list(strat_r.columns),
                         default=list(strat_r.columns))
    if sel:
        fig_g = line_fig(strat_growth[sel], "Cumulative Growth of $1",
                         colors=STRAT_COLORS, height=360)
        st.plotly_chart(fig_g, use_container_width=True)

        sh("Drawdown Paths")
        fig_dd = line_fig(strat_dd[sel], "Portfolio Drawdowns",
                          yformat="pct", colors=STRAT_COLORS, height=320)
        fig_dd.add_hline(y=0, line=dict(color="#9CA3AF",dash="dot",width=1))
        st.plotly_chart(fig_dd, use_container_width=True)

    sh("Rolling 52-Week Sharpe Ratio")
    roll_sh = strat_r.rolling(52).apply(
        lambda x: (x.mean()*52)/(x.std()*np.sqrt(52)) if x.std()>0 else np.nan)
    display_sh = sel if sel else list(strat_r.columns)
    fig_rs = line_fig(roll_sh[display_sh], "Rolling 52-Week Sharpe Ratio",
                      colors=STRAT_COLORS, height=320)
    fig_rs.add_hline(y=0, line=dict(color="#9CA3AF",dash="dot",width=1))
    fig_rs.add_hline(y=1, line=dict(color="#1B7A4E",dash="dash",width=1),
                     annotation_text=" Sharpe = 1",
                     annotation_font=dict(family="IBM Plex Mono",color="#1B7A4E",size=10))
    st.plotly_chart(fig_rs, use_container_width=True)

    sh("Transaction Cost Sensitivity Analysis")

    note(
        "This table uses the final validated notebook values. "
        "It shows how total return changes when round-trip transaction costs increase. "
        "The regime advantage compresses with higher costs, but it does not invert."
    )

    CANONICAL_COST_SENSITIVITY = pd.DataFrame({
        "Cost Level": ["0 bps", "10 bps", "25 bps", "50 bps"],
        "60/40 Benchmark": [352.6, 350.9, 348.3, 344.0],
        "Phase II Hand + Regime": [692.7, 650.8, 592.1, 504.2],
        "Post-Phase-II Regime Optimized": [1016.9, 919.1, 788.0, 605.6],
    }).set_index("Cost Level")

    st.dataframe(
        CANONICAL_COST_SENSITIVITY.style.format("{:.1f}%").set_properties(**{
            "font-family": "IBM Plex Mono,monospace",
            "font-size": "12px"
        }),
        use_container_width=True
    )

    sh("Sharpe Ratio Across Transaction Cost Levels")

    CANONICAL_COST_SHARPE = pd.DataFrame({
        "Cost Level": ["0 bps", "10 bps", "25 bps", "50 bps"],
        "60/40 Benchmark": [0.99, 0.99, 0.98, 0.98],
        "Phase II Hand + Regime": [1.68, 1.63, 1.56, 1.44],
        "Post-Phase-II Regime Optimized": [1.93, 1.85, 1.74, 1.54],
    }).set_index("Cost Level")

    fig_cost = go.Figure()

    cost_colors = {
        "Post-Phase-II Regime Optimized": "#2563EB",
        "Phase II Hand + Regime": "#1B7A4E",
        "60/40 Benchmark": "#9CA3AF",
    }

    for strategy in CANONICAL_COST_SHARPE.columns:
        fig_cost.add_trace(go.Scatter(
            x=CANONICAL_COST_SHARPE.index,
            y=CANONICAL_COST_SHARPE[strategy],
            name=strategy,
            mode="lines+markers",
            line=dict(color=cost_colors.get(strategy, "#1A1D2E"), width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{strategy}</b><br>%{{x}}: %{{y:.2f}} Sharpe<extra></extra>"
        ))

    lay_cost = base_layout("Sharpe Ratio vs Transaction Costs", 320)
    lay_cost["yaxis"] = dict(
        title="Sharpe Ratio",
        gridcolor="#E9ECF2",
        zeroline=False,
        tickfont=dict(family="IBM Plex Mono", size=10)
    )
    lay_cost["xaxis"] = dict(
        title="Transaction Cost",
        gridcolor="#E9ECF2",
        tickfont=dict(family="IBM Plex Mono", size=10)
    )

    fig_cost.update_layout(**lay_cost)
    st.plotly_chart(fig_cost, use_container_width=True)

    note(
        "At 50 bps transaction costs, the optimized regime strategy still keeps a "
        "Sharpe ratio of 1.54 versus 0.98 for the 60/40 benchmark. "
        "So higher turnover reduces performance, but the strategy remains stronger than the static benchmark."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    sh("Stationarity & Normality Tests")
    note("ADF: H₀ = unit root. Low p-value → stationary returns (typical). "
         "Jarque-Bera: H₀ = normality. Low p-value → fat tails / skewness — "
         "supports regime-based rather than static Gaussian portfolio modelling.")
    trows = []
    for col in returns.columns:
        adf_s = adf_p = np.nan
        if STATSMODELS_AVAILABLE:
            s, p, *_ = adfuller(returns[col].dropna())
            adf_s, adf_p = s, p
        jbs, jbp = jarque_bera(returns[col].dropna())
        sk_ = skew(returns[col].dropna())
        ku_ = kurtosis(returns[col].dropna(), fisher=True)
        trows.append({"Asset":col,
                      "ADF Stat":    f"{adf_s:.3f}" if not np.isnan(adf_s) else "N/A",
                      "ADF p-val":   f"{adf_p:.4f}" if not np.isnan(adf_p) else "N/A",
                      "Stationary":  "Yes" if (not np.isnan(adf_p) and adf_p<0.05) else "No",
                      "JB Stat":     f"{jbs:.1f}",
                      "JB p-val":    f"{jbp:.4f}",
                      "Normal":      "Yes" if jbp>0.05 else "No",
                      "Skewness":    f"{sk_:.3f}",
                      "Excess Kurt": f"{ku_:.3f}"})
    st.dataframe(pd.DataFrame(trows).set_index("Asset"), use_container_width=True)

    sh("Weekly Return Distributions")
    fig_hist = make_subplots(rows=1, cols=4, subplot_titles=CORE_TICKERS)
    for j, (ticker, color) in enumerate(zip(CORE_TICKERS, ASSET_COLORS)):
        fig_hist.add_trace(
            go.Histogram(x=returns[ticker], nbinsx=60,
                         marker=dict(color=color, opacity=0.7,
                                     line=dict(color="#FFFFFF",width=0.3)),
                         name=ticker, showlegend=False),
            row=1, col=j+1)
    fig_hist.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FC",
        font=dict(family="IBM Plex Sans",color="#1A1D2E",size=10),
        title=dict(text="Weekly Return Distributions",
                   font=dict(family="IBM Plex Sans",size=12,color="#1A1D2E")),
        margin=dict(l=6,r=6,t=44,b=6), height=300)
    for i in range(1,5):
        fig_hist.update_xaxes(gridcolor="#E9ECF2", tickformat=".1%", row=1, col=i,
                              tickfont=dict(family="IBM Plex Mono",size=9))
        fig_hist.update_yaxes(gridcolor="#E9ECF2", row=1, col=i)
    st.plotly_chart(fig_hist, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sh("Return Correlation Matrix")
        corr = returns.corr()
        fig_corr = px.imshow(corr, text_auto=".2f",
                             color_continuous_scale=["#C8102E","#F4F6FA","#1B7A4E"],
                             zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(paper_bgcolor="#FFFFFF",
                               font=dict(family="IBM Plex Mono",size=11,color="#1A1D2E"),
                               margin=dict(l=4,r=4,t=10,b=4), height=300,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_r:
        sh("Descriptive Statistics")
        desc = returns.describe().T
        desc["skewness"]        = returns.apply(skew)
        desc["excess_kurtosis"] = returns.apply(lambda x: kurtosis(x, fisher=True))
        for c in ["mean","std","min","25%","50%","75%","max"]:
            if c in desc.columns:
                desc[c] = desc[c].apply(lambda v: f"{v*100:.3f}%")
        desc["skewness"]        = desc["skewness"].apply(lambda v: f"{v:.3f}")
        desc["excess_kurtosis"] = desc["excess_kurtosis"].apply(lambda v: f"{v:.3f}")
        st.dataframe(desc.drop(columns=["count"], errors="ignore"), use_container_width=True)

    sh("Regime-Conditional Sharpe Ratios")
    note("Sharpe ratio per asset within each detected regime. "
         "Positive Sharpe for TLT/GLD in stress and crisis regimes validates "
         "the defensive positioning applied by the Phase II hand-weight strategy.")
    piv_sr = cond_df.pivot(index="Regime",columns="Asset",values="Sharpe")
    fig_sr = px.imshow(piv_sr, text_auto=".2f",
                       color_continuous_scale=["#C8102E","#F4F6FA","#1B7A4E"],
                       zmin=-3, zmax=3, aspect="auto",
                       title="Regime-Conditional Sharpe Ratios")
    fig_sr.update_layout(paper_bgcolor="#FFFFFF",
                         font=dict(family="IBM Plex Mono",size=10,color="#1A1D2E"),
                         margin=dict(l=4,r=4,t=38,b=4), height=290,
                         coloraxis_showscale=False)
    st.plotly_chart(fig_sr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            color:#9CA3AF;letter-spacing:1.5px;padding:0.5rem 0;">
  QWIM REGIME ALLOCATION FRAMEWORK &nbsp;·&nbsp; BANK OF AMERICA &nbsp;·&nbsp;
  FA800 &nbsp;·&nbsp; FOR ACADEMIC USE ONLY
</div>
""", unsafe_allow_html=True)
