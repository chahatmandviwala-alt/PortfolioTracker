import json
from pathlib import Path
from datetime import datetime, date

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


st.markdown("""
<style>
/* Hide the 'Press Enter to submit form' helper text everywhere */
[data-testid="InputInstructions"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG & STYLING
# =========================

st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>

    div[data-testid="stTabs"] > div:nth-child(1) {
        margin-top: -20px !important;
        padding-top: 0 !important;
    }

    .block-container {
        padding-top: 0.5rem !important;
    }

    div[data-testid="column"] {
        margin-top: -10px !important;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Hide the + / - buttons on st.number_input */
div[data-testid="stNumberInput"] button {
    display: none !important;
}

/* Optional: make the input full-width once buttons are gone */
div[data-testid="stNumberInput"] input {
    padding-right: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Compact top header with KPIs */
.mobile-header {
    margin-bottom: 0.6rem;
}

/* Bigger, clean title */
.mobile-header-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

/* KPI container */
.kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
}

/* Clean KPI cards */
.kpi {
    flex: 1 1 calc(50% - 12px);
    border-radius: 12px;
    padding: 10px 12px;
    border: 1px solid rgba(255, 255, 255, 0.15); /* subtle but visible */
    background: rgba(255, 255, 255, 0.02);       /* almost flat */
    text-align: center;
}

/* Icons */
.kpi-icon {
    display: block;
    font-size: 1.25rem;
    margin-bottom: 2px;
}

/* Label */
.kpi-label {
    font-size: 0.85rem;
    opacity: 0.8;
    white-space: nowrap;
}

/* Value */
.kpi-value {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 2px;
    line-height: 1.2;
}

/* Value color accents */
.kpi-main .kpi-value {
    color: #4ea8ff; /* blue */
}

.kpi-invested .kpi-value {
    color: #ffb347; /* orange */
}

.kpi-positive .kpi-value {
    color: #19c37d; /* green */
}

.kpi-negative .kpi-value {
    color: #ff4b4b; /* red */
}

.kpi-neutral .kpi-value {
    color: #e5e5e5; /* gray */
}

.kpi-assets .kpi-value {
    color: #a78bfa; /* purple */
}

/* Desktop: 4 in a row */
@media (min-width: 800px) {
    .kpi {
        flex: 1 1 0;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stTabs"] button {
    padding-top: 0px !important;     /* default ~10px */
    padding-bottom: 6px !important;
    padding-left: 0px !important;
    padding-right: 0px !important;

    font-size: 0.85rem !important;   /* smaller label text */
    height: 30px !important;         /* reduce full tab height */
}

div[data-testid="stTabs"] button p {
    font-size: 0.85rem !important;   /* tab label text */
}

/* Optional: reduce gap between tabs a bit */
div[data-testid="stTabs"] button + button {
    margin-left: -2px !important;
}

</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_BASE_CCY = "SEK"

def get_user_paths(username: str) -> tuple[Path, Path]:
    """
    Map a username -> user-specific trades/settings files.
    """
    safe = "".join(c for c in username if c.isalnum() or c in "-_").lower()
    if not safe:
        st.error("Username may only contain letters, numbers, '-' and '_'.")
        st.stop()

    trades_file = DATA_DIR / f"trades_{safe}.csv"
    settings_file = DATA_DIR / f"settings_{safe}.json"
    return trades_file, settings_file

INPUT_COLS = [
    "trade_date",
    "asset",
    "asset_type",
    "side",
    "quantity",
    "price_trade_ccy",
    "trade_ccy",
    "charges_trade_ccy",
    "stt_trade_ccy",
    "fx_to_base",
    # Live pricing fields
    "ticker",
    "last_price_trade_ccy",
    "last_price_base",
]

# =========================
# SETTINGS PERSISTENCE
# =========================

def load_settings(settings_file: Path) -> dict:
    if settings_file.exists():
        try:
            return json.loads(settings_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(settings: dict, settings_file: Path) -> None:
    try:
        settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        # Failing to save settings should not crash the app
        pass


# =========================
# DATA LOGIC
# =========================


def mask_number(value: float, mask: bool, decimals: int = 0):
    if not mask:
        return f"{value:,.{decimals}f}"
    return "••••••"

@st.cache_data(show_spinner=False)
def fetch_historical_fx(date_obj: datetime, from_ccy: str, to_ccy: str) -> float:
    """
    Get FX rate (from_ccy → to_ccy) for a given date, cached by Streamlit.
    """
    from_ccy = (from_ccy or "").upper()
    to_ccy = (to_ccy or "").upper()
    if not from_ccy or not to_ccy:
        return 1.0
    if from_ccy == to_ccy:
        return 1.0
    if pd.isna(date_obj):
        return 0.0

    date_str = date_obj.strftime("%Y-%m-%d")
    url = f"https://api.frankfurter.dev/v1/{date_str}"
    try:
        resp = requests.get(
            url, params={"base": from_ccy, "symbols": to_ccy}, timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"][to_ccy])
        return rate if rate > 0 else 1.0
    except Exception:
        # In case of error, fall back to 1.0 to avoid breaking calculations
        return 1.0


@st.cache_data(show_spinner=False)
def fetch_fx_timeseries(
    from_ccy: str, to_ccy: str, start_date: date, end_date: date
) -> dict[date, float]:
    """
    Fetch a whole time series of FX rates (from_ccy -> to_ccy) between start_date and end_date.
    Returns a dict mapping Python date -> rate.
    If the range call fails for some reason, returns {} and caller can fall back to per-day calls.
    """
    from_ccy = (from_ccy or "").upper()
    to_ccy = (to_ccy or "").upper()
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return {}

    url = f"https://api.frankfurter.dev/v1/{start_date.isoformat()}..{end_date.isoformat()}"
    try:
        resp = requests.get(
            url,
            params={"base": from_ccy, "symbols": to_ccy},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        out: dict[date, float] = {}
        for date_str, daily_rates in data.get("rates", {}).items():
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                rate = float(daily_rates[to_ccy])
                if rate > 0:
                    out[dt] = rate
            except Exception:
                continue

        return out
    except Exception:
        # If anything goes wrong (network, JSON, etc.), just return {}
        # The caller will then fall back to fetch_historical_fx per day.
        return {}


def add_derived_columns(
    df: pd.DataFrame, base_ccy: str | None = None, recalc_fx: bool = False
) -> pd.DataFrame:
    d = df.copy()

    # Ensure stable row order marker for FIFO tie-breaking
    if "_row" not in d.columns:
        d["_row"] = np.arange(len(d))

    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d["side"] = d["side"].astype(str).str.upper().str.strip()
    d["asset"] = d["asset"].astype(str).str.strip()
    d["asset_type"] = d["asset_type"].astype(str).str.strip()
    d["trade_ccy"] = d["trade_ccy"].astype(str).str.upper().str.strip()

    # --- Recalculate FX against current base_ccy (batched + cached) ---
    if recalc_fx and base_ccy:
        base_ccy = base_ccy.upper()

        if "fx_to_base" not in d.columns:
            d["fx_to_base"] = 1.0

        # Trades already in base currency: fx = 1
        d.loc[d["trade_ccy"] == base_ccy, "fx_to_base"] = 1.0

        # Only rows needing FX
        mask = (
            (d["trade_ccy"] != base_ccy)
            & d["trade_date"].notna()
            & (d["trade_ccy"] != "")
        )

        if mask.any():
            sub = d.loc[mask, ["trade_date", "trade_ccy"]].drop_duplicates()

            # (date, trade_ccy) -> fx_to_base
            rate_map: dict[tuple[date, str], float] = {}

            # Group by trade currency so we can batch the API calls
            for ccy, g in sub.groupby("trade_ccy"):
                dates = g["trade_date"].dt.date
                start_dt = dates.min()
                end_dt = dates.max()

                # Try fast batched range fetch first
                ts = fetch_fx_timeseries(ccy, base_ccy, start_dt, end_dt)

                for dt in dates.unique():
                    rate: float | None = None

                    # 1) Use time-series result if available
                    if ts:
                        rate = ts.get(dt)

                    # 2) If missing (weekend/holiday or API issue), fall back to single-date endpoint
                    if rate is None:
                        rate = fetch_historical_fx(
                            datetime.combine(dt, datetime.min.time()),
                            ccy,
                            base_ccy,
                        )

                    # 3) Final safety: never let it be <= 0 or None
                    if rate is None or rate <= 0:
                        rate = 1.0

                    rate_map[(dt, ccy)] = rate

            # Map back into the main DataFrame
            d["_key"] = list(zip(d["trade_date"].dt.date, d["trade_ccy"]))

            d.loc[mask, "fx_to_base"] = d.loc[mask, "_key"].map(rate_map).fillna(
                d.loc[mask, "fx_to_base"]
            )
            d.drop(columns=["_key"], inplace=True, errors="ignore")

    # Numeric conversions
    num_cols = [
        "quantity",
        "price_trade_ccy",
        "charges_trade_ccy",
        "stt_trade_ccy",
        "fx_to_base",
    ]
    for col in num_cols:
        d[col] = (
            d[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

    d["gross_trade_value"] = d["quantity"] * d["price_trade_ccy"]
    buy_mask = d["side"] == "BUY"
    sell_mask = d["side"] == "SELL"

    # Total cash flow in trade currency INCLUDING STT
    d["total_trade_value_trade"] = 0.0
    d.loc[buy_mask, "total_trade_value_trade"] = (
        d["gross_trade_value"] + d["charges_trade_ccy"] + d["stt_trade_ccy"]
    )
    d.loc[sell_mask, "total_trade_value_trade"] = (
        d["gross_trade_value"] - d["charges_trade_ccy"] - d["stt_trade_ccy"]
    )

    # Tax basis (EXCLUDE STT) - used for cost basis in trade currency and INR P&L
    d["total_for_tax"] = 0.0
    d.loc[buy_mask, "total_for_tax"] = d["gross_trade_value"] + d["charges_trade_ccy"]
    d.loc[sell_mask, "total_for_tax"] = d["gross_trade_value"] - d["charges_trade_ccy"]

    # Per-unit tax basis in trade currency (EXCLUDES STT)
    d["effective_unit_trade"] = np.where(
        d["quantity"] > 0, d["total_for_tax"] / d["quantity"], 0.0
    )

    # Base-currency economic basis (INCLUDE STT, then FX)
    d["total_for_base"] = d["total_trade_value_trade"]
    d["effective_unit_base"] = np.where(
        d["quantity"] > 0,
        d["total_for_base"] * d["fx_to_base"] / d["quantity"],
        0.0,
    )

    # Base-currency tax basis EXCLUDING STT (for INR realized P&L rule)
    d["effective_unit_base_no_stt"] = np.where(
        d["quantity"] > 0,
        d["total_for_tax"] * d["fx_to_base"] / d["quantity"],
        0.0,
    )

    return d


def load_trades(base_ccy: str, recalc_fx: bool, data_file: Path) -> pd.DataFrame:
    if data_file.exists():
        df = pd.read_csv(data_file, sep=";")
    else:
        df = pd.DataFrame(columns=INPUT_COLS)

    # Ensure required columns exist (new string columns included)
    for col in INPUT_COLS:
        if col not in df.columns:
            if col in ("asset", "asset_type", "side", "trade_ccy", "ticker"):
                df[col] = ""
            else:
                df[col] = 0.0

    # Attach a stable row index for FIFO tie-breaking
    df["_row"] = np.arange(len(df))

    return add_derived_columns(df[INPUT_COLS + ["_row"]], base_ccy=base_ccy, recalc_fx=recalc_fx)


def save_trades(df: pd.DataFrame, data_file: Path) -> None:
    to_save = df[INPUT_COLS].copy()
    to_save["trade_date"] = pd.to_datetime(
        to_save["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M")
    to_save.to_csv(data_file, index=False, sep=";")


# ============ LIVE PRICE UPDATER ============

def update_prices_in_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw trades DataFrame (with 'ticker' & 'fx_to_base'),
    fetches latest prices from yfinance using the 'ticker' column,
    and fills 'last_price_trade_ccy' and 'last_price_base'.

    Returns a NEW DataFrame (does not modify in-place).
    """
    d = df.copy()

    # Make sure columns exist
    if "ticker" not in d.columns:
        d["ticker"] = ""
    if "last_price_trade_ccy" not in d.columns:
        d["last_price_trade_ccy"] = 0.0
    if "last_price_base" not in d.columns:
        d["last_price_base"] = 0.0

    # Collect unique, non-empty tickers
    tickers = (
        d["ticker"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "NaN": None})
        .dropna()
        .unique()
        .tolist()
    )

    if not tickers:
        return d

    # --- inner helper: fetch price for ONE ticker (trade ccy) ---
    def fetch_one(ticker: str) -> tuple[str, float | None]:
        price = None
        try:
            t = yf.Ticker(ticker)

            # fast_info first
            try:
                fi = getattr(t, "fast_info", None)
                if fi is not None:
                    for attr_name in ["last_price", "regularMarketPrice", "last_price_raw"]:
                        if price is not None:
                            break
                        # attr-style
                        try:
                            val = getattr(fi, attr_name, None)
                            if val is not None:
                                price = float(val)
                        except Exception:
                            pass
                        # dict-style
                        if price is None and isinstance(fi, dict):
                            val = fi.get(attr_name)
                            if val is not None:
                                price = float(val)
            except Exception:
                pass

            # fallback to info
            if price is None:
                try:
                    info = t.info
                    val = info.get("regularMarketPrice")
                    if val is not None:
                        price = float(val)
                except Exception:
                    pass

        except Exception:
            price = None

        return ticker, price

    # --- fetch all prices in parallel ---
    price_map: dict[str, float | None] = {}
    max_workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(future_map):
            ticker, price = fut.result()
            price_map[ticker] = price

    # --- write prices back into df ---
    for idx, row in d.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        price = price_map.get(ticker)
        if price is None:
            # leave whatever is there
            continue

        d.at[idx, "last_price_trade_ccy"] = price

        # Use existing fx_to_base (already in your CSV) to convert
        fx = float(row.get("fx_to_base", 1.0) or 1.0)
        d.at[idx, "last_price_base"] = price * fx

    return d


# =========================
# CALCULATION ENGINES
# =========================

@st.cache_data(show_spinner=False)
def fifo_summary_by_unit(
    df: pd.DataFrame,
    unit_col: str,
    realized_col: str,
    pos_cost_col: str,
    avg_cost_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["asset", "position_qty", realized_col, pos_cost_col, avg_cost_col]
        )

    records = []
    for asset, grp in df.groupby("asset"):
        # Ensure BUY processed before SELL if dates are identical (intraday fix)
        # and preserve original row order within the same date+side
        grp = grp.assign(
            _side_rank=np.where(grp["side"].str.upper() == "BUY", 0, 1)
        ).sort_values(["trade_date", "_side_rank", "_row"])

        lots: list[list[float]] = []
        realized = 0.0

        for _, row in grp.iterrows():
            side = str(row["side"]).upper()
            qty = float(row["quantity"])
            unit = float(row[unit_col])
            if qty <= 0:
                continue

            if side == "BUY":
                lots.append([qty, unit])
            elif side == "SELL":
                sell_qty = qty
                while sell_qty > 0 and lots:
                    take = min(sell_qty, lots[0][0])
                    realized += (unit - lots[0][1]) * take
                    lots[0][0] -= take
                    sell_qty -= take
                    if lots[0][0] == 0:
                        lots.pop(0)

        pos_qty = sum(l[0] for l in lots)
        pos_cost = sum(l[0] * l[1] for l in lots)
        avg_cost = pos_cost / pos_qty if pos_qty > 0 else 0.0
        records.append(
            {
                "asset": asset,
                "position_qty": pos_qty,
                realized_col: realized,
                pos_cost_col: pos_cost,
                avg_cost_col: avg_cost,
            }
        )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def fifo_realized_for_period(
    df: pd.DataFrame, unit_col: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df[df["trade_date"] <= end_dt].copy()
    records = []

    ccy_map = df.groupby("asset")["trade_ccy"].first().to_dict()

    for asset, grp in d.groupby("asset"):
        grp = grp.assign(
            _side_rank=np.where(grp["side"].str.upper() == "BUY", 0, 1)
        ).sort_values(["trade_date", "_side_rank", "_row"])

        lots: list[list[float]] = []
        p_realized = p_buy = p_sell = p_qty = 0.0

        for _, row in grp.iterrows():
            side = str(row["side"]).upper()
            qty = float(row["quantity"])
            unit = float(row[unit_col])
            tdate = row["trade_date"]
            if qty <= 0:
                continue

            if side == "BUY":
                lots.append([qty, unit])
            elif side == "SELL":
                sell_qty = qty
                temp_real = temp_buy = temp_sell = temp_match = 0.0
                while sell_qty > 0 and lots:
                    take = min(sell_qty, lots[0][0])
                    temp_match += take
                    temp_real += (unit - lots[0][1]) * take
                    temp_buy += lots[0][1] * take
                    temp_sell += unit * take
                    lots[0][0] -= take
                    sell_qty -= take
                    if lots[0][0] == 0:
                        lots.pop(0)

                if start_dt <= tdate <= end_dt:
                    p_realized += temp_real
                    p_buy += temp_buy
                    p_sell += temp_sell
                    p_qty += temp_match

        if p_qty > 0:
            records.append(
                {
                    "asset": asset,
                    "quantity_sold": p_qty,
                    "realized": p_realized,
                    "buy_value": p_buy,
                    "sell_value": p_sell,
                    "currency": ccy_map.get(asset, ""),
                }
            )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def compute_portfolio(trades: pd.DataFrame, base_ccy: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    d = trades.sort_values("trade_date")

    bf = fifo_summary_by_unit(
        d,
        "effective_unit_base",
        "realized_pl_base",
        "position_cost_base",
        "avg_cost_base",
    )
    tf = fifo_summary_by_unit(
        d,
        "effective_unit_trade",
        "realized_pl_trade",
        "position_cost_trade",
        "avg_cost_trade",
    )

    res = (
        bf.merge(tf, on=["asset", "position_qty"], how="outer")
        .merge(
            d.groupby("asset", as_index=False)["trade_ccy"].first(),
            on="asset",
            how="left",
        )
        .fillna(0.0)
    )

    # Filter out closed positions
    res = res[res["position_qty"].abs() > 1e-6].copy()

    # Final cleanup
    res["Invested (Base)"] = res["position_cost_base"]
    res = res.rename(
        columns={
            "asset": "Asset",
            "position_qty": "Position",
            "avg_cost_trade": "Avg Price",
            "trade_ccy": "Currency",
        }
    )
    return res[["Asset", "Position", "Avg Price", "Currency", "Invested (Base)"]]


# =========================
# UI LAYOUT
# =========================

# =========================
# USER LOGIN via URL PATH
# =========================

# Get the path, e.g. "/alex"
path = st.experimental_get_query_params().get('__streamlit_path', [""])[0]

# Extract username from path
# Example: "/alex" -> "alex"
url_username = path.strip("/")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = url_username or ""

# Auto login if username in URL
if url_username and not st.session_state.logged_in:
    st.session_state.username = url_username
    st.session_state.logged_in = True

# Show login box ONLY if not logged in
if not st.session_state.logged_in:
    st.header("🪪 Login")

    username_input = st.text_input("Username", key="username_input").strip()
    st.caption("Enter a username to load your portfolio.")

    if username_input:
        st.session_state.username = username_input
        st.session_state.logged_in = True
        
        # Redirect to /username
        st.experimental_set_query_params(__streamlit_path=username_input)
        st.rerun()

# After login
username = st.session_state.username

if not username:
    st.stop()

# Map username -> file paths (per user)
DATA_FILE, SETTINGS_FILE = get_user_paths(username)

# Load settings for THIS user
_settings = load_settings(SETTINGS_FILE)
_last_base = _settings.get("base_ccy", DEFAULT_BASE_CCY)

# --- Sidebar contents (only shown after username is set) ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Base settings
    base_ccy = st.text_input(
        "Base Currency",
        value=_last_base,
        key="base_ccy",          # give it its own key
    ).upper()
    hide_values = st.toggle("Hide Portfolio Values", key="hide_values")

    # Refresh prices button
    if st.button("🔄 Refresh market value"):
        if DATA_FILE.exists():
            raw = pd.read_csv(DATA_FILE, sep=";")
        else:
            raw = pd.DataFrame(columns=INPUT_COLS)

        updated = update_prices_in_trades(raw)
        save_trades(updated, DATA_FILE)
        st.success("Market value updated.")
        st.rerun()

    st.divider()

    # --- Portfolio file import / export (per-user) ---
    if DATA_FILE.exists():
        csv_bytes = DATA_FILE.read_bytes()
        st.download_button(
            label="⬇️ Download portfolio",
            data=csv_bytes,
            file_name=f"{username}_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No portfolio to download.")

    uploaded_file = st.file_uploader(
        "Upload/Replace portfolio",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if st.button("⬆️ Upload", use_container_width=True):
            try:
                DATA_FILE.write_bytes(uploaded_file.getvalue())
                st.success("Portfolio uploaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")

    st.divider()
    st.caption(f"Data Source: `{DATA_FILE}`")

recalc_fx = base_ccy != _last_base

trades_df = load_trades(base_ccy, recalc_fx=recalc_fx, data_file=DATA_FILE)

# 🔧 NEW: recompute last_price_base for the CURRENT base currency
if not trades_df.empty and "last_price_trade_ccy" in trades_df.columns:
    trades_df["last_price_base"] = trades_df["last_price_trade_ccy"] * trades_df["fx_to_base"]

if recalc_fx:
    save_trades(trades_df, DATA_FILE)
    _settings["base_ccy"] = base_ccy
    save_settings(_settings, SETTINGS_FILE)

portfolio_df = compute_portfolio(trades_df, base_ccy)

# --- NEW: derive current market value per asset from last_price_base ---
# --- latest prices per asset (both base and trade CCY) ---
if not trades_df.empty and "last_price_base" in trades_df.columns:
    latest_base_map = (
        trades_df.groupby("asset")["last_price_base"]
        .max()
        .fillna(0.0)
        .to_dict()
    )
else:
    latest_base_map = {}

if not trades_df.empty and "last_price_trade_ccy" in trades_df.columns:
    latest_trade_map = (
        trades_df.groupby("asset")["last_price_trade_ccy"]
        .max()
        .fillna(0.0)
        .to_dict()
    )
else:
    latest_trade_map = {}

if not portfolio_df.empty:
    # map both prices into portfolio_df
    portfolio_df["Last Price (Base)"] = portfolio_df["Asset"].map(latest_base_map).fillna(0.0)
    portfolio_df["Last Price (Trade)"] = portfolio_df["Asset"].map(latest_trade_map).fillna(0.0)

    # If base price is missing (0), use cost basis as "market value"
    portfolio_df["Market Value"] = np.where(
        portfolio_df["Last Price (Base)"] > 0,
        portfolio_df["Position"] * portfolio_df["Last Price (Base)"],
        portfolio_df["Invested (Base)"],
    )
else:
    portfolio_df["Last Price (Base)"] = []
    portfolio_df["Last Price (Trade)"] = []
    portfolio_df["Market Value"] = []


# Keep both totals for now (we still use Invested in treemap/table later)
total_invested = (
    portfolio_df["Invested (Base)"].sum() if not portfolio_df.empty else 0.0
)
total_valuation = (
    portfolio_df["Market Value"].sum() if not portfolio_df.empty else 0.0
)

total_realized_all_time = 0.0
if not trades_df.empty:
    _temp_fy = fifo_realized_for_period(
        trades_df, "effective_unit_base", datetime.min, datetime.max
    )
    if not _temp_fy.empty:
        total_realized_all_time = _temp_fy["realized"].sum()

# --- COMPACT HEADER ROW: TITLE + METRICS ON SAME LINE ---


# ----- COMPACT HEADER -----

# Prepare values as strings (respecting hide_values)
valuation_str = mask_number(total_valuation, hide_values, 0)
invested_str = mask_number(total_invested, hide_values, 0)

unrealized_pct = (
    (total_valuation - total_invested) / total_invested * 100
    if total_invested != 0 else 0
)
unrealized_str = mask_number(unrealized_pct, hide_values, 2)
if not hide_values:
    unrealized_str = f"{unrealized_str}%"

assets_str = str(len(portfolio_df)) if not hide_values else "•••••"

header_html = f"""
<div class="mobile-header">
  <div class="mobile-header-title">📈 Investment Portfolio</div>
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Market Value ({base_ccy})</div>
      <div class="kpi-value">{valuation_str}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Invested ({base_ccy})</div>
      <div class="kpi-value">{invested_str}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Unrealized P&amp;L</div>
      <div class="kpi-value">{unrealized_str}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Active assets</div>
      <div class="kpi-value">{assets_str}</div>
    </div>
  </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("---")

# MAIN TABS
tab_portfolio, tab_new_trade, tab_history, tab_tax = st.tabs(
    ["📊 Holdings", "➕ New Trade", "🧾 Trade History", "💰 Realized P/L"]
)

# --- TAB 1: PORTFOLIO ---
with tab_portfolio:
    # --- PORTFOLIO CONTENT ---
    if portfolio_df.empty:
        st.info("Your portfolio is empty. Add a trade above!")
    else:
        col_chart, col_table = st.columns([1, 1.2])

        # ===== LEFT: TREEMAP (by current Market Value) =====
        with col_chart:
            chart_data = portfolio_df.copy()

            # Currency symbol logic
            if base_ccy.upper() == "SEK":
                ccy_symbol = "kr"
            elif base_ccy.upper() == "INR":
                ccy_symbol = "₹"
            elif base_ccy.upper() == "USD":
                ccy_symbol = "$"
            elif base_ccy.upper() == "EUR":
                ccy_symbol = "€"
            else:
                ccy_symbol = base_ccy.upper() + " "

            # Share % based on current Market Value
            total_mv = chart_data["Market Value"].sum()
            chart_data["Share (%)"] = (
                chart_data["Market Value"] / total_mv * 100
                if total_mv != 0 else 0
            )

            # Treemap by Market Value
            fig = px.treemap(
                chart_data,
                path=["Asset"],
                values="Market Value",
                custom_data=["Share (%)", "Market Value"],
            )

            if hide_values:
                # Hide market value, show only asset + share
                fig.update_traces(
                    texttemplate="%{label}<br>%{customdata[0]:.1f}%",
                    textinfo="label+text",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Share: %{customdata[0]:.1f}%<br>"
                        "Market value: ••••••"
                        "<extra></extra>"
                    ),
                )
            else:
                hovertemplate = (
                    f"<b>%{{label}}</b><br>"
                    "Share: %{customdata[0]:.1f}%<br>"
                    f"Market value: {ccy_symbol} %{{customdata[1]:.0f}}"
                    "<extra></extra>"
                )

                fig.update_traces(
                    texttemplate="%{label}<br>%{customdata[0]:.1f}%",
                    textinfo="label+text",
                    hovertemplate=hovertemplate,
                )

            fig.update_layout(
                height=390,
                margin=dict(t=0, l=0, r=0, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)

        # ===== RIGHT: TABLE (with Market Value column) =====
        with col_table:
            portfolio_df = portfolio_df.copy()

            # --- LTP in trade CCY: use last_price_trade if available, otherwise Avg Price ---
            ltp_trade = np.where(
                portfolio_df["Last Price (Trade)"] > 0,
                portfolio_df["Last Price (Trade)"],
                portfolio_df["Avg Price"],
            )
            portfolio_df["LTP"] = ltp_trade

            # Share % based on Market Value
            total_mv = portfolio_df["Market Value"].sum()
            portfolio_df["Share (%)"] = (
                portfolio_df["Market Value"] / total_mv * 100
                if total_mv != 0 else 0
            )

            portfolio_df = portfolio_df.sort_values("Share (%)", ascending=False)

            # Base currency symbol for display
            if base_ccy.upper() == "SEK":
                ccy_symbol = "kr "
            elif base_ccy.upper() == "INR":
                ccy_symbol = "₹ "
            elif base_ccy.upper() == "USD":
                ccy_symbol = "$ "
            elif base_ccy.upper() == "EUR":
                ccy_symbol = "€ "
            else:
                ccy_symbol = base_ccy.upper() + " "

            if hide_values:
                portfolio_df["Position"] = "•••"
                portfolio_df["LTP"] = "••••••"
                portfolio_df["Avg Price"] = "••••••"
                portfolio_df["Invested (Base)"] = "••••••"
                portfolio_df["Market Value"] = "••••••"
            else:
                portfolio_df["Position"] = portfolio_df["Position"].apply(
                    lambda x: f"{x:.4f}"
                )
                # Map trade currency → symbol
                def ccy_symbol_trade(ccy):
                    c = ccy.upper()
                    if c == "USD": return "$"
                    if c == "EUR": return "€"
                    if c == "INR": return "₹"
                    if c == "SEK": return "kr"
                    return c + " "  # fallback

                # Avg Price (trade CCY, 1 decimal)
                portfolio_df["Avg Price"] = portfolio_df.apply(
                    lambda row: f"{ccy_symbol_trade(row['Currency'])}{row['Avg Price']:,.1f}",
                    axis=1,
                )                
                # LTP shown in trade CCY, 1 decimal place
                portfolio_df["LTP"] = portfolio_df.apply(
                lambda row: f"{ccy_symbol_trade(row['Currency'])}{row['LTP']:,.1f}",
                axis=1,
                )
                # Cost basis and Market value with base CCY symbol
                portfolio_df["Invested (Base)"] = portfolio_df["Invested (Base)"].apply(
                    lambda x: f"{ccy_symbol}{x:,.0f}"
                )
                portfolio_df["Market Value"] = portfolio_df["Market Value"].apply(
                    lambda x: f"{ccy_symbol}{x:,.0f}"
                )

            # Show only the requested columns, in the requested order
            display_cols = [
                "Asset",
                "Position",        # shown as "Qty"
                "LTP",
                "Avg Price",
                "Market Value",
                "Invested (Base)",  # "Cost basis"
                "Share (%)",
            ]

            st.dataframe(
                portfolio_df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Asset": st.column_config.TextColumn("Asset", width="medium"),
                    "Position": st.column_config.TextColumn("Qty", width="small"),
                    "LTP": st.column_config.TextColumn("LTP", width="small"),
                    "Avg Price": st.column_config.TextColumn("Avg Price", width="small"),
                    "Market Value": st.column_config.TextColumn(
                        "Market value", width="small"
                    ),
                    "Invested (Base)": st.column_config.TextColumn(
                        "Cost basis", width="small"
                    ),
                    "Share (%)": st.column_config.NumberColumn(
                        "Share (%)", format="%.2f%%", width="small"
                    ),
                },
            )

# --- TAB 2: REGISTER NEW TRADE ---
with tab_new_trade:

    spacer, auto_col = st.columns([9, 1])
    with auto_col:
        auto_fx = st.checkbox("Auto FX", value=True)

    with st.form("add_trade_form", clear_on_submit=True):
        # Row 1: Key Identifiers (+ ticker)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            t_date = st.date_input("Date", value=datetime.today())
        with c2:
            t_asset = st.text_input(
                "Asset", placeholder="e.g. NVDA"
            ).strip()
        with c3:
            t_ticker = st.text_input(
                "Ticker (yfinance)",
            ).strip()
        with c4:
            t_side = st.selectbox("Side", ["BUY", "SELL"])
        with c5:
            t_type = st.selectbox(
                "Type", ["Stock", "Crypto", "Fund", "ETF"]
            )

        # Row 2: Numbers
        c5, c6, c7, c8, c9, c10 = st.columns(6)
        with c5:
            t_qty = st.number_input(
                "Qty", min_value=0.0, value=None, step=None, format="%.4f"
            )
        with c6:
            t_price = st.number_input(
                "Price", min_value=0.0, value=None, step=None, format="%.2f"
            )
        with c7:
            t_ccy = st.text_input(
                "Currency", value="USD"
            ).upper()
        with c8:
            t_fees = st.number_input(
                "Fees/Brokerage", min_value=0.0, value=None, step=None, format="%.2f"
            )
        with c9:
            t_stt = st.number_input("Tax/STT", min_value=0.0, value=None, step=None, format="%.2f")
        # we already have auto_fx above; maybe show info text instead of checkbox again
        with c10:
            t_manual_fx = st.number_input(
                f"Manual FX (to {base_ccy})",
                min_value=0.0,
                format="%.6f",
                disabled=auto_fx,  # now this updates correctly on rerun
            )

        if st.form_submit_button("💾 Save Trade", type="primary", use_container_width=True):
            if not t_asset or t_qty <= 0:
                st.error("Invalid Asset or Quantity.")
            else:
                save_dt = datetime.combine(t_date, datetime.min.time())
                if auto_fx:
                    fetched = fetch_historical_fx(save_dt, t_ccy, base_ccy)
                    final_fx = fetched if fetched > 0 else 1.0
                else:
                    final_fx = t_manual_fx or 1.0

                # Load existing trades to:
                # - preserve existing rows
                # - reuse ticker if user left ticker blank
                if DATA_FILE.exists():
                    existing = pd.read_csv(DATA_FILE, sep=";")
                else:
                    existing = pd.DataFrame(columns=INPUT_COLS)

                ticker_value = t_ticker

                # If ticker left blank, reuse previous ticker for same asset (if any)
                if not ticker_value:
                    if not existing.empty and "ticker" in existing.columns:
                        matches = (
                            existing.loc[existing["asset"] == t_asset, "ticker"]
                            .astype(str)
                            .str.strip()
                            .replace({"": None, "nan": None, "NaN": None})
                            .dropna()
                        )
                        if not matches.empty:
                            ticker_value = matches.iloc[-1]

                new_trade = {
                    "trade_date": save_dt,
                    "asset": t_asset,
                    "asset_type": t_type,
                    "side": t_side,
                    "quantity": t_qty,
                    "price_trade_ccy": t_price,
                    "trade_ccy": t_ccy,
                    "charges_trade_ccy": t_fees,
                    "stt_trade_ccy": t_stt,
                    "fx_to_base": final_fx,
                    "ticker": ticker_value or "",
                    "last_price_trade_ccy": 0.0,
                    "last_price_base": 0.0,
                }

                combined = pd.concat(
                    [existing[INPUT_COLS] if not existing.empty else pd.DataFrame(columns=INPUT_COLS),
                     pd.DataFrame([new_trade])],
                    ignore_index=True,
                )
                save_trades(combined, DATA_FILE)
                st.success(f"Saved {t_side} {t_asset}!")
                st.rerun()


# --- TAB 3: HISTORY ---
with tab_history:
    spacer, auto_col = st.columns([5, 1])
    with auto_col:
        edit_mode = st.toggle("Enable Edit Mode")

    if trades_df.empty:
        st.info("No trades recorded.")
    elif not edit_mode:
        display_df = trades_df.copy()
        display_df["trade_date"] = display_df["trade_date"].dt.date
        display_df = display_df.sort_values("trade_date", ascending=False)

        display_df = display_df.copy()

        if hide_values:
            display_df["quantity"] = "••••••"
        else:
            display_df["quantity"] = display_df["quantity"].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_df[
                [
                    "trade_date",
                    "asset",
                    "side",
                    "quantity",
                    "price_trade_ccy",
                    "trade_ccy",
                    "fx_to_base",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "trade_date": "Date",
                "asset": "Asset",
                "side": "Side",
                "quantity": st.column_config.TextColumn("Quantity"),
                "price_trade_ccy": st.column_config.NumberColumn(
                    "Price", format="%.2f"
                ),
                "trade_ccy": "Trade Currency",
                "fx_to_base": st.column_config.NumberColumn(
                    "FX", format="%.4f"
                ),
            },
        )
    else:
        edited_df = st.data_editor(
            trades_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "trade_date": st.column_config.DatetimeColumn(
                    "Date", format="YYYY-MM-DD HH:mm"
                ),
                "side": "Side",
                "quantity": "Quantity",
                "trade_ccy": "Trade Currency",
            },
        )
        if st.button("Save Changes", type="primary"):
            save_trades(edited_df, DATA_FILE)
            st.success("History updated.")
            st.rerun()

# --- TAB 4: TAX / P&L ---
with tab_tax:
    today = datetime.today().date()
    default_start = date(today.year, 1, 1)

    # ========= 1. PERIOD + CURRENCY (COMPACT) =========
    top_col1, top_col2 = st.columns([2, 1])

    with top_col1:
        # Period presets instead of always-visible date inputs
        period_option = st.selectbox(
            "Period",
            ["Year to date", "Last calendar year", "Last 12 months", "Custom range"],
            index=0,
        )

        if period_option == "Year to date":
            fy_start = default_start
            fy_end = today

        elif period_option == "Last calendar year":
            last_year = today.year - 1
            fy_start = date(last_year, 1, 1)
            fy_end = date(last_year, 12, 31)

        elif period_option == "Last 12 months":
            fy_end = today
            # naive 12 months back; good enough for UI
            fy_start = date(today.year - (1 if today.month == 12 else 0),
                            today.month % 12 + 1, 1)
            # That’s a bit “calendar-ish” – if you prefer exact 365 days:
            # fy_start = today - timedelta(days=365)

        else:  # "Custom range"
            with st.expander("Custom date range", expanded=True):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    fy_start = st.date_input(
                        "Start",
                        value=default_start,
                    )
                with dcol2:
                    fy_end = st.date_input(
                        "End",
                        value=today,
                    )

        # Small summary line, very compact
        st.caption(f"From {fy_start} to {fy_end}")

    with top_col2:
        pnl_mode = st.radio(
            "Reporting Currency",
            [f"Base ({base_ccy})", "Trade Currency"],
            key="report_ccy_radio",
            horizontal=True,
        )

    start_dt = datetime.combine(fy_start, datetime.min.time())
    end_dt = datetime.combine(fy_end, datetime.max.time())

    # ========= 2. COMPUTE LOGIC =========
    if "Base" in pnl_mode:
        # Base currency view
        unit_col = "effective_unit_base"
        # SPECIAL RULE: for INR, exclude STT from both cost and proceeds
        if base_ccy.upper() == "INR":
            unit_col = "effective_unit_base_no_stt"

        pl_df = fifo_realized_for_period(
            trades_df, unit_col, start_dt, end_dt
        )

        # Currency formatting depending on base currency
        if base_ccy.upper() == "INR":
            cost_fmt = "₹ %.2f"
            pnl_fmt = "₹ %.2f"
        elif base_ccy.upper() == "SEK":
            cost_fmt = "kr %.2f"
            pnl_fmt = "kr %.2f"
        else:
            cost_fmt = f"{base_ccy} %.2f"
            pnl_fmt = f"{base_ccy} %.2f"

        header_ccy = base_ccy
        columns_to_show = [
            "asset",
            "quantity_sold",
            "buy_value",
            "sell_value",
            "realized",
        ]
        currency_col_config = {}

        if base_ccy.upper() == "INR":
            st.caption(
                "STT is excluded from both cost and proceeds when reporting realized P&L in INR."
            )

    else:
        # Trade Currency View
        pl_df = fifo_realized_for_period(
            trades_df, "effective_unit_trade", start_dt, end_dt
        )
        cost_fmt = "%.2f"
        pnl_fmt = "%.2f"
        header_ccy = "Traded CCY"
        columns_to_show = [
            "asset",
            "quantity_sold",
            "buy_value",
            "sell_value",
            "realized",
            "currency",
        ]
        currency_col_config = {
            "currency": st.column_config.TextColumn(
                "Currency", width="small"
            )
        }

    # ========= 3. RENDER TABLE + SUMMARY =========
    if pl_df.empty:
        st.info(f"No realized P&L found between {fy_start} and {fy_end}.")
    else:
        total_pl = pl_df["realized"].sum()
        pl_df = pl_df.sort_values("realized", ascending=False)

        # Slim Net P&L (instead of tall metric)
        if "Base" in pnl_mode:
            if hide_values:
                pl_value = "••••••"
            else:
                pl_value = f"{total_pl:,.2f}"

            net_label = f"Net P&L ({base_ccy})"
            st.markdown(
                f"<div style='font-size:0.85rem; opacity:0.8;'>{net_label}</div>"
                f"<div style='font-size:1.2rem; font-weight:700; margin-bottom:0.2rem;'>{pl_value}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "ℹ️ Total P&L is not summed here because assets may have different trade currencies."
            )

        base_column_config = {
            "asset": st.column_config.TextColumn("Asset"),
            "quantity_sold": st.column_config.TextColumn("Sold Qty"),
            "buy_value": st.column_config.TextColumn(f"Cost ({header_ccy})"),
            "sell_value": st.column_config.TextColumn(f"Proceeds ({header_ccy})"),
            "realized": st.column_config.TextColumn(f"P&L ({header_ccy})"),
        }

        final_column_config = {**base_column_config, **currency_col_config}

        display_pl = pl_df.copy()

        if hide_values:
            display_pl["quantity_sold"] = "••••••"
            display_pl["buy_value"] = "••••••"
            display_pl["sell_value"] = "••••••"
            display_pl["realized"] = "••••••"
        else:
            display_pl["quantity_sold"] = display_pl["quantity_sold"].apply(
                lambda x: f"{x:.2f}"
            )
            display_pl["buy_value"] = display_pl["buy_value"].apply(
                lambda x: f"{x:,.2f}"
            )
            display_pl["sell_value"] = display_pl["sell_value"].apply(
                lambda x: f"{x:,.2f}"
            )
            display_pl["realized"] = display_pl["realized"].apply(
                lambda x: f"{x:,.2f}"
            )

        st.dataframe(
            display_pl[columns_to_show],
            use_container_width=True,
            hide_index=True,
            column_config=final_column_config,
        )


























