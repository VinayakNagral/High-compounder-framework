"""High Compounder Framework v2.1

Streamlit stock screener for non-financial Indian listed companies.

Important: this is a research-screening tool, not investment advice. Yahoo Finance
fields are not authoritative; verify all investment-critical numbers against filings.
"""
from __future__ import annotations

import html
import io
import math
import time
from urllib.request import Request, urlopen
import warnings
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(
    page_title="High Compounder Framework v2.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1B2A4A"

# UPDATED D/E THRESHOLD
MAX_DEBT_TO_EQUITY = 1.0

BANKING_KW = (
    "bank", "finance", "insurance", "nbfc", "housing finance",
    "credit", "lending", "microfinance", "consumer finance",
)

TIER_MIDPOINT = {"FULL": 13.5, "STANDARD": 9.0, "HALF": 5.0, "WATCH": 0.0}
TIER_SIZE = {
    "FULL": "12–15%", "STANDARD": "8–10%", "HALF": "4–6%", "WATCH": "0%"
}

# =============================================================================
# PRESENTATION
# =============================================================================
st.markdown(
f"""
<style>
.main-title{{font-size:1.8rem;font-weight:750;color:{NAVY};}}
.sub-title{{font-size:.88rem;color:#6B7280;}}
.metric-card{{background:#fff;border:1px solid #E5E7EB;border-radius:9px;padding:.8rem;text-align:center;}}
.metric-val{{font-family:monospace;font-size:1.25rem;font-weight:700;color:{NAVY};}}
.metric-lbl{{font-size:.7rem;color:#6B7280;text-transform:uppercase;letter-spacing:.4px;}}
.layer-row{{padding:.5rem .75rem;border-radius:5px;margin:4px 0;border-left:3px solid;font-size:.86rem;}}
.layer-pass{{background:#ECFDF5;border-color:#059669;}}
.layer-fail{{background:#FEF2F2;border-color:#DC2626;}}
.layer-warn{{background:#FFFBEB;border-color:#D97706;}}
.verdict-box{{background:#F9FAFB;padding:1rem;border-radius:8px;border:1px solid #E5E7EB;line-height:1.7;}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GENERIC HELPERS
# =============================================================================
def is_valid(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def percentage_0_100(value: Any) -> float | None:
    value = safe_float(value)
    if value is None or value < 0 or value > 100:
        return None
    return value


def fmt(value: Any, spec: str = ".1f", suffix: str = "", prefix: str = "") -> str:
    if not is_valid(value):
        return "N/A"
    return f"{prefix}{float(value):{spec}}{suffix}"


def clean_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return ""
    return ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"


def fy_label(value: Any) -> str:
    try:
        dt = pd.Timestamp(value)
        fy = dt.year + 1 if dt.month > 3 else dt.year
        return f"FY{str(fy)[-2:]}"
    except Exception:
        return ""


def _normalise_label(label: Any) -> str:
    return " ".join(str(label).lower().replace("&", "and").replace("_", " ").split())


def safe_get(
    df: pd.DataFrame | None,
    labels: str | tuple[str, ...] | list[str],
    col: Any = 0
) -> float | None:
    """Read a financial statement row using exact/normalised aliases.

    `col` can be a positional integer or an actual column label.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    aliases = [labels] if isinstance(labels, str) else list(labels)

    try:
        if isinstance(col, int):
            if col < 0 or col >= len(df.columns):
                return None
            col_label = df.columns[col]
        else:
            col_label = col
            if col_label not in df.columns:
                return None

        index_lookup = {_normalise_label(x): x for x in df.index}

        for alias in aliases:
            actual = alias if alias in df.index else index_lookup.get(_normalise_label(alias))
            if actual is None:
                continue

            value = df.loc[actual, col_label]

            if isinstance(value, pd.Series):
                value = value.iloc[0]

            return safe_float(value)

    except Exception:
        return None

    return None


def _statement_date(col: Any) -> pd.Timestamp | None:
    try:
        dt = pd.Timestamp(col)
        return dt if not pd.isna(dt) else None
    except Exception:
        return None


def _match_statement_column(
    df: pd.DataFrame | None,
    target: Any,
    tolerance_days: int = 45
) -> Any | None:
    """Match statement columns by reporting date, not by column position."""

    if df is None or df.empty:
        return None

    target_dt = _statement_date(target)

    if target_dt is None:
        return None

    candidates = []

    for col in df.columns:
        col_dt = _statement_date(col)

        if col_dt is None:
            continue

        delta = abs((col_dt - target_dt).days)

        if delta == 0:
            return col

        if delta <= tolerance_days:
            candidates.append((delta, col))

    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def latest_statement_column(df: pd.DataFrame | None) -> Any | None:
    if df is None or df.empty:
        return None

    dated = [
        (pd.Timestamp(c), c)
        for c in df.columns
        if _statement_date(c) is not None
    ]

    return max(dated, key=lambda x: x[0])[1] if dated else df.columns[0]


# =============================================================================
# FINANCIAL FIELD GETTERS
# =============================================================================
def get_revenue(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    value = safe_get(
        fin,
        ("Total Revenue", "Operating Revenue", "Revenue", "Net Revenue"),
        col
    )

    return value if is_valid(value) and value > 0 else None


def get_net_income(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        fin,
        (
            "Net Income Common Stockholders",
            "Net Income",
            "Net Income Continuous Operations"
        ),
        col
    )


def get_ebitda(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(fin, ("EBITDA", "Normalized EBITDA"), col)


def get_ebit(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        fin,
        ("EBIT", "Operating Income", "Operating Profit", "Normalized EBIT"),
        col
    )


def get_cost_of_revenue(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    value = safe_get(
        fin,
        (
            "Cost Of Revenue",
            "Cost of Goods Sold",
            "Cost Of Goods And Services Sold"
        ),
        col
    )

    return abs(value) if is_valid(value) else None


def get_equity(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    value = safe_get(
        bs,
        (
            "Stockholders Equity",
            "Total Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ),
        col
    )

    return value if is_valid(value) and value > 0 else None


def get_total_assets(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(bs, "Total Assets", col)


def get_current_liabilities(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(bs, "Current Liabilities", col)


def get_cfo(cf: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        cf,
        (
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities"
        ),
        col
    )


def get_fcf(cf: pd.DataFrame | None, col: Any = 0) -> float | None:
    """FCF = CFO - absolute capex. Never substitute CFO when capex is missing."""

    reported = safe_get(cf, "Free Cash Flow", col)

    if is_valid(reported):
        return reported

    cfo = get_cfo(cf, col)
    capex = safe_get(cf, ("Capital Expenditure", "Capital Expenditures"), col)

    if is_valid(cfo) and is_valid(capex):
        return cfo - abs(capex)

    return None


def get_receivables(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        bs,
        (
            "Accounts Receivable",
            "Net Receivables",
            "Receivables",
            "Accounts Receivable Net"
        ),
        col
    )


def get_inventory(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(bs, ("Inventory", "Inventories"), col)


def get_payables(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        bs,
        (
            "Accounts Payable",
            "Payables And Accrued Expenses",
            "Current Accounts Payable"
        ),
        col
    )


def get_cash(bs: pd.DataFrame | None, col: Any = 0) -> float | None:
    return safe_get(
        bs,
        (
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash Financial",
            "Cash And Short Term Investments"
        ),
        col
    )


def get_total_debt(
    bs: pd.DataFrame | None,
    col: Any = 0
) -> tuple[float | None, str]:
    """Return gross interest-bearing debt; never substitute net debt for gross debt."""

    total = safe_get(
        bs,
        ("Total Debt", "Total Debt And Lease Obligations"),
        col
    )

    if is_valid(total) and total >= 0:
        return total, "total debt line"

    short_debt = safe_get(
        bs,
        (
            "Current Debt And Capital Lease Obligation",
            "Current Debt",
            "Short Term Debt",
            "Current Portion Of Long Term Debt"
        ),
        col
    )

    long_debt = safe_get(
        bs,
        (
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt"
        ),
        col
    )

    if (
        is_valid(short_debt)
        and is_valid(long_debt)
        and short_debt >= 0
        and long_debt >= 0
    ):
        return short_debt + long_debt, "short-term + long-term debt"

    return None, "gross debt unavailable"


def get_interest_expense(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    """Only accept gross interest/finance expense fields, not net interest income."""

    value = safe_get(
        fin,
        (
            "Interest Expense",
            "Interest Expense Non Operating",
            "Interest And Debt Expense",
            "Finance Costs"
        ),
        col
    )

    return abs(value) if is_valid(value) else None


def get_diluted_shares(
    fin: pd.DataFrame | None,
    qfin: pd.DataFrame | None,
    bs: pd.DataFrame | None,
    info: dict
) -> float | None:
    """Weighted-average diluted shares for EPS calculations only."""

    for frame in (qfin, fin):
        for field in ("Diluted Average Shares", "Basic Average Shares"):
            value = safe_get(frame, field, 0)

            if is_valid(value) and value > 0:
                return value

    # Last resort only; current shares are not identical to weighted-average shares.
    value = safe_float(info.get("sharesOutstanding"))

    return value if value and value > 0 else None


def get_current_shares(bs: pd.DataFrame | None, info: dict) -> float | None:
    """Current shares outstanding; do not use weighted-average EPS shares for market cap."""

    value = safe_float(info.get("sharesOutstanding"))

    if value and value > 0:
        return value

    for field in ("Ordinary Shares Number", "Share Issued"):
        value = safe_get(bs, field, 0)

        if is_valid(value) and value > 0:
            return value

    return None


def get_eps(fin: pd.DataFrame | None, col: Any = 0) -> float | None:
    value = safe_get(fin, "Diluted EPS", col)

    if value is None:
        value = safe_get(fin, "Basic EPS", col)

    return value


# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock(ticker: str) -> dict | None:
    debug = {
        "ticker": ticker,
        "warnings": [],
        "errors": [],
        "sources": []
    }

    try:
        security = yf.Ticker(ticker)

        try:
            info = dict(security.info or {})
        except Exception as exc:
            info = {}
            debug["warnings"].append(
                f"Yahoo info unavailable: {type(exc).__name__}"
            )

        statements: dict[str, pd.DataFrame | None] = {}

        statement_getters = (
            ("fin", ("income_stmt", "financials")),
            ("bs", ("balance_sheet",)),
            ("cf", ("cash_flow", "cashflow")),
            ("qfin", ("quarterly_income_stmt", "quarterly_financials")),
        )

        for key, attributes in statement_getters:
            raw = None
            last_error = None

            for attribute in attributes:
                try:
                    raw = getattr(security, attribute)

                    if raw is not None and not raw.empty:
                        debug["sources"].append(f"{key} from {attribute}")
                        break

                except Exception as exc:
                    last_error = exc

            statements[key] = raw.copy() if raw is not None and not raw.empty else None

            if statements[key] is None and last_error is not None:
                debug["warnings"].append(
                    f"{key} unavailable: {type(last_error).__name__}"
                )

        try:
            hist = security.history(period="1y", auto_adjust=True)

            prices = (
                hist["Close"].dropna()
                if hist is not None and not hist.empty and "Close" in hist
                else None
            )

        except Exception as exc:
            prices = None
            debug["warnings"].append(
                f"Price history unavailable: {type(exc).__name__}"
            )

        if (
            not is_valid(info.get("currentPrice"))
            and prices is not None
            and len(prices)
        ):
            info["currentPrice"] = float(prices.iloc[-1])
            debug["sources"].append("Current price from adjusted history")

        # Yahoo insider/institutional ownership is informational only.
        # It is NOT treated as Indian promoter shareholding.
        yahoo_holdings = {
            "insider_pct": _normalise_yahoo_pct(info.get("heldPercentInsiders")),
            "institutional_pct": _normalise_yahoo_pct(
                info.get("heldPercentInstitutions")
            ),
        }

        try:
            major = security.major_holders
            parsed = _parse_major_holders(major)

            for key, value in parsed.items():
                if value is not None:
                    yahoo_holdings[key] = value

        except Exception:
            pass

        if (
            not any(
                is_valid(info.get(k))
                for k in (
                    "currentPrice",
                    "regularMarketPrice",
                    "previousClose"
                )
            )
            and all(
                statements.get(k) is None
                for k in ("fin", "bs", "cf")
            )
        ):
            debug["errors"].append(
                "No usable price or financial statements returned"
            )
            return None

        debug["data_years"] = {
            "annual_income_statement": (
                statements["fin"].shape[1]
                if statements["fin"] is not None else 0
            ),
            "quarterly_income_statement": (
                statements["qfin"].shape[1]
                if statements["qfin"] is not None else 0
            ),
            "balance_sheet": (
                statements["bs"].shape[1]
                if statements["bs"] is not None else 0
            ),
            "cash_flow": (
                statements["cf"].shape[1]
                if statements["cf"] is not None else 0
            ),
            "price_days": len(prices) if prices is not None else 0,
        }

        return {
            **statements,
            "info": info,
            "prices": prices,
            "yahoo_holdings": yahoo_holdings,
            "debug": debug,
        }

    except Exception as exc:
        debug["errors"].append(
            f"Fetch failed: {type(exc).__name__}: {exc}"
        )
        return None


def _normalise_yahoo_pct(value: Any) -> float | None:
    value = safe_float(value)

    if value is None:
        return None

    if 0 <= value <= 1:
        return round(value * 100, 2)

    if 0 <= value <= 100:
        return round(value, 2)

    return None


def _parse_major_holders(frame: pd.DataFrame | None) -> dict:
    out = {
        "insider_pct": None,
        "institutional_pct": None
    }

    if frame is None or frame.empty:
        return out

    for idx, row in frame.iterrows():
        label_parts = [str(idx)] + [
            str(v) for v in row.values if isinstance(v, str)
        ]

        label = " ".join(label_parts).lower()

        numeric = [safe_float(v) for v in row.values]
        numeric = [v for v in numeric if v is not None]

        if not numeric:
            continue

        value = numeric[0]
        pct = _normalise_yahoo_pct(value)

        if pct is None:
            continue

        if "insider" in label:
            out["insider_pct"] = pct

        elif (
            "institution" in label
            and "float" not in label
            and "number" not in label
        ):
            out["institutional_pct"] = pct

    return out


# =============================================================================
# STATEMENT ALIGNMENT AND CORE METRICS
# =============================================================================
def build_multi(
    fin: pd.DataFrame | None,
    bs: pd.DataFrame | None,
    cf: pd.DataFrame | None
) -> tuple[list[dict], list[str]]:
    """Create annual records by matching actual statement reporting dates."""

    warnings_list: list[str] = []

    if fin is None or bs is None or cf is None:
        return [], ["One or more annual statements are unavailable"]

    records = []

    for fcol in fin.columns:
        dt = _statement_date(fcol)

        if dt is None:
            continue

        bcol = _match_statement_column(bs, fcol)
        ccol = _match_statement_column(cf, fcol)

        if bcol is None or ccol is None:
            warnings_list.append(
                f"Could not align all statements for {dt.date()}; period skipped"
            )
            continue

        records.append({
            "period_end": dt,
            "year": fy_label(dt) or str(dt.year),
            "revenue": get_revenue(fin, fcol),
            "net_income": get_net_income(fin, fcol),
            "eps": get_eps(fin, fcol),
            "ebitda": get_ebitda(fin, fcol),
            "ebit": get_ebit(fin, fcol),
            "cost_of_revenue": get_cost_of_revenue(fin, fcol),
            "receivables": get_receivables(bs, bcol),
            "inventory": get_inventory(bs, bcol),
            "equity": get_equity(bs, bcol),
            "current_liabilities": get_current_liabilities(bs, bcol),
            "total_assets": get_total_assets(bs, bcol),
            "cfo": get_cfo(cf, ccol),
            "payables": get_payables(bs, bcol),
            "interest_expense": get_interest_expense(fin, fcol),
            "cash": get_cash(bs, bcol),
            "debt": get_total_debt(bs, bcol)[0],
            "debt_source": get_total_debt(bs, bcol)[1],
        })

    records.sort(key=lambda x: x["period_end"])

    if len(records) < 2:
        warnings_list.append(
            "Fewer than two aligned annual periods are available"
        )

    # Compute ROE/ROCE using average capital where a prior period exists.
    for i, row in enumerate(records):
        prior = records[i - 1] if i > 0 else None

        avg_eq = (
            (row["equity"] + prior["equity"]) / 2
            if prior
            and is_valid(row["equity"])
            and is_valid(prior["equity"])
            else row["equity"]
        )

        ce_now = (
            row["total_assets"] - row["current_liabilities"]
            if is_valid(row["total_assets"])
            and is_valid(row["current_liabilities"])
            else None
        )

        ce_prior = (
            prior["total_assets"] - prior["current_liabilities"]
            if prior
            and is_valid(prior.get("total_assets"))
            and is_valid(prior.get("current_liabilities"))
            else None
        )

        avg_ce = (
            (ce_now + ce_prior) / 2
            if is_valid(ce_now) and is_valid(ce_prior)
            else ce_now
        )

        row["roe"] = (
            row["net_income"] / avg_eq * 100
            if is_valid(row["net_income"])
            and is_valid(avg_eq)
            and avg_eq > 0
            else None
        )

        row["roce"] = (
            row["ebit"] / avg_ce * 100
            if is_valid(row["ebit"])
            and is_valid(avg_ce)
            and avg_ce > 0
            else None
        )

    return records, warnings_list


def annual_cagr(
    frame: pd.DataFrame | None,
    getter: Callable,
    min_years: float = 2.5
) -> tuple[float | None, str, str]:
    """Calculate multi-year CAGR only with positive, complete observations."""

    if frame is None or frame.shape[1] < 3:
        return None, "", "Insufficient annual history"

    values = []

    for col in frame.columns:
        dt = _statement_date(col)
        value = getter(frame, col)

        if dt is not None and is_valid(value):
            values.append((dt, float(value)))

    values.sort(key=lambda x: x[0])

    if len(values) < 3:
        return None, "", "Fewer than three valid annual observations"

    start_dt, start_value = values[0]
    end_dt, end_value = values[-1]

    years = (end_dt - start_dt).days / 365.25

    if years < min_years or start_value <= 0 or end_value <= 0:
        return None, "", "Invalid or short annual period"

    if any(v <= 0 for _, v in values):
        return None, "", "Non-positive intermediate value; CAGR not meaningful"

    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100

    if not is_valid(cagr) or cagr <= -99 or cagr > 500:
        return None, "", "CAGR outside validation range"

    label = f"{fy_label(start_dt)}–{fy_label(end_dt)}"

    return (
        round(cagr, 1),
        label,
        f"Annual CAGR over {years:.1f} years"
    )


def quarterly_ttm_yoy(
    qfin: pd.DataFrame | None,
    getter: Callable
) -> tuple[float | None, str, str]:
    """Use latest 4 quarters vs preceding 4 quarters; never compare overlapping windows."""

    if qfin is None or qfin.shape[1] < 8:
        return None, "", "At least eight quarterly observations required"

    cols = sorted(
        [
            (_statement_date(c), c)
            for c in qfin.columns
            if _statement_date(c) is not None
        ],
        key=lambda x: x[0],
        reverse=True
    )

    if len(cols) < 8:
        return None, "", "At least eight dated quarterly observations required"

    latest_cols = [c for _, c in cols[:4]]
    prior_cols = [c for _, c in cols[4:8]]

    latest_vals = [getter(qfin, c) for c in latest_cols]
    prior_vals = [getter(qfin, c) for c in prior_cols]

    if any(
        not is_valid(v) or v <= 0
        for v in latest_vals + prior_vals
    ):
        return None, "", "Incomplete or non-positive TTM observations"

    latest_ttm = sum(latest_vals)
    prior_ttm = sum(prior_vals)

    if prior_ttm <= 0:
        return None, "", "Prior TTM denominator is non-positive"

    growth = (latest_ttm / prior_ttm - 1) * 100

    if is_valid(growth):
        return (
            round(growth, 1),
            "TTM YoY (1Y proxy)",
            "TTM growth proxy; not a multi-year CAGR"
        )

    return None, "", "Growth unavailable"


def current_price(data: dict) -> float | None:
    info = data["info"]

    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        value = safe_float(info.get(key))

        if value and value > 0:
            return value

    prices = data.get("prices")

    if prices is not None and len(prices):
        value = safe_float(prices.iloc[-1])
        return value if value and value > 0 else None

    return None


def current_market_cap(
    data: dict,
    price: float | None
) -> tuple[float | None, str]:
    reported = safe_float(data["info"].get("marketCap"))

    if reported is not None and reported > 0:
        shares = get_current_shares(data.get("bs"), data["info"])

        if shares and price:
            computed = shares * price

            if (
                computed > 0
                and max(reported / computed, computed / reported) > 3
            ):
                data["debug"]["warnings"].append(
                    "Reported market cap differs from price × current shares by more than 3x; verify share units"
                )

        return reported, "Yahoo reported market cap"

    shares = get_current_shares(data.get("bs"), data["info"])

    if price and shares:
        return price * shares, "Price × current shares"

    return None, "Market cap unavailable"


def _annual_pe_from_quarters(
    data: dict,
    price: float
) -> tuple[float | None, str]:
    qfin = data.get("qfin")

    if qfin is None or qfin.shape[1] < 4:
        return None, "not calculable"

    cols = sorted(
        [
            (_statement_date(c), c)
            for c in qfin.columns
            if _statement_date(c) is not None
        ],
        key=lambda x: x[0],
        reverse=True
    )

    if len(cols) < 4:
        return None, "not calculable"

    vals = [
        get_net_income(qfin, c)
        for _, c in cols[:4]
    ]

    if any(not is_valid(v) for v in vals):
        return None, "incomplete TTM net income"

    ttm_ni = sum(vals)

    shares = get_diluted_shares(
        data.get("fin"),
        qfin,
        data.get("bs"),
        data["info"]
    )

    if ttm_ni > 0 and shares and shares > 0:
        eps = ttm_ni / shares
        pe = price / eps if eps > 0 else None

        if is_valid(pe) and pe > 0:
            return round(pe, 1), "price / TTM diluted EPS"

    return None, "non-positive TTM earnings"


def extract_pe(
    data: dict,
    price: float | None
) -> tuple[float | None, str]:
    info = data["info"]

    trailing = safe_float(info.get("trailingPE"))

    if trailing is not None and trailing > 0:
        return round(trailing, 2), "Yahoo trailing P/E"

    if price and price > 0:
        pe, source = _annual_pe_from_quarters(data, price)

        if pe is not None:
            return pe, source

        forward = safe_float(info.get("forwardPE"))

        if forward is not None and forward > 0:
            return round(forward, 2), "Yahoo forward P/E (fallback)"

    return None, "P/E unavailable"


def extract_debt_to_equity(
    data: dict,
    multi: list[dict]
) -> tuple[float | None, str]:
    """Use statement-derived gross debt / positive equity; no ambiguous Yahoo unit conversion."""

    if not multi:
        return None, "No aligned balance sheet"

    latest = multi[-1]
    debt = latest.get("debt")
    equity = latest.get("equity")

    if (
        not is_valid(debt)
        or not is_valid(equity)
        or debt < 0
        or equity <= 0
    ):
        return None, latest.get("debt_source", "Gross debt/equity unavailable")

    return debt / equity, latest.get("debt_source", "Balance sheet")


def extract_interest_coverage(
    data: dict,
    multi: list[dict]
) -> tuple[float | None, str]:
    if multi:
        latest = multi[-1]
        ebit = latest.get("ebit")
        interest = latest.get("interest_expense")

        if is_valid(ebit) and is_valid(interest) and interest > 0:
            return ebit / interest, "EBIT / gross interest expense"

        if (
            is_valid(interest)
            and interest == 0
            and is_valid(ebit)
            and ebit >= 0
        ):
            return None, "Zero interest expense; coverage not meaningful"

    # Yahoo's precomputed ratio is a last-resort informational fallback.
    info_value = safe_float(data["info"].get("interestCoverage"))

    if info_value is not None and info_value >= 0:
        return info_value, "Yahoo interestCoverage fallback"

    return None, "Not calculable"


def extract_one_year_return(data: dict) -> float | None:
    prices = data.get("prices")

    if prices is None or len(prices) < 200:
        return None

    try:
        end_date = pd.Timestamp(prices.index[-1])
        target = end_date - pd.Timedelta(days=365)
        earlier = prices.loc[prices.index <= target]

        if earlier.empty:
            return None

        end_value = float(prices.iloc[-1])
        start_value = float(earlier.iloc[-1])

        if start_value <= 0:
            return None

        return (end_value / start_value - 1) * 100

    except Exception:
        return None


# =============================================================================
# FUNDAMENTAL SCREEN
# =============================================================================
def is_financial_company(info: dict, ticker: str = "") -> bool:
    sector = str(info.get("sector") or "").lower()
    industry = str(info.get("industry") or "").lower()
    name = str(
        info.get("shortName") or info.get("longName") or ticker
    ).lower()

    joined = f" {sector} {industry} {name} "

    return any(keyword in joined for keyword in BANKING_KW)


def run_layer1(
    data: dict,
    ticker: str,
    multi: list[dict]
) -> dict:
    info = data["info"]

    price = current_price(data)
    mcap, mcap_source = current_market_cap(data, price)
    mcap_cr = mcap / 1e7 if is_valid(mcap) else None

    latest = multi[-1] if multi else {}

    roe = latest.get("roe")
    roce = latest.get("roce")

    de, de_source = extract_debt_to_equity(data, multi)
    int_cov, int_cov_source = extract_interest_coverage(data, multi)

    sales, sales_label, sales_source = annual_cagr(
        data.get("fin"), get_revenue
    )

    if sales is None:
        sales, sales_label, sales_source = quarterly_ttm_yoy(
            data.get("qfin"), get_revenue
        )

    pat, pat_label, pat_source = annual_cagr(
        data.get("fin"), get_net_income
    )

    if pat is None:
        pat, pat_label, pat_source = quarterly_ttm_yoy(
            data.get("qfin"), get_net_income
        )

    eps_growth, eps_label, eps_source = annual_cagr(
        data.get("fin"), get_eps
    )

    if eps_growth is None:
        eps_growth, eps_label, eps_source = quarterly_ttm_yoy(
            data.get("qfin"), get_eps
        )

    pe, pe_source = extract_pe(data, price)

    growth_floor = 15.0

    relaxed = bool(
        is_valid(roe)
        and roe > 25
        and is_valid(mcap)
        and mcap > 500e9
    )

    if relaxed:
        growth_floor = 10.0

    checks = {
        "mcap_pass": is_valid(mcap) and mcap >= 150e9,
        "roe_pass": is_valid(roe) and roe > 15,
        "roce_pass": is_valid(roce) and roce > 18,

        # UPDATED: D/E must be strictly below 1.00x.
        "de_pass": (
            is_valid(de)
            and de >= 0
            and de < MAX_DEBT_TO_EQUITY
        ),

        "growth_pass": (
            is_valid(sales)
            and is_valid(pat)
            and sales > growth_floor
            and pat > growth_floor
        ),
    }

    fundamental_pass = all(checks.values())

    return {
        "ticker": ticker,
        "name": str(
            info.get("shortName")
            or info.get("longName")
            or ticker.replace(".NS", "")
        ),
        "sector": str(info.get("sector") or info.get("industry") or "Unknown"),
        "price": price,
        "mcap": mcap,
        "mcap_cr": mcap_cr,
        "mcap_source": mcap_source,
        "pe": pe,
        "pe_source": pe_source,
        "de": de,
        "de_source": de_source,
        "roe": roe,
        "roce": roce,
        "int_cov": int_cov,
        "int_cov_source": int_cov_source,
        "sales_cagr": sales,
        "sales_label": sales_label,
        "sales_source": sales_source,
        "pat_cagr": pat,
        "pat_label": pat_label,
        "pat_source": pat_source,
        "eps_growth": eps_growth,
        "eps_label": eps_label,
        "eps_source": eps_source,
        "growth_floor": growth_floor,
        "growth_relaxed": relaxed,
        **checks,
        "pass": fundamental_pass,
    }


# =============================================================================
# FORENSIC ACCOUNTING AND WORKING CAPITAL
# =============================================================================
def _ratio(num: Any, den: Any) -> float | None:
    if is_valid(num) and is_valid(den) and float(den) != 0:
        return float(num) / float(den)

    return None


def run_forensic(records: list[dict]) -> dict:
    if len(records) < 2:
        return {
            "score": None,
            "flags": ["Insufficient aligned annual data"],
            "num_flags": 1,
            "complete": False,
            "cum_cfo_pat": None,
            "cum_cfo_ebitda": None,
            "accrual_ratio": None,
            "ccc": None,
            "ccc_trend": None,
            "margins_by_year": [],
            "margin_trend": "unavailable",
            "cfo_ebitda_trend": []
        }

    rows = sorted(records, key=lambda x: x["period_end"])
    flags: list[str] = []
    deductions: dict[str, float] = {}

    # Receivables / revenue trend: require three consecutive aligned periods.
    receivables_pct = []

    for row in rows:
        ratio = _ratio(row.get("receivables"), row.get("revenue"))

        receivables_pct.append(
            ratio * 100
            if is_valid(ratio) and row.get("revenue", 0) > 0
            else None
        )

    valid_last3 = receivables_pct[-3:] if len(receivables_pct) >= 3 else []

    if (
        len(valid_last3) == 3
        and all(is_valid(x) for x in valid_last3)
        and valid_last3[0] < valid_last3[1] < valid_last3[2]
    ):
        flags.append(
            "Receivables as a percentage of revenue rose in three consecutive years"
        )
        deductions["receivables"] = 15

    # Inventory growth vs revenue growth, latest two aligned years.
    latest, prior = rows[-1], rows[-2]

    inv_latest, inv_prior = latest.get("inventory"), prior.get("inventory")
    rev_latest, rev_prior = latest.get("revenue"), prior.get("revenue")

    inv_growth = rev_growth = None

    if all(
        is_valid(x) and x > 0
        for x in (inv_latest, inv_prior, rev_latest, rev_prior)
    ):
        inv_growth = (inv_latest / inv_prior - 1) * 100
        rev_growth = (rev_latest / rev_prior - 1) * 100

        if inv_growth >= rev_growth + 10:
            flags.append(
                f"Inventory growth ({inv_growth:.1f}%) exceeded revenue growth ({rev_growth:.1f}%) by at least 10 percentage points"
            )
            deductions["inventory"] = 15

    # Multi-year cash conversion: paired observations only, >=3 years.
    paired_pat = [
        r for r in rows
        if is_valid(r.get("cfo")) and is_valid(r.get("net_income"))
    ]

    paired_ebitda = [
        r for r in rows
        if is_valid(r.get("cfo")) and is_valid(r.get("ebitda"))
    ]

    cum_cfo_pat = cum_cfo_ebitda = None
    cash_deductions = 0.0

    paired_years_sufficient = (
        len(paired_pat) >= 3 and len(paired_ebitda) >= 3
    )

    complete = False

    if len(paired_pat) >= 3:
        total_cfo_pat = sum(r["cfo"] for r in paired_pat)
        total_pat = sum(r["net_income"] for r in paired_pat)

        if total_pat > 0:
            cum_cfo_pat = total_cfo_pat / total_pat

            if cum_cfo_pat < 0.5:
                flags.append(
                    f"Critical cumulative CFO/PAT: {cum_cfo_pat:.2f}x"
                )
                cash_deductions += 30

            elif cum_cfo_pat < 0.7:
                flags.append(
                    f"Weak cumulative CFO/PAT: {cum_cfo_pat:.2f}x"
                )
                cash_deductions += 20

    if len(paired_ebitda) >= 3:
        total_cfo_ebitda = sum(r["cfo"] for r in paired_ebitda)
        total_ebitda = sum(r["ebitda"] for r in paired_ebitda)

        if total_ebitda > 0:
            cum_cfo_ebitda = total_cfo_ebitda / total_ebitda

            if cum_cfo_ebitda < 0.5:
                flags.append(
                    f"Critical cumulative CFO/EBITDA: {cum_cfo_ebitda:.2f}x"
                )
                cash_deductions += 20

            elif cum_cfo_ebitda < 0.7:
                flags.append(
                    f"Weak cumulative CFO/EBITDA: {cum_cfo_ebitda:.2f}x"
                )
                cash_deductions += 15

    cfo_ebitda_trend = []

    for row in rows:
        ratio = _ratio(row.get("cfo"), row.get("ebitda"))

        if is_valid(ratio) and row.get("ebitda", 0) > 0:
            cfo_ebitda_trend.append((row["year"], ratio))

    if (
        len(cfo_ebitda_trend) >= 2
        and is_valid(cum_cfo_ebitda)
        and cfo_ebitda_trend[-1][1] < cfo_ebitda_trend[-2][1]
        and cfo_ebitda_trend[-1][1] < 0.5
        and cum_cfo_ebitda < 0.7
    ):
        flags.append("CFO/EBITDA conversion deteriorated in the latest year")
        cash_deductions += 10

    if len([r for r in rows if is_valid(r.get("cfo"))]) >= 3:
        cumulative_cfo = sum(
            r["cfo"] for r in rows if is_valid(r.get("cfo"))
        )

        if cumulative_cfo < 0:
            flags.append("Cumulative operating cash flow is negative")
            cash_deductions += 25

    # Accrual ratio: latest year using average total assets.
    accrual_ratio = None

    if (
        is_valid(latest.get("net_income"))
        and is_valid(latest.get("cfo"))
    ):
        ta_now, ta_prior = (
            latest.get("total_assets"),
            prior.get("total_assets")
        )

        if (
            is_valid(ta_now)
            and is_valid(ta_prior)
            and ta_now > 0
            and ta_prior > 0
        ):
            avg_assets = (ta_now + ta_prior) / 2

            accrual_ratio = (
                latest["net_income"] - latest["cfo"]
            ) / avg_assets * 100

            if accrual_ratio > 20:
                flags.append(
                    f"High accrual ratio: {accrual_ratio:.1f}%"
                )
                cash_deductions += 20

            elif accrual_ratio > 10:
                flags.append(
                    f"Elevated accrual ratio: {accrual_ratio:.1f}%"
                )
                cash_deductions += 10

    # Avoid repeatedly subtracting more than 50 points for overlapping symptoms.
    cash_deductions = min(cash_deductions, 50)

    score = max(
        0.0,
        100.0
        - deductions.get("receivables", 0)
        - deductions.get("inventory", 0)
        - cash_deductions
    )

    complete = (
        paired_years_sufficient
        and is_valid(cum_cfo_pat)
        and is_valid(cum_cfo_ebitda)
    )

    # EBITDA margins.
    margins = []

    for row in rows:
        ratio = _ratio(row.get("ebitda"), row.get("revenue"))

        if is_valid(ratio) and row.get("revenue", 0) > 0:
            margins.append((row["year"], ratio * 100))

    margin_trend = "stable"

    if len(margins) >= 2:
        if margins[-1][1] > margins[0][1] + 2:
            margin_trend = "expanding"
        elif margins[-1][1] < margins[0][1] - 3:
            margin_trend = "contracting"

    # Cash conversion cycle: average balance sheet values.
    # DSO uses revenue; DIO and DPO use cost of revenue.
    ccc_rows = []

    for i, row in enumerate(rows):
        previous = rows[i - 1] if i > 0 else None

        if previous is None:
            continue

        revenue = row.get("revenue")
        cogs = row.get("cost_of_revenue")

        required = (
            row.get("receivables"),
            previous.get("receivables"),
            row.get("inventory"),
            previous.get("inventory"),
            row.get("payables"),
            previous.get("payables"),
            revenue,
            cogs
        )

        if not all(is_valid(v) for v in required) or revenue <= 0 or cogs <= 0:
            continue

        avg_receivables = (
            row["receivables"] + previous["receivables"]
        ) / 2

        avg_inventory = (
            row["inventory"] + previous["inventory"]
        ) / 2

        avg_payables = (
            row["payables"] + previous["payables"]
        ) / 2

        dso = avg_receivables / revenue * 365
        dio = avg_inventory / cogs * 365
        dpo = avg_payables / cogs * 365

        ccc_rows.append({
            "year": row["year"],
            "dso": dso,
            "dio": dio,
            "dpo": dpo,
            "ccc": dso + dio - dpo
        })

    ccc_latest = ccc_rows[-1] if ccc_rows else None
    ccc_prior = ccc_rows[-2] if len(ccc_rows) >= 2 else None
    ccc_trend = None

    if ccc_latest and ccc_prior:
        change = ccc_latest["ccc"] - ccc_prior["ccc"]

        ccc_trend = (
            "worsening" if change > 10
            else "improving" if change < -5
            else "stable"
        )

        if ccc_prior["dpo"] > 0:
            dpo_growth = (
                ccc_latest["dpo"] / ccc_prior["dpo"] - 1
            ) * 100

            if (
                dpo_growth > 30
                and is_valid(cum_cfo_pat)
                and cum_cfo_pat < 0.7
            ):
                flags.append(
                    f"Payables stretch: DPO increased {dpo_growth:.0f}% while cumulative CFO/PAT is weak"
                )

    return {
        "score": round(score),
        "flags": flags,
        "num_flags": len(flags),
        "complete": complete,
        "paired_pat_years": len(paired_pat),
        "paired_ebitda_years": len(paired_ebitda),
        "cum_cfo_pat": cum_cfo_pat,
        "cum_cfo_ebitda": cum_cfo_ebitda,
        "accrual_ratio": accrual_ratio,
        "cfo_ebitda_trend": cfo_ebitda_trend,
        "margins_by_year": margins,
        "margin_trend": margin_trend,
        "ccc": ccc_latest,
        "ccc_trend": ccc_trend,
        "receivables_pct": receivables_pct,
        "cash_deduction_capped": cash_deductions,
    }


# =============================================================================
# MOAT PROXY, CYCLICALITY, EPS MOMENTUM
# =============================================================================
def run_moat(records: list[dict]) -> dict:
    rows = sorted(records, key=lambda x: x["period_end"])

    roe_rows = [
        (r["year"], r.get("roe"))
        for r in rows
        if is_valid(r.get("roe"))
    ]

    if not roe_rows:
        return {
            "pct": 0,
            "consistency": "no data",
            "years_above_15": 0,
            "total_years": 0,
            "roe_by_year": []
        }

    values = [v for _, v in roe_rows]
    above = sum(v > 15 for v in values)
    total = len(values)
    pct = round(above / total * 100)

    if total >= 5 and above == total:
        consistency = "established moat proxy"

    elif (
        total >= 2
        and values[-1] > 15
        and values[-2] > 15
        and values[-1] > values[0]
    ):
        consistency = "emerging moat proxy"

    else:
        consistency = "weak/unproven moat"

    return {
        "pct": pct,
        "consistency": consistency,
        "years_above_15": above,
        "total_years": total,
        "roe_by_year": roe_rows,
    }


def run_cyclical(records: list[dict]) -> dict:
    rows = sorted(records, key=lambda x: x["period_end"])

    values = [
        (r["year"], r.get("roe"))
        for r in rows
        if is_valid(r.get("roe"))
    ]

    if len(values) < 4:
        return {
            "peak": False,
            "latest": values[-1][1] if values else None,
            "average_prior": None,
            "median": None,
            "roe_by_year": values,
            "note": "At least four annual ROE observations required"
        }

    historical = [v for _, v in values[:-1]]
    latest = values[-1][1]
    avg_prior = float(np.mean(historical))
    median = float(np.median(historical))

    peak = avg_prior > 0 and latest > 2 * avg_prior and latest > 20

    return {
        "peak": peak,
        "latest": latest,
        "average_prior": avg_prior,
        "median": median,
        "roe_by_year": values,
        "note": "Compared with prior-year average"
    }


def _quarter_eps(qfin: pd.DataFrame | None) -> list[tuple[pd.Timestamp, float]]:
    if qfin is None or qfin.empty:
        return []

    output = []

    for col in qfin.columns:
        dt = _statement_date(col)

        if dt is None:
            continue

        eps = get_eps(qfin, col)

        if is_valid(eps):
            output.append((dt, float(eps)))

    return sorted(output, key=lambda x: x[0], reverse=True)


def _yoy_eps(current: float, prior: float) -> tuple[float | None, str]:
    if prior > 0:
        return (current / prior - 1) * 100, "calculable"

    if current > 0 and prior <= 0:
        return None, "turned profitable from zero/loss"

    if current <= 0 and prior > 0:
        return (current / prior - 1) * 100, "profit to loss/decline"

    return None, "both periods loss-making or zero-base"


def run_momentum(qfin: pd.DataFrame | None) -> dict:
    eps_rows = _quarter_eps(qfin)

    if len(eps_rows) < 6:
        return {
            "available": False,
            "score": 40,
            "latest_yoy": None,
            "prior_yoy": None,
            "status": "Insufficient quarterly EPS history",
            "eps_rows": eps_rows
        }

    latest, prior_q = eps_rows[0][1], eps_rows[1][1]
    same_q_last_year, prior_same_q_last_year = eps_rows[4][1], eps_rows[5][1]

    latest_yoy, latest_status = _yoy_eps(latest, same_q_last_year)
    prior_yoy, prior_status = _yoy_eps(prior_q, prior_same_q_last_year)

    if is_valid(latest_yoy) and is_valid(prior_yoy):

        if latest_yoy > 0 and prior_yoy > 0:
            score = 85 if latest_yoy >= prior_yoy else 70

            status = (
                "Positive YoY EPS growth in both latest quarters"
                if score == 85
                else "Positive but decelerating YoY EPS growth"
            )

        elif latest_yoy < 0 and prior_yoy < 0:
            score, status = 15, "Negative YoY EPS growth in both latest quarters"

        else:
            score, status = 40, "Mixed YoY EPS momentum"

    elif (
        "turned profitable" in latest_status
        or "turned profitable" in prior_status
    ):
        score, status = 65, "Profitability transition; percentage growth not comparable"

    else:
        score, status = 40, "EPS growth not fully calculable"

    return {
        "available": True,
        "score": score,
        "latest_yoy": latest_yoy,
        "prior_yoy": prior_yoy,
        "latest_status": latest_status,
        "prior_status": prior_status,
        "status": status,
        "eps_rows": eps_rows[:6],
    }


# =============================================================================
# OWNERSHIP: YAHOO DATA IS INFORMATIONAL; PROMOTER DATA MUST BE VERIFIED
# =============================================================================
def load_ownership_overrides(uploaded_file) -> dict:
    if uploaded_file is None:
        return {}

    try:
        df = pd.read_csv(uploaded_file)
        required = {"ticker", "promoter_pct"}

        if not required.issubset({c.lower() for c in df.columns}):
            st.sidebar.error(
                "Ownership CSV needs ticker and promoter_pct columns."
            )
            return {}

        df.columns = [c.lower().strip() for c in df.columns]
        output = {}

        for _, row in df.iterrows():
            raw_ticker = row.get("ticker", "")

            if pd.isna(raw_ticker) or not str(raw_ticker).strip():
                continue

            ticker = clean_ticker(raw_ticker)

            output[ticker] = {
                "promoter_pct": percentage_0_100(row.get("promoter_pct")),
                "institutional_pct": percentage_0_100(row.get("institutional_pct")),
                "promoter_pledge_pct": percentage_0_100(row.get("promoter_pledge_pct")),
                "as_of": str(row.get("as_of", "")),
                "source": str(row.get("source", "User-provided CSV")),
            }

        return output

    except Exception as exc:
        st.sidebar.error(f"Could not read ownership CSV: {exc}")
        return {}


def run_holdings_analysis(
    yahoo_holdings: dict,
    override: dict | None
) -> dict:
    override = override or {}

    result = {
        "yahoo_insider_pct": yahoo_holdings.get("insider_pct"),
        "yahoo_institutional_pct": yahoo_holdings.get("institutional_pct"),
        "promoter_pct": percentage_0_100(override.get("promoter_pct")),
        "institutional_pct": percentage_0_100(override.get("institutional_pct")),
        "promoter_pledge_pct": percentage_0_100(override.get("promoter_pledge_pct")),
        "as_of": override.get("as_of", ""),
        "source": override.get("source", ""),
        "flags": [],
        "score_adj": 0,
    }

    promoter = result["promoter_pct"]
    inst = result["institutional_pct"]
    pledge = result["promoter_pledge_pct"]

    if is_valid(promoter):

        if promoter < 20:
            result["flags"].append(
                f"Verified promoter holding is low ({promoter:.1f}%); maximum STANDARD tier"
            )
            result["score_adj"] = -10

        elif promoter > 75:
            result["flags"].append(
                f"Verified promoter holding is highly concentrated ({promoter:.1f}%); review minority-shareholder governance"
            )
            result["score_adj"] = -5

    else:
        result["flags"].append(
            "Verified promoter holding unavailable; Yahoo insider ownership is not a substitute"
        )

    if is_valid(inst) and inst < 5:
        result["flags"].append(
            f"Verified institutional ownership is low ({inst:.1f}%)"
        )

    if is_valid(pledge) and pledge > 0:
        result["flags"].append(
            f"Promoter pledge reported at {pledge:.1f}% of promoter holding; manual review required"
        )

    return result


# =============================================================================
# VALUATION
# =============================================================================
def run_valuation(
    pe: float | None,
    eps_growth: float | None,
    sales_growth: float | None,
    data: dict,
    market_cap: float | None
) -> dict:
    peg = None
    growth_used = None
    turnaround_adjustment = False

    # PEG uses per-share earnings growth, not total PAT growth.
    if (
        is_valid(pe)
        and pe > 0
        and is_valid(eps_growth)
        and eps_growth > 0
    ):
        growth_used = eps_growth

        if (
            is_valid(sales_growth)
            and sales_growth > 0
            and eps_growth > 2 * sales_growth
        ):
            growth_used = sales_growth
            turnaround_adjustment = True

        peg = pe / growth_used

    fcf_yield = None
    cf = data.get("cf")

    if is_valid(market_cap) and market_cap > 0 and cf is not None:
        fcf = get_fcf(cf, latest_statement_column(cf))

        if is_valid(fcf):
            fcf_yield = fcf / market_cap * 100

    return {
        "peg": round(peg, 2) if is_valid(peg) else None,
        "peg_growth": round(growth_used, 1) if is_valid(growth_used) else None,
        "peg_turnaround": turnaround_adjustment,
        "fcf_yield": round(fcf_yield, 2) if is_valid(fcf_yield) else None,
        "pe_extreme": is_valid(pe) and pe > 80,
    }


# =============================================================================
# ALLOCATION AND COMPOSITE RANKING
# =============================================================================
def get_tier(
    forensic_score: float | None,
    num_flags: int,
    cyclical_peak: bool,
    peg: float | None,
    one_year_return: float | None,
    moat_consistency: str,
    holdings: dict,
    interest_coverage: float | None,
    debt_to_equity: float | None,
    fundamental_pass: bool,
    forensic_complete: bool
) -> tuple[str, str, str]:
    """Deterministic tiering. Fundamental failures remain capped at HALF in single-stock review."""

    if forensic_score is None:
        return (
            "WATCH",
            TIER_SIZE["WATCH"],
            "Insufficient aligned financial data for a reliable forensic score"
        )

    if forensic_score < 50:
        return "WATCH", TIER_SIZE["WATCH"], "Forensic score below 50"

    if is_valid(peg) and peg > 5:
        return "WATCH", TIER_SIZE["WATCH"], "PEG above 5.0"

    if not is_valid(peg):
        base = "HALF"
    elif peg > 1.5:
        base = "HALF"

    elif (
        forensic_score >= 85
        and num_flags == 0
        and not cyclical_peak
        and moat_consistency in (
            "established moat proxy",
            "emerging moat proxy"
        )
        and peg <= 1.0
        and forensic_complete
    ):
        base = "FULL"

    elif forensic_score >= 70 and num_flags <= 1 and peg <= 1.5:
        base = "STANDARD"

    else:
        base = "HALF"

    reasons = [f"Base assessment: {base}"]

    rank = {
        "WATCH": 0,
        "HALF": 1,
        "STANDARD": 2,
        "FULL": 3
    }

    def cap_to(maximum: str, reason: str):
        nonlocal base

        if rank[base] > rank[maximum]:
            base = maximum
            reasons.append(reason)

    if not fundamental_pass:
        cap_to("HALF", "One or more fundamental hard filters failed")

    if is_valid(interest_coverage) and interest_coverage < 3:
        cap_to("HALF", "Interest coverage below 3x")

    elif (
        not is_valid(interest_coverage)
        and is_valid(debt_to_equity)
        and debt_to_equity >= 0.1
    ):
        cap_to(
            "STANDARD",
            "Interest coverage unavailable despite material debt"
        )

    if is_valid(peg) and peg > 3:
        cap_to("HALF", "PEG above 3.0")

    if not is_valid(peg):
        cap_to("HALF", "PEG unavailable")

    if not forensic_complete:
        cap_to(
            "STANDARD",
            "Limited multi-year forensic data; FULL tier unavailable"
        )

    if holdings.get("score_adj", 0) <= -10:
        cap_to("STANDARD", "Verified promoter holding below 20%")

    if (
        base == "FULL"
        and is_valid(one_year_return)
        and one_year_return < -30
    ):
        cap_to(
            "HALF",
            "One-year share-price return below -30%; investment thesis requires review"
        )

    return base, TIER_SIZE[base], "; ".join(reasons)


def score_peg(peg: float | None) -> float:
    if not is_valid(peg) or peg <= 0:
        return 0.0

    if peg <= 0.5:
        return 100.0

    if peg <= 1.0:
        return 100 - (peg - 0.5) * 40

    if peg <= 1.5:
        return 80 - (peg - 1.0) * 40

    if peg <= 2.0:
        return 60 - (peg - 1.5) * 40

    if peg <= 3.0:
        return 40 - (peg - 2.0) * 30

    return max(0.0, 10 - (peg - 3.0) * 5)


def score_cfo_quality(cfo_pat: float | None) -> float:
    if not is_valid(cfo_pat):
        return 0.0

    if cfo_pat >= 1.2:
        return 100.0

    if cfo_pat >= 1.0:
        return 90.0

    if cfo_pat >= 0.7:
        return 65.0 + (cfo_pat - 0.7) / 0.3 * 25.0

    if cfo_pat >= 0.5:
        return 35.0 + (cfo_pat - 0.5) / 0.2 * 30.0

    return max(0.0, cfo_pat / 0.5 * 35.0)


def score_roe(roe: float | None) -> float:
    if not is_valid(roe):
        return 0.0

    # 5% ROE -> 0; 35% or higher -> 100.
    return float(np.clip((roe - 5) / 30 * 100, 0, 100))


def composite_score(
    l1: dict,
    acct: dict,
    moat: dict,
    momentum: dict,
    val: dict
) -> float:
    components = {
        "peg": score_peg(val.get("peg")),
        "forensic": float(acct.get("score") or 0),
        "momentum": float(momentum.get("score", 40)),
        "cfo": score_cfo_quality(acct.get("cum_cfo_pat")),
        "moat": float(moat.get("pct", 0)),
        "roe": score_roe(l1.get("roe")),
    }

    weights = {
        "peg": .25,
        "forensic": .20,
        "momentum": .15,
        "cfo": .15,
        "moat": .15,
        "roe": .10
    }

    return round(
        sum(components[k] * weights[k] for k in weights),
        2
    )


# =============================================================================
# OWNERSHIP / UNIVERSE FILES
# =============================================================================
def get_universe_tickers(
    max_stocks: int,
    uploaded_csv=None
) -> tuple[list[str], str]:

    if uploaded_csv is not None:
        try:
            frame = pd.read_csv(uploaded_csv)

            symbol_col = next(
                (
                    c for c in frame.columns
                    if str(c).strip().lower() in ("symbol", "ticker")
                ),
                None
            )

            if symbol_col is None:
                raise ValueError(
                    "Uploaded CSV must contain a Symbol or Ticker column"
                )

            symbols = [
                clean_ticker(x)
                for x in frame[symbol_col].dropna().tolist()
            ]

            symbols = list(dict.fromkeys(x for x in symbols if x))

            if not symbols:
                raise ValueError("No valid symbols found")

            if max_stocks < len(symbols):
                selected = pd.Series(symbols).sample(
                    n=max_stocks,
                    random_state=42
                ).tolist()

                return (
                    selected,
                    f"User-uploaded constituent CSV · deterministic sample of {max_stocks}/{len(frame)}"
                )

            return symbols[:max_stocks], "User-uploaded constituent CSV"

        except Exception as exc:
            raise ValueError(
                f"Could not parse universe CSV: {exc}"
            ) from exc

    urls = [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]

    errors = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,*/*;q=0.8",
        "Referer": "https://www.niftyindices.com/indices/equity/broad-based-indices",
    }

    for url in urls:
        try:
            request = Request(url, headers=headers)

            with urlopen(request, timeout=15) as response:
                payload = response.read()

            frame = pd.read_csv(io.BytesIO(payload))

            symbol_col = next(
                (
                    c for c in frame.columns
                    if str(c).strip().lower() == "symbol"
                ),
                None
            )

            if symbol_col and len(frame) >= 450:
                symbols = list(
                    dict.fromkeys(
                        clean_ticker(x)
                        for x in frame[symbol_col].dropna().tolist()
                    )
                )

                if max_stocks < len(symbols):
                    symbols = pd.Series(symbols).sample(
                        n=max_stocks,
                        random_state=42
                    ).tolist()

                    source_label = (
                        f"Live Nifty 500 CSV · deterministic sample "
                        f"of {max_stocks}/{len(frame)}"
                    )

                    return symbols, source_label

                return symbols[:max_stocks], "Live Nifty 500 constituent CSV"

            errors.append(
                f"{url}: unexpected CSV schema or fewer than 450 rows"
            )

        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}")

    raise RuntimeError(
        "Live Nifty 500 download failed. Upload the latest official "
        "Nifty 500 CSV. " + "; ".join(errors)
    )


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================
def analyse_stock(
    ticker: str,
    ownership_overrides: dict | None = None
) -> dict | None:

    ticker = clean_ticker(ticker)
    data = fetch_stock(ticker)

    if not data:
        return None

    if is_financial_company(data["info"], ticker):
        return {
            "ticker": ticker,
            "excluded": True,
            "excluded_reason": (
                "Financial company; use a separate financial-sector framework"
            )
        }

    multi, alignment_warnings = build_multi(
        data.get("fin"),
        data.get("bs"),
        data.get("cf")
    )

    data["debug"]["warnings"].extend(alignment_warnings)

    l1 = run_layer1(data, ticker, multi)
    acct = run_forensic(multi)
    moat = run_moat(multi)
    cyc = run_cyclical(multi)
    momentum = run_momentum(data.get("qfin"))
    one_year_return = extract_one_year_return(data)

    val = run_valuation(
        l1.get("pe"),
        l1.get("eps_growth"),
        l1.get("sales_cagr"),
        data,
        l1.get("mcap")
    )

    override = (ownership_overrides or {}).get(ticker, {})

    holdings = run_holdings_analysis(
        data.get("yahoo_holdings", {}),
        override
    )

    tier, size, tier_reason = get_tier(
        acct.get("score"),
        acct.get("num_flags", 0),
        cyc.get("peak", False),
        val.get("peg"),
        one_year_return,
        moat.get("consistency", ""),
        holdings,
        l1.get("int_cov"),
        l1.get("de"),
        l1.get("pass", False),
        acct.get("complete", False)
    )

    composite = composite_score(
        l1, acct, moat, momentum, val
    )

    return {
        "ticker": ticker,
        "data": data,
        "multi": multi,
        "l1": l1,
        "acct": acct,
        "moat": moat,
        "cyc": cyc,
        "momentum": momentum,
        "return_1y": one_year_return,
        "valuation": val,
        "holdings": holdings,
        "tier": tier,
        "size": size,
        "tier_reason": tier_reason,
        "composite": composite,
        "excluded": False,
    }


def result_to_row(result: dict) -> dict:
    l1 = result["l1"]
    acct = result["acct"]
    moat = result["moat"]
    val = result["valuation"]
    mom = result["momentum"]
    holdings = result["holdings"]

    latest_period = result["multi"][-1] if result.get("multi") else {}

    net_debt_cr = None

    if (
        is_valid(latest_period.get("debt"))
        and is_valid(latest_period.get("cash"))
    ):
        net_debt_cr = (
            latest_period["debt"] - latest_period["cash"]
        ) / 1e7

    return {
        "ticker": result["ticker"].replace(".NS", "").replace(".BO", ""),
        "name": l1["name"],
        "sector": l1["sector"],
        "price": l1["price"],
        "market_cap_cr": l1["mcap_cr"],
        "pe": l1["pe"],
        "peg": val["peg"],
        "roe": l1["roe"],
        "roce": l1["roce"],
        "de": l1["de"],
        "sales_cagr": l1["sales_cagr"],
        "pat_cagr": l1["pat_cagr"],
        "eps_growth": l1["eps_growth"],
        "score": acct["score"],
        "flags": acct["num_flags"],
        "cfo_pat": acct["cum_cfo_pat"],
        "cfo_ebitda": acct["cum_cfo_ebitda"],
        "accrual_pct": acct["accrual_ratio"],
        "moat_pct": moat["pct"],
        "moat_status": moat["consistency"],
        "earnings_momentum_score": mom["score"],
        "eps_yoy": mom.get("latest_yoy"),
        "fcf_yield": val["fcf_yield"],
        "interest_coverage": l1["int_cov"],
        "net_debt_cr": net_debt_cr,
        "promoter_pct": holdings["promoter_pct"],
        "promoter_pledge_pct": holdings["promoter_pledge_pct"],
        "return_1y": result["return_1y"],
        "tier": result["tier"],
        "size": result["size"],
        "tier_reason": result["tier_reason"],
        "composite": result["composite"],
    }


# =============================================================================
# LAYER DISPLAY AND VERDICT
# =============================================================================
def layer_rows(result: dict) -> list[tuple[str, str, str]]:
    l1 = result["l1"]
    acct = result["acct"]
    moat = result["moat"]
    val = result["valuation"]
    mom = result["momentum"]

    rows = []

    checks = [
        (
            "Market capitalisation",
            l1["mcap_pass"],
            f"₹{fmt(l1['mcap_cr'], ',.0f')} Cr; minimum ₹15,000 Cr"
        ),
        (
            "ROE",
            l1["roe_pass"],
            f"{fmt(l1['roe'], '.1f')}%; threshold >15%"
        ),
        (
            "ROCE",
            l1["roce_pass"],
            f"{fmt(l1['roce'], '.1f')}%; threshold >18%"
        ),
        (
            "Gross debt / equity",
            l1["de_pass"],
            f"{fmt(l1['de'], '.2f')}x; threshold <{MAX_DEBT_TO_EQUITY:.2f}x ({l1['de_source']})"
        ),
        (
            "Revenue and PAT growth",
            l1["growth_pass"],
            f"Revenue {fmt(l1['sales_cagr'], '.1f')}% ({l1['sales_label']}); PAT {fmt(l1['pat_cagr'], '.1f')}% ({l1['pat_label']}); floor {l1['growth_floor']}%"
        ),
    ]

    for label, passed, detail in checks:
        rows.append(
            (label, "pass" if passed else "fail", detail)
        )

    ic = l1.get("int_cov")

    if is_valid(ic):
        rows.append((
            "Interest coverage",
            "fail" if ic < 3 else "warn" if ic < 8 else "pass",
            f"{ic:.1f}x ({l1['int_cov_source']}); below 3x caps allocation at HALF"
        ))
    else:
        rows.append((
            "Interest coverage",
            "warn",
            l1["int_cov_source"]
        ))

    rows.append((
        "Forensic accounting",
        "pass" if acct.get("score", 0) >= 85 else "warn" if acct.get("score", 0) >= 50 else "fail",
        f"Score {fmt(acct.get('score'), '.0f')}/100; {acct.get('num_flags', 0)} flags; multi-year data {'complete' if acct.get('complete') else 'limited'}"
    ))

    if acct.get("ccc"):
        ccc = acct["ccc"]

        rows.append((
            "Cash conversion cycle",
            "warn" if acct.get("ccc_trend") == "worsening" else "pass",
            f"{ccc['ccc']:.1f} days; DSO {ccc['dso']:.1f}, DIO {ccc['dio']:.1f}, DPO {ccc['dpo']:.1f}; {acct.get('ccc_trend') or 'trend unavailable'}"
        ))

    else:
        rows.append((
            "Cash conversion cycle",
            "warn",
            "Not calculable; average balances and COGS are required"
        ))

    rows.append((
        "ROE consistency / moat proxy",
        "pass" if moat["consistency"] == "established moat proxy" else "warn" if moat["consistency"] == "emerging moat proxy" else "fail",
        f"ROE >15% in {moat['years_above_15']}/{moat['total_years']} years; {moat['consistency']} — qualitative moat research still required"
    ))

    hold = result["holdings"]

    rows.append((
        "Ownership",
        "warn" if hold["flags"] else "pass",
        "; ".join(hold["flags"]) if hold["flags"] else "No manual ownership warning; verify source and date"
    ))

    peg = val.get("peg")

    rows.append((
        "PEG valuation",
        "pass" if is_valid(peg) and peg <= 1 else "warn" if is_valid(peg) and peg <= 1.5 else "fail" if is_valid(peg) else "warn",
        f"PEG {fmt(peg, '.2f')}; based on diluted EPS growth {fmt(val.get('peg_growth'), '.1f')}%"
    ))

    fcf = val.get("fcf_yield")

    rows.append((
        "FCF yield",
        "pass" if is_valid(fcf) and fcf > 3 else "warn",
        f"{fmt(fcf, '.1f')}%; negative FCF requires context; missing capex is not treated as zero"
    ))

    if val.get("pe_extreme"):
        rows.append((
            "Extreme P/E warning",
            "warn",
            f"P/E {l1['pe']:.1f}x exceeds 80x; warning only, not an automatic rejection"
        ))

    rows.append((
        "Earnings momentum",
        "pass" if mom.get("score", 40) >= 70 else "fail" if mom.get("score", 40) <= 20 else "warn",
        f"{mom.get('status')}; latest YoY EPS {fmt(mom.get('latest_yoy'), '+.1f')}%"
    ))

    cyc = result["cyc"]

    rows.append((
        "Cyclical ROE",
        "warn" if cyc.get("peak") else "pass",
        cyc.get("note", "")
    ))

    rows.append((
        "One-year price return",
        "fail" if is_valid(result["return_1y"]) and result["return_1y"] < -30 else "warn" if not is_valid(result["return_1y"]) else "pass",
        f"{fmt(result['return_1y'], '+.1f')}%; a decline triggers thesis review"
    ))

    return rows


def build_verdict(result: dict) -> str:
    l1 = result["l1"]
    acct = result["acct"]
    val = result["valuation"]
    tier = result["tier"]
    size = result["size"]

    opening = {
        "FULL": "This company meets the framework's highest quantitative conviction requirements.",
        "STANDARD": "This company clears much of the quality assessment but does not meet every FULL-tier condition.",
        "HALF": "This company has a mixed profile and should receive limited exposure, if any, pending further research.",
        "WATCH": "The framework does not currently support deploying capital in this company.",
    }[tier]

    details = [opening]

    details.append(
        f"ROE is {fmt(l1['roe'], '.1f')}%, ROCE is {fmt(l1['roce'], '.1f')}%, gross D/E is {fmt(l1['de'], '.2f')}x, and forensic score is {fmt(acct['score'], '.0f')}/100."
    )

    details.append(
        f"Revenue CAGR is {fmt(l1['sales_cagr'], '.1f')}%, PAT CAGR is {fmt(l1['pat_cagr'], '.1f')}%, and diluted EPS growth used for PEG is {fmt(l1['eps_growth'], '.1f')}%."
    )

    details.append(
        f"PEG is {fmt(val['peg'], '.2f')} and FCF yield is {fmt(val['fcf_yield'], '.1f')}%."
    )

    if acct.get("flags"):
        details.append(
            "Forensic / working-capital flags: "
            + "; ".join(acct["flags"][:4]) + "."
        )

    if result["holdings"].get("flags"):
        details.append(
            "Ownership review: "
            + "; ".join(result["holdings"]["flags"][:3]) + "."
        )

    details.append(
        f"Allocation rule: {result['tier_reason']}. Suggested tier size: {size}."
    )

    details.append(
        "This is a screening output, not a final investment recommendation. Review annual reports, concalls, governance, industry structure and valuation assumptions before investing."
    )

    return " ".join(details)


def build_scorecard_html(result: dict) -> str:
    """Create a printable one-page HTML summary for a single company."""

    l1 = result["l1"]
    acct = result["acct"]
    val = result["valuation"]

    rows = []

    for label, status, detail in layer_rows(result):
        status_text = {
            "pass": "PASS",
            "warn": "REVIEW",
            "fail": "FAIL"
        }.get(status, status.upper())

        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td><strong>{status_text}</strong></td>"
            f"<td>{html.escape(detail)}</td>"
            "</tr>"
        )

    metrics = [
        ("P/E", fmt(l1.get("pe"), ".1f")),
        ("PEG", fmt(val.get("peg"), ".2f")),
        ("ROE", fmt(l1.get("roe"), ".1f", "%")),
        ("ROCE", fmt(l1.get("roce"), ".1f", "%")),
        ("Debt / equity", fmt(l1.get("de"), ".2f", "x")),
        ("Forensic score", fmt(acct.get("score"), ".0f") + "/100"),
        ("FCF yield", fmt(val.get("fcf_yield"), ".1f", "%")),
        ("1Y return", fmt(result.get("return_1y"), "+.1f", "%")),
    ]

    metric_html = "".join(
        f"<td><small>{html.escape(label)}</small><br><b>{html.escape(value)}</b></td>"
        for label, value in metrics
    )

    verdict = html.escape(build_verdict(result))

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(l1['name'])} — High Compounder</title>
    <style>body{{font-family:Arial,sans-serif;color:#172033;margin:28px}}h1{{color:{NAVY}}}table{{border-collapse:collapse;width:100%;margin:14px 0}}td,th{{border:1px solid #ddd;padding:8px;text-align:left;font-size:12px}}small{{color:#687386}}.tier{{display:inline-block;padding:6px 12px;border-radius:14px;background:#eaf0f8;font-weight:bold}}.verdict{{padding:14px;background:#f7f8fa;border:1px solid #ddd;line-height:1.6}}@media print{{body{{margin:10mm}}}}</style></head><body>
    <h1>High Compounder Framework v2.1</h1><h2>{html.escape(l1['name'])}</h2>
    <p>{html.escape(result['ticker'])} · {html.escape(l1['sector'])} · CMP ₹{fmt(l1.get('price'), ',.2f')} · {datetime.now():%d %b %Y}</p>
    <p class='tier'>{html.escape(result['tier'])} · {html.escape(result['size'])}</p>
    <table><tr>{metric_html}</tr></table><h3>Layer results</h3>
    <table><tr><th>Layer</th><th>Result</th><th>Details</th></tr>{''.join(rows)}</table>
    <h3>Investment assessment</h3><div class='verdict'>{verdict}</div>
    <p style='font-size:10px;color:#777;margin-top:20px'>Quantitative screening output, not investment advice. Verify data against company filings and conduct independent research.</p>
    </body></html>"""


# =============================================================================
# BATCH SCREENER AND SECTOR CONSTRAINT
# =============================================================================
def analyse_for_funnel(
    ticker: str,
    ownership_overrides: dict
) -> tuple[dict | None, str]:

    data = fetch_stock(ticker)

    if not data:
        return None, "fetch_fail"

    if is_financial_company(data["info"], ticker):
        return None, "financial"

    multi, align_warnings = build_multi(
        data.get("fin"),
        data.get("bs"),
        data.get("cf")
    )

    if len(multi) < 2:
        return None, "data"

    data["debug"]["warnings"].extend(align_warnings)

    l1 = run_layer1(data, ticker, multi)

    # Batch mode is a strict hard-filter screen.
    # Single-stock mode can show exceptions capped at HALF.
    for key, reason in (
        ("mcap_pass", "mcap"),
        ("de_pass", "debt"),
        ("roe_pass", "roe"),
        ("roce_pass", "roce"),
        ("growth_pass", "growth")
    ):
        if not l1[key]:
            return None, reason

    acct = run_forensic(multi)

    if acct.get("score") is None:
        return None, "data"

    moat = run_moat(multi)
    cyc = run_cyclical(multi)
    momentum = run_momentum(data.get("qfin"))
    ret = extract_one_year_return(data)

    val = run_valuation(
        l1["pe"],
        l1["eps_growth"],
        l1["sales_cagr"],
        data,
        l1["mcap"]
    )

    holdings = run_holdings_analysis(
        data.get("yahoo_holdings", {}),
        ownership_overrides.get(ticker, {})
    )

    tier, size, tier_reason = get_tier(
        acct["score"],
        acct["num_flags"],
        cyc.get("peak", False),
        val.get("peg"),
        ret,
        moat["consistency"],
        holdings,
        l1.get("int_cov"),
        l1.get("de"),
        l1["pass"],
        acct.get("complete", False)
    )

    result = {
        "ticker": ticker,
        "data": data,
        "multi": multi,
        "l1": l1,
        "acct": acct,
        "moat": moat,
        "cyc": cyc,
        "momentum": momentum,
        "return_1y": ret,
        "valuation": val,
        "holdings": holdings,
        "tier": tier,
        "size": size,
        "tier_reason": tier_reason,
        "composite": composite_score(l1, acct, moat, momentum, val),
        "excluded": False,
    }

    return result, "pass"


def select_with_sector_limit(
    candidates: pd.DataFrame,
    top_n: int,
    max_sector_pct: float = 30.0
) -> pd.DataFrame:
    """Keep top-ranked candidates and replace weakest overrepresented-sector names.

    The cap is measured as a share of the selected portfolio's midpoint allocation.
    If there are not enough alternative sectors, the function keeps the best available
    names and the UI reports that the cap could not be fully met.
    """

    if candidates.empty:
        return candidates

    ranked = candidates.sort_values("composite", ascending=False).copy()
    selected = ranked.head(top_n).copy()
    remaining = ranked.loc[~ranked.index.isin(selected.index)].copy()

    def sector_weights(frame: pd.DataFrame) -> tuple[dict, float]:
        alloc = frame["tier"].map(TIER_MIDPOINT).fillna(0.0)
        total = float(alloc.sum())
        grouped = alloc.groupby(frame["sector"]).sum().to_dict()
        return grouped, total

    max_iter = len(ranked) + 1

    for _ in range(max_iter):
        grouped, total = sector_weights(selected)

        if total <= 0:
            break

        over = [
            (sector, amount / total * 100)
            for sector, amount in grouped.items()
            if amount / total * 100 > max_sector_pct
        ]

        if not over:
            break

        over_sector = max(over, key=lambda x: x[1])[0]

        sector_rows = selected[
            selected["sector"] == over_sector
        ].sort_values("composite", ascending=True)

        if sector_rows.empty:
            break

        weakest_idx = sector_rows.index[0]
        replacement = None
        best_share = float("inf")

        current_share = grouped.get(over_sector, 0.0) / total * 100

        for idx, candidate in remaining.sort_values(
            "composite",
            ascending=False
        ).iterrows():

            if candidate["sector"] == over_sector:
                continue

            test = pd.concat([
                selected.drop(index=weakest_idx),
                candidate.to_frame().T
            ])

            weights, test_total = sector_weights(test)

            if test_total <= 0:
                continue

            proposed_share = weights.get(over_sector, 0.0) / test_total * 100

            if proposed_share < current_share and proposed_share < best_share:
                replacement = (idx, candidate)
                best_share = proposed_share

                if proposed_share <= max_sector_pct:
                    break

        if replacement is None:
            break

        replacement_idx, replacement_row = replacement

        selected = selected.drop(index=weakest_idx)
        selected.loc[replacement_idx] = replacement_row

        remaining = remaining.drop(index=replacement_idx)
        remaining.loc[weakest_idx] = candidates.loc[weakest_idx]

    return selected.sort_values("composite", ascending=False)


# =============================================================================
# STREAMLIT SIDEBAR INPUTS
# =============================================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    ["Single Stock", "Compare", "Auto Top 10", "How It Works"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data inputs")

universe_upload = st.sidebar.file_uploader(
    "Latest official Nifty 500 CSV (optional)",
    type=["csv"],
    key="universe_csv"
)

ownership_upload = st.sidebar.file_uploader(
    "Verified ownership CSV (optional)",
    type=["csv"],
    key="ownership_csv",
    help="Columns: ticker, promoter_pct, institutional_pct, promoter_pledge_pct, as_of, source"
)

ownership_overrides = load_ownership_overrides(ownership_upload)

st.sidebar.caption(
    "Yahoo insider ownership is informational only; it is not treated as Indian promoter shareholding."
)

if st.sidebar.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("High Compounder Framework v2.1 · Research tool")


# =============================================================================
# SINGLE STOCK PAGE
# =============================================================================
st.markdown(
    '<p class="main-title">High Compounder Framework v2.1</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Systematic Indian equity screener · corrected data alignment, cash-flow tests and allocation rules</p>',
    unsafe_allow_html=True
)

st.markdown("---")

if page == "Single Stock":

    col1, col2 = st.columns([4, 1])

    with col1:
        ticker_input = st.text_input(
            "Enter NSE ticker",
            value="LUPIN",
            placeholder="LUPIN, INFY, TCS"
        ).strip().upper()

    with col2:
        st.write("")
        run_button = st.button(
            "Analyse",
            type="primary",
            use_container_width=True
        )

    quick_cols = st.columns(6)

    for col, symbol in zip(
        quick_cols,
        ["LUPIN", "BSE", "DIXON", "INFY", "MAZDOCK", "HCLTECH"]
    ):
        if col.button(symbol, use_container_width=True):
            ticker_input, run_button = symbol, True

    if run_button:

        ticker = clean_ticker(ticker_input)

        with st.spinner(f"Analysing {ticker}..."):
            result = analyse_stock(ticker, ownership_overrides)

        if result is None:
            st.error(
                "Unable to fetch usable data. Check the ticker or try again later."
            )

        elif result.get("excluded"):
            st.warning(
                f"{ticker}: {result['excluded_reason']}. No non-financial allocation tier has been assigned."
            )

        else:
            l1 = result["l1"]

            st.subheader(l1["name"])

            st.caption(
                f"{ticker} · {l1['sector']} · CMP ₹{fmt(l1['price'], ',.2f')} · {datetime.now():%d %b %Y %H:%M}"
            )

            if result["data"]["debug"].get("warnings"):
                with st.expander("Data-quality warnings", expanded=False):
                    for message in result["data"]["debug"]["warnings"]:
                        st.warning(message)

            cards = st.columns(8)

            card_values = [
                ("Forensic", f"{fmt(result['acct']['score'], '.0f')}/100"),
                ("P/E", fmt(l1["pe"], '.1f')),
                ("PEG", fmt(result["valuation"]["peg"], '.2f')),
                ("ROE", fmt(l1["roe"], '.1f', '%')),
                ("ROCE", fmt(l1["roce"], '.1f', '%')),
                ("D/E", fmt(l1["de"], '.2f', 'x')),
                ("FCF yield", fmt(result["valuation"]["fcf_yield"], '.1f', '%')),
                ("Tier", f"{result['tier']} {result['size']}"),
            ]

            for col, (label, value) in zip(cards, card_values):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-lbl">{html.escape(label)}</div><div class="metric-val">{html.escape(value)}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### Layer-by-layer assessment")

            for label, status, detail in layer_rows(result):
                icon = (
                    "✅" if status == "pass"
                    else "❌" if status == "fail"
                    else "⚠️"
                )

                st.markdown(
                    f'<div class="layer-row layer-{status}">{icon} <strong>{html.escape(label)}:</strong> {html.escape(detail)}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("### Investment verdict")

            st.markdown(
                f'<div class="verdict-box">{html.escape(build_verdict(result))}</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Composite research score: {result['composite']}/100 · Allocation rationale: {result['tier_reason']}"
            )

            t1, t2, t3, t4 = st.tabs([
                "Forensic & cash flow",
                "ROE & cyclicality",
                "Momentum & ownership",
                "Debug"
            ])

            with t1:
                st.metric(
                    "Forensic score",
                    fmt(result["acct"].get("score"), '.0f') + "/100"
                )

                st.write("**Flags**")

                if result["acct"]["flags"]:
                    for flag in result["acct"]["flags"]:
                        st.warning(flag)
                else:
                    st.success(
                        "No threshold flags in the available data."
                    )

                st.write({
                    "Cumulative CFO/PAT": result["acct"].get("cum_cfo_pat"),
                    "Cumulative CFO/EBITDA": result["acct"].get("cum_cfo_ebitda"),
                    "Accrual ratio (%)": result["acct"].get("accrual_ratio"),
                    "Cash-flow observations": result["acct"].get("paired_pat_years"),
                    "Forensic coverage complete": result["acct"].get("complete"),
                    "CCC": result["acct"].get("ccc"),
                    "CCC trend": result["acct"].get("ccc_trend"),
                    "EBITDA margin trend": result["acct"].get("margin_trend"),
                })

            with t2:
                st.write("**Annual ROE history (average equity denominator)**")

                st.dataframe(
                    pd.DataFrame(
                        result["moat"]["roe_by_year"],
                        columns=["Financial year", "ROE %"]
                    ),
                    hide_index=True,
                    use_container_width=True
                )

                st.write({
                    "ROE consistency": result["moat"]["consistency"],
                    "Years above 15%": result["moat"]["years_above_15"],
                    "Cyclical peak warning": result["cyc"]["peak"],
                    "Prior-year average ROE": result["cyc"].get("average_prior"),
                    "Historical median ROE": result["cyc"].get("median"),
                })

                st.info(
                    "ROE consistency is only a quantitative moat proxy. Confirm the actual competitive advantage through qualitative research."
                )

            with t3:
                mom = result["momentum"]

                st.write({
                    "Earnings momentum": mom.get("status"),
                    "Latest quarterly EPS YoY (%)": mom.get("latest_yoy"),
                    "Previous quarterly EPS YoY (%)": mom.get("prior_yoy"),
                    "Momentum score": mom.get("score"),
                })

                if mom.get("eps_rows"):
                    st.dataframe(
                        pd.DataFrame(
                            mom["eps_rows"],
                            columns=["Quarter end", "EPS"]
                        ),
                        hide_index=True,
                        use_container_width=True
                    )

                st.write("**Ownership**")

                st.write({
                    "Verified promoter holding (%)": result["holdings"].get("promoter_pct"),
                    "Verified institutional holding (%)": result["holdings"].get("institutional_pct"),
                    "Promoter pledge (% of promoter holding)": result["holdings"].get("promoter_pledge_pct"),
                    "Yahoo insider holding (not promoter data) (%)": result["holdings"].get("yahoo_insider_pct"),
                    "Ownership source": result["holdings"].get("source"),
                    "Ownership as of": result["holdings"].get("as_of"),
                })

                for flag in result["holdings"].get("flags", []):
                    st.warning(flag)

            with t4:
                st.write("**Data availability**")
                st.json(result["data"]["debug"].get("data_years", {}))

                st.write("**Data warnings**")
                st.json(result["data"]["debug"].get("warnings", []))

                st.write("**Metric sources**")

                st.json({
                    k: l1.get(k)
                    for k in [
                        "mcap_source",
                        "de_source",
                        "pe_source",
                        "int_cov_source",
                        "sales_source",
                        "pat_source",
                        "eps_source"
                    ]
                })

            export = result_to_row(result)

            dl1, dl2 = st.columns(2)

            with dl1:
                st.download_button(
                    "Download analysis row (CSV)",
                    pd.DataFrame([export]).to_csv(index=False),
                    file_name=f"{ticker.replace('.', '_')}_high_compounder_v21.csv",
                    mime="text/csv",
                )

            with dl2:
                st.download_button(
                    "Download printable scorecard (HTML)",
                    build_scorecard_html(result),
                    file_name=f"{ticker.replace('.', '_')}_scorecard_v21.html",
                    mime="text/html",
                )


# =============================================================================
# COMPARE PAGE
# =============================================================================
elif page == "Compare":

    st.subheader("Compare companies")

    c1, c2, c3 = st.columns(3)

    symbols = [
        c1.text_input("Stock 1", "LUPIN"),
        c2.text_input("Stock 2", "INFY"),
        c3.text_input("Stock 3 (optional)", ""),
    ]

    if st.button("Compare", type="primary"):

        results = []

        for symbol in symbols:
            if not symbol.strip():
                continue

            with st.spinner(f"Analysing {symbol}..."):
                item = analyse_stock(
                    clean_ticker(symbol),
                    ownership_overrides
                )

            if item and not item.get("excluded"):
                results.append(item)

            elif item and item.get("excluded"):
                st.warning(
                    f"{symbol}: excluded as a financial company."
                )

            else:
                st.warning(f"Could not analyse {symbol}.")

        if len(results) >= 2:

            frame = pd.DataFrame([
                result_to_row(r)
                for r in results
            ])

            columns = [
                "ticker", "sector", "tier", "composite", "score",
                "pe", "peg", "roe", "roce", "de",
                "sales_cagr", "pat_cagr", "cfo_pat", "moat_pct",
                "earnings_momentum_score", "fcf_yield"
            ]

            st.dataframe(
                frame[[c for c in columns if c in frame.columns]],
                hide_index=True,
                use_container_width=True
            )

            for r in results:
                with st.expander(
                    f"{r['l1']['name']} · {r['tier']} {r['size']}"
                ):
                    st.write(build_verdict(r))

        else:
            st.error(
                "At least two valid non-financial companies are required."
            )


# =============================================================================
# AUTO TOP 10 PAGE
# =============================================================================
elif page == "Auto Top 10":

    st.subheader("Nifty 500 screening and ranked shortlist")

    st.caption(
        "Upload the latest official constituent CSV if the live Nifty 500 download is unavailable."
    )

    left, right = st.columns(2)

    max_stocks = left.slider(
        "Number of constituents to screen (500 = full universe)",
        20, 500, 500, 10
    )

    top_n = right.slider(
        "Number of shortlist companies",
        5, 30, 10
    )

    sector_cap = st.checkbox(
        "Apply 30% sector allocation cap",
        value=True
    )

    if st.button(
        "Run Nifty 500 screen",
        type="primary",
        use_container_width=True
    ):

        try:
            tickers, universe_source = get_universe_tickers(
                max_stocks,
                universe_upload
            )

        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.caption(
            f"Universe source: {universe_source} · {len(tickers)} tickers loaded"
        )

        funnel = {
            k: 0 for k in [
                "total",
                "financial",
                "mcap",
                "debt",
                "roe",
                "roce",
                "growth",
                "data",
                "fetch_fail",
                "pass"
            ]
        }

        funnel["total"] = len(tickers)

        results = []

        progress = st.progress(0)
        status = st.empty()

        for index, ticker in enumerate(tickers, start=1):

            status.caption(
                f"Screening {ticker} ({index}/{len(tickers)})"
            )

            result, reason = analyse_for_funnel(
                ticker,
                ownership_overrides
            )

            funnel[reason] = funnel.get(reason, 0) + 1

            if result:
                results.append(result_to_row(result))

            progress.progress(index / max(1, len(tickers)))

            if index % 10 == 0:
                time.sleep(0.5)

        progress.empty()
        status.empty()

        st.markdown("### Screening funnel")

        st.write(
            f"Universe: **{funnel['total']}** · "
            f"Financial companies: **{funnel['financial']}** · "
            f"Passed all hard filters and data checks: **{funnel['pass']}** · "
            f"Fetch failures: **{funnel['fetch_fail']}**"
        )

        funnel_rows = [
            ("Financial exclusion", funnel["financial"]),
            ("Market cap", funnel["mcap"]),
            ("Debt / equity", funnel["debt"]),
            ("ROE", funnel["roe"]),
            ("ROCE", funnel["roce"]),
            ("Growth", funnel["growth"]),
            ("Insufficient data", funnel["data"]),
            ("Fetch failure", funnel["fetch_fail"]),
        ]

        st.dataframe(
            pd.DataFrame(
                funnel_rows,
                columns=["Elimination reason", "Companies"]
            ),
            hide_index=True,
            use_container_width=True
        )

        if funnel["fetch_fail"] > funnel["total"] * 0.2:
            st.warning(
                "More than 20% of the universe failed to fetch. "
                "Treat this screening run as incomplete and repeat after checking data access."
            )

        if not results:
            st.error(
                "No stocks passed all fundamental filters and data requirements."
            )
            st.stop()

        df = pd.DataFrame(results)
        df = df[df["tier"] != "WATCH"].copy()
        df = df.sort_values("composite", ascending=False)

        selected = (
            select_with_sector_limit(df, top_n, 30.0)
            if sector_cap
            else df.head(top_n)
        )

        if sector_cap and not selected.empty:
            weights = selected["tier"].map(TIER_MIDPOINT).fillna(0)
            total_weight = weights.sum()

            if total_weight > 0:
                max_sector_share = (
                    weights.groupby(selected["sector"]).sum()
                    / total_weight * 100
                ).max()

                if max_sector_share > 30.0:
                    st.warning(
                        f"The 30% sector allocation cap could not be fully met with the available qualifying candidates. Highest sector share is {max_sector_share:.1f}%."
                    )

        if selected.empty:
            st.warning(
                "All qualifying candidates were removed by the WATCH rule. Review the data and allocation rules."
            )
            st.stop()

        st.markdown("### Ranked shortlist")

        display_cols = [
            "ticker", "name", "sector", "tier", "size", "composite",
            "score", "pe", "peg", "roe", "roce", "de",
            "sales_cagr", "pat_cagr", "cfo_pat", "moat_pct",
            "earnings_momentum_score", "fcf_yield",
            "interest_coverage", "promoter_pct"
        ]

        st.dataframe(
            selected[[c for c in display_cols if c in selected.columns]],
            hide_index=True,
            use_container_width=True
        )

        total_midpoint = selected["tier"].map(TIER_MIDPOINT).sum()

        if total_midpoint > 100:
            st.warning(
                f"The sum of suggested midpoint weights is {total_midpoint:.1f}% — above 100%. These are individual tier ranges, not a portfolio budget. Apply a separate portfolio-level sizing constraint."
            )
        else:
            st.info(
                f"Sum of suggested midpoint weights: {total_midpoint:.1f}%. This is not a portfolio optimisation result."
            )

        csv_columns = [
            c for c in display_cols + [
                "eps_growth",
                "cfo_ebitda",
                "accrual_pct",
                "promoter_pledge_pct",
                "return_1y",
                "tier_reason"
            ]
            if c in selected.columns
        ]

        st.download_button(
            "Download shortlist (CSV)",
            selected[csv_columns].to_csv(index=False),
            file_name=f"high_compounder_v21_top_{len(selected)}_{datetime.now():%d%b%Y}.csv",
            mime="text/csv",
        )


# =============================================================================
# HOW IT WORKS PAGE
# =============================================================================
else:

    st.subheader("High Compounder Framework v2.1 — methodology")

    st.markdown("""
### Stage 1 — Universe and exclusions
Start with the current Nifty 500 constituent list. Exclude banks, NBFCs, insurers, lenders and other credit-intermediation businesses because these require a different set of ratios. A broad "Financial Services" sector label alone is not sufficient to exclude a company: exchanges and market-infrastructure businesses may have operating economics that can be screened separately. Upload the official constituent CSV if the live download is unavailable; the app deliberately does not silently substitute an old hardcoded list.

### Stage 2 — Fundamental quality screen
- Market capitalisation: at least ₹15,000 crore.
- ROE: above 15%, calculated using average shareholder equity where available.
- ROCE: above 18%, calculated using average capital employed where available.
- Gross debt/equity: below 1.00x, calculated from statement debt and equity. Missing debt is N/A, not zero.
- Revenue CAGR and PAT CAGR: both above 15%; mature compounder exception applies at market cap above ₹50,000 crore and ROE above 25%, reducing the threshold to 10%.

Annual multi-year CAGR is preferred. A quarterly TTM year-on-year proxy is used only when eight complete quarterly observations exist and is labelled as a one-year proxy, not a multi-year CAGR.

### Stage 3 — Debt servicing
Interest coverage is EBIT divided by gross interest expense. Net interest income is not substituted for gross interest expense. Below 3x caps the allocation at HALF. Missing data is shown as unavailable and requires verification.

### Stage 4 — Forensic accounting
Review receivables, inventory, cumulative CFO/PAT, cumulative CFO/EBITDA and the accrual ratio. Cumulative cash-flow ratios require at least three paired annual observations. The cash-conversion-related deductions are capped at 50 points to reduce repeated penalties for overlapping symptoms; the final forensic score is floored at zero.

### Stage 5 — Working capital
Cash conversion cycle uses average receivables, inventory and payables balances. DSO uses revenue; DIO and DPO use cost of revenue. Missing inputs are not treated as zero.

### Stage 6 — ROE consistency and moat research
Historical ROE consistency is a quantitative proxy, not proof of a moat. An established moat proxy requires at least five years of ROE above 15%. Qualitative research into pricing power, brand, switching costs, distribution, technology and customer retention remains necessary.

### Stage 7 — Ownership and governance
Yahoo insider ownership is displayed only as an informational signal; it is not treated as Indian promoter holding. Verified promoter, institutional and pledge data can be supplied through a CSV with columns `ticker`, `promoter_pct`, `institutional_pct`, `promoter_pledge_pct`, `as_of`, and `source`. Pledge disclosures trigger manual review rather than an invented automatic penalty.

### Stage 8 — Valuation
PEG uses diluted EPS growth rather than total PAT growth. The growth rate is adjusted to revenue growth when diluted EPS growth is more than twice revenue growth. FCF yield is calculated only when both CFO and capex are available. P/E above 80x is flagged but is not automatically treated as an exclusion.

### Stage 9 — Earnings momentum and cyclicality
Quarterly EPS momentum compares each of the latest two quarters with the same quarter one year earlier, reducing seasonal distortion. Loss-to-profit transitions are labelled rather than converted into misleading percentage growth. Cyclical ROE warnings require at least four annual observations and compare the latest ROE with the average of prior years.

### Stage 10 — Allocation and ranking
FULL requires strong forensic quality, no forensic flags, a qualifying moat proxy, no cyclical peak warning and PEG at or below 1.0. STANDARD and HALF are assigned through explicit score, flag and valuation rules. PEG above 3.0 and interest coverage below 3.0x cap the allocation at HALF; PEG above 5.0 or forensic score below 50 results in WATCH. A fundamental failure in single-stock review caps the tier at HALF; batch screening remains strict.

The composite score uses six normalised 0–100 inputs: PEG 25%, forensic score 20%, earnings momentum 15%, CFO quality 15%, ROE consistency 15%, and normalised ROE 10%. Share-price return is used only as a risk override, not as an earnings momentum score.

### Important limitations
The framework has not been historically backtested. Yahoo Finance data can be missing, stale or inconsistently mapped. Sector-relative benchmarking, related-party transactions, management quality, full promoter pledge analysis and a complete sell discipline remain outside the automated model. Verify the data against annual reports and exchange disclosures before investing.
    """)

    st.caption(
        "High Compounder Framework v2.1 · Quantitative research tool · Not investment advice"
    )
