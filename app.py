"""
High Compounder Framework v2.8
Author: Vinayak Nagral

Streamlit screener for non-financial Indian listed companies. Yahoo Finance data is an
unofficial, best-effort source; verify every investment-critical number against exchange
filings and company annual reports before any institutional use.

v2.8 changes vs v2.7:
- Fixed strict CAGR calculation (was returning N/A for most companies)
- Fixed CFO/PAT 5Y and trend windows (gracefully handle 3-5 years of Yahoo data)
- Fixed EPS field selection (tolerates isolated missing values)
- Fixed moat proxy strict contiguity requirement
- Restored reported-FCF as last-resort fallback
- Loosened 1Y return date tolerance from 7 to 15 days
- Removed sector cap entirely
"""
from __future__ import annotations

import html
import io
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="High Compounder Framework v2.8",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "2.8"
AUTHOR = "Vinayak Nagral"
NAVY = "#1B2A4A"

MIN_MARKET_CAP = 150e9            # Rs 15,000 Cr
MAX_AUTOMATED_PE = 500.0          # sanity guard; higher values require manual verification
ANNUAL_GAP_MIN_DAYS = 300
ANNUAL_GAP_MAX_DAYS = 430
MIN_ROE = 15.0
MIN_ROCE = 18.0
MAX_DEBT_TO_EQUITY = 1.0
MIN_GROWTH = 15.0
MATURE_GROWTH = 10.0
MATURE_MCAP = 500e9               # Rs 50,000 Cr
MATURE_ROE = 25.0
INVENTORY_MATERIAL_PCT = 3.0      # inventory below 3% of revenue = immaterial
NEGLIGIBLE_INTEREST_PCT = 1.0     # interest below 1% of EBIT = negligible

FINANCIAL_KW = [
    "bank", "finance", "financial", "insurance", "nbfc",
    "housing finance", "credit", "lending", "microfinance",
]
# These are market-infrastructure businesses, not deposit-taking lenders or insurers.
# They are deliberately allowed through, as specified in the SOP.
MARKET_INFRASTRUCTURE_TICKERS = {"BSE.NS", "MCX.NS", "CDSL.NS"}

TIER_MIDPOINT = {"FULL": 13.5, "STANDARD": 9.0, "HALF": 5.0, "WATCH": 0.0}
TIER_SIZE = {"FULL": "12-15%", "STANDARD": "8-10%", "HALF": "4-6%", "WATCH": "0%"}
MAX_FULL_POSITIONS = 5
TIER_COLOR = {"FULL": "#059669", "STANDARD": "#2563EB", "HALF": "#D97706", "WATCH": "#DC2626"}
TIER_RANK = {"WATCH": 0, "HALF": 1, "STANDARD": 2, "FULL": 3}

UNIVERSE_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
]
UNIVERSE_CACHE_FILE = Path(__file__).resolve().parent / "last_nifty500.csv"

SHORT_DEBT_LABELS = [
    "Current Debt", "Short Term Debt", "Short Term Borrowings",
    "Current Portion Of Long Term Debt",
]
LONG_DEBT_LABELS = [
    "Long Term Debt", "Long Term Borrowings", "Long Term Debt Non Current",
]
ALL_DEBT_LABELS = [
    "Total Debt", "Total Debt And Lease Obligations",
    "Current Debt And Capital Lease Obligation",
    "Long Term Debt And Capital Lease Obligation",
] + SHORT_DEBT_LABELS + LONG_DEBT_LABELS

PLEDGE_COLUMNS = ("pledged_pct_of_promoter_holding", "promoter_pledge_pct", "pledge_pct")

# ============================================================
# STYLES
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.main-title{font-family:'DM Sans',sans-serif;font-size:1.8rem;font-weight:700;color:__NAVY__;}
.sub-title{font-size:.85rem;color:#6B7280;}
.stock-name{font-size:1.6rem;font-weight:700;color:__NAVY__;}
.stock-meta{font-size:.85rem;color:#6B7280;}
.metric-card{background:#fff;border:1px solid #E5E7EB;border-radius:8px;padding:.8rem;text-align:center;}
.metric-val{font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:600;color:__NAVY__;}
.metric-lbl{font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.5px;}
.tier-badge{display:inline-block;padding:4px 13px;border-radius:20px;font-weight:600;font-size:.83rem;}
.tier-full{background:#ECFDF5;color:#059669;border:1px solid #059669;}
.tier-standard{background:#EFF6FF;color:#2563EB;border:1px solid #2563EB;}
.tier-half{background:#FFFBEB;color:#D97706;border:1px solid #D97706;}
.tier-watch{background:#FEF2F2;color:#DC2626;border:1px solid #DC2626;}
.layer-row{padding:.5rem .8rem;border-radius:4px;margin:4px 0;font-size:.84rem;border-left:3px solid;}
.layer-pass{background:#ECFDF5;border-color:#059669;}
.layer-fail{background:#FEF2F2;border-color:#DC2626;}
.layer-warn{background:#FFFBEB;border-color:#D97706;}
.verdict-box{background:#F9FAFB;padding:1.1rem;border-radius:8px;border:1px solid #E5E7EB;line-height:1.8;font-size:.9rem;}
.flag-item{background:#FFFBEB;padding:.45rem .8rem;border-radius:4px;margin:3px 0;border-left:3px solid #D97706;font-size:.85rem;}
.leading-tag{background:#DBEAFE;color:#1D4ED8;padding:2px 5px;border-radius:3px;font-size:.65rem;font-weight:600;margin-left:4px;}
.score-box{padding:1rem;border-radius:8px;text-align:center;}
.score-green{background:#ECFDF5;border:1px solid #059669;}
.score-yellow{background:#FFFBEB;border:1px solid #D97706;}
.score-red{background:#FEF2F2;border:1px solid #DC2626;}
</style>
""".replace("__NAVY__", NAVY)

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# GENERIC UTILITIES
# ============================================================
def valid(x):
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError, OverflowError):
        return False


def number(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fmt(v, spec=".1f", suffix="", prefix=""):
    if not valid(v):
        return "N/A"
    return f"{prefix}{float(v):{spec}}{suffix}"


def clean_ticker(ticker):
    ticker = str(ticker).strip().upper()
    if not ticker:
        return ""
    return ticker if ticker.endswith((".NS", ".BO")) else ticker + ".NS"


def short_ticker(ticker):
    return str(ticker).replace(".NS", "").replace(".BO", "")


def fy_label(dt):
    try:
        dt = pd.Timestamp(dt)
        year = dt.year + 1 if dt.month > 3 else dt.year
        return f"FY{str(year)[-2:]}"
    except Exception:
        return ""


def norm_label(x):
    return " ".join(str(x).lower().replace("&", "and").replace("_", " ").split())


def safe_get(df, labels, col=0):
    """Read a statement row by alias; skips aliases whose value is missing."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(labels, str):
        labels = [labels]
    try:
        if isinstance(col, (int, np.integer)):
            if col < 0 or col >= len(df.columns):
                return None
            col = df.columns[col]
        elif col not in df.columns:
            return None
        lookup = {norm_label(x): x for x in df.index}
        for label in labels:
            actual = label if label in df.index else lookup.get(norm_label(label))
            if actual is None:
                continue
            value = df.loc[actual, col]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            v = number(value)
            if v is not None:
                return v
    except Exception:
        return None
    return None


def field_exists(df, labels):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if isinstance(labels, str):
        labels = [labels]
    available = {norm_label(x) for x in df.index}
    return any(norm_label(x) in available for x in labels)


def statement_date(col):
    try:
        dt = pd.Timestamp(col)
        return None if pd.isna(dt) else dt
    except Exception:
        return None


def consecutive_annual_periods(earlier, later):
    """True only when two statement dates are plausibly adjacent annual periods."""
    d1, d2 = statement_date(earlier), statement_date(later)
    if d1 is None or d2 is None:
        return False
    days = (d2 - d1).days
    return ANNUAL_GAP_MIN_DAYS <= days <= ANNUAL_GAP_MAX_DAYS


def dated_columns(df, newest_first=True):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    cols = [(statement_date(c), c) for c in df.columns]
    cols = [x for x in cols if x[0] is not None]
    return sorted(cols, key=lambda x: x[0], reverse=newest_first)


def match_column(df, target, tolerance=45):
    """Match statement columns by reporting date, never by position."""
    target = statement_date(target)
    if target is None:
        return None
    best = None
    for dt, col in dated_columns(df):
        days = abs((dt - target).days)
        if days <= tolerance and (best is None or days < best[0]):
            best = (days, col)
    return best[1] if best else None


def latest_column(df):
    cols = dated_columns(df)
    return cols[0][1] if cols else None


# ============================================================
# FINANCIAL FIELD GETTERS
# ============================================================
def get_revenue(df, col=0):
    v = safe_get(df, ["Total Revenue", "Operating Revenue", "Revenue", "Net Revenue"], col)
    return v if valid(v) and v > 0 else None


def get_pat(df, col=0):
    return safe_get(df, ["Net Income Common Stockholders", "Net Income",
                         "Net Income Continuous Operations"], col)


def get_ebit(df, col=0):
    return safe_get(df, ["EBIT", "Operating Income", "Operating Profit", "Normalized EBIT"], col)


def get_ebitda(df, col=0):
    return safe_get(df, ["EBITDA", "Normalized EBITDA"], col)


def get_cogs(df, col=0):
    v = safe_get(df, ["Cost Of Revenue", "Cost of Goods Sold",
                      "Cost Of Goods And Services Sold"], col)
    return abs(v) if valid(v) else None


def get_equity(df, col=0):
    v = safe_get(df, ["Stockholders Equity", "Total Stockholders Equity",
                      "Common Stock Equity", "Total Equity Gross Minority Interest"], col)
    return v if valid(v) and v > 0 else None


def get_assets(df, col=0):
    return safe_get(df, "Total Assets", col)


def get_current_liabilities(df, col=0):
    return safe_get(df, ["Current Liabilities", "Total Current Liabilities"], col)


def get_cfo(df, col=0):
    return safe_get(df, ["Operating Cash Flow", "Total Cash From Operating Activities",
                         "Cash Flow From Continuing Operating Activities"], col)


def get_fcf(df, col=0):
    """FCF = CFO - |capex|. Last-resort fallback to reported FCF if both inputs
    are missing (prevents FCF yield from always being N/A when capex isn't under
    a recognised alias)."""
    cfo = get_cfo(df, col)
    capex = safe_get(df, ["Capital Expenditure", "Capital Expenditures",
                          "Capital Expenditure Reported"], col)
    if valid(cfo) and valid(capex):
        return cfo - abs(capex)
    reported = safe_get(df, "Free Cash Flow", col)
    if valid(reported):
        return reported
    return None


def get_receivables(df, col=0):
    return safe_get(df, ["Accounts Receivable", "Net Receivables", "Receivables"], col)


def get_inventory(df, col=0):
    return safe_get(df, ["Inventory", "Inventories"], col)


def get_payables(df, col=0):
    return safe_get(df, ["Accounts Payable", "Payables And Accrued Expenses",
                         "Current Accounts Payable"], col)


def get_interest_expense(df, col=0):
    """Gross interest / finance cost only - never net interest income."""
    v = safe_get(df, ["Interest Expense", "Interest Expense Non Operating",
                      "Interest And Debt Expense", "Finance Costs"], col)
    return abs(v) if valid(v) else None


def get_cash(df, col=0):
    return safe_get(df, ["Cash And Cash Equivalents",
                         "Cash Cash Equivalents And Short Term Investments",
                         "Cash And Short Term Investments", "Cash Financial"], col)


def get_lease_liabilities(df, col=0):
    total = safe_get(df, ["Capital Lease Obligations"], col)
    if valid(total) and total >= 0:
        return total
    parts = [
        safe_get(df, ["Current Capital Lease Obligation"], col),
        safe_get(df, ["Long Term Capital Lease Obligation"], col),
    ]
    parts = [p for p in parts if valid(p) and p >= 0]
    return sum(parts) if parts else None


def get_gross_debt(df, col, leases=None):
    """Prefer complete ex-lease borrowings, otherwise use a reported total-debt line."""
    short = safe_get(df, SHORT_DEBT_LABELS, col)
    long_ = safe_get(df, LONG_DEBT_LABELS, col)
    if valid(short) and valid(long_) and short >= 0 and long_ >= 0:
        return short + long_, "Short-term + long-term borrowings (ex-leases)"

    current_with_leases = safe_get(df, "Current Debt And Capital Lease Obligation", col)
    long_with_leases = safe_get(df, "Long Term Debt And Capital Lease Obligation", col)
    if (valid(current_with_leases) and valid(long_with_leases)
            and current_with_leases >= 0 and long_with_leases >= 0):
        return (current_with_leases + long_with_leases,
                "Current + long-term debt/lease obligations (includes leases)")

    total = safe_get(df, ["Total Debt", "Total Debt And Lease Obligations",
                          "Total Debt Including Lease Liabilities"], col)
    if valid(total) and total >= 0:
        return total, "Reported total debt (may include leases)"
    return None, "Borrowing data incomplete - verify manually"


def get_equity_shares(bs, info):
    value = number(info.get("sharesOutstanding"))
    if valid(value) and value > 0:
        return value
    latest_bs_col = latest_column(bs)
    value = safe_get(bs, ["Ordinary Shares Number", "Share Issued"], latest_bs_col) if latest_bs_col is not None else None
    if valid(value) and value > 0:
        return value
    return None


# ============================================================
# EPS HELPERS
# ============================================================
def choose_eps_field(df):
    """Prefer diluted EPS; basic EPS only if diluted is unavailable.
    Requires the LATEST reported period to have a valid value (prevents selecting
    a field that only has old data, which would produce stale growth/momentum)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    cols = dated_columns(df)  # newest first
    if not cols:
        return None
    latest_col = cols[0][1]

    # Growth path: latest value present AND 3+ total valid observations
    for field in ("Diluted EPS", "Basic EPS"):
        if valid(safe_get(df, field, latest_col)):
            total_count = sum(valid(safe_get(df, field, c)) for c in df.columns)
            if total_count >= 3:
                return field

    # Momentum fallback: latest value present AND 2+ total valid observations
    for field in ("Diluted EPS", "Basic EPS"):
        if valid(safe_get(df, field, latest_col)):
            total_count = sum(valid(safe_get(df, field, c)) for c in df.columns)
            if total_count >= 2:
                return field

    return None
    return lambda f, c: safe_get(f, field, c)


# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock(ticker):
    """Fetch price, quote metadata and statements with visible fallback diagnostics."""
    debug = {"ticker": ticker, "warnings": [], "sources": [], "valuation": {}}
    try:
        t = yf.Ticker(ticker)

        info = {}
        info_errors = []
        for label, getter in (
            ("ticker.info", lambda: t.info),
            ("ticker.get_info", lambda: t.get_info()),
        ):
            try:
                candidate = dict(getter() or {})
                if candidate:
                    info = candidate
                    debug["sources"].append(f"company info: {label}")
                    break
                info_errors.append(f"{label} returned empty data")
            except Exception as exc:
                info_errors.append(f"{label}: {type(exc).__name__}")

        fast = {}
        try:
            fast_obj = t.fast_info
            if fast_obj is not None:
                for key in (
                    "currency", "exchange", "quote_type", "last_price",
                    "previous_close", "market_cap", "shares"
                ):
                    try:
                        value = fast_obj.get(key)
                        if value is not None:
                            fast[key] = value
                    except Exception:
                        continue
                if fast:
                    debug["sources"].append("quote data: fast_info fallback")
        except Exception as exc:
            debug["warnings"].append(
                f"Yahoo fast quote data unavailable ({type(exc).__name__})"
            )

        fast_aliases = {
            "currency": "currency",
            "exchange": "exchange",
            "quoteType": "quote_type",
            "currentPrice": "last_price",
            "regularMarketPrice": "last_price",
            "previousClose": "previous_close",
            "marketCap": "market_cap",
            "sharesOutstanding": "shares",
        }
        for info_key, fast_key in fast_aliases.items():
            if not valid(info.get(info_key)) and valid(fast.get(fast_key)):
                info[info_key] = fast[fast_key]

        if not info:
            detail = "; ".join(info_errors) if info_errors else "no quote profile returned"
            debug["warnings"].append(
                "Yahoo company profile is unavailable; using statement/price fallbacks. " + detail
            )

        frames = {}
        getters = {
            "fin": ["income_stmt", "financials"],
            "bs": ["balance_sheet"],
            "cf": ["cash_flow", "cashflow"],
            "qfin": ["quarterly_income_stmt", "quarterly_financials"],
        }
        for key, attributes in getters.items():
            frame = None
            for attribute in attributes:
                try:
                    candidate = getattr(t, attribute)
                    if candidate is not None and not candidate.empty:
                        frame = candidate.copy()
                        debug["sources"].append(f"{key}: {attribute}")
                        break
                except Exception:
                    continue
            frames[key] = frame

        try:
            hist = t.history(period="2y", auto_adjust=False)
            prices = hist["Close"].dropna().sort_index() if hist is not None and not hist.empty else None
        except Exception as exc:
            prices = None
            debug["warnings"].append(
                f"Yahoo price history unavailable ({type(exc).__name__})"
            )

        if not valid(info.get("currentPrice")) and prices is not None and len(prices):
            info["currentPrice"] = float(prices.iloc[-1])
            debug["sources"].append("price: 2y history close fallback")

        holdings = {"yahoo_insider_pct": None, "yahoo_institutional_pct": None}
        for key, field in (("yahoo_insider_pct", "heldPercentInsiders"),
                           ("yahoo_institutional_pct", "heldPercentInstitutions")):
            value = number(info.get(field))
            if valid(value):
                holdings[key] = value * 100 if 0 <= value <= 1 else value

        if not valid(info.get("sharesOutstanding")):
            shares = get_equity_shares(frames.get("bs"), {})
            if valid(shares):
                info["sharesOutstanding"] = shares

        usable = (valid(info.get("currentPrice"))
                  or frames.get("fin") is not None
                  or frames.get("bs") is not None)
        if not usable:
            debug["warnings"].append("No usable quote, income statement or balance sheet data")
            return None

        debug["data_years"] = {k: (v.shape[1] if v is not None else 0) for k, v in frames.items()}
        debug["data_years"]["price_days"] = len(prices) if prices is not None else 0
        debug["quote_fields"] = {
            "currency": info.get("currency"),
            "financialCurrency": info.get("financialCurrency"),
            "trailingPE": info.get("trailingPE"),
            "trailingEps": info.get("trailingEps"),
            "marketCap": info.get("marketCap"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

        return {**frames, "info": info, "prices": prices,
                "holdings": holdings, "debug": debug}
    except Exception as exc:
        debug["warnings"].append(str(exc))
        return None


# ============================================================
# STATEMENT ALIGNMENT
# ============================================================
def build_multi(fin, bs, cf):
    """Build annual records from income statement dates; align BS and CF independently."""
    if fin is None or not isinstance(fin, pd.DataFrame) or fin.empty:
        return [], ["Annual income statement unavailable; annual analysis cannot be built"]

    rows, warns = [], []
    if bs is None or not isinstance(bs, pd.DataFrame) or bs.empty:
        warns.append("Annual balance sheet unavailable; ROE, ROCE and debt metrics will be N/A")
    if cf is None or not isinstance(cf, pd.DataFrame) or cf.empty:
        warns.append("Annual cash-flow statement unavailable; forensic cash metrics will be provisional")

    for fdt, fcol in dated_columns(fin, newest_first=False):
        bcol = match_column(bs, fdt)
        ccol = match_column(cf, fdt)
        if bcol is None:
            warns.append(f"Balance sheet not aligned for {fdt.date()}; balance-sheet metrics unavailable")
        if ccol is None:
            warns.append(f"Cash-flow statement not aligned for {fdt.date()}; cash-flow metrics unavailable")

        leases = get_lease_liabilities(bs, bcol) if bcol is not None else None
        debt, debt_source = get_gross_debt(bs, bcol, leases) if bcol is not None else (None, "Balance sheet not aligned")
        ebit = get_ebit(fin, fcol)
        interest = get_interest_expense(fin, fcol)
        debt_assumed = False

        if (bcol is not None and debt is None and not field_exists(bs, ALL_DEBT_LABELS)):
            negligible = (
                valid(interest)
                and (interest == 0 or (valid(ebit) and ebit > 0
                                       and interest / ebit * 100 < NEGLIGIBLE_INTEREST_PCT))
            )
            if negligible:
                debt = 0.0
                debt_source = "Assumed debt-free: no borrowing lines, negligible interest (verify)"
                debt_assumed = True

        rows.append({
            "period_end": fdt,
            "year": fy_label(fdt),
            "revenue": get_revenue(fin, fcol),
            "pat": get_pat(fin, fcol),
            "ebit": ebit,
            "ebitda": get_ebitda(fin, fcol),
            "cogs": get_cogs(fin, fcol),
            "interest_expense": interest,
            "equity": get_equity(bs, bcol) if bcol is not None else None,
            "assets": get_assets(bs, bcol) if bcol is not None else None,
            "current_liabilities": get_current_liabilities(bs, bcol) if bcol is not None else None,
            "receivables": get_receivables(bs, bcol) if bcol is not None else None,
            "inventory": get_inventory(bs, bcol) if bcol is not None else None,
            "payables": get_payables(bs, bcol) if bcol is not None else None,
            "cash": get_cash(bs, bcol) if bcol is not None else None,
            "gross_debt": debt,
            "debt_source": debt_source,
            "debt_assumed": debt_assumed,
            "lease_liabilities": leases,
            "cfo": get_cfo(cf, ccol) if ccol is not None else None,
            "balance_aligned": bcol is not None,
            "cashflow_aligned": ccol is not None,
        })

    rows.sort(key=lambda x: x["period_end"])
    if len(rows) < 2:
        warns.append("Fewer than two annual income-statement periods")

    for i, row in enumerate(rows):
        prev = rows[i - 1] if i > 0 and consecutive_annual_periods(rows[i - 1]["period_end"], row["period_end"]) else None
        if i == 0:
            avg_eq = row["equity"]
        elif prev is not None and valid(row["equity"]) and valid(prev.get("equity")):
            avg_eq = (row["equity"] + prev["equity"]) / 2
        else:
            avg_eq = None

        ce = (row["assets"] - row["current_liabilities"]
              if valid(row["assets"]) and valid(row["current_liabilities"]) else None)
        if i == 0:
            avg_ce = ce
        else:
            prev_ce = (prev["assets"] - prev["current_liabilities"]
                       if prev is not None and valid(prev.get("assets"))
                       and valid(prev.get("current_liabilities")) else None)
            avg_ce = (ce + prev_ce) / 2 if valid(ce) and ce > 0 and valid(prev_ce) and prev_ce > 0 else None

        row["roe"] = (row["pat"] / avg_eq * 100
                      if valid(row["pat"]) and valid(avg_eq) and avg_eq > 0 else None)
        row["roce"] = (row["ebit"] / avg_ce * 100
                       if valid(row["ebit"]) and valid(avg_ce) and avg_ce > 0 else None)

    return rows, warns


# ============================================================
# GROWTH
# ============================================================
def annual_cagr(frame, getter, min_years=2.5):
    """Annual CAGR respecting recent performance.
    Rules:
    - Latest reported year must be present and positive (prevents hiding recent losses).
    - Uses the longest consecutive-positive window ENDING at the latest year.
    - Caps extreme low-base growth (>200%) with a visible warning instead of silently
      publishing fake-cheap PEG values (EMMVEE-style distortion).
    """
    cols = dated_columns(frame, newest_first=False)
    if frame is None or getter is None or len(cols) < 3:
        return None, "", "Insufficient annual data"

    # Collect all observations
    all_values = []
    for dt, col in cols:
        v = getter(frame, col)
        all_values.append((dt, number(v)))

    # SAFETY: latest year must exist and be positive.
    # A loss or missing value in the latest year must NOT be hidden by positive history.
    latest_dt, latest_v = all_values[-1]
    if not valid(latest_v):
        return None, "", "Latest reported year missing; CAGR not calculated"
    if latest_v <= 0:
        return None, "", f"Latest reported year {fy_label(latest_dt)} non-positive; CAGR not meaningful"

    # Keep only positive observations
    values = [(dt, float(v)) for dt, v in all_values if valid(v) and v > 0]
    if len(values) < 3:
        return None, "", "Fewer than three valid positive annual observations"

    # Latest must be the end of our window (we verified it's positive above)
    if values[-1][0] != latest_dt:
        return None, "", "Latest year not in positive-observation window"

    # Walk BACKWARDS from the latest year, adding each consecutive positive year.
    # This guarantees the CAGR window ends at the latest reported year, not some
    # older clean stretch that ignores recent deterioration.
    best_run = [values[-1]]
    for i in range(len(values) - 2, -1, -1):
        if consecutive_annual_periods(values[i][0], best_run[0][0]):
            best_run.insert(0, values[i])
        else:
            break

    if len(best_run) < 3:
        return None, "", "Fewer than three consecutive positive years ending at latest period"

    start_dt, start_v = best_run[0]
    end_dt, end_v = best_run[-1]
    years = (end_dt - start_dt).days / 365.25
    if years < min_years:
        return None, "", "Historical period too short"
    try:
        growth = ((end_v / start_v) ** (1 / years) - 1) * 100
    except (ValueError, ZeroDivisionError, OverflowError):
        return None, "", "CAGR calculation failed"
    if not valid(growth) or growth > 500:
        return None, "", "CAGR outside accepted range"

    note = f"Annual CAGR over {years:.1f} years ending at {fy_label(end_dt)}"
    # Flag low-base distortions (prevents EMMVEE-style fake-cheap PEG)
    if growth > 200:
        note += " (WARNING: >200% growth, likely low-base distortion; verify manually)"

    return (round(growth, 1), f"{fy_label(start_dt)}-{fy_label(end_dt)}", note)

def quarterly_ttm_growth(qfin, getter):
    """TTM YoY proxy using eight complete, consecutive quarterly observations."""
    if qfin is None or getter is None:
        return None, "", "Quarterly data unavailable"
    cols = dated_columns(qfin)
    if len(cols) < 8:
        return None, "", "Eight quarters required"
    selected = cols[:8]
    if any(not (70 <= (selected[i][0] - selected[i + 1][0]).days <= 110)
           for i in range(7)):
        return None, "", "Quarterly dates are not consecutive; TTM proxy unavailable"
    latest = [getter(qfin, c) for _, c in selected[:4]]
    prior = [getter(qfin, c) for _, c in selected[4:8]]
    if any(not valid(x) or x <= 0 for x in latest + prior):
        return None, "", "Incomplete or non-positive TTM history"
    prior_total = float(sum(prior))
    if prior_total <= 0:
        return None, "", "Prior TTM total is non-positive"
    growth = (float(sum(latest)) / prior_total - 1) * 100
    if not valid(growth):
        return None, "", "TTM growth unavailable"
    return round(growth, 1), "TTM YoY proxy", "One-year proxy, not a multi-year CAGR"


# ============================================================
# PRICE, MARKET CAP, P/E
# ============================================================
def current_price(data):
    info = data["info"]
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        v = number(info.get(key))
        if valid(v) and v > 0:
            return v
    prices = data.get("prices")
    if prices is not None and len(prices):
        return float(prices.iloc[-1])
    return None


def get_market_cap(data, price):
    info = data.get("info", {}) or {}
    reported = number(info.get("marketCap"))
    shares = number(info.get("sharesOutstanding"))

    if valid(reported) and reported > 0:
        if valid(shares) and shares > 0 and valid(price) and price > 0:
            implied = price * shares
            mismatch = abs(reported / implied - 1) if implied > 0 else None
            if valid(mismatch) and mismatch > 0.30:
                debug = data.setdefault("debug", {})
                debug.setdefault("warnings", []).append(
                    f"Yahoo market cap conflicts with price x shares by {mismatch:.0%}; "
                    "market-cap-dependent calculations withheld for manual verification."
                )
                debug.setdefault("valuation", {}).update({
                    "reported_market_cap": reported, "price_x_shares": implied,
                    "market_cap_relative_mismatch": mismatch,
                })
                return None, "Yahoo market cap conflicts with price x shares; verify manually"
        return reported, "Yahoo reported market cap"

    shares = get_equity_shares(data.get("bs"), info)
    if valid(shares) and valid(price) and price > 0:
        return price * shares, "Price x shares outstanding (fallback)"
    return None, "Not available"


def get_pe(data, price):
    """Use trailing P/E only when inputs pass basic consistency checks."""
    info = data.get("info", {}) or {}
    debug = data.setdefault("debug", {})
    debug.setdefault("valuation", {})

    reported_pe = number(info.get("trailingPE"))
    reported_eps = number(info.get("trailingEps"))
    quote_ccy = str(info.get("currency") or "").upper()
    financial_ccy = str(info.get("financialCurrency") or "").upper()
    currency_conflict = bool(quote_ccy and financial_ccy and quote_ccy != financial_ccy)
    eps_pe = (price / reported_eps if valid(price) and price > 0
              and valid(reported_eps) and reported_eps > 0 else None)

    debug["valuation"].update({
        "price": number(price),
        "quote_currency": quote_ccy or "not supplied",
        "financial_currency": financial_ccy or "not supplied",
        "yahoo_trailing_pe": reported_pe,
        "yahoo_trailing_eps": reported_eps,
        "price_divided_by_trailing_eps": eps_pe,
        "automated_pe_sanity_limit": MAX_AUTOMATED_PE,
    })

    if currency_conflict:
        debug.setdefault("warnings", []).append(
            f"P/E and FCF calculations blocked: quote currency {quote_ccy} differs from "
            f"financial statement currency {financial_ccy}."
        )
    else:
        if valid(reported_pe) and 0 < reported_pe <= MAX_AUTOMATED_PE:
            if valid(eps_pe) and eps_pe > 0:
                mismatch = abs(eps_pe / reported_pe - 1)
                if mismatch <= 0.20 and eps_pe <= MAX_AUTOMATED_PE:
                    debug["valuation"].update({"selected_source": "Yahoo trailing P/E (EPS cross-check passed)",
                                               "selected_pe": reported_pe})
                    return round(reported_pe, 2), "Yahoo trailing P/E (cross-checked)"
                debug.setdefault("warnings", []).append(
                    f"Yahoo P/E rejected: reported {reported_pe:.2f}x conflicts with "
                    f"price / trailing EPS {eps_pe:.2f}x."
                )
            else:
                debug.setdefault("warnings", []).append(
                    "Yahoo trailing P/E has no independent EPS cross-check; treated as lower-confidence data."
                )
                debug["valuation"].update({"selected_source": "Yahoo trailing P/E (not cross-checked)",
                                           "selected_pe": reported_pe})
                return round(reported_pe, 2), "Yahoo trailing P/E (not cross-checked)"
        elif valid(reported_pe):
            debug.setdefault("warnings", []).append(
                f"Yahoo trailing P/E {reported_pe:.2f}x exceeds the automated sanity limit; manual verification required."
            )

        if valid(eps_pe) and 0 < eps_pe <= MAX_AUTOMATED_PE:
            debug["valuation"].update({"selected_source": "Price / Yahoo trailing EPS",
                                       "selected_pe": eps_pe})
            return round(eps_pe, 2), "Price / Yahoo trailing EPS"

        if valid(eps_pe) and eps_pe > MAX_AUTOMATED_PE:
            debug.setdefault("warnings", []).append(
                f"Price / trailing EPS implies {eps_pe:.2f}x P/E; automated valuation suppressed for manual review."
            )

        qfin = data.get("qfin")
        cols = dated_columns(qfin) if qfin is not None else []
        if len(cols) >= 4:
            latest = cols[:4]
            dates = [dt for dt, _ in latest]
            gaps_ok = all(70 <= (dates[i] - dates[i + 1]).days <= 110 for i in range(3))
            quarterly_pat = [get_pat(qfin, col) for _, col in latest]
            if gaps_ok and all(valid(x) for x in quarterly_pat):
                ttm_pat = float(sum(quarterly_pat))
                mcap, mcap_source = get_market_cap(data, price)
                debug["valuation"].update({
                    "quarterly_pat_latest_four": quarterly_pat,
                    "ttm_pat": ttm_pat,
                    "market_cap": mcap,
                    "market_cap_source": mcap_source,
                    "quarter_dates": [str(d.date()) for d in dates],
                })
                if valid(mcap) and mcap > 0 and valid(ttm_pat) and ttm_pat > 0:
                    pe = mcap / ttm_pat
                    annual_pat = None
                    annual_fin = data.get("fin")
                    annual_col = latest_column(annual_fin)
                    if annual_col is not None:
                        annual_pat = get_pat(annual_fin, annual_col)
                    annual_check = (ttm_pat / annual_pat if valid(annual_pat) and annual_pat > 0 else None)
                    debug["valuation"]["ttm_pat_to_latest_annual_pat"] = annual_check
                    annual_scale_ok = (not valid(annual_check) or 0.50 <= annual_check <= 2.00)

                    eps_scale_check = None
                    annual_eps_ok = True
                    shares = number(info.get("sharesOutstanding"))
                    if valid(annual_pat) and annual_pat > 0 and valid(shares) and shares > 0 and annual_col is not None:
                        annual_eps_field = next((field for field in ("Diluted EPS", "Basic EPS")
                                                 if valid(safe_get(annual_fin, field, annual_col))), None)
                        reported_annual_eps = (safe_get(annual_fin, annual_eps_field, annual_col)
                                               if annual_eps_field else None)
                        if valid(reported_annual_eps) and reported_annual_eps > 0:
                            derived_annual_eps = annual_pat / shares
                            eps_scale_check = derived_annual_eps / reported_annual_eps
                            annual_eps_ok = 0.70 <= eps_scale_check <= 1.30
                            debug["valuation"].update({
                                "annual_eps_field": annual_eps_field,
                                "reported_annual_eps": reported_annual_eps,
                                "pat_divided_by_shares_eps": derived_annual_eps,
                                "pat_to_reported_eps_scale_check": eps_scale_check,
                            })
                            if not annual_eps_ok:
                                debug.setdefault("warnings", []).append(
                                    "Annual PAT / shares does not reconcile to reported annual EPS; "
                                    "market-cap / TTM PAT P/E fallback suppressed for unit verification."
                                )

                    if 0 < pe <= MAX_AUTOMATED_PE and annual_scale_ok and annual_eps_ok:
                        debug["valuation"].update({"selected_source": "Market cap / TTM PAT",
                                                   "selected_pe": pe})
                        if not (quote_ccy and financial_ccy):
                            debug.setdefault("warnings", []).append(
                                "Calculated P/E uses Yahoo statement units; currency metadata is incomplete. Verify with filings."
                            )
                        return round(pe, 2), "Market cap / TTM PAT (calculated fallback)"
                    debug.setdefault("warnings", []).append(
                        f"Calculated P/E suppressed (candidate {pe:.2f}x; annual/TTM PAT scale check {annual_check}; "
                        f"PAT/share EPS scale check {eps_scale_check})."
                    )

    if currency_conflict:
        note = "P/E unavailable because quote and financial statement currencies conflict"
    else:
        note = "P/E unavailable or failed consistency checks; verify trailing EPS and PAT in filings"
    debug["valuation"].update({"selected_source": "Unavailable", "selected_pe": None,
                               "reason": note})
    debug.setdefault("warnings", []).append(note)
    return None, note


# ============================================================
# DEBT AND INTEREST COVERAGE
# ============================================================
def get_de_ratio(multi):
    if not multi:
        return None, "No aligned balance sheet", None, False
    latest = multi[-1]
    debt, equity = latest.get("gross_debt"), latest.get("equity")
    leases, assumed = latest.get("lease_liabilities"), latest.get("debt_assumed", False)
    if not valid(equity) or equity <= 0:
        return None, "Equity unavailable or non-positive", leases, assumed
    if not valid(debt):
        return None, latest.get("debt_source", "Debt unavailable"), leases, assumed
    if debt < 0:
        return None, "Invalid negative debt", leases, assumed
    return debt / equity, latest.get("debt_source", ""), leases, assumed


def get_interest_coverage(multi):
    if not multi:
        return None, "No aligned data"
    latest = multi[-1]
    ebit, interest, debt = latest.get("ebit"), latest.get("interest_expense"), latest.get("gross_debt")
    if valid(debt) and debt == 0:
        return None, "Debt-free; coverage not applicable"
    if valid(ebit) and valid(interest) and interest > 0:
        return ebit / interest, "EBIT / gross interest expense"
    if valid(interest) and interest == 0 and valid(debt) and debt > 0:
        return None, "Zero interest despite borrowing; verify"
    return None, "Not calculable"


# ============================================================
# LAYER 1 - FUNDAMENTAL SCREEN
# ============================================================
def is_financial(info, ticker=""):
    ticker = clean_ticker(ticker)
    if ticker in MARKET_INFRASTRUCTURE_TICKERS:
        return False
    combined = " ".join([
        str(info.get("sector") or ""),
        str(info.get("industry") or ""),
        str(info.get("shortName") or info.get("longName") or ticker),
    ]).lower()
    return any(k in combined for k in FINANCIAL_KW)


def run_layer1(data, ticker, multi):
    info = data["info"]
    price = current_price(data)
    mcap, mcap_source = get_market_cap(data, price)
    latest = multi[-1] if multi else {}
    roe, roce = latest.get("roe"), latest.get("roce")
    de, de_source, leases, debt_assumed = get_de_ratio(multi)
    int_cov, int_cov_source = get_interest_coverage(multi)

    sc, sl, ss = annual_cagr(data.get("fin"), get_revenue)
    if sc is None:
        sc, sl, ss = quarterly_ttm_growth(data.get("qfin"), get_revenue)
    pc, pl, ps = annual_cagr(data.get("fin"), get_pat)
    if pc is None:
        pc, pl, ps = quarterly_ttm_growth(data.get("qfin"), get_pat)

    eps_basis = "Unavailable"
    eg, el, es = None, "", "EPS history unavailable"
    annual_field = choose_eps_field(data.get("fin"))
    if annual_field:
        eg, el, es = annual_cagr(data.get("fin"), eps_getter(data.get("fin")))
        if eg is not None:
            eps_basis = f"Annual {annual_field}"
    if eg is None:
        q_field = choose_eps_field(data.get("qfin"))
        if q_field:
            eg, el, es = quarterly_ttm_growth(data.get("qfin"), eps_getter(data.get("qfin")))
            if eg is not None:
                eps_basis = f"Quarterly {q_field} (TTM proxy)"

    pe, pe_source = get_pe(data, price)

    relaxed = bool(valid(roe) and roe > MATURE_ROE and valid(mcap) and mcap > MATURE_MCAP)
    floor = MATURE_GROWTH if relaxed else MIN_GROWTH

    checks = {
        "mcap_pass": bool(valid(mcap) and mcap >= MIN_MARKET_CAP),
        "roe_pass": bool(valid(roe) and roe > MIN_ROE),
        "roce_pass": bool(valid(roce) and roce > MIN_ROCE),
        "de_pass": bool(valid(de) and 0 <= de < MAX_DEBT_TO_EQUITY),
        "growth_pass": bool(valid(sc) and valid(pc) and sc > floor and pc > floor),
    }
    names = {"mcap_pass": "Market cap", "roe_pass": "ROE", "roce_pass": "ROCE",
             "de_pass": "Debt/equity", "growth_pass": "Growth"}
    failed = [names[k] for k, v in checks.items() if not v]

    debt, cash = latest.get("gross_debt"), latest.get("cash")
    net_debt_cr = (debt - cash) / 1e7 if valid(debt) and valid(cash) else None

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or short_ticker(ticker),
        "sector": info.get("sector") or info.get("industry") or "Unknown",
        "price": price, "mcap": mcap,
        "mcap_cr": mcap / 1e7 if valid(mcap) else None, "mcap_source": mcap_source,
        "pe": pe, "pe_source": pe_source,
        "roe": roe, "roce": roce,
        "de": de, "de_source": de_source, "leases": leases, "debt_assumed": debt_assumed,
        "int_cov": int_cov, "int_cov_source": int_cov_source,
        "net_debt_cr": net_debt_cr,
        "sales_cagr": sc, "sales_label": sl, "sales_source": ss,
        "pat_cagr": pc, "pat_label": pl, "pat_source": ps,
        "eps_growth": eg, "eps_label": el, "eps_source": es, "eps_basis": eps_basis,
        "growth_floor": floor, "growth_relaxed": relaxed,
        **checks,
        "failed": failed,
        "pass": not failed,
    }


# ============================================================
# FORENSIC ACCOUNTING + WORKING CAPITAL
# ============================================================
def empty_forensic():
    return {
        "score": None, "flags": ["Insufficient aligned financial history"], "num_flags": 1,
        "complete": False, "cum_cfo_pat": None, "cum_cfo_ebitda": None,
        "recent3_cfo_pat": None, "recent3_cfo_ebitda": None,
        "long_window_cfo_pat": None, "long_window_cfo_ebitda": None,
        "five_year_cfo_pat": None, "five_year_cfo_ebitda": None,
        "prior2_cfo_pat": None, "prior2_cfo_ebitda": None,
        "long_window_label": "unavailable", "prior_window_label": "unavailable",
        "cash_conversion_trend": "unavailable",
        "cash_conversion_delta": None, "recent3_years": [], "long_window_years": [],
        "accrual_ratio": None, "cfo_ebitda_trend": [], "margins_by_year": [],
        "margin_trend": "unavailable", "ccc": None, "ccc_trend": None,
        "int_coverage_trend": [], "cash_deductions": 0,
        "paired_pat_years": 0, "paired_ebitda_years": 0,
    }


def cash_conversion_window(window_rows, denominator_key, min_years=3):
    """Return cumulative CFO/denominator only for a complete consecutive window."""
    if len(window_rows) < min_years:
        return None
    if not all(consecutive_annual_periods(window_rows[i]["period_end"], window_rows[i + 1]["period_end"])
               for i in range(len(window_rows) - 1)):
        return None
    if not all(valid(r.get("cfo")) and valid(r.get(denominator_key)) for r in window_rows):
        return None
    denominator = float(sum(r[denominator_key] for r in window_rows))
    if denominator <= 0:
        return None
    return float(sum(r["cfo"] for r in window_rows)) / denominator


def run_forensic(records):
    if len(records) < 2:
        return empty_forensic()

    rows = sorted(records, key=lambda x: x["period_end"])
    latest, prior = rows[-1], rows[-2]
    adjacent_latest = consecutive_annual_periods(prior["period_end"], latest["period_end"])
    score, flags, cash_ded = 100, [], 0

    # 1. Receivables as % of revenue rising for three consecutive annual periods
    recv_pct = []
    for r in rows:
        rev, recv = r.get("revenue"), r.get("receivables")
        recv_pct.append(recv / rev * 100 if valid(rev) and rev > 0 and valid(recv) and recv >= 0 else None)
    for i in range(2, len(recv_pct)):
        consecutive = (consecutive_annual_periods(rows[i - 2]["period_end"], rows[i - 1]["period_end"])
                       and consecutive_annual_periods(rows[i - 1]["period_end"], rows[i]["period_end"]))
        w = recv_pct[i - 2:i + 1]
        if consecutive and all(valid(x) for x in w) and w[0] < w[1] < w[2]:
            flags.append(f"Receivables rising: {w[0]:.1f}% -> {w[1]:.1f}% -> {w[2]:.1f}% of revenue")
            score -= 15
            break

    # 2. Inventory bloat
    def material(r):
        inv, rev, assets = r.get("inventory"), r.get("revenue"), r.get("assets")
        if not valid(inv) or inv <= 0:
            return False
        by_rev = valid(rev) and rev > 0 and inv / rev * 100 >= INVENTORY_MATERIAL_PCT
        by_assets = valid(assets) and assets > 0 and inv / assets >= 0.02
        return bool(by_rev or by_assets)

    il, ip = latest.get("inventory"), prior.get("inventory")
    rl, rp = latest.get("revenue"), prior.get("revenue")
    if adjacent_latest and all(valid(x) and x > 0 for x in (il, ip, rl, rp)) and material(latest) and material(prior):
        ig, rg = (il / ip - 1) * 100, (rl / rp - 1) * 100
        if ig >= rg + 10:
            flags.append(f"Material inventory growth {ig:.1f}% vs revenue growth {rg:.1f}%")
            score -= 15

    # Primary scoring window: latest 3 consecutive years
    # Long context window: whatever consecutive annual history is available (3-5 years)
    recent_rows = rows[-3:]
    recent3_periods_consecutive = len(recent_rows) == 3 and all(
        consecutive_annual_periods(recent_rows[i]["period_end"], recent_rows[i + 1]["period_end"])
        for i in range(2)
    )

    # Walk backwards from the latest year and include every row that is consecutive
    long_rows = [rows[-1]]
    for i in range(len(rows) - 2, -1, -1):
        if consecutive_annual_periods(rows[i]["period_end"], long_rows[0]["period_end"]):
            long_rows.insert(0, rows[i])
        else:
            break
    long_rows = long_rows[-min(5, len(long_rows)):]

    recent3_cfo_pat = cash_conversion_window(recent_rows, "pat")
    recent3_cfo_ebitda = cash_conversion_window(recent_rows, "ebitda")
    long_window_cfo_pat = cash_conversion_window(long_rows, "pat",
                                                  min_years=min(3, len(long_rows)))
    long_window_cfo_ebitda = cash_conversion_window(long_rows, "ebitda",
                                                     min_years=min(3, len(long_rows)))

    recent3_pat_pairs = sum(valid(r.get("cfo")) and valid(r.get("pat")) for r in recent_rows)
    recent3_ebitda_pairs = sum(valid(r.get("cfo")) and valid(r.get("ebitda")) for r in recent_rows)
    long_window_years = [r["year"] for r in long_rows]
    recent3_years = [r["year"] for r in recent_rows]

    long_window_label = (f"{len(long_rows)}Y" if len(long_rows) >= 3
                         else "not available")
    five_year_cfo_pat = long_window_cfo_pat if len(long_rows) == 5 else None
    five_year_cfo_ebitda = long_window_cfo_ebitda if len(long_rows) == 5 else None

    # Trend comparison: compare recent 3Y with whatever consecutive prior years exist.
    # Accepts 1 or 2 prior years. With Yahoo's typical 4-year history, this gives us
    # recent 3Y vs preceding 1Y. Better than "unavailable" even if noisier than 2Y.
    prior_rows_candidates = [r for r in long_rows if r not in recent_rows]
    prior_rows = []
    if prior_rows_candidates and recent_rows:
        tail = prior_rows_candidates[-1]
        if consecutive_annual_periods(tail["period_end"], recent_rows[0]["period_end"]):
            # Take up to 2 prior years, whichever are consecutive
            prior_rows = prior_rows_candidates[-2:] if len(prior_rows_candidates) >= 2 else prior_rows_candidates[-1:]
            # If we took 2, verify they're also consecutive with each other
            if len(prior_rows) == 2 and not consecutive_annual_periods(
                prior_rows[0]["period_end"], prior_rows[1]["period_end"]
            ):
                prior_rows = prior_rows[-1:]  # keep only the one adjacent to recent

    prior_window_label = f"{len(prior_rows)}Y" if prior_rows else "unavailable"
    # min_years=1 allows a single-year prior window (noisier but useful signal)
    prior2_cfo_pat = (cash_conversion_window(prior_rows, "pat", min_years=1)
                      if prior_rows else None)
    prior2_cfo_ebitda = (cash_conversion_window(prior_rows, "ebitda", min_years=1)
                         if prior_rows else None)
    cum_cfo_pat = recent3_cfo_pat
    cum_cfo_ebitda = recent3_cfo_ebitda
    cash_conversion_delta = None
      cash_conversion_trend = (
        f"unavailable - no consecutive prior years (actual history: {long_window_label})"
    )
      if valid(recent3_cfo_pat) and valid(prior2_cfo_pat):
        cash_conversion_delta = recent3_cfo_pat - prior2_cfo_pat
        # Require a bigger delta for 1Y prior windows (noisier); gentler for 2Y
        threshold = 0.25 if len(prior_rows) == 1 else 0.15
        noisy_suffix = " [noisy - 1Y prior only]" if len(prior_rows) == 1 else ""

        if cash_conversion_delta >= threshold:
            cash_conversion_trend = f"improving (vs preceding {prior_window_label}){noisy_suffix}"
        elif cash_conversion_delta <= -threshold:
            cash_conversion_trend = f"deteriorating (vs preceding {prior_window_label}){noisy_suffix}"
            # Only raise a hard flag on 2Y comparisons (1Y is too volatile)
            if len(prior_rows) >= 2:
                flags.append(
                    f"Cash conversion deteriorating: recent 3Y CFO/PAT {recent3_cfo_pat:.2f}x "
                    f"vs preceding {prior_window_label} {prior2_cfo_pat:.2f}x"
                )
        else:
            cash_conversion_trend = f"broadly stable (vs preceding {prior_window_label}){noisy_suffix}"

    # 3. Recent cumulative CFO / PAT
    if valid(cum_cfo_pat):
        if cum_cfo_pat < 0.5:
            flags.append(f"Critical recent 3Y CFO/PAT: {cum_cfo_pat:.2f}x")
            cash_ded += 30
        elif cum_cfo_pat < 0.7:
            flags.append(f"Weak recent 3Y CFO/PAT: {cum_cfo_pat:.2f}x")
            cash_ded += 20

    # 4. Recent cumulative CFO / EBITDA
    if valid(cum_cfo_ebitda):
        if cum_cfo_ebitda < 0.5:
            flags.append(f"Critical recent 3Y CFO/EBITDA: {cum_cfo_ebitda:.2f}x")
            cash_ded += 20
        elif cum_cfo_ebitda < 0.7:
            flags.append(f"Weak recent 3Y CFO/EBITDA: {cum_cfo_ebitda:.2f}x")
            cash_ded += 15

    paired_ebitda_positive = [r for r in rows if valid(r.get("cfo"))
                              and valid(r.get("ebitda")) and r["ebitda"] > 0]
    cfo_ebitda_trend = [(r["year"], r["cfo"] / r["ebitda"]) for r in paired_ebitda_positive]
    latest_ebitda_ratio = (latest["cfo"] / latest["ebitda"]
                           if valid(latest.get("cfo")) and valid(latest.get("ebitda"))
                           and latest["ebitda"] > 0 else None)
    prior_ebitda_ratio = (prior["cfo"] / prior["ebitda"]
                          if valid(prior.get("cfo")) and valid(prior.get("ebitda"))
                          and prior["ebitda"] > 0 else None)
    if (adjacent_latest and valid(latest_ebitda_ratio) and valid(prior_ebitda_ratio)
            and valid(cum_cfo_ebitda) and latest_ebitda_ratio < prior_ebitda_ratio
            and latest_ebitda_ratio < 0.5 and cum_cfo_ebitda < 0.7):
        flags.append(f"CFO/EBITDA deteriorating: {prior_ebitda_ratio:.2f} -> {latest_ebitda_ratio:.2f}")
        cash_ded += 10

    recent3_cfo_values = [r.get("cfo") for r in recent_rows]
    if (recent3_periods_consecutive and len(recent3_cfo_values) == 3
            and all(valid(x) for x in recent3_cfo_values) and sum(recent3_cfo_values) < 0):
        flags.append("Negative cumulative operating cash flow over the recent 3Y window")
        cash_ded += 25

    # 5. Latest-year accrual ratio
    accrual = None
    ni, cfo = latest.get("pat"), latest.get("cfo")
    ta_l, ta_p = latest.get("assets"), prior.get("assets")
    if adjacent_latest and valid(ni) and valid(cfo) and valid(ta_l) and valid(ta_p) and ta_l > 0 and ta_p > 0:
        accrual = (ni - cfo) / ((ta_l + ta_p) / 2) * 100
        if accrual > 20:
            flags.append(f"High accrual ratio: {accrual:.1f}%")
            cash_ded += 20
        elif accrual > 10:
            flags.append(f"Elevated accrual ratio: {accrual:.1f}%")
            cash_ded += 10

    cash_ded = min(cash_ded, 50)
    score = max(0, score - cash_ded)

    # EBITDA margin trend (context only)
    margins = [(r["year"], r["ebitda"] / r["revenue"] * 100) for r in rows
               if valid(r.get("ebitda")) and valid(r.get("revenue")) and r["revenue"] > 0]
    margin_trend = "stable"
    if len(margins) >= 2:
        if margins[-1][1] > margins[0][1] + 2:
            margin_trend = "expanding"
        elif margins[-1][1] < margins[0][1] - 3:
            margin_trend = "contracting"

    int_cov_trend = [(r["year"], r["ebit"] / r["interest_expense"]) for r in rows
                     if valid(r.get("ebit")) and valid(r.get("interest_expense")) and r["interest_expense"] > 0]

    # Cash conversion cycle
    ccc_rows = []
    for i in range(1, len(rows)):
        cur, prev = rows[i], rows[i - 1]
        if not consecutive_annual_periods(prev["period_end"], cur["period_end"]):
            continue
        rev, cogs = cur.get("revenue"), cur.get("cogs")
        needed = [cur.get("receivables"), prev.get("receivables"),
                  cur.get("payables"), prev.get("payables"), rev, cogs]
        if not all(valid(x) for x in needed) or rev <= 0 or cogs <= 0:
            continue
        notes = []
        dso = (cur["receivables"] + prev["receivables"]) / 2 / rev * 365
        dpo = (cur["payables"] + prev["payables"]) / 2 / cogs * 365
        inv_c, inv_p = cur.get("inventory"), prev.get("inventory")
        if (valid(inv_c) and valid(inv_p) and inv_c >= 0 and inv_p >= 0
                and inv_c / rev * 100 >= INVENTORY_MATERIAL_PCT):
            dio = (inv_c + inv_p) / 2 / cogs * 365
        else:
            dio = 0.0
            notes.append("inventory immaterial/not reported, DIO excluded")
        reliable = True
        if cogs / rev < 0.10:
            reliable = False
            notes.append("cost of revenue under 10% of sales, DIO/DPO may be overstated")
        ccc_rows.append({"year": cur["year"], "period_end": cur["period_end"],
                         "dso": dso, "dio": dio, "dpo": dpo,
                         "ccc": dso + dio - dpo, "notes": notes, "reliable": reliable})

    ccc_latest = ccc_rows[-1] if ccc_rows else None
    ccc_prior = ccc_rows[-2] if len(ccc_rows) >= 2 else None
    ccc_trend = None
    if (ccc_latest and ccc_prior and
            consecutive_annual_periods(ccc_prior["period_end"], ccc_latest["period_end"])):
        diff = ccc_latest["ccc"] - ccc_prior["ccc"]
        ccc_trend = "worsening" if diff > 10 else "improving" if diff < -5 else "stable"
        if ccc_prior["dpo"] > 0:
            dpo_growth = (ccc_latest["dpo"] / ccc_prior["dpo"] - 1) * 100
            if dpo_growth > 30 and valid(cum_cfo_pat) and cum_cfo_pat < 0.7:
                flags.append(f"Payables stretch: DPO +{dpo_growth:.0f}% with weak recent 3Y cash conversion")

    annual_periods = [r["year"] for r in rows]
    complete = bool(
        recent3_periods_consecutive
        and recent3_pat_pairs == 3
        and recent3_ebitda_pairs == 3
        and valid(cum_cfo_pat) and valid(cum_cfo_ebitda)
    )

    return {
        "score": round(score), "flags": flags, "num_flags": len(flags), "complete": complete,
        "cum_cfo_pat": cum_cfo_pat, "cum_cfo_ebitda": cum_cfo_ebitda, "accrual_ratio": accrual,
        "cfo_ebitda_trend": cfo_ebitda_trend, "margins_by_year": margins,
        "margin_trend": margin_trend, "ccc": ccc_latest, "ccc_trend": ccc_trend,
        "int_coverage_trend": int_cov_trend, "cash_deductions": cash_ded,
        "paired_pat_years": recent3_pat_pairs, "paired_ebitda_years": recent3_ebitda_pairs,
        "annual_periods": annual_periods,
        "history_start": annual_periods[0] if annual_periods else None,
        "history_end": annual_periods[-1] if annual_periods else None,
        "recent3_years": recent3_years,
        "recent3_cfo_pat": recent3_cfo_pat,
        "recent3_cfo_ebitda": recent3_cfo_ebitda,
        "recent3_paired_pat_years": recent3_pat_pairs,
        "recent3_paired_ebitda_years": recent3_ebitda_pairs,
        "long_window_years": long_window_years,
        "long_window_label": long_window_label,
        "long_window_cfo_pat": long_window_cfo_pat,
        "long_window_cfo_ebitda": long_window_cfo_ebitda,
        "five_year_cfo_pat": five_year_cfo_pat,
        "five_year_cfo_ebitda": five_year_cfo_ebitda,
        "prior2_cfo_pat": prior2_cfo_pat,
        "prior2_cfo_ebitda": prior2_cfo_ebitda,
        "prior_window_label": prior_window_label,
        "cash_conversion_trend": cash_conversion_trend,
        "cash_conversion_delta": cash_conversion_delta,
    }


# ============================================================
# MOAT PROXY AND CYCLICALITY
# ============================================================
def run_moat(records):
    rows = sorted(records, key=lambda x: x["period_end"])
    roe_rows = [(r["year"], r["roe"], r["period_end"]) for r in rows if valid(r.get("roe"))]
    if not roe_rows:
        return {"pct": 0, "consistency": "no data", "years_above_15": 0,
                "total_years": 0, "roe_by_year": []}

    values = [v for _, v, _ in roe_rows]
    total = len(values)
    above = sum(v > MIN_ROE for v in values)

    if total >= 4 and above == total:
        consistency = "established moat proxy"
    elif (total >= 3 and values[-1] > MIN_ROE and values[-2] > MIN_ROE
          and values[-1] >= float(np.mean(values[:-2]))):
        consistency = "emerging moat proxy"
    else:
        consistency = "weak/unproven moat"

    return {"pct": round(above / total * 100), "consistency": consistency,
            "years_above_15": above, "total_years": total,
            "roe_by_year": [(y, v) for y, v, _ in roe_rows]}


def run_cyclical(records):
    rows = sorted(records, key=lambda x: x["period_end"])
    valid_rows = [r for r in rows if valid(r.get("roe"))]
    values = [r["roe"] for r in valid_rows]
    if len(values) < 4 or not all(
        consecutive_annual_periods(valid_rows[i]["period_end"], valid_rows[i + 1]["period_end"])
        for i in range(len(valid_rows) - 1)
    ):
        return {"peak": None, "latest": values[-1] if values else None,
                "average_prior": None, "median": None,
                "note": "At least four consecutive annual ROE values required; FULL is not permitted without this check"}
    hist = values[:-1]
    avg_prior, med = float(np.mean(hist)), float(np.median(hist))
    latest = values[-1]
    peak = bool(avg_prior > 0 and latest > 2 * avg_prior and latest > 20)
    return {"peak": peak, "latest": latest, "average_prior": avg_prior, "median": med,
            "note": "Latest ROE vs average of prior years"}


# ============================================================
# EARNINGS MOMENTUM (YoY, same quarter last year)
# ============================================================
def quarterly_series(qfin):
    field = choose_eps_field(qfin)
    if field is not None:
        getter, basis = (lambda f, c: safe_get(f, field, c)), f"Quarterly {field}"
    else:
        getter, basis = get_pat, "Quarterly PAT (EPS unavailable)"
    records = []
    for dt, col in dated_columns(qfin):
        v = getter(qfin, col)
        if valid(v):
            records.append((dt, v))
    return records, basis


def match_same_quarter(records, current_date):
    best = None
    for dt, value in records:
        if dt >= current_date:
            continue
        days = (current_date - dt).days
        if 330 <= days <= 400 and (best is None or abs(days - 365) < best[0]):
            best = (abs(days - 365), value)
    return best[1] if best else None


def yoy_growth(current, previous):
    if previous > 0:
        return (current / previous - 1) * 100, "calculable"
    if previous <= 0 < current:
        return None, "turned profitable"
    if current <= 0 < previous:
        return None, "moved from profit to loss"
    return None, "both periods loss-making"


def run_momentum(qfin):
    records, basis = quarterly_series(qfin)
    out = {"available": False, "score": None, "latest_yoy": None, "prior_yoy": None,
           "status": "Insufficient quarterly data", "basis": basis, "records": records[:8]}
    if len(records) < 2:
        return out

    latest_dt, latest_value = records[0]
    latest_prev = match_same_quarter(records, latest_dt)
    ly, ls = (yoy_growth(latest_value, latest_prev)
              if valid(latest_prev) else (None, "no same quarter last year"))

    second_dt, second_value = records[1]
    second_is_adjacent = 70 <= (latest_dt - second_dt).days <= 110
    if second_is_adjacent:
        second_prev = match_same_quarter(records, second_dt)
        py, ps = (yoy_growth(second_value, second_prev)
                  if valid(second_prev) else (None, "no same quarter last year"))
    else:
        py, ps = None, "previous reported quarter is not adjacent; not used"

    if valid(ly) and valid(py):
        if ly > 0 and py > 0:
            score = 90 if ly > py else 75
            status = "Accelerating YoY earnings growth" if score == 90 else "Positive but decelerating YoY growth"
        elif ly < 0 and py < 0:
            score, status = 15, "YoY earnings declining in both recent quarters"
        else:
            score, status = 40, "Mixed YoY earnings momentum"
    elif valid(ly):
        score = 70 if ly > 0 else 25
        status = "Latest quarter only: " + ("YoY growth positive" if ly > 0 else "YoY decline")
    elif "turned profitable" in (ls, ps):
        score, status = 60, "Turned profitable; % growth not comparable"
    elif "moved from profit to loss" in (ls, ps):
        score, status = 10, "Moved from profit to loss"
    else:
        score, status = None, "No matching same-quarter comparison"

    out.update({"available": score is not None, "score": score,
                "latest_yoy": ly, "prior_yoy": py, "status": status})
    return out


# ============================================================
# OWNERSHIP
# ============================================================
def load_ownership(upload):
    if upload is None:
        return {}
    try:
        df = pd.read_csv(upload)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"ticker", "promoter_pct"}.issubset(df.columns):
            st.sidebar.error("Ownership CSV needs at least: ticker, promoter_pct")
            return {}
        pledge_col = next((c for c in PLEDGE_COLUMNS if c in df.columns), None)
        out = {}
        for _, row in df.iterrows():
            raw = row.get("ticker")
            if pd.isna(raw) or not str(raw).strip():
                continue
            t = clean_ticker(raw)
            promoter = number(row.get("promoter_pct"))
            inst = number(row.get("institutional_pct"))
            pledge = number(row.get(pledge_col)) if pledge_col else None
            for x in (promoter, inst, pledge):
                if valid(x) and not 0 <= x <= 100:
                    raise ValueError(f"{t}: percentages must be between 0 and 100")
            as_of_raw = row.get("as_of", "")
            source_raw = row.get("source", "User CSV")
            as_of_text = "" if pd.isna(as_of_raw) else str(as_of_raw).strip()
            source_text = "User CSV" if pd.isna(source_raw) or not str(source_raw).strip() else str(source_raw).strip()
            out[t] = {"promoter_pct": promoter, "institutional_pct": inst, "pledge_pct": pledge,
                      "as_of": as_of_text, "source": source_text}
        return out
    except Exception as exc:
        st.sidebar.error(f"Ownership CSV error: {exc}")
        return {}


def run_holdings(yahoo, override):
    override = override or {}
    promoter = number(override.get("promoter_pct"))
    inst = number(override.get("institutional_pct"))
    pledge = number(override.get("pledge_pct"))
    flags, adj = [], 0

    if valid(promoter):
        if promoter < 20:
            flags.append(f"Low promoter holding {promoter:.1f}% (maximum STANDARD)")
            adj = -10
        elif promoter > 75:
            flags.append(f"High promoter concentration {promoter:.1f}% (review minority-shareholder protections)")
    else:
        flags.append("Verified promoter data unavailable (upload ownership CSV)")

    if valid(inst) and inst < 5:
        flags.append(f"Institutional ownership below 5% ({inst:.1f}%)")
    if valid(pledge) and pledge > 0:
        flags.append(f"Promoter pledge {pledge:.1f}% of promoter holding")

    ownership_supplied = any(valid(x) for x in (promoter, inst, pledge))
    as_of = str(override.get("as_of") or "").strip()
    if ownership_supplied:
        if not as_of:
            flags.append("Ownership snapshot date missing; freshness cannot be verified")
        else:
            parsed_as_of = pd.to_datetime(as_of, errors="coerce", dayfirst=True)
            if pd.isna(parsed_as_of):
                flags.append("Ownership snapshot date invalid; freshness cannot be verified")
            else:
                snapshot_date = pd.Timestamp(parsed_as_of).tz_localize(None).normalize()
                age_days = (pd.Timestamp.now().normalize() - snapshot_date).days
                if age_days < -30:
                    flags.append("Ownership snapshot date is unexpectedly in the future")
                elif age_days > 180:
                    flags.append(f"Ownership snapshot is stale ({age_days} days old); refresh shareholding data")

    return {"promoter_pct": promoter, "institutional_pct": inst, "pledge_pct": pledge,
            "yahoo_insider_pct": yahoo.get("yahoo_insider_pct"),
            "yahoo_institutional_pct": yahoo.get("yahoo_institutional_pct"),
            "flags": flags, "score_adj": adj,
            "as_of": as_of or None, "source": override.get("source")}


# ============================================================
# VALUATION
# ============================================================
def run_valuation(pe, eps_growth, sales_growth, data, mcap):
    peg, growth_used, growth_raw, turnaround, eps_negative = None, None, None, False, False
    if not valid(pe) or pe <= 0:
        note = "P/E unavailable or non-positive"
    elif not valid(eps_growth):
        note = "EPS growth unavailable"
    elif eps_growth <= 0:
        eps_negative = True
        note = f"EPS growth {eps_growth:.1f}% is not positive; PEG not meaningful"
    else:
        growth_raw = float(eps_growth)
        if valid(sales_growth) and sales_growth > 0 and eps_growth > 2 * sales_growth:
            growth_raw, turnaround = float(sales_growth), True
        growth_used = growth_raw
        peg = pe / growth_used
        note = f"P/E {pe:.1f} / growth {growth_used:.1f}%"
        if turnaround:
            note += " (sales growth used: EPS growth > 2x sales)"
        if growth_used > 30:
            note += " (growth above 30%; verify sustainability manually; no automatic cap)"

    fcf_yield = None
    cf = data.get("cf")
    col = latest_column(cf)
    info = data.get("info", {}) or {}
    quote_ccy = str(info.get("currency") or "").upper()
    financial_ccy = str(info.get("financialCurrency") or "").upper()
    currency_conflict = bool(quote_ccy and financial_ccy and quote_ccy != financial_ccy)
    if valid(mcap) and mcap > 0 and col is not None and not currency_conflict:
        fcf = get_fcf(cf, col)
        if valid(fcf):
            fcf_yield = fcf / mcap * 100
    elif currency_conflict:
        data.setdefault("debug", {}).setdefault("warnings", []).append(
            "FCF yield withheld because quote and statement currencies conflict."
        )

    return {"peg": round(peg, 2) if valid(peg) else None,
            "peg_growth": growth_used, "peg_growth_raw": growth_raw,
            "peg_turnaround": turnaround, "eps_negative": eps_negative, "peg_note": note,
            "fcf_yield": fcf_yield, "pe_extreme": bool(valid(pe) and pe > 80)}


# ============================================================
# COMPOSITE RANKING
# ============================================================
def score_peg(peg):
    if not valid(peg) or peg <= 0:
        return None
    if peg <= 0.5:
        return 100.0
    if peg <= 1:
        return 80 + (1 - peg) * 40
    if peg <= 1.5:
        return 60 + (1.5 - peg) * 40
    if peg <= 2:
        return 40 + (2 - peg) * 40
    if peg <= 3:
        return 10 + (3 - peg) * 30
    return max(0.0, 10 - (peg - 3) * 5)


def score_cfo(x):
    if not valid(x):
        return None
    if x >= 1.2:
        return 100.0
    if x >= 1:
        return 90.0
    if x >= 0.7:
        return 65 + (x - 0.7) / 0.3 * 25
    if x >= 0.5:
        return 35 + (x - 0.5) / 0.2 * 30
    return max(0.0, x / 0.5 * 35)


def score_roe(roe):
    if not valid(roe):
        return None
    return float(np.clip((roe - 5) / 30 * 100, 0, 100))


WEIGHTS = {"PEG": 25, "Forensic": 20, "Earnings momentum": 15,
           "CFO quality": 15, "Moat consistency": 15, "ROE": 10}


def calculate_composite(l1, acct, moat, mom, val):
    comps = {
        "PEG": score_peg(val.get("peg")),
        "Forensic": acct.get("score") if acct.get("complete") else None,
        "Earnings momentum": mom.get("score"),
        "CFO quality": score_cfo(acct.get("cum_cfo_pat")) if acct.get("complete") else None,
        "Moat consistency": moat.get("pct") if moat.get("total_years", 0) >= 2 else None,
        "ROE": score_roe(l1.get("roe")),
    }
    coverage = sum(WEIGHTS[k] for k, v in comps.items() if valid(v))
    if coverage == 0:
        return None, 0, comps
    score = sum(comps[k] * WEIGHTS[k] for k in comps if valid(comps[k])) / coverage
    return round(score, 2), coverage, comps


# ============================================================
# ALLOCATION RULES
# ============================================================
def get_tier(score, num_flags, cyclical_peak, peg, eps_negative, one_year_ret,
             moat_status, holdings, interest_coverage, de, failed_filters, forensic_complete):
    if not valid(score):
        return "WATCH", TIER_SIZE["WATCH"], "Forensic score unavailable (insufficient aligned data)"
    if score < 50:
        return "WATCH", TIER_SIZE["WATCH"], f"Forensic score {score:.0f} below 50"
    if valid(peg) and peg > 5:
        return "WATCH", TIER_SIZE["WATCH"], f"PEG {peg:.2f} above 5"
    if len(failed_filters) >= 2:
        return "WATCH", TIER_SIZE["WATCH"], f"Failed {len(failed_filters)} hard filters: {', '.join(failed_filters)}"

    moat_ok = moat_status in ("established moat proxy", "emerging moat proxy")
    if (score >= 85 and num_flags == 0 and cyclical_peak is False and moat_ok
            and valid(peg) and peg <= 1 and forensic_complete):
        tier, why = "FULL", "all FULL conditions met"
    elif score >= 70 and num_flags <= 1 and valid(peg) and peg <= 1.5:
        tier = "STANDARD"
        miss = []
        if score < 85:
            miss.append(f"score {score:.0f} < 85")
        if num_flags:
            miss.append(f"{num_flags} flag")
        if not moat_ok:
            miss.append("moat not proven")
        if peg > 1:
            miss.append(f"PEG {peg:.2f} > 1.0")
        if cyclical_peak is True:
            miss.append("cyclical peak")
        elif cyclical_peak is None:
            miss.append("cyclical data unavailable")
        if not forensic_complete:
            miss.append("forensic data incomplete")
        why = "short of FULL: " + (", ".join(miss) or "mixed")
    else:
        tier = "HALF"
        miss = []
        if score < 70:
            miss.append(f"score {score:.0f} < 70")
        if num_flags > 1:
            miss.append(f"{num_flags} flags")
        if not valid(peg):
            miss.append("PEG not available")
        elif peg > 1.5:
            miss.append(f"PEG {peg:.2f} > 1.5")
        why = "short of STANDARD: " + (", ".join(miss) or "mixed profile")

    reasons = [f"Base {tier} ({why})"]

    def cap(target, reason):
        nonlocal tier
        if TIER_RANK[tier] > TIER_RANK[target]:
            tier = target
            reasons.append(reason)

    if len(failed_filters) == 1:
        cap("HALF", f"Failed hard filter: {failed_filters[0]}")
    if valid(interest_coverage) and interest_coverage < 3:
        cap("HALF", "Interest coverage below 3x")
    elif not valid(interest_coverage) and valid(de) and de >= 0.1:
        cap("STANDARD", "Interest coverage unavailable despite material debt")
    if not valid(peg):
        cap("HALF", "EPS growth negative; PEG not meaningful" if eps_negative else "PEG unavailable")
    elif peg > 3:
        cap("HALF", "PEG above 3")
    if not forensic_complete:
        cap("STANDARD", "Multi-year forensic data incomplete")
    if holdings.get("score_adj", 0) <= -10:
        cap("STANDARD", "Promoter holding below 20%")
    pledge = holdings.get("pledge_pct")
    if valid(pledge):
        if pledge >= 50:
            cap("HALF", "Promoter pledge 50%+ of promoter holding")
        elif pledge > 25:
            cap("STANDARD", "Promoter pledge above 25% of promoter holding")
    if tier == "FULL" and valid(one_year_ret) and one_year_ret < -30:
        cap("HALF", "1Y price fall over 30%; re-check thesis")

    return tier, TIER_SIZE[tier], "; ".join(reasons)


# ============================================================
# FULL ANALYSIS
# ============================================================
def one_year_return(data):
    """Calculate a genuine ~365-day close-to-close price return; otherwise N/A."""
    prices = data.get("prices")
    if prices is None or len(prices) < 200:
        return None
    try:
        latest_date = pd.Timestamp(prices.index[-1]).tz_localize(None)
        if (pd.Timestamp.now().normalize() - latest_date.normalize()).days > 15:
            return None
        target = latest_date - pd.Timedelta(days=365)
        index = pd.DatetimeIndex(prices.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        eligible = [(d, float(v)) for d, v in zip(index, prices.values) if d <= target]
        if not eligible:
            return None
        old_date, old = eligible[-1]
        new = float(prices.iloc[-1])
        if (target - old_date).days > 15 or old <= 0 or new <= 0:
            return None
        return (new / old - 1) * 100
    except Exception:
        return None


def full_analysis(ticker, ownership):
    ticker = clean_ticker(ticker)
    data = fetch_stock(ticker)
    if not data:
        return None
    if is_financial(data["info"], ticker):
        return {"bank": True, "ticker": ticker,
                "name": data["info"].get("shortName") or short_ticker(ticker)}

    multi, align_warn = build_multi(data.get("fin"), data.get("bs"), data.get("cf"))
    data["debug"]["warnings"].extend(align_warn)
    if multi and multi[-1].get("debt_assumed"):
        data["debug"]["warnings"].append("No borrowing lines in Yahoo data; treated as debt-free. Verify in annual report.")

    l1 = run_layer1(data, ticker, multi)
    acct = run_forensic(multi)
    moat = run_moat(multi)
    cyc = run_cyclical(multi)
    mom = run_momentum(data.get("qfin"))
    ret = one_year_return(data)
    val = run_valuation(l1["pe"], l1["eps_growth"], l1["sales_cagr"], data, l1["mcap"])
    holdings = run_holdings(data.get("holdings", {}), ownership.get(ticker, {}))

    decision_flags = list(acct.get("flags", []))
    decision_flags.extend(f for f in holdings.get("flags", []) if "unavailable" not in f.lower())
    if valid(l1.get("int_cov")) and 3 <= l1["int_cov"] < 8:
        decision_flags.append(f"Interest coverage {l1['int_cov']:.1f}x (3x-8x caution band)")
    if val.get("pe_extreme"):
        decision_flags.append(f"Extreme valuation: P/E {l1['pe']:.1f}x above 80x")

    tier, size, reason = get_tier(
        acct.get("score"), len(decision_flags), cyc.get("peak"),
        val.get("peg"), val.get("eps_negative", False), ret, moat.get("consistency", ""),
        holdings, l1.get("int_cov"), l1.get("de"), l1.get("failed", []),
        acct.get("complete", False),
    )
    composite, coverage, comps = calculate_composite(l1, acct, moat, mom, val)

    return {"bank": False, "ticker": ticker, "data": data, "multi": multi, "l1": l1,
            "acct": acct, "moat": moat, "cyc": cyc, "mom": mom, "ret": ret, "val": val,
            "holdings": holdings, "decision_flags": decision_flags,
            "tier": tier, "size": size, "reason": reason,
            "composite": composite, "ranking_coverage": coverage, "components": comps}


def result_row(r):
    l1, a, v, m, h = r["l1"], r["acct"], r["val"], r["moat"], r["holdings"]
    return {
        "ticker": short_ticker(r["ticker"]), "name": l1["name"], "sector": l1["sector"],
        "tier": r["tier"], "size": r["size"], "composite": r["composite"],
        "coverage": r["ranking_coverage"], "forensic": a.get("score"), "flags": a.get("num_flags"),
        "price": l1["price"], "mcap_cr": l1["mcap_cr"], "pe": l1["pe"], "peg": v["peg"],
        "roe": l1["roe"], "roce": l1["roce"], "de": l1["de"], "int_cov": l1["int_cov"],
        "sales_cagr": l1["sales_cagr"], "pat_cagr": l1["pat_cagr"], "eps_growth": l1["eps_growth"],
        "cfo_pat": a.get("recent3_cfo_pat"), "cfo_ebitda": a.get("recent3_cfo_ebitda"),
        "cfo_pat_3y": a.get("recent3_cfo_pat"), "cfo_ebitda_3y": a.get("recent3_cfo_ebitda"),
        "cfo_pat_long_window": a.get("long_window_cfo_pat"),
        "cfo_ebitda_long_window": a.get("long_window_cfo_ebitda"),
        "cfo_pat_5y": a.get("five_year_cfo_pat"),
        "cfo_ebitda_5y": a.get("five_year_cfo_ebitda"),
        "cfo_pat_prior2y": a.get("prior2_cfo_pat"),
        "cash_conversion_trend": a.get("cash_conversion_trend"),
        "accrual_pct": a.get("accrual_ratio"), "moat_pct": m.get("pct"), "moat": m.get("consistency"),
        "momentum": r["mom"].get("score"), "fcf_yield": v["fcf_yield"],
        "net_debt_cr": l1["net_debt_cr"], "promoter_pct": h.get("promoter_pct"),
        "pledge_pct": h.get("pledge_pct"), "return_1y": r["ret"], "reason": r["reason"],
    }


# ============================================================
# LAYERS, VERDICT, SCORECARD
# ============================================================
def gen_layers(r):
    """Returns (name, status, detail). Names ending in ' *' are leading indicators."""
    l1, a, v, m, mom, cyc, h = r["l1"], r["acct"], r["val"], r["moat"], r["mom"], r["cyc"], r["holdings"]
    L = []

    L.append(("Market cap", "pass" if l1["mcap_pass"] else "fail",
              f"Rs {fmt(l1['mcap_cr'], ',.0f')} Cr (minimum Rs 15,000 Cr)"))
    L.append(("ROE", "pass" if l1["roe_pass"] else "fail",
              f"{fmt(l1['roe'])}% (minimum >15%, average equity)"))
    L.append(("ROCE", "pass" if l1["roce_pass"] else "fail",
              f"{fmt(l1['roce'])}% (minimum >18%)"))

    lease_txt = (f"; leases Rs {l1['leases'] / 1e7:,.0f} Cr shown separately"
                 if valid(l1["leases"]) and l1["leases"] > 0 else "")
    L.append(("Debt / equity", "pass" if l1["de_pass"] else "fail",
              f"{fmt(l1['de'], '.2f')}x (maximum <{MAX_DEBT_TO_EQUITY:.2f}x) - {l1['de_source']}{lease_txt}"))

    floor = l1["growth_floor"]
    below = []
    if not valid(l1["sales_cagr"]) or l1["sales_cagr"] <= floor:
        below.append("Sales")
    if not valid(l1["pat_cagr"]) or l1["pat_cagr"] <= floor:
        below.append("PAT")
    g_txt = (f"Sales {fmt(l1['sales_cagr'])}% ({l1['sales_label'] or 'n/a'}), "
             f"PAT {fmt(l1['pat_cagr'])}% ({l1['pat_label'] or 'n/a'}); floor {floor:.0f}%")
    if l1["growth_relaxed"]:
        g_txt += " (mature compounder relaxation)"
    if below:
        g_txt += f"; below floor: {', '.join(below)}"
    L.append(("Sales & PAT growth", "pass" if l1["growth_pass"] else "fail", g_txt))

    ic = l1["int_cov"]
    if valid(ic):
        st_ic = "pass" if ic >= 8 else "warn" if ic >= 3 else "fail"
        L.append(("Interest coverage", st_ic, f"{ic:.1f}x - {l1['int_cov_source']}"))
    elif valid(l1["de"]) and l1["de"] == 0:
        L.append(("Interest coverage", "pass", l1["int_cov_source"]))
    else:
        L.append(("Interest coverage", "warn", l1["int_cov_source"]))

    s = a.get("score")
    provisional = not a.get("complete")
    st_f = ("warn" if s is None else "fail" if s < 50 else "warn" if provisional
            else "pass" if s >= 85 else "warn")
    L.append(("Forensic accounting", st_f,
              f"{fmt(s, '.0f')}/100{' (provisional - incomplete data)' if provisional else ''}, "
              f"{a.get('num_flags', 0)} flags; recent 3Y CFO/PAT {fmt(a.get('cum_cfo_pat'), '.2f')}x, "
              f"recent 3Y CFO/EBITDA {fmt(a.get('cum_cfo_ebitda'), '.2f')}x; "
              f"{a.get('long_window_label', 'longer window')} CFO/PAT {fmt(a.get('long_window_cfo_pat'), '.2f')}x "
              f"({a.get('cash_conversion_trend', 'trend n/a')}); accrual {fmt(a.get('accrual_ratio'))}%"))

    ccc = a.get("ccc")
    if ccc:
        st_c = "warn" if a.get("ccc_trend") == "worsening" or not ccc["reliable"] else "pass"
        c_txt = (f"{ccc['ccc']:.0f} days (DSO {ccc['dso']:.0f}, DIO {ccc['dio']:.0f}, DPO {ccc['dpo']:.0f}); "
                 f"{a.get('ccc_trend') or 'trend n/a'}")
        if ccc["notes"]:
            c_txt += "; " + "; ".join(ccc["notes"])
        L.append(("Working capital *", st_c, c_txt))
    else:
        L.append(("Working capital *", "warn", "Insufficient working-capital data"))

    mc = m.get("consistency", "no data")
    st_m = ("pass" if mc == "established moat proxy" else "warn" if mc in ("emerging moat proxy", "no data")
            else "fail")
    L.append(("Moat proxy", st_m,
              f"ROE >15% in {m.get('years_above_15', 0)}/{m.get('total_years', 0)} years - {mc}"))

    own_bits = []
    if valid(h.get("promoter_pct")):
        own_bits.append(f"Promoter {h['promoter_pct']:.1f}%")
    if valid(h.get("institutional_pct")):
        own_bits.append(f"Institutions {h['institutional_pct']:.1f}%")
    own_txt = ", ".join(own_bits)
    if h["flags"]:
        own_txt = (own_txt + "; " if own_txt else "") + "; ".join(h["flags"])
    L.append(("Promoter / holdings *", "warn" if h["flags"] else "pass", own_txt or "No ownership warning"))

    peg = v.get("peg")
    if valid(peg):
        st_p = "pass" if peg <= 1 else "warn" if peg <= 1.5 else "fail"
        L.append(("PEG valuation", st_p, f"PEG {peg:.2f} - {v['peg_note']}"))
    else:
        L.append(("PEG valuation", "fail" if v.get("eps_negative") else "warn", v.get("peg_note", "N/A")))

       if valid(fcf):
        # Near-zero (±0.5%) is effectively "no FCF cushion" - flag as warn, not pass
        if fcf > 3:
            st_y, suffix = "pass", ""
        elif fcf > 0.5:
            st_y, suffix = "warn", " - thin"
        elif fcf > -0.5:
            st_y, suffix = "warn", " - effectively zero FCF"
        else:
            st_y, suffix = "fail", " - negative"
        L.append(("FCF yield", st_y, f"{fcf:.1f}%{suffix}"))

    if v.get("pe_extreme"):
        L.append(("Extreme valuation", "warn", f"P/E {l1['pe']:.1f}x above 80x"))

    ms = mom.get("score")
    st_mo = "warn" if ms is None else "pass" if ms >= 70 else "fail" if ms <= 25 else "warn"
    mo_txt = (f"{mom.get('status')}; latest YoY {fmt(mom.get('latest_yoy'), '+.1f')}%, "
              f"previous YoY {fmt(mom.get('prior_yoy'), '+.1f')}% ({mom.get('basis')})")
    L.append(("Earnings momentum *", st_mo, mo_txt))

    if cyc.get("peak"):
        L.append(("Cyclical ROE", "warn",
                  f"PEAK: latest {fmt(cyc.get('latest'))}% vs prior average {fmt(cyc.get('average_prior'))}% - use median {fmt(cyc.get('median'))}%"))
    elif valid(cyc.get("average_prior")):
        L.append(("Cyclical ROE", "pass",
                  f"Not at peak: latest {fmt(cyc.get('latest'))}% vs prior average {fmt(cyc.get('average_prior'))}% (median {fmt(cyc.get('median'))}%)"))
    else:
        L.append(("Cyclical ROE", "warn", cyc.get("note", "Unavailable")))

    ret = r.get("ret")
    if valid(ret):
        L.append(("1Y price", "pass" if ret > 0 else "warn" if ret > -30 else "fail",
                  f"{ret:+.1f}%" + (" - re-check thesis" if ret <= -30 else "")))
    else:
        L.append(("1Y price", "warn", "Unavailable"))

    return L


def gen_verdict(r):
    l1, a, v, m, mom, cyc, h = r["l1"], r["acct"], r["val"], r["moat"], r["mom"], r["cyc"], r["holdings"]
    tier = r["tier"]
    openings = {
        "FULL": "meets every condition for maximum conviction.",
        "STANDARD": "clears the quality bar but misses at least one FULL condition.",
        "HALF": "has a mixed profile; strengths exist but so do unresolved risks.",
        "WATCH": "does not currently justify deploying capital.",
    }
    parts = [f"{l1['name']} {openings[tier]}"]
    good, bad = [], []

    if l1["roe_pass"]:
        good.append(f"ROE {l1['roe']:.1f}%")
    if l1["roce_pass"]:
        good.append(f"ROCE {l1['roce']:.1f}%")
    if l1["growth_pass"]:
        good.append(f"sales growth {l1['sales_cagr']:.1f}% and PAT growth {l1['pat_cagr']:.1f}%")
    de = l1["de"]
    if valid(de) and de == 0:
        good.append("debt-free" + (" (assumed, verify)" if l1["debt_assumed"] else ""))
    elif valid(de) and de < 0.5:
        good.append(f"low leverage (D/E {de:.2f}x)")
    if valid(l1["int_cov"]) and l1["int_cov"] > 15:
        good.append(f"interest coverage {l1['int_cov']:.0f}x")
    cp = a.get("cum_cfo_pat")
    if valid(cp) and cp >= 1:
        good.append(f"recent 3Y cash conversion {cp:.2f}x (cash exceeds profit)")
    elif valid(cp) and cp >= 0.7:
        good.append(f"recent 3Y cash conversion {cp:.2f}x")
    margins = a.get("margins_by_year", [])
    if a.get("margin_trend") == "expanding" and len(margins) >= 2:
        good.append(f"EBITDA margins expanding {margins[0][1]:.1f}% -> {margins[-1][1]:.1f}%")
    if m.get("consistency") in ("established moat proxy", "emerging moat proxy"):
        good.append(m["consistency"])
    if valid(v.get("peg")) and v["peg"] <= 1:
        good.append(f"PEG {v['peg']:.2f}")
    if valid(v.get("fcf_yield")) and v["fcf_yield"] > 3:
        good.append(f"FCF yield {v['fcf_yield']:.1f}%")
    if valid(mom.get("score")) and mom["score"] >= 70:
        good.append("positive earnings momentum")

    if l1["failed"]:
        bad.append("failed hard filters: " + ", ".join(l1["failed"]))
    bad.extend(a.get("flags", [])[:3])
    if a.get("margin_trend") == "contracting" and len(margins) >= 2:
        bad.append(f"EBITDA margins contracting {margins[0][1]:.1f}% -> {margins[-1][1]:.1f}%")
    if a.get("ccc_trend") == "worsening":
        bad.append("cash conversion cycle worsening")
    if v.get("eps_negative"):
        bad.append("EPS growth not positive")
    elif valid(v.get("peg")) and v["peg"] > 1.5:
        bad.append(f"PEG {v['peg']:.2f} (expensive for its growth)")
    if valid(v.get("fcf_yield")) and v["fcf_yield"] < 0:
        bad.append(f"negative FCF yield {v['fcf_yield']:.1f}%")
    if valid(mom.get("score")) and mom["score"] <= 25:
        bad.append("earnings declining year-on-year")
    if cyc.get("peak"):
        bad.append("ROE at possible cyclical peak")
    if valid(r.get("ret")) and r["ret"] < -30:
        bad.append(f"share price down {abs(r['ret']):.0f}% in a year")
    if v.get("pe_extreme"):
        bad.append(f"P/E {l1['pe']:.0f}x above 80")
    bad.extend(f for f in h.get("flags", []) if "unavailable" not in f)

    if good:
        parts.append("Strengths: " + "; ".join(good) + ".")
    if bad:
        parts.append("Concerns: " + "; ".join(bad) + ".")
    parts.append(f"Allocation: {tier} ({r['size']}). Reason: {r['reason']}.")
    parts.append("Screening output only - verify with annual reports, concalls and governance review before investing.")
    return " ".join(parts)


SCORECARD_CSS = """
@page{size:A4;margin:10mm;}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:Segoe UI,Arial,sans-serif;color:#1a1a1a;font-size:11px;}
table{border-collapse:collapse;width:100%;}
.lead{background:#DBEAFE;color:#1D4ED8;padding:1px 4px;border-radius:2px;font-size:7px;font-weight:bold;margin-left:3px;}
"""


def gen_scorecard(r):
    l1, a, v, m, h = r["l1"], r["acct"], r["val"], r["moat"], r["holdings"]
    esc = html.escape
    now = datetime.now()
    tier_bg = TIER_COLOR.get(r["tier"], "#888")

    int_cov_str = ("Debt-free" if valid(l1["de"]) and l1["de"] == 0 else fmt(l1["int_cov"], ".1f", "x"))
    metrics = [
        ("PE", fmt(l1["pe"], ".1f")), ("PEG", fmt(v["peg"], ".2f")),
        ("ROE", fmt(l1["roe"], ".1f", "%")), ("ROCE", fmt(l1["roce"], ".1f", "%")),
        ("D/E", fmt(l1["de"], ".2f", "x")), ("Int Cov", int_cov_str),
        ("FCF Yield", fmt(v["fcf_yield"], ".1f", "%")),
        ("Promoter", fmt(h.get("promoter_pct"), ".1f", "%")),
        ("1Y Ret", fmt(r["ret"], "+.1f", "%")),
    ]
    met_html = "".join(
        f'<td style="padding:10px 4px;text-align:center;border-right:1px solid #ddd;">'
        f'<div style="color:#888;font-size:8px;margin-bottom:3px;">{esc(k)}</div>'
        f'<div style="font-family:Courier New,monospace;font-weight:bold;font-size:13px;color:{NAVY};">{esc(val)}</div></td>'
        for k, val in metrics
    )

    layer_html = ""
    for name, status, detail in gen_layers(r):
        icon = {"pass": "&#10003;", "fail": "&#10007;", "warn": "&#9888;"}[status]
        color = {"pass": "#059669", "fail": "#DC2626", "warn": "#D97706"}[status]
        lead = name.endswith(" *")
        clean = name[:-2] if lead else name
        tag = '<span class="lead">LEAD</span>' if lead else ""
        layer_html += (
            f'<tr><td style="padding:4px 6px;color:{color};font-size:12px;width:16px;">{icon}</td>'
            f'<td style="padding:4px 6px;font-size:10px;font-weight:bold;white-space:nowrap;">{esc(clean)}{tag}</td>'
            f'<td style="padding:4px 6px;font-size:9px;color:#555;">{esc(detail)}</td></tr>'
        )

    items = [
        (f"Sales CAGR ({l1['sales_label'] or 'n/a'})", fmt(l1["sales_cagr"], ".1f", "%")),
        (f"PAT CAGR ({l1['pat_label'] or 'n/a'})", fmt(l1["pat_cagr"], ".1f", "%")),
        ("EPS growth (PEG input)", fmt(l1["eps_growth"], ".1f", "%")),
        ("Recent 3Y CFO/PAT (score)", fmt(a.get("recent3_cfo_pat"), ".2f", "x")),
        ("Recent 3Y CFO/EBITDA (score)", fmt(a.get("recent3_cfo_ebitda"), ".2f", "x")),
        (f"{a.get('long_window_label', 'Longer')} CFO/PAT (context)", fmt(a.get("long_window_cfo_pat"), ".2f", "x")),
        (f"Preceding {a.get('prior_window_label', 'n/a')} CFO/PAT", fmt(a.get("prior2_cfo_pat"), ".2f", "x")),
        ("Cash conversion trend", a.get("cash_conversion_trend", "unavailable")),
        ("Accrual ratio", fmt(a.get("accrual_ratio"), ".1f", "%")),
        (f"Moat ({m.get('total_years', 0)} yrs)", f"{m.get('pct', 0)}% - {m.get('consistency', 'n/a')}"),
    ]
    nd = l1.get("net_debt_cr")
    if valid(nd):
        items.append(("Net debt", f"Rs {nd:,.0f} Cr" + (" (net cash)" if nd < 0 else "")))
    if valid(l1.get("leases")) and l1["leases"] > 0:
        items.append(("Lease liabilities", f"Rs {l1['leases'] / 1e7:,.0f} Cr"))
    margins = a.get("margins_by_year", [])
    if len(margins) >= 2:
        arrow = {"expanding": " up", "contracting": " down"}.get(a.get("margin_trend"), "")
        items.append(("EBITDA margins", " -> ".join(f"{x:.1f}%" for _, x in margins) + arrow))
    ccc = a.get("ccc")
    if ccc:
        items.append(("Cash cycle", f"{ccc['ccc']:.0f}d (R:{ccc['dso']:.0f} I:{ccc['dio']:.0f} P:{ccc['dpo']:.0f})"))
    items.append(("Composite", f"{fmt(r['composite'], '.1f')} ({r['ranking_coverage']}% data)"))

    growth_html = "".join(
        f'<tr><td style="padding:4px 8px;font-size:9px;color:#888;">{esc(k)}</td>'
        f'<td style="padding:4px 8px;font-size:10px;font-family:Courier New,monospace;text-align:right;color:{NAVY};">{esc(val)}</td></tr>'
        for k, val in items
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{esc(str(l1['name']))} Scorecard</title><style>{SCORECARD_CSS}</style></head><body>
<div style="background:{NAVY};color:white;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;">
<div><div style="font-size:12px;font-weight:bold;letter-spacing:0.5px;">HIGH COMPOUNDER FRAMEWORK v{VERSION}</div>
<div style="font-size:9px;margin-top:3px;font-weight:bold;">{esc(AUTHOR)} | {now:%d %b %Y}</div></div>
<div style="text-align:right;font-size:8px;">Quantitative screening<br>Not investment advice</div></div>
<div style="padding:14px 20px;border-bottom:3px solid {NAVY};display:flex;justify-content:space-between;align-items:flex-end;">
<div><div style="font-size:22px;font-weight:bold;color:{NAVY};">{esc(str(l1['name']))}</div>
<div style="font-size:10px;color:#888;margin-top:3px;">{esc(r['ticker'])} | {esc(str(l1['sector']))} | Last available price Rs {fmt(l1['price'], ',.2f')} | report generated {now:%d %b %Y, %I:%M %p}</div></div>
<div style="text-align:right;"><div style="background:{tier_bg};color:white;padding:5px 16px;border-radius:14px;font-size:12px;font-weight:bold;display:inline-block;">{r['tier']} {r['size']}</div>
<div style="font-size:9px;color:#888;margin-top:4px;">Forensic {fmt(a.get('score'), '.0f')}/100{' (provisional)' if not a.get('complete') else ''}</div></div></div>
<table style="border-bottom:1px solid #ddd;"><tr>{met_html}</tr></table>
<div style="display:flex;border-bottom:1px solid #ddd;">
<div style="flex:1;padding:12px 20px;border-right:1px solid #ddd;"><div style="font-size:10px;font-weight:bold;color:{NAVY};margin-bottom:8px;">LAYER RESULTS</div><table>{layer_html}</table></div>
<div style="width:260px;padding:12px 20px;"><div style="font-size:10px;font-weight:bold;color:{NAVY};margin-bottom:8px;">GROWTH &amp; CASH</div><table>{growth_html}</table></div></div>
<div style="padding:12px 20px;border-bottom:1px solid #ddd;"><div style="font-size:10px;font-weight:bold;color:{NAVY};margin-bottom:6px;">INVESTMENT VERDICT</div>
<div style="font-size:10px;line-height:1.7;">{esc(gen_verdict(r))}</div></div>
<div style="padding:8px 20px;font-size:7px;color:#aaa;">Quantitative framework output. Not a research recommendation. Data via Yahoo Finance; verify against company filings. {esc(AUTHOR)} - personal use only.</div>
</body></html>"""


# ============================================================
# BATCH SCREENING
# ============================================================
def analyse_with_funnel(ticker, ownership, pe_diagnostics=None):
    r = full_analysis(ticker, ownership)
    if r is None:
        return None, "fetch_fail"
    if r.get("bank"):
        return None, "financial"
    l1 = r["l1"]

    if pe_diagnostics is not None:
        pe_diagnostics["checked_nonfinancial"] += 1
        if not valid(l1.get("pe")):
            pe_diagnostics["missing_pe"] += 1
            pe_diagnostics["missing_pe_tickers"].append(ticker)
    for key, reason in (("mcap_pass", "mcap"), ("de_pass", "debt"), ("roe_pass", "roe"),
                        ("roce_pass", "roce"), ("growth_pass", "growth")):
        if not l1[key]:
            return None, reason
    if not r["acct"].get("complete"):
        return None, "data"
    if r["composite"] is None or r["ranking_coverage"] < 80:
        return None, "ranking_data"
    return r, "pass"


def parse_universe_csv(payload):
    frame = pd.read_csv(io.BytesIO(payload))
    col = next((c for c in frame.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    if col is None:
        raise ValueError("CSV needs a Symbol column")
    symbols = list(dict.fromkeys(clean_ticker(x) for x in frame[col].dropna() if str(x).strip()))
    if len(symbols) < 450:
        raise ValueError(f"Only {len(symbols)} symbols found; expected about 500")
    return symbols


def save_universe(payload):
    try:
        UNIVERSE_CACHE_FILE.write_bytes(payload)
    except Exception:
        pass


@st.cache_data(ttl=86400, show_spinner=False)
def download_universe():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
               "Accept": "text/csv,*/*"}
    errors = []
    for url in UNIVERSE_URLS:
        try:
            with urlopen(Request(url, headers=headers), timeout=15) as resp:
                payload = resp.read()
            parse_universe_csv(payload)
            return payload, ""
        except Exception as exc:
            errors.append(f"{url.split('/')[2]}: {type(exc).__name__}")
    return None, "; ".join(errors)


def get_nifty_tickers(max_stocks, uploaded=None):
    if uploaded is not None:
        payload = uploaded.getvalue()
        symbols = parse_universe_csv(payload)
        save_universe(payload)
        return symbols[:max_stocks], "Uploaded official Nifty 500 CSV"

    payload, errors = download_universe()
    if payload is not None:
        save_universe(payload)
        return parse_universe_csv(payload)[:max_stocks], "Live official Nifty 500 CSV"

    if UNIVERSE_CACHE_FILE.exists():
        try:
            symbols = parse_universe_csv(UNIVERSE_CACHE_FILE.read_bytes())
            saved = datetime.fromtimestamp(UNIVERSE_CACHE_FILE.stat().st_mtime)
            return symbols[:max_stocks], f"Saved list from {saved:%d %b %Y} (live download failed)"
        except Exception as exc:
            errors += f"; saved file: {type(exc).__name__}"

    raise RuntimeError("Could not load the Nifty 500 list. Download the constituents CSV from "
                       "niftyindices.com and upload it in the sidebar. Details: " + errors)


def apply_portfolio_weights(df):
    """Apply tier midpoints, a maximum of five FULL positions, and a 100% exposure cap."""
    df = df.copy()
    raw = df["tier"].map(TIER_MIDPOINT).fillna(0.0).astype(float)
    df["portfolio_size_band"] = df["tier"].map(TIER_SIZE).fillna("0%")
    df["allocation_adjustment"] = ""

    full_rows = df[df["tier"] == "FULL"].sort_values("composite", ascending=False)
    capped_indices = full_rows.index[MAX_FULL_POSITIONS:]
    if len(capped_indices):
        raw.loc[capped_indices] = TIER_MIDPOINT["STANDARD"]
        df.loc[capped_indices, "portfolio_size_band"] = TIER_SIZE["STANDARD"] + " (portfolio cap)"
        df.loc[capped_indices, "allocation_adjustment"] = (
            f"Company still qualifies as FULL; allocation capped at STANDARD midpoint because "
            f"the portfolio limit is {MAX_FULL_POSITIONS} FULL positions"
        )

    total = float(raw.sum())
    scale = min(1.0, 100.0 / total) if total > 0 else 0.0
    df["tier_midpoint"] = raw
    df["portfolio_weight"] = (raw * scale).round(2)
    return df, scale


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("High Compounder")
page = st.sidebar.radio("Navigation", ["Single Stock", "Compare", "Auto Top 10", "How It Works"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.subheader("Data inputs")
universe_upload = st.sidebar.file_uploader("Official Nifty 500 CSV (optional)", type=["csv"], key="universe_csv")
ownership_upload = st.sidebar.file_uploader("Verified ownership CSV (optional)", type=["csv"], key="ownership_csv")
ownership = load_ownership(ownership_upload)
st.sidebar.caption("Ownership CSV columns: ticker, promoter_pct, institutional_pct, "
                   "pledged_pct_of_promoter_holding (or promoter_pledge_pct), as_of, source")
st.sidebar.markdown("---")
if st.sidebar.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"{AUTHOR} | v{VERSION} | Yahoo Finance via yfinance")

# ============================================================
# HEADER
# ============================================================
st.markdown(f'<p class="main-title">High Compounder Framework v{VERSION}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Systematic Indian equity screener - quality, forensics, valuation and momentum</p>',
            unsafe_allow_html=True)
st.warning("Research screening aid only. Yahoo Finance is an unofficial, best-effort data source; verify all financials, share counts, debt, EPS and valuation against exchange filings before presenting an investment view. This model is not backtested.")
st.markdown("---")


def show_layers(r):
    for name, status, detail in gen_layers(r):
        icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
        lead = name.endswith(" *")
        clean = name[:-2] if lead else name
        tag = '<span class="leading-tag">LEADING</span>' if lead else ""
        st.markdown(f'<div class="layer-row layer-{status}">{icon} <strong>{html.escape(clean)}</strong>{tag}: '
                    f'{html.escape(detail)}</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: SINGLE STOCK
# ============================================================
if page == "Single Stock":
    c1, c2 = st.columns([4, 1])
    with c1:
        ticker_input = st.text_input("Enter NSE ticker", "LUPIN").strip().upper()
    with c2:
        st.write("")
        run_button = st.button("Analyse", type="primary", use_container_width=True)

    quick = st.columns(6)
    for col, symbol in zip(quick, ["LUPIN", "BSE", "DIXON", "INFY", "HCLTECH", "MAZDOCK"]):
        if col.button(symbol, use_container_width=True):
            ticker_input, run_button = symbol, True

    if run_button and ticker_input:
        ticker = clean_ticker(ticker_input)
        with st.spinner(f"Analysing {short_ticker(ticker)}..."):
            r = full_analysis(ticker, ownership)

        if r is None:
            st.error("Could not fetch usable data. Check the NSE symbol (e.g. LUPIN, INFY) or retry later.")
        elif r.get("bank"):
            st.warning(f"{r['name']} is a bank/financial company. This framework is for non-financials only.")
        else:
            l1, a, v = r["l1"], r["acct"], r["val"]
            st.markdown(f'<p class="stock-name">{html.escape(str(l1["name"]))}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="stock-meta">{ticker} | {html.escape(str(l1["sector"]))} | '
                        f'Last available price Rs {fmt(l1["price"], ",.2f")} | report generated {datetime.now():%d %b %Y, %I:%M %p}</p>',
                        unsafe_allow_html=True)

            if r["data"]["debug"]["warnings"]:
                with st.expander("Data-quality warnings"):
                    for w in r["data"]["debug"]["warnings"]:
                        st.warning(w)

            cards = [
                ("Forensic", f"{fmt(a.get('score'), '.0f')}/100"), ("P/E", fmt(l1["pe"], ".1f")),
                ("PEG", fmt(v["peg"], ".2f")), ("ROE", fmt(l1["roe"], ".1f", "%")),
                ("ROCE", fmt(l1["roce"], ".1f", "%")), ("D/E", fmt(l1["de"], ".2f", "x")),
                ("1Y Return", fmt(r["ret"], "+.1f", "%")),
            ]
            cols = st.columns(8)
            for col, (label, value) in zip(cols, cards):
                col.markdown(f'<div class="metric-card"><div class="metric-lbl">{label}</div>'
                             f'<div class="metric-val">{html.escape(value)}</div></div>', unsafe_allow_html=True)
            cols[7].markdown(f'<div class="metric-card"><div class="metric-lbl">Tier</div>'
                             f'<span class="tier-badge tier-{r["tier"].lower()}">{r["tier"]} {r["size"]}</span></div>',
                             unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Layer-by-layer assessment")
            show_layers(r)

            st.markdown("---")
            st.subheader("Investment verdict")
            st.markdown(f'<div class="verdict-box">{html.escape(gen_verdict(r))}</div>', unsafe_allow_html=True)
            st.caption(f"Composite {fmt(r['composite'], '.1f')} | data coverage {r['ranking_coverage']}% | "
                       f"decision flags: {len(r.get('decision_flags', []))} | allocation: {r['reason']}")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download scorecard (HTML - open, Ctrl+P, save as PDF)", gen_scorecard(r),
                                   file_name=f"{short_ticker(ticker)}_scorecard_v{VERSION}.html", mime="text/html")
            with d2:
                st.download_button("Download analysis row (CSV)", pd.DataFrame([result_row(r)]).to_csv(index=False),
                                   file_name=f"{short_ticker(ticker)}_v{VERSION}.csv", mime="text/csv")

            t1, t2, t3, t4 = st.tabs(["Forensic", "ROE & valuation", "Momentum & holdings", "Debug"])

            with t1:
                s = a.get("score")
                css = "score-green" if valid(s) and s >= 85 else "score-yellow" if valid(s) and s >= 50 else "score-red"
                st.markdown(f'<div class="score-box {css}"><h2>{fmt(s, ".0f")}/100</h2></div>', unsafe_allow_html=True)
                if not a.get("complete"):
                    st.caption("Provisional score: some multi-year cash-flow data is missing.")
                recent_years = a.get("recent3_years", [])
                long_years = a.get("long_window_years", [])
                st.caption(
                    "Forensic cash-conversion scoring uses the latest three consecutive annual periods. "
                    "A longer aggregate is shown separately (3-5 years, whatever is available). "
                    "Trend compares recent 3Y with the preceding 1-2 consecutive years when they exist."
                )
                st.caption("Scoring window: " + (", ".join(recent_years) if recent_years else "not available"))
                st.caption(
                    f"Longer reference window ({a.get('long_window_label', 'unavailable')}): "
                    + (", ".join(long_years) if long_years else "not available")
                )
                k1, k2, k3 = st.columns(3)
                k1.metric("Recent 3Y CFO/PAT (scoring)", fmt(a.get("recent3_cfo_pat"), ".2f", "x"))
                k2.metric("Longer-window CFO/PAT", fmt(a.get("long_window_cfo_pat"), ".2f", "x"),
                          help=f"Uses up to 5 consecutive years when available. Current window: {a.get('long_window_label', 'unavailable')}")
                k3.metric("Latest-year accrual ratio", fmt(a.get("accrual_ratio"), ".1f", "%"))
                k4, k5, k6 = st.columns(3)
                k4.metric("Recent 3Y CFO/EBITDA", fmt(a.get("recent3_cfo_ebitda"), ".2f", "x"))
                k5.metric("Longer-window CFO/EBITDA", fmt(a.get("long_window_cfo_ebitda"), ".2f", "x"))
                k6.metric("Cash conversion trend", a.get("cash_conversion_trend", "unavailable").title())
                if a.get("cash_conversion_delta") is not None:
                    st.caption(
                        f"Recent 3Y CFO/PAT is {a['cash_conversion_delta']:+.2f}x versus the preceding "
                        f"{a.get('prior_window_label', 'n/a')}. These comparison windows do not overlap."
                    )
                if a.get("margins_by_year"):
                    st.markdown("**EBITDA margins:** " + " -> ".join(f"{y}: {x:.1f}%" for y, x in a["margins_by_year"])
                                + f" ({a.get('margin_trend')})")
                if a.get("int_coverage_trend"):
                    st.markdown("**Interest coverage trend:** " + " -> ".join(f"{y}: {x:.1f}x" for y, x in a["int_coverage_trend"]))
                ccc = a.get("ccc")
                if ccc:
                    st.markdown(f"**Cash conversion cycle:** {ccc['ccc']:.0f} days (DSO {ccc['dso']:.0f} + DIO {ccc['dio']:.0f} "
                                f"- DPO {ccc['dpo']:.0f}), {a.get('ccc_trend') or 'trend n/a'}")
                    for note in ccc["notes"]:
                        st.caption(note)
                if a.get("flags"):
                    for f in a["flags"]:
                        st.markdown(f'<div class="flag-item">⚠ {html.escape(f)}</div>', unsafe_allow_html=True)
                else:
                    st.success("No forensic flags in the available data.")

            with t2:
                if r["moat"]["roe_by_year"]:
                    st.dataframe(pd.DataFrame(r["moat"]["roe_by_year"], columns=["Year", "ROE %"]).round(1),
                                 hide_index=True, use_container_width=True)
                st.markdown(f"**Moat proxy:** {r['moat']['consistency']} "
                            f"({r['moat']['years_above_15']}/{r['moat']['total_years']} years above 15%)")
                st.markdown(f"**PEG:** {fmt(v['peg'], '.2f')} - {v['peg_note']}")
                st.markdown(f"**EPS basis:** {l1['eps_basis']} | **FCF yield:** {fmt(v['fcf_yield'], '.1f', '%')}")
                st.markdown(f"**D/E source:** {l1['de_source']}")
                if valid(l1["net_debt_cr"]):
                    nd = l1["net_debt_cr"]
                    st.markdown(f"**Net {'cash' if nd < 0 else 'debt'}:** Rs {abs(nd):,.0f} Cr")
                st.info("Consistently high ROE is a moat proxy only. Confirm the real advantage through qualitative research.")

            with t3:
                mom = r["mom"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Latest quarter YoY", fmt(mom.get("latest_yoy"), "+.1f", "%"))
                m2.metric("Previous quarter YoY", fmt(mom.get("prior_yoy"), "+.1f", "%"))
                m3.metric("Momentum score", fmt(mom.get("score"), ".0f"))
                st.caption(f"{mom.get('status')} ({mom.get('basis')})")
                if mom.get("records"):
                    st.dataframe(pd.DataFrame([(d.strftime("%b %Y"), round(x, 2)) for d, x in mom["records"]],
                                              columns=["Quarter", "Value"]), hide_index=True, use_container_width=True)
                st.markdown("---")
                hd = r["holdings"]
                h1, h2, h3 = st.columns(3)
                h1.metric("Promoter (verified)", fmt(hd.get("promoter_pct"), ".1f", "%"))
                h2.metric("Institutional (verified)", fmt(hd.get("institutional_pct"), ".1f", "%"))
                h3.metric("Pledge (% of promoter)", fmt(hd.get("pledge_pct"), ".1f", "%"))
                st.caption(f"Yahoo insider % (not promoter data): {fmt(hd.get('yahoo_insider_pct'), '.1f', '%')} | "
                           f"source: {hd.get('source') or 'n/a'} | as of: {hd.get('as_of') or 'n/a'}")
                for f in hd["flags"]:
                    st.warning(f)

            with t4:
                st.write("**Valuation data check (no terminal required)**")
                st.json(r["data"]["debug"].get("valuation", {}))
                st.write("**Yahoo quote fields**")
                st.json(r["data"]["debug"].get("quote_fields", {}))
                st.write("**Data availability (columns returned)**")
                st.json(r["data"]["debug"].get("data_years", {}))
                st.write("**Data sources**")
                st.json(r["data"]["debug"].get("sources", []))
                st.write("**Decision chain**")
                st.json({
                    "Market cap": l1["mcap_source"], "P/E": l1["pe_source"], "D/E": l1["de_source"],
                    "Interest coverage": l1["int_cov_source"], "Sales growth": l1["sales_source"],
                    "PAT growth": l1["pat_source"], "EPS growth": l1["eps_source"], "EPS basis": l1["eps_basis"],
                    "Failed hard filters": l1["failed"], "Forensic complete": a.get("complete"),
                    "Ranking components": {k: (round(x, 1) if valid(x) else None) for k, x in r["components"].items()},
                    "Allocation": r["reason"],
                })


# ============================================================
# PAGE: COMPARE
# ============================================================
elif page == "Compare":
    st.subheader("Compare companies")
    c1, c2, c3 = st.columns(3)
    inputs = [c1.text_input("Company 1", "LUPIN"), c2.text_input("Company 2", "INFY"),
              c3.text_input("Company 3 (optional)", "")]
    if st.button("Compare", type="primary", use_container_width=True):
        results = []
        for t in inputs:
            if not t.strip():
                continue
            with st.spinner(f"Analysing {t.strip().upper()}..."):
                r = full_analysis(t, ownership)
            if r is None:
                st.warning(f"Could not analyse {t}")
            elif r.get("bank"):
                st.warning(f"{t} is a financial company - excluded")
            else:
                results.append(r)
        if len(results) < 2:
            st.error("At least two valid non-financial companies are needed.")
        else:
            table = pd.DataFrame([result_row(r) for r in results])
            show = ["ticker", "tier", "composite", "forensic", "pe", "peg", "roe", "roce", "de",
                    "sales_cagr", "pat_cagr", "eps_growth", "cfo_pat_3y", "cfo_pat_long_window",
                    "cash_conversion_trend", "moat", "momentum", "fcf_yield", "return_1y"]
            st.dataframe(table[show].round(2), hide_index=True, use_container_width=True)
            for r in results:
                with st.expander(f"{r['l1']['name']} - {r['tier']} {r['size']}", expanded=True):
                    st.markdown(f'<div class="verdict-box">{html.escape(gen_verdict(r))}</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: AUTO TOP 10
# ============================================================
elif page == "Auto Top 10":
    st.subheader("Nifty 500 screen and ranked shortlist")
    c1, c2 = st.columns(2)
    max_stocks = c1.slider("Companies to screen (500 = full universe)", 20, 500, 500, 10)
    top_n = c2.slider("Shortlist size", 5, 30, 10)

    if st.button("Run screen", type="primary", use_container_width=True):
        try:
            tickers, source = get_nifty_tickers(max_stocks, universe_upload)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.caption(f"Universe: {source} | {len(tickers)} companies")

        funnel = {k: 0 for k in ["financial", "mcap", "debt", "roe", "roce", "growth",
                                 "data", "ranking_data", "fetch_fail", "pass"]}
        rows, full = [], {}
        pe_diagnostics = {
            "checked_nonfinancial": 0,
            "missing_pe": 0,
            "missing_pe_tickers": [],
        }
        progress, status = st.progress(0), st.empty()
        for i, t in enumerate(tickers, start=1):
            status.caption(f"Screening {short_ticker(t)} ({i}/{len(tickers)})")
            r, reason = analyse_with_funnel(t, ownership, pe_diagnostics)
            funnel[reason] += 1
            if r is not None:
                row = result_row(r)
                rows.append(row)
                full[row["ticker"]] = r
            progress.progress(i / len(tickers))
            if i % 5 == 0:
                time.sleep(0.5)
        progress.empty()
        status.empty()

        st.markdown("### Screening funnel")
        labels = [("Financial companies excluded", "financial"), ("Market cap below Rs 15,000 Cr", "mcap"),
                  ("D/E 1.0x or more / unavailable", "debt"), ("ROE 15% or less", "roe"),
                  ("ROCE 18% or less", "roce"), ("Growth below floor", "growth"),
                  ("Insufficient forensic data", "data"), ("Insufficient ranking data", "ranking_data"),
                  ("Fetch failure", "fetch_fail"), ("Passed everything", "pass")]
        st.dataframe(pd.DataFrame([(n, funnel[k]) for n, k in labels], columns=["Stage", "Companies"]),
                     hide_index=True, use_container_width=True)

        p1, p2 = st.columns(2)
        p1.metric("Non-financial stocks with valuation checked", pe_diagnostics["checked_nonfinancial"])
        p2.metric("P/E unavailable", pe_diagnostics["missing_pe"])
        st.caption(
            "P/E-unavailable is an independent diagnostic count. These companies may also appear "
            "under another funnel rejection reason; the counts are not additive. If PEG is unavailable, "
            "the stock cannot reach the 80% ranking-data coverage requirement."
        )
        if pe_diagnostics["missing_pe_tickers"]:
            with st.expander("View companies with unavailable P/E"):
                st.write(", ".join(sorted(pe_diagnostics["missing_pe_tickers"])))

        fetch_failure_rate = funnel["fetch_fail"] / max(1, len(tickers))
        if fetch_failure_rate > 0.20:
            st.error(
                f"INCOMPLETE RUN: {funnel['fetch_fail']} of {len(tickers)} stocks "
                f"({fetch_failure_rate:.1%}) failed to fetch. No Top 10 or portfolio weights are produced. "
                "Clear the app cache and repeat later; do not use this partial run for allocation."
            )
            st.stop()
        if len(tickers) < 500:
            st.warning("Partial-universe run: this is a test subset, not a complete Nifty 500 result.")

        if not rows:
            st.error("No companies passed all filters.")
            st.stop()

        df = pd.DataFrame(rows)
        df = df[df["composite"].notna() & (df["tier"] != "WATCH")]
        df = df.sort_values("composite", ascending=False).reset_index(drop=True)
        if df.empty:
            st.warning("All qualifying companies were rated WATCH.")
            st.stop()

        selected = df.head(top_n).copy()
        selected, scale = apply_portfolio_weights(selected)

        if scale < 1:
            st.warning(f"Tier midpoints totalled {selected['tier_midpoint'].sum():.1f}%; weights scaled to 100%.")
        else:
            st.info(f"Suggested weights total {selected['portfolio_weight'].sum():.1f}%; "
                    f"cash {100 - selected['portfolio_weight'].sum():.1f}%.")

        st.markdown("### Ranked shortlist")
              show = ["ticker", "name", "sector", "tier", "portfolio_size_band", "portfolio_weight",
                "allocation_adjustment", "composite", "coverage", "forensic",
                "pe", "peg", "roe", "roce", "de", "sales_cagr", "pat_cagr", "eps_growth",
                "cfo_pat_3y", "cfo_ebitda_3y", "cfo_pat_long_window", "cfo_ebitda_long_window",
                "cash_conversion_trend",
                "moat_pct", "momentum", "fcf_yield", "int_cov", "promoter_pct", "pledge_pct", "return_1y"]
        st.dataframe(selected[show].round(2), hide_index=True, use_container_width=True)

               export = selected[show + ["cfo_pat_5y", "cfo_ebitda_5y", "cfo_pat_prior2y",
                                  "accrual_pct", "net_debt_cr", "reason"]]
        st.download_button("Download shortlist (CSV)", export.to_csv(index=False),
                           file_name=f"high_compounder_v{VERSION}_top{len(selected)}_{datetime.now():%d%b%Y}.csv",
                           mime="text/csv")

        st.markdown("### Company details")
        icons = {"FULL": "🟢", "STANDARD": "🔵", "HALF": "🟡", "WATCH": "🔴"}
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            r = full.get(row["ticker"])
            with st.expander(f"#{rank} {row['ticker']} - {icons.get(row['tier'], '')} {row['tier']} | "
                             f"weight {row['portfolio_weight']:.1f}% | composite {row['composite']:.1f}",
                             expanded=rank <= 3):
                if r is None:
                    continue
                show_layers(r)
                if row.get("allocation_adjustment"):
                    st.info(row["allocation_adjustment"])
                st.markdown(f'<div class="verdict-box">{html.escape(gen_verdict(r))}</div>', unsafe_allow_html=True)
                st.download_button("Scorecard (HTML)", gen_scorecard(r),
                                   file_name=f"{row['ticker']}_scorecard_v{VERSION}.html",
                                   mime="text/html", key=f"sc_{row['ticker']}")


# ============================================================
# PAGE: HOW IT WORKS
# ============================================================
else:
    st.subheader(f"High Compounder Framework v{VERSION} - methodology")
    st.markdown("""
**1. Universe.** Nifty 500 (live official CSV, uploaded CSV, or the last saved list). Banks, NBFCs, insurers and lenders are excluded - they need NIM, CASA, NPA and capital-adequacy metrics instead.

**2. Hard filters.** Market cap at least Rs 15,000 Cr, ROE above 15%, ROCE above 18% (both on average capital), gross D/E below 1.0x, sales and PAT CAGR above 15% (10% for companies above Rs 50,000 Cr with ROE above 25%). Auto Top 10 requires all five. In single-stock view, one failure caps the tier at HALF and two or more give WATCH.

**3. Debt.** Gross borrowings = short-term + long-term borrowing lines (excluding leases) when both are reported; otherwise Yahoo's Total Debt (may include leases, so D/E may be slightly overstated). A single partial borrowing line is never accepted. A company is treated as debt-free only when Yahoo shows no borrowing lines at all AND interest expense is reported near zero; otherwise D/E is N/A and must be verified manually. Interest coverage below 3x caps at HALF.

**4. Forensics (score starts at 100).** Receivables rising 3 years in a row (-15), material inventory growing 10+ points faster than revenue (-15), trailing 3Y CFO/PAT (-20 / -30), trailing 3Y CFO/EBITDA (-15 / -20), worsening annual CFO/EBITDA (-10), negative trailing 3Y CFO (-25), accrual ratio (-10 / -20). The latest three consecutive annual periods are the primary scoring window. A separate longer aggregate uses whatever consecutive annual history is available (3-5 years). Trend compares recent 3Y with the preceding 1-2 consecutive years when they exist. Cash-flow deductions are capped at 50 and the score cannot go below 0.

**5. Working capital.** CCC = DSO + DIO - DPO on average balances. DIO is excluded when inventory is immaterial.

**6. Moat proxy.** Established: ROE above 15% in every year (minimum 4 years). Emerging: at least 3 years, latest two above 15%, and latest ROE at least the earlier-years' average.

**7. Ownership (leading).** Upload verified promoter, institutional and pledge data. Promoter below 20% caps at STANDARD; pledge above 25% caps at STANDARD, 50%+ caps at HALF.

**8. Valuation.** P/E uses Yahoo trailing P/E when within a broad sanity range and cross-checks it against price / trailing EPS when available. Otherwise the app uses validated trailing EPS or market capitalisation / four consecutive quarters of PAT. PEG = trailing P/E / selected observed growth (sales growth used if EPS growth is more than twice sales growth). Growth above 30% is flagged for analyst review but is not mechanically capped. Negative EPS growth makes PEG not meaningful. FCF yield is calculated only as CFO minus capex when both inputs are present (reported FCF used as last-resort fallback). P/E above 80 is a warning.

**9. Momentum (leading).** EPS vs the same quarter last year for the latest two consecutive reported quarters.

**10. Cyclicality.** Requires at least four consecutive annual ROE values. Peak warning when latest ROE is more than twice the average of prior years and above 20%.

**11. Tiers.** FULL 12-15%: score 85+, zero flags, moat proxy, PEG at most 1.0, no cyclical peak, complete data. STANDARD 8-10%: score 70+, at most one flag, PEG at most 1.5. HALF 4-6%: everything else that is not WATCH. WATCH 0%: score below 50, PEG above 5, or two or more hard-filter failures.

**12. Ranking and portfolio sizing.** PEG 25%, forensic 20%, earnings momentum 15%, CFO quality 15%, moat consistency 15%, ROE 10%. Each factor is scored 0-100; missing factors are excluded and at least 80% coverage is required. The Top 10 limits the portfolio to five FULL positions (excess FULL names keep their research tier but receive a STANDARD-sized allocation), and caps total exposure at 100%.

**v2.8 fixes vs v2.7:**
- CAGR calculation now uses the longest clean positive window instead of demanding every data point be perfect - fixes most "growth unavailable" rejections
- 5Y CFO/PAT and trend windows gracefully handle 3-4 years of Yahoo data instead of showing N/A
- EPS field selection restored to v2.4 tolerance (3+ valid obs, not every obs valid)
- Moat proxy no longer requires strict date contiguity
- FCF fallback to reported Free Cash Flow restored as last resort
- 1Y return date tolerance loosened from 7 to 15 days
- Sector cap removed

**Limitations.** Yahoo Finance access through yfinance is unofficial, can be rate-limited or changed, and data can be wrong, restated or missing. This app is not an institutional data terminal. Verify every key figure in exchange filings/annual reports. Thresholds are mostly absolute, not sector-relative; portfolio liquidity, volatility, correlation, drawdown and factor exposures are not modelled. No empirical backtest or live track record is supplied. Management quality, related-party transactions and sell decisions require separate analyst work.
    """)
    st.caption(f"{AUTHOR} | High Compounder Framework v{VERSION} | Not investment advice")
