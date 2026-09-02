import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="High Compounder Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
NAVY = '#1B2A4A'
BANKING_KW = [
    'bank', 'finance', 'insurance', 'nbfc',
    'housing finance', 'credit', 'lending', 'microfinance',
]

DEFAULT_ACCT = {
    'score': None, 'flags': [], 'num_flags': 0,
    'cum_cfo_pat': None, 'cum_cfo_ebitda': None,
    'single_cfo_pat': None, 'cfo_ebitda_trend': [],
    'recv_pcts': [], 'recv_years': [],
    'margins_by_year': [], 'margin_trend': '',
    'ccc': None, 'ccc_trend': None, 'accrual_ratio': None,
}
DEFAULT_CYC = {'peak': False, 'roe_by_year': {}}
DEFAULT_MOAT = {
    'pct': 0, 'consistency': 'no data',
    'years_above_15': 0, 'total_years': 0, 'roe_by_year': [],
}
DEFAULT_MOM = {'available': False}
DEFAULT_HOLDINGS = {
    'insider_pct': None, 'inst_pct': None, 'inst_count': None,
    'flags': [], 'score_adj': 0,
}

# ============================================================
# STYLES
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.main-title{{font-family:'DM Sans',sans-serif;font-size:1.75rem;font-weight:700;color:{NAVY};}}
.sub-title{{font-size:0.85rem;color:#6B7280;}}
.stock-name{{font-size:1.5rem;font-weight:700;color:{NAVY};}}
.stock-meta{{font-size:0.85rem;color:#6B7280;}}
.metric-card{{background:#fff;border:1px solid #E5E7EB;border-radius:8px;padding:0.8rem;text-align:center;}}
.metric-val{{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:600;color:{NAVY};}}
.metric-lbl{{font-size:0.7rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;}}
.tier-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:600;font-size:0.85rem;}}
.tier-full{{background:#ECFDF5;color:#059669;border:1px solid #059669;}}
.tier-standard{{background:#EFF6FF;color:#2563EB;border:1px solid #2563EB;}}
.tier-half{{background:#FFFBEB;color:#D97706;border:1px solid #D97706;}}
.tier-watch{{background:#FEF2F2;color:#DC2626;border:1px solid #DC2626;}}
.layer-row{{padding:0.5rem 0.8rem;border-radius:4px;margin:3px 0;font-size:0.85rem;border-left:3px solid;}}
.layer-pass{{background:#ECFDF5;border-color:#059669;}}
.layer-fail{{background:#FEF2F2;border-color:#DC2626;}}
.layer-warn{{background:#FFFBEB;border-color:#D97706;}}
.verdict-box{{background:#F9FAFB;padding:1.2rem;border-radius:8px;border:1px solid #E5E7EB;line-height:1.8;font-size:0.9rem;}}
.flag-item{{background:#FFFBEB;padding:0.4rem 0.8rem;border-radius:4px;margin:3px 0;border-left:3px solid #D97706;font-size:0.85rem;}}
.clean-item{{background:#ECFDF5;padding:0.4rem 0.8rem;border-radius:4px;margin:3px 0;border-left:3px solid #059669;font-size:0.85rem;}}
.score-card{{padding:1rem;border-radius:8px;text-align:center;}}
.score-green{{background:#ECFDF5;border:1px solid #059669;}}
.score-yellow{{background:#FFFBEB;border:1px solid #D97706;}}
.score-red{{background:#FEF2F2;border:1px solid #DC2626;}}
.debug-row{{font-family:'JetBrains Mono',monospace;font-size:0.8rem;padding:3px 0;border-bottom:1px solid #f0f0f0;}}
.debug-label{{color:#6B7280;display:inline-block;width:180px;}}
.debug-val{{color:{NAVY};}}
.debug-ok{{color:#059669;}}
.debug-bad{{color:#DC2626;}}
.debug-fb{{color:#D97706;font-size:0.7rem;}}
.leading-tag{{display:inline-block;padding:2px 6px;border-radius:3px;font-size:0.65rem;font-weight:600;background:#DBEAFE;color:#1D4ED8;margin-left:4px;vertical-align:middle;}}
.lagging-tag{{display:inline-block;padding:2px 6px;border-radius:3px;font-size:0.65rem;font-weight:600;background:#E5E7EB;color:#6B7280;margin-left:4px;vertical-align:middle;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def is_valid(val):
    if val is None:
        return False
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError, OverflowError):
        return False

def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            v = df.loc[label].iloc[col]
            if pd.notna(v):
                return float(v)
    except Exception:
        pass
    return None

def fmt(val, f=".1f", suffix="", prefix=""):
    if not is_valid(val):
        return "N/A"
    return f"{prefix}{val:{f}}{suffix}"

def fy_label(dt):
    try:
        return f"FY{str(dt.year)[2:]}" if dt.month <= 3 else f"FY{str(dt.year + 1)[2:]}"
    except Exception:
        return ""

# ============================================================
# FINANCIAL FIELD GETTERS
# ============================================================
def get_revenue(fin, col=0):
    for field in ['Total Revenue', 'Operating Revenue', 'Revenue', 'Net Revenue']:
        v = safe_get(fin, field, col)
        if is_valid(v) and v > 0:
            return v
    return None

def get_ni(fin, col=0):
    for field in ['Net Income', 'Net Income Common Stockholders', 'Net Income Continuous Operations']:
        v = safe_get(fin, field, col)
        if v is not None:
            return v
    return None

def get_ebitda(fin, col=0):
    for field in ['EBITDA', 'Normalized EBITDA']:
        v = safe_get(fin, field, col)
        if is_valid(v) and v > 0:
            return v
    return None

def get_ebit(fin, col=0):
    for field in ['EBIT', 'Operating Income', 'Operating Profit', 'Normalized EBIT']:
        v = safe_get(fin, field, col)
        if v is not None:
            return v
    return None

def get_equity(bs, col=0):
    for field in ['Stockholders Equity', 'Total Stockholders Equity',
                  'Common Stock Equity', 'Total Equity Gross Minority Interest']:
        v = safe_get(bs, field, col)
        if is_valid(v) and v > 0:
            return v
    return None

def get_cfo(cf, col=0):
    for field in ['Operating Cash Flow', 'Total Cash From Operating Activities',
                  'Cash Flow From Continuing Operating Activities']:
        v = safe_get(cf, field, col)
        if v is not None:
            return v
    return None

def get_fcf(cf, col=0):
    v = safe_get(cf, 'Free Cash Flow', col)
    if v is not None:
        return v
    cfo = get_cfo(cf, col)
    capex = safe_get(cf, 'Capital Expenditure', col)
    if is_valid(cfo) and is_valid(capex):
        return cfo + capex  # capex is negative in yfinance
    return cfo

def get_receivables(bs, col=0):
    for field in ['Accounts Receivable', 'Net Receivables', 'Receivables']:
        v = safe_get(bs, field, col)
        if v is not None:
            return v
    return None

def get_payables(bs, col=0):
    for field in ['Accounts Payable', 'Payables And Accrued Expenses', 'Current Accounts Payable']:
        v = safe_get(bs, field, col)
        if v is not None:
            return v
    return None

def get_interest_expense(fin, col=0):
    """Get interest expense (returned as positive number)."""
    for field in ['Interest Expense', 'Interest Expense Non Operating',
                  'Net Interest Income', 'Net Non Operating Interest Income Expense']:
        v = safe_get(fin, field, col)
        if v is not None:
            return abs(v)
    return None

def get_cash(bs, col=0):
    """Get cash and equivalents from balance sheet."""
    for field in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments',
                  'Cash Financial', 'Cash And Short Term Investments']:
        v = safe_get(bs, field, col)
        if is_valid(v) and v >= 0:
            return v
    return None

def get_total_debt(bs, col=0):
    """Get total debt from balance sheet."""
    for field in ['Total Debt', 'Long Term Debt And Capital Lease Obligation',
                  'Long Term Debt', 'Net Debt']:
        v = safe_get(bs, field, col)
        if is_valid(v) and v >= 0:
            return v
    return None

def get_shares(fin, bs, qfin, info):
    for src in [qfin, fin]:
        if src is not None:
            for field in ['Diluted Average Shares', 'Basic Average Shares']:
                v = safe_get(src, field, 0)
                if is_valid(v) and v > 1000:
                    return v
    if bs is not None:
        for field in ['Ordinary Shares Number', 'Share Issued']:
            v = safe_get(bs, field, 0)
            if is_valid(v) and v > 1000:
                return v
    v = info.get('sharesOutstanding')
    if is_valid(v) and v > 1000:
        return float(v)
    return None

def validate_shares(price, shares, info):
    mcap_reported = info.get('marketCap')
    if not all(is_valid(x) and x > 0 for x in [price, shares, mcap_reported]):
        return shares, False
    computed = price * shares
    ratio = mcap_reported / computed
    if 0.33 < ratio < 3.0:
        return shares, False
    for multiplier in [1000, 100, 10, 0.001, 0.01, 0.1]:
        test_ratio = mcap_reported / (price * shares * multiplier)
        if 0.5 < test_ratio < 2.0:
            return shares * multiplier, True
    return mcap_reported / price, True

def is_banking(info, name=""):
    s = (info.get('sector', '') or '').lower()
    ind = (info.get('industry', '') or '').lower()
    n = (info.get('shortName', '') or name or '').lower()
    return any(k in s or k in ind or k in n for k in BANKING_KW)

# ============================================================
# DATA FETCHING (improved with holdings)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker):
    debug = {'ticker': ticker, 'fallbacks': [], 'warnings': []}
    try:
        t = yf.Ticker(ticker)
        try:
            info = {k: v for k, v in (dict(t.info) if t.info else {}).items() if v is not None}
        except Exception:
            info = {}
        try:
            fi = t.fast_info
            if fi:
                for attr, key in [('last_price', 'currentPrice'), ('previous_close', 'previousClose'),
                                  ('market_cap', 'marketCap'), ('shares', 'sharesOutstanding')]:
                    if not is_valid(info.get(key)):
                        try:
                            v = getattr(fi, attr, None)
                            if is_valid(v) and v > 0:
                                info[key] = float(v)
                                debug['fallbacks'].append(f'{key} from fast_info')
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        except Exception:
            fin = None
        try:
            bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        except Exception:
            bs = None
        try:
            cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        except Exception:
            cf = None
        try:
            qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None
        except Exception:
            qfin = None
        qbs = None  # removed: quarterly BS is unaudited and caused deployment issues
        ph = None
        try:
            h = t.history(period="1y", auto_adjust=True)
            if h is not None and not h.empty and 'Close' in h.columns:
                ph = h['Close'].dropna()
                if len(ph) > 0 and not is_valid(info.get('currentPrice')):
                    info['currentPrice'] = float(ph.iloc[-1])
                    debug['fallbacks'].append('price from 1y history')
        except Exception:
            pass
        if ph is None or len(ph) == 0:
            try:
                h = t.history(period="5d", auto_adjust=True)
                if h is not None and not h.empty and 'Close' in h.columns:
                    ph = h['Close'].dropna()
                    if len(ph) > 0 and not is_valid(info.get('currentPrice')):
                        info['currentPrice'] = float(ph.iloc[-1])
                        debug['fallbacks'].append('price from 5d history')
            except Exception:
                pass
        if not is_valid(info.get('sharesOutstanding')):
            sh = get_shares(fin, bs, qfin, {})
            if sh:
                info['sharesOutstanding'] = sh
                debug['fallbacks'].append('shares from financials')
        price_for_val = None
        for f in ['currentPrice', 'regularMarketPrice', 'previousClose']:
            v = info.get(f)
            if is_valid(v) and v > 0:
                price_for_val = float(v)
                break
        if is_valid(info.get('sharesOutstanding')) and price_for_val:
            corrected, was_fixed = validate_shares(price_for_val, info['sharesOutstanding'], info)
            if was_fixed:
                info['sharesOutstanding'] = corrected
                debug['warnings'].append(f'Shares unit corrected to {corrected:,.0f}')

        # ---- NEW: Fetch holdings data (leading indicator) ----
        holdings = dict(DEFAULT_HOLDINGS)
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                for idx_row in range(len(mh)):
                    try:
                        val_raw = mh.iloc[idx_row, 0]
                        label_raw = str(mh.iloc[idx_row, 1]).lower() if mh.shape[1] > 1 else ''
                        val_clean = float(str(val_raw).replace('%', '').strip())
                        if 'insider' in label_raw or 'promoter' in label_raw:
                            holdings['insider_pct'] = round(val_clean, 1)
                        elif 'institution' in label_raw and 'float' not in label_raw:
                            holdings['inst_pct'] = round(val_clean, 1)
                        elif 'number' in label_raw:
                            holdings['inst_count'] = int(val_clean)
                    except (ValueError, IndexError):
                        continue
                debug['fallbacks'].append('holdings from major_holders')
        except Exception:
            pass

        has_price = any(is_valid(info.get(f)) for f in ['currentPrice', 'regularMarketPrice', 'previousClose'])
        has_fin = fin is not None and fin.shape[1] >= 1
        has_name = any(info.get(f) for f in ['shortName', 'longName'])
        if not has_price and not has_fin and not has_name:
            return None
        debug['data_years'] = {
            'annual': fin.shape[1] if fin is not None else 0,
            'quarterly': qfin.shape[1] if qfin is not None else 0,
            'balance_sheet': bs.shape[1] if bs is not None else 0,
            
            'cashflow': cf.shape[1] if cf is not None else 0,
            'price_days': len(ph) if ph is not None else 0,
        }
        return {"info": info, "fin": fin, "bs": bs, "cf": cf, "qfin": qfin,
                "qbs": qbs, "ph": ph, "debug": debug, "holdings": holdings}
    except Exception:
        return None

# ============================================================
# METRIC EXTRACTORS
# ============================================================
def ext_price(sd):
    for f in ['currentPrice', 'regularMarketPrice', 'regularMarketPreviousClose', 'previousClose']:
        v = sd['info'].get(f)
        if is_valid(v) and v > 0:
            return round(float(v), 2)
    ph = sd.get('ph')
    if ph is not None and len(ph) > 0:
        v = float(ph.iloc[-1])
        if is_valid(v) and v > 0:
            return round(v, 2)
    return None

def ext_name(sd, ticker):
    for f in ['shortName', 'longName', 'displayName']:
        v = sd['info'].get(f)
        if v and str(v).strip() and str(v).strip().lower() != 'none':
            return str(v).strip()
    return ticker.replace('.NS', '').replace('.BO', '')

def ext_sector(sd):
    for f in ['sector', 'industry']:
        v = sd['info'].get(f)
        if v and str(v).strip() and str(v).strip().lower() != 'none':
            return str(v).strip()
    return "—"

def ext_pe(sd, price):
    info = sd['info']
    for f in ['trailingPE', 'forwardPE']:
        v = info.get(f)
        if is_valid(v) and 1 < v < 500:
            return round(float(v), 1), f
    if not is_valid(price) or price <= 0:
        return None, 'no price'
    for f in ['trailingEps', 'forwardEps']:
        eps = info.get(f)
        if is_valid(eps) and eps > 0:
            pe = price / eps
            if 1 < pe < 500:
                return round(pe, 1), f'price/{f}'
    qfin = sd.get('qfin')
    if qfin is not None and qfin.shape[1] >= 4:
        ni_vals = [get_ni(qfin, i) for i in range(4)]
        valid_vals = [v for v in ni_vals if v is not None]
        if len(valid_vals) >= 3:
            ttm = sum(valid_vals)
            if ttm > 0:
                sh = get_shares(sd.get('fin'), sd.get('bs'), qfin, info)
                if is_valid(sh) and sh > 0:
                    pe = price / (ttm / sh)
                    if 1 < pe < 500:
                        return round(pe, 1), 'TTM_quarterly'
    fin = sd.get('fin')
    if fin is not None:
        ni = get_ni(fin, 0)
        if is_valid(ni) and ni > 0:
            sh = get_shares(fin, sd.get('bs'), sd.get('qfin'), info)
            if is_valid(sh) and sh > 0:
                pe = price / (ni / sh)
                if 1 < pe < 500:
                    return round(pe, 1), 'annual_NI/shares'
    return None, 'not calculable'

def ext_mcap(sd, price):
    v = sd['info'].get('marketCap')
    if is_valid(v) and v > 0:
        return float(v)
    if is_valid(price) and price > 0:
        sh = get_shares(sd.get('fin'), sd.get('bs'), sd.get('qfin'), sd['info'])
        if is_valid(sh) and sh > 0:
            return price * sh
    return None

def ext_de(sd):
    info_de = sd['info'].get('debtToEquity')
    if is_valid(info_de):
        # yfinance reports D/E as percentage (e.g. 45 = 0.45x)
        # BUG FIX: detect if already in decimal form
        val = float(info_de)
        if val > 5:  # likely percentage form (e.g. 45.2 meaning 0.452x)
            return round(val / 100, 2)
        else:  # already in ratio form (e.g. 0.45)
            return round(val, 2)
    bs = sd.get('bs')
    if bs is not None:
        debt = get_total_debt(bs, 0) or 0
        eq = get_equity(bs, 0)
        if is_valid(eq) and eq > 0:
            return round(debt / eq, 2)
    return 0.0

# NEW: Interest Coverage Ratio
def ext_interest_coverage(sd):
    """EBIT / Interest Expense. Higher = safer. Below 3 = danger."""
    fin = sd.get('fin')
    if fin is None:
        return None, 'no data'
    ebit = get_ebit(fin, 0)
    interest = get_interest_expense(fin, 0)
    if is_valid(ebit) and is_valid(interest) and interest > 0:
        return round(ebit / interest, 1), 'EBIT/Interest'
    # Fallback: try from info dict
    v = sd['info'].get('interestCoverage')
    if is_valid(v):
        return round(float(v), 1), 'info'
    return None, 'not calculable'

# NEW: Net Debt
def ext_net_debt(sd):
    """Total Debt - Cash. Negative = net cash positive."""
    bs = sd.get('bs')
    if bs is None:
        return None
    debt = get_total_debt(bs, 0) or 0
    cash = get_cash(bs, 0) or 0
    return round((debt - cash) / 1e7, 0)  # in crores

def ext_cagr(fin, getter, min_years=0.8):
    if fin is None or fin.shape[1] < 2:
        return None, None, None, None
    lv, ld, ov, od = None, None, None, None
    for i in range(fin.shape[1]):
        v = getter(fin, i)
        if is_valid(v) and v > 0:
            d = fin.columns[i]
            if lv is None:
                lv, ld = v, d
            ov, od = v, d
    if lv is None or ov is None or ld == od:
        return None, None, None, None
    try:
        yrs = abs((ld - od).days) / 365.25
    except Exception:
        yrs = float(abs(fin.columns.tolist().index(ld) - fin.columns.tolist().index(od)))
    if yrs < min_years:
        return None, None, None, None
    label = f"{fy_label(od)}-{fy_label(ld)}" if fy_label(od) and fy_label(ld) else f"{round(yrs)}Y"
    try:
        c = ((lv / ov) ** (1.0 / yrs) - 1) * 100
        if is_valid(c) and -99 < c < 500:
            dbg = f"annual: {ov / 1e7:.0f}Cr -> {lv / 1e7:.0f}Cr over {yrs:.1f}y"
            return round(c, 1), label, round(yrs, 1), dbg
    except Exception:
        pass
    return None, None, None, None

def ext_cagr_q(qfin, getter):
    if qfin is None or qfin.shape[1] < 5:
        return None, None, None, None
    n = qfin.shape[1]
    lt_vals = [getter(qfin, i) for i in range(min(4, n))]
    lt = sum(v for v in lt_vals if v is not None)
    if n >= 8:
        ot_vals = [getter(qfin, i) for i in range(4, 8)]
        ot = sum(v for v in ot_vals if v is not None)
        try:
            yrs = abs((qfin.columns[0] - qfin.columns[4]).days) / 365.25
        except Exception:
            yrs = 1.0
    else:
        ot_vals = [getter(qfin, i) for i in range(n - 4, n)]
        ot = sum(v for v in ot_vals if v is not None)
        try:
            yrs = abs((qfin.columns[0] - qfin.columns[n - 4]).days) / 365.25
        except Exception:
            yrs = 1.0
    if lt and lt > 0 and ot and ot > 0 and yrs > 0.5:
        try:
            c = ((lt / ot) ** (1.0 / yrs) - 1) * 100
            if is_valid(c) and -99 < c < 500:
                label = f"{fy_label(qfin.columns[min(n - 1, 7)])}-{fy_label(qfin.columns[0])}"
                dbg = f"quarterly TTM: {ot / 1e7:.0f}Cr -> {lt / 1e7:.0f}Cr over {yrs:.1f}y"
                return round(c, 1), label, round(yrs, 1), dbg
        except Exception:
            pass
    return None, None, None, None

def ext_roe(fin, bs, info):
    ni = get_ni(fin, 0) if fin is not None else None
    eq = get_equity(bs, 0) if bs is not None else None
    if is_valid(ni) and is_valid(eq) and eq > 0:
        return round(ni / eq * 100, 1), 'manual'
    v = info.get('returnOnEquity')
    if is_valid(v):
        return round(float(v) * 100, 1), 'info'
    return None, 'none'

def ext_roce(fin, bs):
    ebit = get_ebit(fin, 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if is_valid(ebit) and is_valid(ta) and is_valid(cl) and (ta - cl) > 0:
        return round(ebit / (ta - cl) * 100, 1)
    return None

def ext_1y_ret(sd):
    ph = sd.get('ph')
    if ph is None or len(ph) < 30:
        return None
    try:
        cur = float(ph.iloc[-1])
        if not is_valid(cur) or cur <= 0:
            return None
        tgt = ph.index[-1] - pd.Timedelta(days=365)
        mask = ph.index <= tgt
        if mask.sum() > 0:
            past = float(ph.loc[mask].iloc[-1])
        else:
            past = float(ph.iloc[0])
        if is_valid(past) and past > 0:
            r = round((cur / past - 1) * 100, 1)
            return r if is_valid(r) else None
    except Exception:
        pass
    return None

# ============================================================
# LAYER 1 — QUANTITATIVE CORE SCREEN
# ============================================================
def run_layer1(sd, ticker):
    r = {}
    info = sd['info']
    fin = sd.get('fin')
    bs = sd.get('bs')
    r['price'] = ext_price(sd)
    mcap = ext_mcap(sd, r['price'])
    r['mcap_cr'] = round(mcap / 1e7) if is_valid(mcap) else None
    r['mcap_pass'] = is_valid(mcap) and mcap > 150e9
    r['pe'], r['pe_source'] = ext_pe(sd, r['price'])
    r['de'] = ext_de(sd)
    r['de_pass'] = is_valid(r['de']) and r['de'] < 0.5
    r['roe'], r['roe_source'] = ext_roe(fin, bs, info)
    r['roe_pass'] = is_valid(r['roe']) and r['roe'] > 15
    r['roce'] = ext_roce(fin, bs)
    r['roce_pass'] = is_valid(r['roce']) and r['roce'] > 18

    # NEW: Interest coverage & net debt
    r['int_cov'], r['int_cov_source'] = ext_interest_coverage(sd)
    r['net_debt_cr'] = ext_net_debt(sd)

    sc, sl, sy, sd1 = ext_cagr(fin, get_revenue)
    r['sales_cagr_debug'] = sd1
    if sc is None:
        sc, sl, sy, sd1 = ext_cagr_q(sd.get('qfin'), get_revenue)
        r['sales_cagr_debug'] = sd1 or 'quarterly fallback'
    if sc is None and is_valid(info.get('revenueGrowth')):
        sc = round(float(info['revenueGrowth']) * 100, 1)
        sl = "1Y"
        r['sales_cagr_debug'] = 'info.revenueGrowth fallback'
    r['sales_cagr'] = sc
    r['sales_cagr_label'] = sl or ""
    pc, pl, py, pd1 = ext_cagr(fin, get_ni)
    r['pat_cagr_debug'] = pd1
    if pc is None:
        pc, pl, py, pd1 = ext_cagr_q(sd.get('qfin'), get_ni)
        r['pat_cagr_debug'] = pd1 or 'quarterly fallback'
    if pc is None and is_valid(info.get('earningsGrowth')):
        pc = round(float(info['earningsGrowth']) * 100, 1)
        pl = "1Y"
        r['pat_cagr_debug'] = 'info.earningsGrowth fallback'
    r['pat_cagr'] = pc
    r['pat_cagr_label'] = pl or ""
    growth_floor = 15
    r['growth_relaxed'] = False
    if is_valid(r['roe']) and r['roe'] > 25 and is_valid(mcap) and mcap > 500e9:
        growth_floor = 10
        r['growth_relaxed'] = True
    r['growth_floor'] = growth_floor
    r['growth_pass'] = (r.get('sales_cagr') or 0) > growth_floor and (r.get('pat_cagr') or 0) > growth_floor
    r['pass'] = all([r['mcap_pass'], r['de_pass'], r['roe_pass'], r['roce_pass'], r['growth_pass']])
    r['name'] = ext_name(sd, ticker)
    r['sector'] = ext_sector(sd)
    return r

# ============================================================
# MULTI-YEAR DATA BUILDER
# ============================================================
def build_multi(fin, bs, cf):
    if fin is None or bs is None or cf is None:
        return None
    yrs = min(fin.shape[1], bs.shape[1], cf.shape[1])
    if yrs < 2:
        return None
    rows = []
    for i in range(yrs):
        rows.append({
            'year': str(fin.columns[i].year) if hasattr(fin.columns[i], 'year') else str(i),
            'revenue': get_revenue(fin, i), 'net_income': get_ni(fin, i),
            'ebitda': get_ebitda(fin, i), 'ebit': get_ebit(fin, i),
            'receivables': get_receivables(bs, i),
            'inventory': safe_get(bs, 'Inventory', i), 'equity': get_equity(bs, i),
            'current_liabilities': safe_get(bs, 'Current Liabilities', i),
            'total_assets': safe_get(bs, 'Total Assets', i),
            'cfo': get_cfo(cf, i), 'payables': get_payables(bs, i),
            'interest_expense': get_interest_expense(fin, i),
            'cash': get_cash(bs, i),
            'total_debt': get_total_debt(bs, i),
        })
    return rows

# ============================================================
# FORENSIC ACCOUNTING (enhanced with accrual ratio)
# ============================================================
def run_forensic(data):
    flags = []
    score = 100
    det = {}
    sd = sorted(data, key=lambda x: x['year'])
    latest = sd[-1]
    prior = sd[-2] if len(sd) >= 2 else None

    # Check 1: Receivables trend
    rp = []
    for d in sd:
        if is_valid(d.get('receivables')) and is_valid(d.get('revenue')) and d['revenue'] > 0:
            rp.append(round(d['receivables'] / d['revenue'] * 100, 1))
        else:
            rp.append(None)
    det['recv_pcts'] = rp
    det['recv_years'] = [d['year'] for d in sd]
    vr = [x for x in rp if x is not None]
    if len(vr) >= 3 and sum(1 for i in range(1, len(vr)) if vr[i] > vr[i - 1]) >= 2:
        flags.append(f"Receivables rising: {vr[-3]}% -> {vr[-2]}% -> {vr[-1]}% of revenue")
        score -= 15

    # Check 2: Inventory bloat
    det['inv_growth'] = None
    det['rev_growth_yoy'] = None
    if prior:
        il, ip = latest.get('inventory'), prior.get('inventory')
        rl, rp2 = latest.get('revenue'), prior.get('revenue')
        if all(is_valid(v) and v > 0 for v in [il, ip, rl, rp2]):
            ig = round((il / ip - 1) * 100, 1)
            rg = round((rl / rp2 - 1) * 100, 1)
            det['inv_growth'] = ig
            det['rev_growth_yoy'] = rg
            if ig > rg + 10:
                flags.append(f"Inventory bloat: {ig}% vs Revenue {rg}%")
                score -= 15

    # Check 3: Cumulative CFO/PAT
    tc = sum(d['cfo'] for d in sd if is_valid(d.get('cfo')))
    tp = sum(d['net_income'] for d in sd if is_valid(d.get('net_income')))
    if tp > 0:
        det['cum_cfo_pat'] = round(tc / tp, 2)
        if det['cum_cfo_pat'] < 0.5:
            flags.append(f"Critical: Cum CFO/PAT {det['cum_cfo_pat']}x")
            score -= 30
        elif det['cum_cfo_pat'] < 0.7:
            flags.append(f"Low cash: Cum CFO/PAT {det['cum_cfo_pat']}x")
            score -= 20
    else:
        det['cum_cfo_pat'] = None
    det['single_cfo_pat'] = None
    if is_valid(latest.get('cfo')) and is_valid(latest.get('net_income')) and latest['net_income'] > 0:
        det['single_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2)

    # Check 4: Cumulative CFO/EBITDA
    te = sum(d['ebitda'] for d in sd if is_valid(d.get('ebitda')) and d['ebitda'] > 0)
    if te > 0:
        det['cum_cfo_ebitda'] = round(tc / te, 2)
        if det['cum_cfo_ebitda'] < 0.5:
            flags.append(f"Critical: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x")
            score -= 20
        elif det['cum_cfo_ebitda'] < 0.7:
            flags.append(f"Weak: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x")
            score -= 15
    else:
        det['cum_cfo_ebitda'] = None
    yr_ratios = []
    for d in sd:
        if is_valid(d.get('cfo')) and is_valid(d.get('ebitda')) and d['ebitda'] > 0:
            yr_ratios.append(round(d['cfo'] / d['ebitda'], 2))
    det['cfo_ebitda_trend'] = yr_ratios
    if (len(yr_ratios) >= 2 and is_valid(det.get('cum_cfo_ebitda'))
            and yr_ratios[-1] < yr_ratios[-2] and yr_ratios[-1] < 0.5 and det['cum_cfo_ebitda'] < 0.7):
        flags.append(f"Deteriorating: CFO/EBITDA {yr_ratios[-2]} -> {yr_ratios[-1]}")
        score -= 10

    # Bonus: Negative cumulative CFO
    if tc < 0:
        flags.append("Negative cumulative CFO")
        score -= 25

    # NEW Check 5: Accrual Ratio — (NI - CFO) / Avg Total Assets
    det['accrual_ratio'] = None
    if prior and is_valid(latest.get('net_income')) and is_valid(latest.get('cfo')):
        ta_l = latest.get('total_assets')
        ta_p = prior.get('total_assets')
        if is_valid(ta_l) and is_valid(ta_p) and (ta_l + ta_p) > 0:
            avg_ta = (ta_l + ta_p) / 2
            ar = round((latest['net_income'] - latest['cfo']) / avg_ta * 100, 1)
            det['accrual_ratio'] = ar
            if ar > 20:
                flags.append(f"High accrual ratio: {ar}% — earnings far exceed cash")
                score -= 20
            elif ar > 10:
                flags.append(f"Elevated accrual ratio: {ar}% — watch earnings quality")
                score -= 10

    # Margins
    margins_by_year = []
    for d in sd:
        if is_valid(d.get('ebitda')) and is_valid(d.get('revenue')) and d['revenue'] > 0:
            margins_by_year.append((d['year'], round(d['ebitda'] / d['revenue'] * 100, 1)))
    det['margins_by_year'] = margins_by_year
    margin_vals = [m[1] for m in margins_by_year]
    if len(margin_vals) >= 2 and margin_vals[-1] > margin_vals[0] + 2:
        det['margin_trend'] = 'expanding'
    elif len(margin_vals) >= 2 and margin_vals[-1] < margin_vals[0] - 3:
        det['margin_trend'] = 'contracting'
    else:
        det['margin_trend'] = 'stable'

    # NEW: Interest coverage trend
    det['int_cov_trend'] = []
    for d in sd:
        ebit_d = d.get('ebit')
        int_d = d.get('interest_expense')
        if is_valid(ebit_d) and is_valid(int_d) and int_d > 0:
            det['int_cov_trend'].append((d['year'], round(ebit_d / int_d, 1)))

    # CCC
    ccc_latest = None
    ccc_prior = None
    for d in sd:
        rev = d.get('revenue')
        if not (is_valid(rev) and rev > 0):
            continue
        recv = d.get('receivables') if is_valid(d.get('receivables')) else 0
        inv = d.get('inventory') if is_valid(d.get('inventory')) else 0
        pay = d.get('payables') if is_valid(d.get('payables')) else 0
        dr = round(recv / rev * 365, 1)
        di = round(inv / rev * 365, 1)
        dp = round(pay / rev * 365, 1)
        ccc = round(dr + di - dp, 1)
        ccc_prior = ccc_latest
        ccc_latest = {'days_recv': dr, 'days_inv': di, 'days_pay': dp, 'ccc': ccc, 'year': d['year']}
    det['ccc'] = ccc_latest
    det['ccc_trend'] = None
    if ccc_latest and ccc_prior:
        diff = ccc_latest['ccc'] - ccc_prior['ccc']
        det['ccc_trend'] = 'improving' if diff < -5 else 'worsening' if diff > 10 else 'stable'
        if ccc_prior['days_pay'] > 0:
            dp_growth = (ccc_latest['days_pay'] / ccc_prior['days_pay'] - 1) * 100
            if dp_growth > 30 and is_valid(det.get('cum_cfo_pat')) and det['cum_cfo_pat'] < 0.7:
                flags.append(f"Payables stretch: +{dp_growth:.0f}% YoY with weak cash ({det['cum_cfo_pat']}x)")

    det['score'] = max(score, 0)
    det['flags'] = flags
    det['num_flags'] = len(flags)
    return det

# ============================================================
# PROMOTER / HOLDINGS ANALYSIS (NEW — leading indicator)
# ============================================================
def run_holdings_analysis(holdings):
    """Analyse holdings data for red flags. Returns enriched holdings dict."""
    result = dict(holdings)
    result['flags'] = []
    result['score_adj'] = 0

    insider = holdings.get('insider_pct')
    inst = holdings.get('inst_pct')

    if is_valid(insider):
        if insider < 20:
            result['flags'].append(f"Very low promoter/insider holding: {insider}%")
            result['score_adj'] -= 10
        elif insider > 75:
            result['flags'].append(f"Very high promoter holding: {insider}% — low float, governance risk")
            result['score_adj'] -= 5

    if is_valid(inst):
        if inst < 5:
            result['flags'].append(f"Negligible institutional ownership: {inst}% — no external oversight")
            result['score_adj'] -= 5

    return result

# ============================================================
# MOAT DURABILITY
# ============================================================
def run_moat(data):
    sd = sorted(data, key=lambda x: x['year'])
    rh = []
    for d in sd:
        if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity'] > 0:
            rh.append((d['year'], round(d['net_income'] / d['equity'] * 100, 1)))
    if not rh:
        return dict(DEFAULT_MOAT)
    vals = [r[1] for r in rh]
    above = sum(1 for v in vals if v > 15)
    total = len(vals)
    pct = round(above / total * 100) if total > 0 else 0
    if above == total and total >= 2:
        cons = 'established moat'
    elif len(vals) >= 2 and vals[-1] > 15 and vals[-2] > 15 and vals[-1] > vals[0]:
        cons = 'emerging moat'
    else:
        cons = 'weak/no moat'
    return {'years_above_15': above, 'total_years': total, 'pct': pct,
            'consistency': cons, 'roe_by_year': rh, 'quarterly_roe': [], 'extended_total': total}

# ============================================================
# CYCLICAL ROE
# ============================================================
def run_cyclical(data):
    sd = sorted(data, key=lambda x: x['year'])
    rv = []
    rby = {}
    for d in sd:
        if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity'] > 0:
            r = round(d['net_income'] / d['equity'] * 100, 1)
            rv.append(r)
            rby[d['year']] = r
    if len(rv) < 2:
        return dict(DEFAULT_CYC)
    return {
        'latest': rv[-1], 'avg': round(float(np.mean(rv)), 1),
        'median': round(float(np.median(rv)), 1),
        'peak': rv[-1] > float(np.mean(rv)) * 2 and rv[-1] > 20,
        'values': rv, 'roe_by_year': rby,
    }

# ============================================================
# EARNINGS MOMENTUM (improved edge-case handling)
# ============================================================
def run_momentum(qfin):
    if qfin is None or qfin.shape[1] < 3:
        return dict(DEFAULT_MOM)
    ev = []
    qs = []
    for i in range(min(4, qfin.shape[1])):
        e = safe_get(qfin, 'Diluted EPS', i) or safe_get(qfin, 'Basic EPS', i)
        if e is not None:
            ev.append(e)
            try:
                qs.append(qfin.columns[i].strftime('%b %Y'))
            except Exception:
                qs.append(str(i))
    if len(ev) < 3:
        return dict(DEFAULT_MOM)
    # BUG FIX: handle zero and negative EPS safely
    lq, pq = None, None
    if ev[1] != 0:
        lq_raw = (ev[0] / ev[1] - 1) * 100
        if is_valid(lq_raw) and -500 < lq_raw < 500:
            lq = round(lq_raw, 1)
    if ev[2] != 0:
        pq_raw = (ev[1] / ev[2] - 1) * 100
        if is_valid(pq_raw) and -500 < pq_raw < 500:
            pq = round(pq_raw, 1)
    return {'available': True, 'latest_qoq': lq, 'prior_qoq': pq, 'eps': ev[:4], 'quarters': qs[:4]}

# ============================================================
# DUAL VALUATION
# ============================================================
def run_valuation(pe, pat_cagr, sales_cagr, price, mcap, sd):
    peg, peg_growth, peg_turnaround = None, None, False
    if is_valid(pe) and pe > 0:
        growth = pat_cagr
        if is_valid(pat_cagr) and is_valid(sales_cagr) and sales_cagr > 0 and pat_cagr > sales_cagr * 2:
            growth = sales_cagr
            peg_turnaround = True
        if is_valid(growth) and growth > 0:
            peg = round(pe / growth, 2)
            peg_growth = growth
    fcf_yield = None
    cf = sd.get('cf')
    if cf is not None and is_valid(mcap) and mcap > 0:
        fcf = get_fcf(cf, 0)
        if is_valid(fcf):
            fcf_yield = round(fcf / mcap * 100, 2)
    return {'peg': peg, 'peg_growth': peg_growth, 'peg_turnaround': peg_turnaround, 'fcf_yield': fcf_yield}

# ============================================================
# POSITION SIZING v4 (enhanced with holdings & interest coverage)
# ============================================================
def get_tier(score, nf, peak, peg, mom_1y, moat_cons, holdings=None, int_cov=None):
    if score is None or score < 50:
        return 'WATCH', '0%'
    if is_valid(peg) and peg > 5:
        return 'WATCH', '0%'
    if is_valid(peg) and peg > 3:
        return 'HALF', '4-6%'

    # NEW: Holdings safety — very low promoter = cap at STANDARD
    holdings_cap = None
    if holdings and holdings.get('score_adj', 0) <= -10:
        holdings_cap = 'STANDARD'

    # NEW: Interest coverage safety — below 3x = cap at HALF
    if is_valid(int_cov) and int_cov < 3:
        return 'HALF', '4-6%'

    moat_ok = moat_cons in ('established moat', 'emerging moat')
    if score >= 85 and nf == 0 and not peak and moat_ok:
        base = 'FULL'
    elif score >= 85 and nf == 0 and not peak:
        base = 'STANDARD'
    elif score >= 70 and nf <= 1:
        base = 'STANDARD'
    else:
        base = 'HALF'

    # Apply holdings cap
    tier_rank = {'WATCH': 0, 'HALF': 1, 'STANDARD': 2, 'FULL': 3}
    if holdings_cap and tier_rank.get(base, 0) > tier_rank.get(holdings_cap, 0):
        base = holdings_cap

    if base == 'FULL' and not is_valid(peg):
        return 'STANDARD', '8-10%'
    if base == 'FULL' and peg < 0.5:
        return 'FULL', '12-15%'
    if base == 'FULL' and is_valid(mom_1y) and mom_1y < -30:
        return 'HALF', '4-6%'
    if base == 'FULL' and peg > 1.5:
        return 'STANDARD', '8-10%'
    return {'FULL': ('FULL', '12-15%'), 'STANDARD': ('STANDARD', '8-10%'), 'HALF': ('HALF', '4-6%')}[base]

# ============================================================
# VERDICT (enhanced)
# ============================================================
def gen_verdict(name, tier, l1, acct, cyc, val, ret, moat, is_bank, holdings=None):
    if is_bank:
        return (f"{name} is a banking/financial stock. This framework uses ROE, D/E, and "
                f"CFO/EBITDA metrics for non-financials. Banks need NIM, CASA, Credit Cost, GNPA. Scores for reference only.")
    peg = val.get('peg'); fy = val.get('fcf_yield'); roe = l1.get('roe'); de = l1.get('de')
    cfo = acct.get('cum_cfo_pat'); mby = acct.get('margins_by_year', []); mt = acct.get('margin_trend', '')
    mp = moat.get('pct', 0); mc = moat.get('consistency', '')
    sc_v, sl = l1.get('sales_cagr'), l1.get('sales_cagr_label', '')
    pc_v, pl = l1.get('pat_cagr'), l1.get('pat_cagr_label', '')
    sents = []
    if tier == 'FULL':
        sents.append(f"{name} passes all layers with conviction.")
    elif tier == 'STANDARD':
        sents.append(f"{name} clears the quality bar but one or more factors prevent full conviction.")
    elif tier == 'HALF':
        sents.append(f"{name} presents a mixed profile — genuine strengths alongside flags that need resolution.")
    else:
        sents.append(f"{name} fails critical checks and does not warrant capital deployment at current readings.")
    q = []
    if is_valid(roe) and roe > 15:
        q.append(f"ROE at {roe}%{' on ' + str(de) + 'x D/E' if is_valid(de) else ''}")
    if is_valid(cfo) and cfo > 1.0:
        q.append(f"cumulative cash conversion at {cfo}x — more cash than reported profit")
    elif is_valid(cfo) and cfo > 0.7:
        q.append(f"cumulative cash conversion at {cfo}x")
    elif is_valid(cfo) and cfo < 0.5:
        q.append(f"cumulative cash conversion at only {cfo}x — structural gap between profit and cash")
    # NEW: Interest coverage in verdict
    int_cov = l1.get('int_cov')
    if is_valid(int_cov):
        if int_cov < 3:
            q.append(f"interest coverage at {int_cov}x — dangerously thin")
        elif int_cov > 15:
            q.append(f"interest coverage at {int_cov}x — debt is irrelevant")
    if mby and len(mby) >= 2:
        margin_str = " -> ".join(f"{m[1]}%" for m in mby)
        if mt == 'expanding':
            q.append(f"EBITDA margins expanding {margin_str}")
        elif mt == 'contracting':
            q.append(f"EBITDA margins contracting {margin_str}")
    if q:
        sents.append(", ".join(q) + ".")
    if is_valid(sc_v) and is_valid(pc_v):
        sents.append(f"Revenue compounded at {sc_v}% across {sl} with PAT at {pc_v}% across {pl}.")
        if l1.get('growth_relaxed'):
            sents.append("Growth threshold relaxed to 10% — mature compounder with ROE above 25%.")
    # NEW: Holdings in verdict
    if holdings and holdings.get('flags'):
        sents.append("Ownership concern: " + "; ".join(holdings['flags'][:2]).lower() + ".")
    # NEW: Accrual ratio in verdict
    ar = acct.get('accrual_ratio')
    if is_valid(ar) and ar > 10:
        sents.append(f"Accrual ratio at {ar}% — reported earnings significantly exceed cash earnings relative to asset base.")
    ccc = acct.get('ccc')
    if ccc and acct.get('ccc_trend') == 'worsening':
        sents.append(f"Cash conversion cycle worsened to {ccc['ccc']} days — more capital trapped in working capital.")
    if is_valid(peg):
        tn = f" (using revenue growth {val.get('peg_growth')}% as PAT CAGR inflated by turnaround)" if val.get('peg_turnaround') else ""
        if peg < 0.5:
            sents.append(f"At PEG {peg}{tn}, the market prices a fraction of earnings growth into the multiple.")
        elif peg < 1.0:
            sents.append(f"PEG at {peg}{tn} indicates growth more than justifies the PE.")
        elif peg < 1.5:
            sents.append(f"PEG at {peg}{tn} — fairly valued.")
        elif peg < 2.0:
            sents.append(f"PEG at {peg}{tn} embeds a valuation premium.")
        else:
            sents.append(f"PEG at {peg}{tn} signals overvaluation.")
    if is_valid(fy) and fy > 5:
        sents.append(f"FCF yield at {fy}% provides additional margin of safety.")
    elif is_valid(fy) and fy < 0:
        sents.append(f"Negative FCF yield at {fy}% — the company consumes cash despite reported profitability.")
    if mc == 'established moat':
        sents.append(f"Moat durability at {mp}% ({moat.get('years_above_15')}/{moat.get('total_years')} years) confirms structural competitive advantage.")
    elif mc == 'emerging moat':
        sents.append(f"Moat is emerging — ROE crossed 15% in latest two years ({moat.get('years_above_15')}/{moat.get('total_years')} above threshold), trajectory positive but unproven.")
    elif moat.get('total_years', 0) >= 2:
        sents.append(f"Moat durability at only {mp}% ({moat.get('years_above_15')}/{moat.get('total_years')} years) raises sustainability questions.")
    if cyc.get('peak'):
        sents.append(f"Current ROE at {cyc.get('latest')}% vs median {cyc.get('median')}% indicates cyclical peak — use normalized figure for valuation.")
    if is_valid(ret) and ret < -30:
        sents.append(f"The {abs(ret):.0f}% drawdown signals the market is repricing something negative.")
    if tier in ['HALF', 'WATCH'] and acct.get('flags'):
        sents.append("Specific flags: " + "; ".join(acct['flags'][:2]).lower() + ".")
    actions = {'FULL': "Position at 12-15% of portfolio.", 'STANDARD': "Position at 8-10%.",
               'HALF': "Half position at 4-6% only — add after next results confirm improvement.",
               'WATCH': "Do not deploy capital. Monitor quarterly."}
    sents.append(actions.get(tier, "Monitor."))
    return " ".join(sents)

# ============================================================
# LAYER DISPLAY (enhanced)
# ============================================================
def gen_layers(l1, acct, cyc, val, mom, ret, moat, holdings=None):
    L = []
    sl, pl = l1.get('sales_cagr_label', ''), l1.get('pat_cagr_label', '')
    gf = l1.get('growth_floor', 15)
    gf_note = f" (relaxed to {gf}%)" if l1.get('growth_relaxed') else ""
    if l1['pass']:
        L.append(("Fundamentals", "pass", f"MCap Rs {fmt(l1.get('mcap_cr'), ',.0f')} Cr | ROE {fmt(l1.get('roe'), '.1f')}% | ROCE {fmt(l1.get('roce'), '.1f')}% | D/E {l1.get('de', '-')} | Sales {fmt(l1.get('sales_cagr'), '.1f')}% ({sl}) | PAT {fmt(l1.get('pat_cagr'), '.1f')}% ({pl}){gf_note}"))
    else:
        fails = []
        if not l1['mcap_pass']:
            fails.append(f"MCap Rs {fmt(l1.get('mcap_cr'), ',.0f')} Cr < Rs 15,000 Cr")
        if not l1['roe_pass']:
            fails.append(f"ROE {fmt(l1.get('roe'), '.1f')}% < 15%")
        if not l1['roce_pass']:
            fails.append(f"ROCE {fmt(l1.get('roce'), '.1f')}% < 18%")
        if not l1['de_pass']:
            fails.append(f"D/E {l1.get('de', '-')} > 0.5")
        if not l1['growth_pass']:
            fails.append(f"Growth: Sales {fmt(l1.get('sales_cagr'), '.1f')}% / PAT {fmt(l1.get('pat_cagr'), '.1f')}% < {gf}%")
        L.append(("Fundamentals", "fail", " | ".join(fails) if fails else "Criteria not met"))

    # NEW: Interest coverage layer
    int_cov = l1.get('int_cov')
    if is_valid(int_cov):
        if int_cov >= 8:
            L.append(("Interest coverage", "pass", f"{int_cov}x — comfortable"))
        elif int_cov >= 3:
            L.append(("Interest coverage", "warn", f"{int_cov}x — adequate"))
        else:
            L.append(("Interest coverage", "fail", f"{int_cov}x — danger zone"))
    else:
        de = l1.get('de', 0)
        if is_valid(de) and de < 0.1:
            L.append(("Interest coverage", "pass", "Minimal debt — N/A"))
        else:
            L.append(("Interest coverage", "warn", "Not calculable"))

    s = acct.get('score')
    nf = acct.get('num_flags', 0)
    if s is not None:
        ar_note = f" | Accrual {acct.get('accrual_ratio', '-')}%" if is_valid(acct.get('accrual_ratio')) else ""
        if s >= 85:
            L.append(("Forensic accounting", "pass", f"Score {s}/100 | {nf} flags | CFO/PAT {fmt(acct.get('cum_cfo_pat'), '.2f')}x | CFO/EBITDA {fmt(acct.get('cum_cfo_ebitda'), '.2f')}x{ar_note}"))
        elif s >= 50:
            L.append(("Forensic accounting", "warn", f"Score {s}/100 | {nf} flags | {' | '.join(acct.get('flags', [])[:2])}"))
        else:
            L.append(("Forensic accounting", "fail", f"Score {s}/100 | {nf} flags | {' | '.join(acct.get('flags', [])[:2])}"))
    ccc = acct.get('ccc')
    ccc_trend = acct.get('ccc_trend')
    if ccc:
        trend_str = f" — {ccc_trend}" if ccc_trend else ""
        detail = f"{ccc['ccc']}d (R:{ccc['days_recv']} I:{ccc['days_inv']} P:{ccc['days_pay']}){trend_str}"
        L.append(("Working capital", "warn" if ccc_trend == 'worsening' else "pass", detail))
    mc = moat.get('consistency', '')
    mt2 = moat.get('total_years', 0)
    if mt2 >= 2:
        st2 = 'pass' if mc == 'established moat' else 'warn' if mc == 'emerging moat' else 'fail'
        ext = f" (+ {len(moat.get('quarterly_roe', []))} quarterly points)" if moat.get('quarterly_roe') else ""
        L.append(("Moat durability", st2, f"ROE>15% in {moat['years_above_15']}/{mt2} yrs ({moat['pct']}%) — {mc}{ext}"))

    # NEW: Promoter / Holdings layer (leading indicator)
    if holdings and is_valid(holdings.get('insider_pct')):
        ins = holdings['insider_pct']
        inst = holdings.get('inst_pct')
        inst_str = f" | Inst {inst}%" if is_valid(inst) else ""
        if holdings.get('flags'):
            L.append(("Promoter/Holdings ★", "warn", f"Insider {ins}%{inst_str} — {'; '.join(holdings['flags'][:1])}"))
        elif ins > 40:
            L.append(("Promoter/Holdings ★", "pass", f"Insider {ins}%{inst_str} — strong skin in game"))
        else:
            L.append(("Promoter/Holdings ★", "pass", f"Insider {ins}%{inst_str}"))
    else:
        L.append(("Promoter/Holdings ★", "warn", "Data unavailable"))

    peg = val.get('peg'); pt = val.get('peg_turnaround'); pg = val.get('peg_growth')
    peg_note = f" (using revenue growth {pg}%)" if pt and is_valid(pg) else ""
    if is_valid(peg):
        if peg < 1.0:
            L.append(("PEG valuation", "pass", f"PEG {peg:.2f}{peg_note}"))
        elif peg < 1.5:
            L.append(("PEG valuation", "warn", f"PEG {peg:.2f}{peg_note} — fair"))
        elif peg < 2.0:
            L.append(("PEG valuation", "warn", f"PEG {peg:.2f}{peg_note} — expensive"))
        else:
            L.append(("PEG valuation", "fail", f"PEG {peg:.2f}{peg_note} — overvalued"))
    else:
        L.append(("PEG valuation", "warn", "Not calculable"))
    fy = val.get('fcf_yield')
    if is_valid(fy):
        if fy > 3:
            L.append(("FCF yield", "pass", f"{fy:.1f}%"))
        elif fy > 0:
            L.append(("FCF yield", "warn", f"{fy:.1f}% — thin"))
        else:
            L.append(("FCF yield", "fail", f"{fy:.1f}% — negative"))
    pe = l1.get('pe')
    if is_valid(pe) and pe > 80:
        L.append(("Adani filter", "fail", f"PE {pe:.0f} > 80"))
    if mom.get('available'):
        lq, pq = mom.get('latest_qoq'), mom.get('prior_qoq')
        if is_valid(lq) and is_valid(pq):
            if lq > 0 and pq > 0:
                L.append(("Earnings momentum", "pass", f"+{lq:.1f}%, +{pq:.1f}% QoQ"))
            elif lq > 0 or pq > 0:
                L.append(("Earnings momentum", "warn", f"{lq:+.1f}%, {pq:+.1f}% QoQ"))
            else:
                L.append(("Earnings momentum", "fail", f"{lq:+.1f}%, {pq:+.1f}% QoQ — declining"))
        else:
            L.append(("Earnings momentum", "warn", "Partial data"))
    else:
        L.append(("Earnings momentum", "warn", "Insufficient data"))
    if cyc.get('peak'):
        L.append(("Cyclical ROE", "warn", f"PEAK: {cyc['latest']}% vs norm {cyc['median']}%"))
    elif cyc.get('roe_by_year'):
        L.append(("Cyclical ROE", "pass", f"Not at peak: {cyc.get('latest', '-')}% vs norm {cyc.get('median', '-')}%"))
    if is_valid(ret):
        if ret > 0:
            L.append(("1Y price", "pass", f"{ret:+.1f}%"))
        elif ret > -20:
            L.append(("1Y price", "warn", f"{ret:+.1f}%"))
        else:
            L.append(("1Y price", "fail", f"{ret:+.1f}%"))
    else:
        L.append(("1Y price", "warn", "Unavailable"))
    return L

# ============================================================
# SCORECARD HTML (enhanced)
# ============================================================
def gen_scorecard(name, ticker, sector, price, tier, size, l1, acct, val, ret, moat, layers, verdict, holdings=None):
    navy = NAVY; now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    pr = f"Rs {price:,.2f}" if is_valid(price) else "—"
    sc = acct.get('score', '—')
    tier_bg = {'FULL': '#059669', 'STANDARD': '#2563EB', 'HALF': '#D97706', 'WATCH': '#DC2626'}.get(tier, '#888')
    sl, pl = l1.get('sales_cagr_label', ''), l1.get('pat_cagr_label', '')
    int_cov = l1.get('int_cov')
    int_cov_str = f"{int_cov}x" if is_valid(int_cov) else "N/A"
    ins_pct = holdings.get('insider_pct', None) if holdings else None
    ins_str = f"{ins_pct}%" if is_valid(ins_pct) else "N/A"
    met = [('PE', fmt(l1['pe'], '.1f')), ('PEG', fmt(val.get('peg'), '.2f')), ('ROE', fmt(l1['roe'], '.1f', '%')),
           ('ROCE', fmt(l1['roce'], '.1f', '%')), ('D/E', fmt(l1['de'], '.2f')), ('Int Cov', int_cov_str),
           ('FCF Yield', fmt(val.get('fcf_yield'), '.1f', '%')), ('Insider', ins_str), ('1Y Ret', fmt(ret, '+.1f', '%'))]
    met_html = "".join(f'<td style="padding:10px 4px;text-align:center;border-right:1px solid #ddd;"><div style="color:#888;font-size:8px;margin-bottom:3px;">{lb}</div><div style="font-family:Courier New,monospace;font-weight:bold;font-size:13px;color:{navy};">{vl}</div></td>' for lb, vl in met)
    layer_html = ""
    for ln, st2, dt in layers:
        ic = '&#10003;' if st2 == 'pass' else '&#10007;' if st2 == 'fail' else '&#9888;'
        cl = '#059669' if st2 == 'pass' else '#DC2626' if st2 == 'fail' else '#D97706'
        lead_tag = ' <span style="background:#DBEAFE;color:#1D4ED8;padding:1px 4px;border-radius:2px;font-size:7px;font-weight:bold;">LEAD</span>' if '★' in ln else ''
        ln_clean = ln.replace(' ★', '')
        layer_html += f'<tr><td style="padding:4px 8px;color:{cl};font-size:12px;width:18px;">{ic}</td><td style="padding:4px 8px;font-size:10px;font-weight:bold;white-space:nowrap;">{ln_clean}{lead_tag}</td><td style="padding:4px 8px;font-size:9px;color:#555;">{dt}</td></tr>'
    growth_items = [(f'Sales CAGR ({sl})', fmt(l1.get('sales_cagr'), '.1f', '%')),
                    (f'PAT CAGR ({pl})', fmt(l1.get('pat_cagr'), '.1f', '%')),
                    ('Cum CFO/PAT', fmt(acct.get('cum_cfo_pat'), '.2f', 'x')),
                    ('Cum CFO/EBITDA', fmt(acct.get('cum_cfo_ebitda'), '.2f', 'x')),
                    ('Accrual ratio', fmt(acct.get('accrual_ratio'), '.1f', '%')),
                    (f'Moat ({moat.get("total_years", 0)} yrs)', f'{moat.get("pct", 0)}% — {moat.get("consistency", "—")}')]
    net_debt = l1.get('net_debt_cr')
    if is_valid(net_debt):
        growth_items.append(('Net debt', f"Rs {net_debt:,.0f} Cr" + (" (net cash)" if net_debt < 0 else "")))
    mby = acct.get('margins_by_year', [])
    if mby and len(mby) >= 2:
        margin_str = " &rarr; ".join(f"{m[1]}%" for m in mby)
        arr = '&#8593;' if acct.get('margin_trend') == 'expanding' else '&#8595;' if acct.get('margin_trend') == 'contracting' else '&rarr;'
        growth_items.append(('EBITDA margins', f'{margin_str} {arr}'))
    if val.get('peg_turnaround'):
        growth_items.append(('PEG note', f'Using revenue growth {val.get("peg_growth")}%'))
    ccc = acct.get('ccc')
    if ccc:
        growth_items.append(('Cash cycle', f"{ccc['ccc']}d (R:{ccc['days_recv']} I:{ccc['days_inv']} P:{ccc['days_pay']})"))
    growth_html = "".join(f'<tr><td style="padding:4px 8px;font-size:9px;color:#888;">{lb}</td><td style="padding:4px 8px;font-size:11px;font-family:Courier New,monospace;text-align:right;color:{navy};">{vl}</td></tr>' for lb, vl in growth_items)
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>{name} Scorecard</title>
<style>@page{{size:A4;margin:10mm;}}*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:Segoe UI,Arial,sans-serif;color:#1a1a1a;font-size:11px;}}table{{border-collapse:collapse;width:100%;}}</style></head><body>
<div style="background:{navy};color:white;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;">
<div><div style="font-size:12px;font-weight:bold;letter-spacing:0.5px;">HIGH COMPOUNDER FRAMEWORK v2</div><div style="font-size:9px;margin-top:3px;font-weight:bold;">Vinayak Nagral  |  {datetime.now().strftime("%d %b %Y")}</div></div>
<div style="text-align:right;font-size:8px;">Quantitative screening<br>Not investment advice</div></div>
<div style="padding:14px 20px;border-bottom:3px solid {navy};display:flex;justify-content:space-between;align-items:flex-end;">
<div><div style="font-size:22px;font-weight:bold;color:{navy};">{name}</div><div style="font-size:10px;color:#888;margin-top:3px;">{ticker}  |  {sector}  |  {pr}  |  CMP as of {now_str}</div></div>
<div style="text-align:right;"><div style="background:{tier_bg};color:white;padding:5px 16px;border-radius:14px;font-size:12px;font-weight:bold;display:inline-block;">{tier}  {size}</div>
<div style="font-size:9px;color:#888;margin-top:4px;">Score {sc}/100</div></div></div>
<table style="border-bottom:1px solid #ddd;"><tr>{met_html}</tr></table>
<div style="display:flex;border-bottom:1px solid #ddd;">
<div style="flex:1;padding:12px 20px;border-right:1px solid #ddd;"><div style="font-size:10px;font-weight:bold;color:{navy};margin-bottom:8px;">LAYER RESULTS</div><table>{layer_html}</table></div>
<div style="width:240px;padding:12px 20px;"><div style="font-size:10px;font-weight:bold;color:{navy};margin-bottom:8px;">GROWTH &amp; CASH</div><table>{growth_html}</table></div></div>
<div style="padding:12px 20px;border-bottom:1px solid #ddd;"><div style="font-size:10px;font-weight:bold;color:{navy};margin-bottom:6px;">INVESTMENT VERDICT</div>
<div style="font-size:10px;line-height:1.7;">{verdict}</div></div>
<div style="padding:8px 20px;font-size:7px;color:#aaa;">Quantitative framework output. Not a research recommendation. Data via Yahoo Finance. For personal use only.</div>
</body></html>'''

# ============================================================
# FULL ANALYSIS
# ============================================================
def full_analysis(ticker):
    sd = fetch(ticker)
    if not sd:
        return None
    info = sd['info']; fin = sd['fin']; bs = sd['bs']; cf = sd['cf']
    qfin = sd['qfin']
    bank = is_banking(info, ticker)
    try:
        l1 = run_layer1(sd, ticker)
    except Exception:
        return None
    try:
        multi = build_multi(fin, bs, cf)
    except Exception:
        multi = None
    try:
        acct = run_forensic(multi) if multi else dict(DEFAULT_ACCT)
    except Exception:
        acct = dict(DEFAULT_ACCT)
    try:
        cyc = run_cyclical(multi) if multi else dict(DEFAULT_CYC)
    except Exception:
        cyc = dict(DEFAULT_CYC)
    try:
        moat = run_moat(multi) if multi else dict(DEFAULT_MOAT)
    except Exception:
        moat = dict(DEFAULT_MOAT)
    try:
        mom = run_momentum(qfin)
    except Exception:
        mom = dict(DEFAULT_MOM)
    try:
        ret = ext_1y_ret(sd)
    except Exception:
        ret = None
    try:
        mcap = ext_mcap(sd, l1['price'])
    except Exception:
        mcap = None
    try:
        val = run_valuation(l1['pe'], l1['pat_cagr'], l1.get('sales_cagr'), l1['price'], mcap, sd)
    except Exception:
        val = {'peg': None, 'peg_growth': None, 'peg_turnaround': False, 'fcf_yield': None}
    # NEW: Holdings analysis
    try:
        holdings = run_holdings_analysis(sd.get('holdings', dict(DEFAULT_HOLDINGS)))
    except Exception:
        holdings = dict(DEFAULT_HOLDINGS)
    try:
        tier, size = get_tier(
            acct.get('score'), acct.get('num_flags', 0), cyc.get('peak', False),
            val.get('peg'), ret, moat.get('consistency', ''),
            holdings, l1.get('int_cov'))
        if not l1['pass'] and tier in ['FULL', 'STANDARD']:
            tier, size = 'HALF', '4-6%'
    except Exception:
        tier, size = 'WATCH', '0%'
    try:
        layers = gen_layers(l1, acct, cyc, val, mom, ret, moat, holdings)
    except Exception:
        layers = [("Error", "fail", "Layer display failed")]
    try:
        verdict = gen_verdict(l1['name'], tier, l1, acct, cyc, val, ret, moat, bank, holdings)
    except Exception:
        verdict = f"{l1.get('name', ticker)} — analysis completed with errors. Review debug tab."
    try:
        html = gen_scorecard(l1['name'], ticker, l1['sector'], l1['price'], tier, size, l1, acct, val, ret, moat, layers, verdict, holdings)
    except Exception:
        html = None
    return {'sd': sd, 'l1': l1, 'acct': acct, 'cyc': cyc, 'moat': moat, 'mom': mom, 'ret': ret, 'val': val,
            'tier': tier, 'size': size, 'layers': layers, 'verdict': verdict, 'html': html,
            'bank': bank, 'info': info, 'fin': fin, 'bs': bs, 'cf': cf, 'qfin': qfin,
            'holdings': holdings}

# ============================================================
# BATCH SCREENER (enhanced with sector cap)
# ============================================================
def analyse_with_funnel(ticker):
    sd = fetch(ticker)
    if not sd:
        return None, 'fetch_fail'
    if is_banking(sd['info']):
        return None, 'banking'
    try:
        l1 = run_layer1(sd, ticker)
    except Exception:
        return None, 'fetch_fail'
    if not l1['mcap_pass']:
        return None, 'mcap'
    if not l1['de_pass']:
        return None, 'debt'
    if not l1['roe_pass']:
        return None, 'roe'
    if not l1['roce_pass']:
        return None, 'roce'
    if not l1['growth_pass']:
        return None, 'growth'
    try:
        multi = build_multi(sd['fin'], sd['bs'], sd['cf'])
    except Exception:
        return None, 'data'
    if not multi:
        return None, 'data'
    try:
        acct = run_forensic(multi)
        cyc = run_cyclical(multi)
        moat = run_moat(multi)
        val = run_valuation(l1['pe'], l1['pat_cagr'], l1.get('sales_cagr'), l1['price'], ext_mcap(sd, l1['price']), sd)
        ret = ext_1y_ret(sd)
        holdings = run_holdings_analysis(sd.get('holdings', dict(DEFAULT_HOLDINGS)))
        tier, size = get_tier(
            acct['score'], acct['num_flags'], cyc.get('peak', False),
            val.get('peg'), ret, moat.get('consistency', ''),
            holdings, l1.get('int_cov'))
    except Exception:
        return None, 'data'
    ccc_obj = acct.get('ccc')
    return {'ticker': ticker.replace('.NS', ''), 'name': l1['name'], 'sector': l1['sector'],
            'price': l1['price'], 'pe': l1['pe'], 'peg': val.get('peg'), 'roe': l1['roe'], 'roce': l1['roce'],
            'score': acct['score'], 'nf': acct['num_flags'], 'cum_cfo': acct.get('cum_cfo_pat'),
            'peak': cyc.get('peak', False), 'ret': ret, 'tier': tier, 'size': size,
            'sc': l1.get('sales_cagr'), 'pc': l1.get('pat_cagr'), 'moat': moat.get('pct', 0),
            'fcf_yield': val.get('fcf_yield'), 'ccc': ccc_obj.get('ccc') if ccc_obj else None,
            'int_cov': l1.get('int_cov'), 'net_debt_cr': l1.get('net_debt_cr'),
            'insider_pct': holdings.get('insider_pct'),
            'accrual_ratio': acct.get('accrual_ratio')}, 'pass'

def apply_sector_cap(results, max_sector_pct=30):
    """Cap any sector at max_sector_pct of total picks. Demote excess to WATCH."""
    if not results:
        return results
    df = pd.DataFrame(results)
    tier_alloc = {'FULL': 13.5, 'STANDARD': 9, 'HALF': 5, 'WATCH': 0}
    df['alloc'] = df['tier'].map(tier_alloc).fillna(0)
    total_alloc = df['alloc'].sum()
    if total_alloc == 0:
        return results

    sector_alloc = df.groupby('sector')['alloc'].sum()
    for sector in sector_alloc.index:
        sector_pct = sector_alloc[sector] / total_alloc * 100
        if sector_pct > max_sector_pct:
            # Demote weakest stocks in this sector
            sector_mask = df['sector'] == sector
            sector_df = df[sector_mask].sort_values('rank', ascending=True)  # weakest first
            while True:
                current_alloc = df.loc[df['sector'] == sector, 'alloc'].sum()
                new_total = df['alloc'].sum()
                if new_total == 0 or current_alloc / new_total * 100 <= max_sector_pct:
                    break
                # Demote the weakest in this sector that isn't already WATCH
                demote_candidates = df[(df['sector'] == sector) & (df['tier'] != 'WATCH')]
                if demote_candidates.empty:
                    break
                worst_idx = demote_candidates['rank'].idxmin()
                if df.loc[worst_idx, 'tier'] == 'FULL':
                    df.loc[worst_idx, 'tier'] = 'STANDARD'
                    df.loc[worst_idx, 'size'] = '8-10%'
                    df.loc[worst_idx, 'alloc'] = 9
                elif df.loc[worst_idx, 'tier'] == 'STANDARD':
                    df.loc[worst_idx, 'tier'] = 'HALF'
                    df.loc[worst_idx, 'size'] = '4-6%'
                    df.loc[worst_idx, 'alloc'] = 5
                else:
                    df.loc[worst_idx, 'tier'] = 'WATCH'
                    df.loc[worst_idx, 'size'] = '0%'
                    df.loc[worst_idx, 'alloc'] = 0
    return df.drop(columns=['alloc']).to_dict('records')

# ============================================================
# NIFTY TICKER LIST (FIXED — comprehensive fallback)
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_nifty_tickers(max_stocks):
    from urllib.request import Request, urlopen
    import io as _io
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    # Try Nifty 500 first, then 200, then 100
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    ]
    for url in urls:
        try:
            req = Request(url, headers=headers)
            resp = urlopen(req, timeout=15)
            df = pd.read_csv(_io.BytesIO(resp.read()))
            if 'Symbol' in df.columns:
                tickers = [s.strip() + ".NS" for s in df['Symbol'].tolist()]
                if len(tickers) >= max_stocks * 0.8:  # BUG FIX: only accept if we got enough
                    return tickers[:max_stocks], url.split('/')[-1].replace('.csv', '')
        except Exception:
            continue

    # Comprehensive Nifty 500 hardcoded fallback (Sep 2026)
    fallback = [
        "360ONE","3MINDIA","ABB","ACC","ACMESOLAR","AIAENG","APLAPOLLO","AUBANK","AWL",
        "AADHARHFC","AARTIIND","AAVAS","ABBOTINDIA","ACE","ADANIENSOL","ADANIENT",
        "ADANIGREEN","ADANIPORTS","ADANIPOWER","ATGL","ABCAPITAL","ABFRL","ABSLAMC",
        "AEGISLOG","AFCONS","AFFLE","AJANTPHARM","ALKEM","AMBER","AMBUJACEM",
        "ANANDRATHI","ANANTRAJ","ANGELONE","APARINDS","APOLLOHOSP","APOLLOTYRE","APTUS",
        "ASAHIINDIA","ASHOKLEY","ASIANPAINT","ASTERDM","ASTRAL","ATUL","AUROPHARMA",
        "DMART","AXISBANK","BEML","BLS","BSE","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
        "BAJAJHLDNG","BAJAJHFL","BALKRISIND","BALRAMCHIN","BANDHANBNK","BANKBARODA",
        "BANKINDIA","MAHABANK","BATAINDIA","BAYERCROP","BERGEPAINT","BDL","BEL",
        "BHARATFORG","BHEL","BPCL","BHARTIARTL","BHARTIHEXA","BIKAJI","BIOCON","BSOFT",
        "BLUEDART","BLUEJET","BLUESTARCO","BOSCHLTD","BRIGADE","BRITANNIA","MAPMYINDIA",
        "CCL","CESC","CGPOWER","CIEINDIA","CRISIL","CANFINHOME","CANBK","CAPLIPOINT",
        "CGCL","CARBORUNIV","CASTROLIND","CEATLTD","CENTRALBK","CDSL","CHALET",
        "CHAMBLFERT","CHENNPETRO","CHOLAHLDNG","CHOLAFIN","CIPLA","CUB","CLEAN",
        "COALINDIA","COCHINSHIP","COFORGE","COLPAL","CAMS","CONCORDBIO","CONCOR",
        "COROMANDEL","CRAFTSMAN","CREDITACC","CROMPTON","CUMMINSIND","CYIENT","DCMSHRIRAM",
        "DLF","DOMS","DABUR","DALBHARAT","DATAPATTNS","DEEPAKFERT","DEEPAKNTR","DELHIVERY",
        "DEVYANI","DIVISLAB","DIXON","LALPATHLAB","DRREDDY","EIDPARRY","EIHOTEL",
        "EICHERMOT","ELECON","ELGIEQUIP","EMAMILTD","EMCURE","ENDURANCE","ENGINERSIN",
        "ERIS","ESCORTS","ETERNAL","EXIDEIND","NYKAA","FEDERALBNK","FACT","FINCABLES",
        "FSL","FIVESTAR","FORCEMOT","FORTIS","GAIL","GMRAIRPORT","GABRIEL","GRSE",
        "GICRE","GILLETTE","GLAND","GLAXO","GLENMARK","MEDANTA","GODIGIT","GPIL",
        "GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP","GRANULES","GRAPHITE","GRASIM",
        "GRAVITA","GESHIP","FLUOROCHEM","GMDCLTD","HEG","HCLTECH","HDFCAMC","HDFCBANK",
        "HDFCLIFE","HFCL","HAVELLS","HEROMOTOCO","HEXT","HINDALCO","HAL","HINDCOPPER",
        "HINDPETRO","HINDUNILVR","HINDZINC","POWERINDIA","HOMEFIRST","HONASA","HONAUT",
        "HUDCO","HYUNDAI","ICICIBANK","ICICIGI","ICICIPRULI","IDBI","IDFCFIRSTB",
        "IFCI","IIFL","IRB","IRCON","ITC","ITI","INDIAMART","INDIANB","IEX","INDHOTEL",
        "IOC","IOB","IRCTC","IRFC","IREDA","IGL","INDUSTOWER","INDUSINDBK","NAUKRI",
        "INFY","INOXWIND","INTELLECT","INDIGO","IPCALAB","JKCEMENT","JBMA","JKTYRE",
        "JMFINANCIL","JSWENERGY","JSWINFRA","JSWSTEEL","JINDALSAW","JSL","JINDALSTEL",
        "JIOFIN","JUBLFOOD","JUBLINGREA","JUBLPHARMA","JWL","JYOTICNC","KPRMILL","KEI",
        "KPITTECH","KAJARIACER","KPIL","KALYANKJIL","KARURVYSYA","KAYNES","KEC",
        "KFINTECH","KIRLOSENG","KOTAKBANK","KIMS","LTF","LTTS","LICHSGFIN","LTFOODS",
        "LT","LATENTVIEW","LAURUSLABS","LEMONTREE","LICI","LINDEINDIA","LLOYDSME",
        "LODHA","LUPIN","MMTC","MRF","MGL","MANAPPURAM","MRPL","MANKIND","MARICO",
        "MARUTI","MFSL","MAXHEALTH","MAZDOCK","MINDACORP","MSUMI","MOTILALOFS","MPHASIS",
        "MCX","MUTHOOTFIN","NATCOPHARM","NBCC","NCC","NHPC","NLCINDIA","NMDC","NTPC",
        "NH","NATIONALUM","NAVA","NAVINFLUOR","NESTLEIND","NETWEB","NEULANDLAB","NEWGEN",
        "NUVAMA","NUVOCO","OBEROIRLTY","ONGC","OIL","OLAELEC","OLECTRA","PAYTM",
        "OFSS","POLICYBZR","PCBL","PGEL","PIIND","PNBHOUSING","PVRINOX","PAGEIND",
        "PATANJALI","PERSISTENT","PETRONET","PFIZER","PHOENIXLTD","PIDILITIND","POLYCAB",
        "POONAWALLA","PFC","POWERGRID","PREMIERENE","PRESTIGE","PNB","RRKABEL","RBLBANK",
        "RECLTD","RHIM","RITES","RADICO","RVNL","RAILTEL","RAINBOW","RKFORGE",
        "REDINGTON","RELIANCE","RPOWER","SBFC","SBICARD","SBILIFE","SJVN","SRF",
        "SAGILITY","MOTHERSON","SAPPHIRE","SARDAEN","SAREGAMA","SCHAEFFLER","SCHNEIDER",
        "SCI","SHREECEM","SHRIRAMFIN","SHYAMMETL","SIEMENS","SOBHA","SOLARINDS",
        "SONACOMS","SONATSOFTW","STARHEALTH","SBIN","SAIL","SUMICHEM","SUNPHARMA",
        "SUNTV","SUNDARMFIN","SUPREMEIND","SUZLON","SYNGENE","SYRMA","TBOTEK",
        "TVSMOTOR","TATACHEM","TATACOMM","TCS","TATACONSUM","TATAELXSI","TATAPOWER",
        "TATASTEEL","TATATECH","TECHM","TECHNOE","TEGA","TEJASNET","RAMCOCEM","THERMAX",
        "TIMKEN","TITAGARH","TITAN","TORNTPHARM","TORNTPOWER","TRENT","TRIDENT",
        "TRITURBINE","TIINDIA","UCOBANK","UNOMINDA","UPL","UTIAMC","ULTRACEMCO",
        "UNIONBANK","UBL","UNITDSPR","USHAMART","VTL","VBL","VEDL","VIJAYA","VMM",
        "IDEA","VOLTAS","WAAREEENER","WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO",
        "WOCKPHARMA","YESBANK","ZFCVINDIA","ZEEL","ZENTEC","ZENSARTECH","ZYDUSLIFE",
        "ZYDUSWELL","ECLERX","AARTIPHARM","AARTI","ABCAP","ABSLAMC","ADANITOTAL",
        "ALKYLAMINE","ALLCARGO","ALOKINDS","ANURAS","ASTRAZEN","AVANTIFEED","BANARISUG",
        "BASF","BBTC","BIRLACORPN","CANFINHOME","CARERATING","CENTURYPLY","CENTURYTEX",
        "CHEMPLASTS","CHOICEIN","CIGNITITEC","COCHINSHIP","DEEPAKFERT","DELTACORP",
        "DHANI","EIHOTEL","ELECTCAST","ESTER","FDC","FINPIPE","GARFIBRES",
        "GENUSPOWER","GHCL","GNFC","GOCOLORS","GOODYEAR","GRINDWELL","GSFC",
        "GSPL","GUJALKALI","GULFOILLUB","HAPPSTMNDS","HATSUN","HGS","HIKAL",
        "HINDWAREAP","IBULHSGFIN","ICRA","IDFC","IGPL","IIFLWAM","INDIACEM",
        "JAMNAAUTO","JCHAC","JUBLINDS","JUSTDIAL","KALYANKJIL","KANSAINER",
        "KSB","LAXMIMACH","LUXIND","MAHINDCIE","MANINFRA","MASTEK","MAXHEALTH",
        "METROPOLIS","MIDHANI","MHRIL","MINDTREE","MMFL","MOREPENLAB","MSTCLTD",
        "MUTHOOTCAP","NIACL","NOCIL","OLECTRA","PGHH","PRISMJOYCE","PRINCEPIPE",
        "QUESS","RATNAMANI","RELAXO","RENUKA","ROSSARI","ROUTE","SAFARI",
        "SANOFI","SHARDACROP","SHILPAMED","SIS","SKFINDIA","SOLARA","SPARC",
        "STLTECH","SUDARSCHEM","SUNDRMFAST","SWANENERGY","TASTYBITE","TEAMLEASE",
        "THYROCARE","TIMETECH","TVSSRICHAK","VAIBHAVGBL","VGUARD","VINATIORGA",
        "VIPIND","VSTIND","WABAG","WESTLIFE","ZENSARTECH",
    ]
    # Deduplicate
    seen = set()
    deduped = []
    for s in fallback:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return [s + ".NS" for s in deduped][:max_stocks], 'hardcoded_nifty500'

# ============================================================
# APP UI
# ============================================================
st.markdown('<p class="main-title">High Compounder Framework v2</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Systematic Indian Equity Screener — Leading + Lagging Indicators</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Single Stock", "Compare", "Auto Top 10", "How It Works"], label_visibility="collapsed")
st.sidebar.markdown("### Portfolio")
portfolio_data = pd.DataFrame({
    "Stock": ["LUPIN","DIXON","SIEMENS","BSE","MCX","ICICIPRULI","EICHER","POWERINDIA","PERSISTENT",
              "KPIT","POLYCAB","HDFCAMC","COFORGE","VBL","CUMMINS"],
    "Tier": ["FULL","FULL","FULL","STD","STD","STD","STD","STD","STD","HALF","HALF","HALF","HALF","HALF","HALF"],
    "Score": [100,100,100,85,85,85,85,85,85,100,55,70,70,70,70],
    "PEG": [0.14,0.54,0.91,0.38,0.51,1.46,1.60,1.07,1.79,1.48,1.67,1.42,1.32,1.52,2.46],
})
st.sidebar.dataframe(portfolio_data, hide_index=True, use_container_width=True)
if st.sidebar.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Vinayak Nagral · Sep 2026")

# ============================
# PAGE: SINGLE STOCK
# ============================
if page == "Single Stock":
    c1, c2 = st.columns([4, 1])
    with c1:
        ti = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="LUPIN, INFY, TCS, DIXON").strip().upper()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Analyse", type="primary", use_container_width=True)
    qc = st.columns(7)
    for i, q in enumerate(["LUPIN", "BSE", "DIXON", "INFY", "HDFCAMC", "MAZDOCK", "HCLTECH"]):
        with qc[i]:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                ti = q; go = True
    if not ti.endswith(".NS"):
        ti += ".NS"
    if go:
        with st.spinner(f"Analysing {ti.replace('.NS', '')}..."):
            res = full_analysis(ti)
        if not res:
            st.error(f"Could not fetch data for **{ti.replace('.NS', '')}**. Check the ticker — NSE symbol like LUPIN, INFY, TCS.")
            st.stop()
        l1, acct, val, mom, ret, moat, cyc = res['l1'], res['acct'], res['val'], res['mom'], res['ret'], res['moat'], res['cyc']
        tier, size, layers, verdict, bank = res['tier'], res['size'], res['layers'], res['verdict'], res['bank']
        holdings = res.get('holdings', dict(DEFAULT_HOLDINGS))
        name, sector, price = l1['name'], l1['sector'], l1['price']
        st.markdown(f'<p class="stock-name">{name}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="stock-meta">{ti} | {sector} | {"Rs " + f"{price:,.2f}" if is_valid(price) else "-"} | CMP as of {datetime.now().strftime("%d %b %Y, %I:%M %p")}</p>', unsafe_allow_html=True)
        if bank:
            st.info("Banking/Financial — framework for non-financials. Scores for reference.")
        st.markdown("---")
        if res.get('html'):
            fn = f"{name.replace(' ', '_')}_Scorecard_{datetime.now().strftime('%d%b%Y')}.html"
            st.download_button("Download scorecard", data=res['html'], file_name=fn, mime="text/html", help="Open in browser, Ctrl+P, Save as PDF")

        cols = st.columns(8)
        score_display = f"{acct.get('score', '-')}/100" if acct.get('score') is not None else "-"
        int_cov = l1.get('int_cov')
        card_data = [
            ("Score", score_display), ("PE", fmt(l1['pe'], '.1f')), ("PEG", fmt(val.get('peg'), '.2f')),
            ("ROE", fmt(l1['roe'], '.1f', '%')), ("Int Cov", fmt(int_cov, '.1f', 'x') if is_valid(int_cov) else "N/A"),
            ("FCF Yield", fmt(val.get('fcf_yield'), '.1f', '%')),
            ("1Y Return", fmt(ret, '+.1f', '%')), ("Tier", "")]
        for i, (lb, vl) in enumerate(card_data):
            with cols[i]:
                if lb == "Tier":
                    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Tier</div><span class="tier-badge tier-{tier.lower()}">{tier} {size}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-lbl">{lb}</div><div class="metric-val">{vl}</div></div>', unsafe_allow_html=True)
        st.markdown("---"); st.subheader("Layer-by-layer breakdown")
        for ln, st2, dt in layers:
            ic = "✅" if st2 == "pass" else "❌" if st2 == "fail" else "⚠️"
            lead_tag = ' <span class="leading-tag">LEADING</span>' if '★' in ln else ''
            ln_clean = ln.replace(' ★', '')
            st.markdown(f'<div class="layer-row layer-{st2}">{ic} <strong>{ln_clean}:</strong>{lead_tag} {dt}</div>', unsafe_allow_html=True)
        st.markdown("---"); st.subheader("Investment verdict")
        st.markdown(f'<div class="verdict-box">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("---")
        t1, t2, t3, t4 = st.tabs(["Forensic", "Cyclical & valuation", "Momentum & Holdings", "Debug"])
        with t1:
            if acct.get('score') is not None:
                sc2 = acct['score']
                css = "score-green" if sc2 >= 85 else "score-yellow" if sc2 >= 50 else "score-red"
                st.markdown(f'<div class="score-card {css}"><h2>{sc2}/100</h2></div>', unsafe_allow_html=True)
                ca, cb, cc_col = st.columns(3)
                with ca:
                    st.markdown("**Cumulative CFO/PAT**")
                    v2 = acct.get('cum_cfo_pat')
                    if is_valid(v2):
                        st.markdown(f"{'✅' if v2 >= 0.7 else '⚠️' if v2 >= 0.5 else '❌'} **{v2}x**")
                with cb:
                    st.markdown("**Cumulative CFO/EBITDA**")
                    v3 = acct.get('cum_cfo_ebitda')
                    if is_valid(v3):
                        st.markdown(f"{'✅' if v3 >= 0.7 else '⚠️' if v3 >= 0.5 else '❌'} **{v3}x**")
                with cc_col:
                    st.markdown("**Accrual Ratio**")
                    ar = acct.get('accrual_ratio')
                    if is_valid(ar):
                        st.markdown(f"{'✅' if ar < 10 else '⚠️' if ar < 20 else '❌'} **{ar}%**")
                    else:
                        st.markdown("N/A")
                mby = acct.get('margins_by_year', [])
                if mby:
                    trend_icon = '📈' if acct.get('margin_trend') == 'expanding' else '📉' if acct.get('margin_trend') == 'contracting' else '➡️'
                    st.markdown(f"**EBITDA Margins:** {' -> '.join(f'{m[0]}: {m[1]}%' for m in mby)} {trend_icon}")
                # Interest coverage trend
                ict = acct.get('int_cov_trend', [])
                if ict:
                    st.markdown(f"**Interest Coverage Trend:** {' -> '.join(f'{y}: {v}x' for y, v in ict)}")
                ccc = acct.get('ccc')
                if ccc:
                    trend_str = acct.get('ccc_trend', '') or ''
                    trend_icon = '📈' if trend_str == 'worsening' else '📉' if trend_str == 'improving' else '➡️'
                    st.markdown(f"**Cash Conversion Cycle:** {ccc['ccc']}d (Recv {ccc['days_recv']}d + Inv {ccc['days_inv']}d - Pay {ccc['days_pay']}d) {trend_str} {trend_icon}")
                if acct.get('flags'):
                    for f in acct['flags']:
                        st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="clean-item">✅ All checks passed — 0 flags</div>', unsafe_allow_html=True)
        with t2:
            cc, cd = st.columns(2)
            with cc:
                if cyc.get('roe_by_year'):
                    st.dataframe(pd.DataFrame(list(cyc['roe_by_year'].items()), columns=['Year', 'ROE%']), hide_index=True, use_container_width=True)
                # NEW: Show quarterly ROE if available
                q_roe = moat.get('quarterly_roe', [])
                if q_roe:
                    st.markdown("**Quarterly rolling ROE (extended moat)**")
                    st.dataframe(pd.DataFrame(q_roe, columns=['Period', 'ROE%']), hide_index=True, use_container_width=True)
                st.markdown(f"**Moat:** {moat.get('years_above_15', 0)}/{moat.get('total_years', 0)} yrs ROE>15% — **{moat.get('consistency', '-')}**")
            with cd:
                peg_note = " (using revenue growth)" if val.get('peg_turnaround') else ""
                st.markdown(f"**PE:** {fmt(l1['pe'], '.1f')} | **PEG:** {fmt(val.get('peg'), '.2f')}{peg_note} | **FCF Yield:** {fmt(val.get('fcf_yield'), '.1f', '%')}")
                st.markdown(f"**Sales CAGR ({l1.get('sales_cagr_label','')}):** {fmt(l1.get('sales_cagr'), '.1f')}% | **PAT CAGR ({l1.get('pat_cagr_label','')}):** {fmt(l1.get('pat_cagr'), '.1f')}%")
                # NEW: Net debt display
                nd = l1.get('net_debt_cr')
                if is_valid(nd):
                    nd_label = f"Rs {abs(nd):,.0f} Cr" + (" net cash" if nd < 0 else " net debt")
                    st.markdown(f"**Net Debt:** {nd_label}")
        with t3:
            # Momentum
            if mom.get('available'):
                mc1, mc2 = st.columns(2)
                mc1.metric("Latest QoQ", fmt(mom.get('latest_qoq'), '+.1f', '%'))
                mc2.metric("Prior QoQ", fmt(mom.get('prior_qoq'), '+.1f', '%'))
                if mom.get('eps'):
                    st.dataframe(pd.DataFrame({'Quarter': mom['quarters'], 'EPS': [round(e, 2) for e in mom['eps']]}), hide_index=True, use_container_width=True)
            else:
                st.info("Insufficient quarterly data for momentum analysis.")
            st.markdown(f"**1Y Return:** {fmt(ret, '+.1f', '%')}")
            # NEW: Holdings section
            st.markdown("---")
            st.markdown("### Promoter & Institutional Holdings")
            h1, h2, h3 = st.columns(3)
            with h1:
                ins = holdings.get('insider_pct')
                st.metric("Insider/Promoter %", f"{ins}%" if is_valid(ins) else "N/A")
            with h2:
                inst = holdings.get('inst_pct')
                st.metric("Institutional %", f"{inst}%" if is_valid(inst) else "N/A")
            with h3:
                inst_c = holdings.get('inst_count')
                st.metric("# Institutions", f"{inst_c:,}" if is_valid(inst_c) else "N/A")
            if holdings.get('flags'):
                for hf in holdings['flags']:
                    st.markdown(f'<div class="flag-item">⚠ {hf}</div>', unsafe_allow_html=True)
            elif is_valid(ins):
                st.markdown('<div class="clean-item">✅ Holdings profile acceptable</div>', unsafe_allow_html=True)
        with t4:
            dbg = res['sd'].get('debug', {})
            dy = dbg.get('data_years', {})
            st.markdown("**Data availability**")
            for k, v in dy.items():
                st.markdown(f'<div class="debug-row"><span class="debug-label">{k}</span> <span class="debug-val">{v}</span> {"✅" if v > 0 else "❌"}</div>', unsafe_allow_html=True)
            st.markdown("**Fallbacks triggered**")
            fbs = dbg.get('fallbacks', [])
            if fbs:
                for fb in fbs:
                    st.markdown(f'<div class="debug-row"><span class="debug-fb">↳ {fb}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="debug-row"><span class="debug-ok">None — all primary sources used</span></div>', unsafe_allow_html=True)
            warns = dbg.get('warnings', [])
            if warns:
                st.markdown("**Warnings**")
                for w in warns:
                    st.markdown(f'<div class="debug-row"><span class="debug-bad">⚠ {w}</span></div>', unsafe_allow_html=True)
            st.markdown("**Decision chain**")
            chain = [
                ('Price', fmt(l1['price'], ',.2f'), 'primary'),
                ('PE', fmt(l1['pe'], '.1f'), l1.get('pe_source', '?')),
                ('ROE', fmt(l1['roe'], '.1f', '%'), l1.get('roe_source', '?')),
                ('ROCE', fmt(l1['roce'], '.1f', '%'), 'EBIT/(TA-CL)'),
                ('D/E', fmt(l1['de'], '.2f'), 'info/manual'),
                ('Int Coverage', fmt(l1.get('int_cov'), '.1f', 'x'), l1.get('int_cov_source', '?')),
                ('Net Debt', fmt(l1.get('net_debt_cr'), ',.0f', ' Cr'), 'Debt-Cash'),
                ('Sales CAGR', fmt(l1.get('sales_cagr'), '.1f', '%'), l1.get('sales_cagr_debug', '?')),
                ('PAT CAGR', fmt(l1.get('pat_cagr'), '.1f', '%'), l1.get('pat_cagr_debug', '?')),
                ('Growth floor', f"{l1.get('growth_floor', 15)}%", 'relaxed' if l1.get('growth_relaxed') else 'standard'),
                ('Forensic score', str(acct.get('score', '?')), f"{acct.get('num_flags', 0)} flags"),
                ('Accrual ratio', fmt(acct.get('accrual_ratio'), '.1f', '%'), 'NI-CFO / AvgAssets'),
                ('Moat', moat.get('consistency', '?'), f"{moat.get('pct', 0)}% ({moat.get('extended_total', moat.get('total_years', 0))} pts)"),
                ('Insider %', fmt(holdings.get('insider_pct'), '.1f', '%'), 'major_holders'),
                ('PEG', fmt(val.get('peg'), '.2f'), f"growth={val.get('peg_growth', '?')}{'(turnaround)' if val.get('peg_turnaround') else ''}"),
                ('1Y Return', fmt(ret, '+.1f', '%'), 'momentum risk' if is_valid(ret) and ret < -30 else 'ok' if is_valid(ret) else 'no data'),
                ('TIER', tier, size),
            ]
            for label, value, source in chain:
                st.markdown(f'<div class="debug-row"><span class="debug-label">{label}</span> <span class="debug-val">{value}</span> <span class="debug-fb">← {source}</span></div>', unsafe_allow_html=True)
            st.markdown("**Raw info fields**")
            for k in ['currentPrice','trailingPE','forwardPE','trailingEps','marketCap','sharesOutstanding','shortName','sector','industry','debtToEquity','returnOnEquity','revenueGrowth','earningsGrowth']:
                v = res['info'].get(k, '—')
                ok = is_valid(v) or (isinstance(v, str) and v.strip() and v != '—')
                st.markdown(f'<div class="debug-row"><span class="debug-label">{k}</span> <span class="debug-val">{str(v)[:50]}</span> {"✅" if ok else "❌"}</div>', unsafe_allow_html=True)
        st.caption("Research only, not investment advice | Yahoo Finance")

# ============================
# PAGE: COMPARE
# ============================
elif page == "Compare":
    st.subheader("Compare stocks")
    c1, c2, c3 = st.columns(3)
    with c1:
        t1_in = st.text_input("Stock 1", value="LUPIN").strip().upper()
    with c2:
        t2_in = st.text_input("Stock 2", value="INFY").strip().upper()
    with c3:
        t3_in = st.text_input("Stock 3 (optional)", value="").strip().upper()
    if st.button("Compare", type="primary", use_container_width=True):
        tickers = [t + ".NS" if not t.endswith(".NS") else t for t in [t1_in, t2_in, t3_in] if t]
        results = {}
        for t in tickers:
            with st.spinner(f"Analysing {t.replace('.NS', '')}..."):
                r = full_analysis(t)
            if r:
                results[t] = r
            else:
                st.warning(f"Could not fetch **{t.replace('.NS', '')}** — check the ticker.")
        if len(results) < 2:
            st.error("Need at least 2 valid stocks."); st.stop()
        best_peg = min(results.items(), key=lambda x: x[1]['val'].get('peg') or 999)
        strongest_moat = max(results.items(), key=lambda x: x[1]['moat'].get('pct', 0))
        st.markdown(f"**{best_peg[1]['l1']['name']}** offers the widest PEG discount at {fmt(best_peg[1]['val'].get('peg'), '.2f')} with {best_peg[1]['acct'].get('score', '-')}/100 forensic quality. **{strongest_moat[1]['l1']['name']}** has the strongest moat at {strongest_moat[1]['moat'].get('pct', 0)}%.")
        st.markdown("---")
        metric_defs = [
            ('Tier', lambda r: r['tier'], None),
            ('Score', lambda r: f"{r['acct'].get('score', '-')}/100", lambda r: r['acct'].get('score') or 0),
            ('PE', lambda r: fmt(r['l1']['pe'], '.1f'), lambda r: -(r['l1'].get('pe') or 999)),
            ('PEG', lambda r: fmt(r['val'].get('peg'), '.2f'), lambda r: -(r['val'].get('peg') or 999)),
            ('ROE', lambda r: fmt(r['l1']['roe'], '.1f', '%'), lambda r: r['l1'].get('roe') or 0),
            ('ROCE', lambda r: fmt(r['l1']['roce'], '.1f', '%'), lambda r: r['l1'].get('roce') or 0),
            ('D/E', lambda r: fmt(r['l1']['de'], '.2f'), lambda r: -(r['l1'].get('de') or 999)),
            ('Int Coverage', lambda r: fmt(r['l1'].get('int_cov'), '.1f', 'x'), lambda r: r['l1'].get('int_cov') or 0),
            ('Sales CAGR', lambda r: fmt(r['l1'].get('sales_cagr'), '.1f', '%'), lambda r: r['l1'].get('sales_cagr') or 0),
            ('PAT CAGR', lambda r: fmt(r['l1'].get('pat_cagr'), '.1f', '%'), lambda r: r['l1'].get('pat_cagr') or 0),
            ('CFO/PAT', lambda r: fmt(r['acct'].get('cum_cfo_pat'), '.2f', 'x'), lambda r: r['acct'].get('cum_cfo_pat') or 0),
            ('Accrual %', lambda r: fmt(r['acct'].get('accrual_ratio'), '.1f', '%'), lambda r: -(r['acct'].get('accrual_ratio') or 999)),
            ('FCF Yield', lambda r: fmt(r['val'].get('fcf_yield'), '.1f', '%'), lambda r: r['val'].get('fcf_yield') or 0),
            ('Moat', lambda r: f"{r['moat'].get('pct', 0)}% ({r['moat'].get('consistency', '')})", lambda r: r['moat'].get('pct', 0)),
            ('Insider %', lambda r: fmt(r.get('holdings', {}).get('insider_pct'), '.1f', '%'), lambda r: r.get('holdings', {}).get('insider_pct') or 0),
            ('1Y Return', lambda r: fmt(r['ret'], '+.1f', '%'), lambda r: r['ret'] or 0),
        ]
        names = [r['l1']['name'] for r in results.values()]
        header = "| Metric | " + " | ".join(names) + " |\n|---|" + "|".join(["---"] * len(names)) + "|\n"
        rows = ""
        for mname, display_fn, rank_fn in metric_defs:
            vals_display = [display_fn(r) for r in results.values()]
            if rank_fn:
                scores = [rank_fn(r) for r in results.values()]
                best_idx = scores.index(max(scores))
                vals_display = [f"**{v}**" if i == best_idx else v for i, v in enumerate(vals_display)]
            rows += f"| {mname} | " + " | ".join(vals_display) + " |\n"
        st.markdown(header + rows)
        st.markdown("---")
        for t, r in results.items():
            with st.expander(f"**{r['l1']['name']}** — {r['tier']} {r['size']}", expanded=True):
                st.markdown(f'<div class="verdict-box">{r["verdict"]}</div>', unsafe_allow_html=True)

# ============================
# PAGE: AUTO TOP 10
# ============================
elif page == "Auto Top 10":
    st.subheader("Automatic Top 10 Picker")
    c1, c2 = st.columns(2)
    with c1:
        mx = st.slider("Stocks to screen", 20, 500, 200, 10)
    with c2:
        tn = st.slider("Show top N", 5, 30, 10)
    sector_cap_on = st.checkbox("Apply 30% sector cap", value=True)
    if st.button("Run screen", type="primary", use_container_width=True):
        tickers, source = get_nifty_tickers(mx)
        st.caption(f"Source: {source} | {len(tickers)} tickers loaded (requested {mx})")
        results = []
        funnel = {'total': len(tickers), 'banking': 0, 'mcap': 0, 'debt': 0, 'roe': 0, 'roce': 0, 'growth': 0, 'data': 0, 'fetch_fail': 0, 'pass': 0}
        prog = st.progress(0)
        batch_size = 5
        for i, t in enumerate(tickers):
            prog.progress((i + 1) / len(tickers), f"{t.replace('.NS', '')} ({i + 1}/{len(tickers)})")
            r, reason = analyse_with_funnel(t)
            funnel[reason] = funnel.get(reason, 0) + 1
            if r:
                results.append(r)
            # BUG FIX: adaptive throttling — slower for larger batches
            if (i + 1) % batch_size == 0:
                if len(tickers) > 200:
                    time.sleep(1.0)
                elif len(tickers) > 100:
                    time.sleep(0.7)
                else:
                    time.sleep(0.5)
        prog.empty()
        st.markdown("### Screening funnel")
        total = funnel['total']; passed = funnel['pass']; failed_fetch = funnel.get('fetch_fail', 0)
        if failed_fetch > total * 0.2:
            st.warning(f"⚠ {failed_fetch} stocks failed to fetch — Yahoo Finance may be rate-limiting. Try fewer stocks or run again later.")
        running = total
        for label, key in [('Universe', None), ('Excl. banking', 'banking'), ('MCap > Rs 15,000 Cr', 'mcap'), ('D/E < 0.5', 'debt'), ('ROE > 15%', 'roe'), ('ROCE > 18%', 'roce'), ('Growth filter', 'growth')]:
            if key:
                running -= funnel.get(key, 0)
            pct = running / total * 100 if total > 0 else 0
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;"><div style="width:140px;font-size:12px;color:#6B7280;">{label}</div><div style="height:24px;background:{NAVY};border-radius:4px;display:flex;align-items:center;padding:0 8px;color:white;font-size:11px;font-weight:500;width:{max(pct, 3)}%;">{running}</div></div>', unsafe_allow_html=True)
        st.markdown(f"**{passed} of {total} passed** ({round(passed / total * 100, 1) if total > 0 else 0}% pass rate) | {failed_fetch} fetch failures")
        if not results:
            st.error("No stocks passed all filters."); st.stop()
        st.markdown("---")
        df = pd.DataFrame(results)
        peg_s = lambda x: (90 if x < 0.5 else 75 if x < 1 else 55 if x < 1.5 else 30 if x < 2 else 10) if is_valid(x) else 10
        mom_s = lambda x: (80 if x > 30 else 65 if x > 0 else 40 if x > -20 else 20) if is_valid(x) else 40
        cfo_s = lambda x: (100 if x > 1.2 else 85 if x > .9 else 65 if x > .7 else 40) if is_valid(x) else 20
        df['rank'] = (df['score'].fillna(0) * .20 + df['peg'].apply(peg_s) * .25 +
                      df['roe'].fillna(0) * .10 + df['ret'].apply(mom_s) * .15 +
                      df['cum_cfo'].apply(cfo_s) * .15 + df['moat'] * .15)
        df = df.sort_values('rank', ascending=False).head(tn)

        # Apply sector cap
        if sector_cap_on:
            capped = apply_sector_cap(df.to_dict('records'), 30)
            df = pd.DataFrame(capped)

        # CSV export
        export_cols = ['ticker','name','sector','price','pe','peg','roe','roce','score','tier','size',
                       'sc','pc','cum_cfo','moat','fcf_yield','ret','int_cov','insider_pct','accrual_ratio','net_debt_cr']
        export_df = df[[c for c in export_cols if c in df.columns]].copy()
        rename_map = {'ticker':'Ticker','name':'Name','sector':'Sector','price':'Price','pe':'PE','peg':'PEG',
                      'roe':'ROE%','roce':'ROCE%','score':'Score','tier':'Tier','size':'Size',
                      'sc':'Sales CAGR%','pc':'PAT CAGR%','cum_cfo':'CFO/PAT','moat':'Moat%',
                      'fcf_yield':'FCF Yield%','ret':'1Y Ret%','int_cov':'Int Coverage',
                      'insider_pct':'Insider%','accrual_ratio':'Accrual%','net_debt_cr':'Net Debt Cr'}
        export_df.columns = [rename_map.get(c, c) for c in export_df.columns]
        st.download_button("Download results (CSV)", export_df.to_csv(index=False), f"top_{tn}_{datetime.now().strftime('%d%b%Y')}.csv", "text/csv")

        for idx, row in df.iterrows():
            rk = list(df.index).index(idx) + 1
            ic = {"FULL": "🟢", "STANDARD": "🔵", "HALF": "🟡", "WATCH": "🔴"}.get(row['tier'], "⚪")
            with st.expander(f"#{rk} {row['ticker']} — {ic} {row['tier']} | Score {row['score']} | PEG {fmt(row['peg'], '.2f')}", expanded=rk <= 3):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("ROE", fmt(row.get('roe'), '.1f', '%'))
                c2.metric("PE", fmt(row.get('pe'), '.1f'))
                c3.metric("FCF Yield", fmt(row.get('fcf_yield'), '.1f', '%'))
                c4.metric("Moat", f"{row.get('moat', 0)}%")
                c5.metric("Int Cov", fmt(row.get('int_cov'), '.1f', 'x') if is_valid(row.get('int_cov')) else 'N/A')
                c6.metric("Insider", fmt(row.get('insider_pct'), '.0f', '%') if is_valid(row.get('insider_pct')) else 'N/A')
                nd = row.get('net_debt_cr')
                nd_str = f" | Net {'cash' if nd and nd < 0 else 'debt'} Rs {abs(nd):,.0f}Cr" if is_valid(nd) else ""
                ar_str = f" | Accrual {row.get('accrual_ratio', '-')}%" if is_valid(row.get('accrual_ratio')) else ""
                st.markdown(f"**{row['sector']}** | Sales {fmt(row.get('sc'), '.1f')}% | PAT {fmt(row.get('pc'), '.1f')}%{nd_str}{ar_str} | **{row['tier']} {row['size']}**")

# ============================
# PAGE: HOW IT WORKS
# ============================
elif page == "How It Works":
    st.subheader("Framework architecture — v2")
    st.markdown("""
**Layer 1 — Quantitative screen:** MCap > Rs 15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, Sales & PAT CAGR > 15%. Mature compounders (MCap > Rs 50,000 Cr + ROE > 25%) get a relaxed 10% growth floor.

**Layer 2 — Interest coverage** *(new):* EBIT / Interest Expense. Below 3x = reject. Completes the debt picture that D/E alone misses — a company can have low D/E but still struggle to service interest if operating profits are thin.

**Layer 3 — Forensic accounting:** Five checks — receivables trend, inventory bloat, cumulative CFO/PAT, cumulative CFO/EBITDA, and **accrual ratio** *(new)*. Accrual ratio catches earnings manipulation relative to asset size — the earliest warning before cash flow deterioration becomes visible.

**Layer 4 — Cash Conversion Cycle:** Days Receivable + Days Inventory - Days Payable. Payables stress detection flags companies stretching suppliers when cash conversion is weak.

**Layer 5 — Moat durability** *(enhanced):* Uses annual ROE history plus **quarterly rolling ROE reconstruction** for longer lookback. If quarterly data reveals periods below 12% ROE hidden within annual averages, established moat gets downgraded to emerging.

**Layer 6 — Promoter / Holdings** *(new — leading indicator):* Insider/promoter holding %, institutional ownership %. Flags very low promoter holding (<20%), excessive concentration (>75%), and negligible institutional oversight (<5%). Caps tier at STANDARD for high-risk ownership structures.

**Layer 7 — Dual valuation:** PEG with turnaround adjustment + FCF Yield + Adani filter (PE > 80).

**Layer 8 — Earnings momentum:** QoQ EPS changes for last 2 quarters.

**Layer 9 — Cyclical ROE:** Peak detection when latest ROE > 2x average and above 20%.

**Layer 10 — Position sizing v4:** Quality + moat + valuation + momentum + **holdings** + **interest coverage** -> FULL/STANDARD/HALF/WATCH. Interest coverage below 3x forces HALF. Weak ownership structure caps at STANDARD.

**Layer 11 — Sector cap** *(new):* In Auto Top 10, optional 30% sector concentration limit prevents correlated blowups.

---
**Leading indicators** (marked with ★): Promoter/institutional holdings, receivables trend, inventory bloat, CCC/payables stress, earnings momentum, accrual ratio.

**Lagging indicators:** ROE, ROCE, D/E, CAGR, cumulative cash conversion, moat consistency, PEG, FCF yield.

**Bug fixes in v2:** D/E unit detection (percentage vs decimal), EPS momentum edge cases for negative earnings, adaptive rate limiting for 500-stock screens, expanded Nifty 500 ticker list with fallback chain.
    """)
    st.caption("Vinayak Nagral | September 2026")
