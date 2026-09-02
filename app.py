import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="High Compounder Framework", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main-header { font-family:'Inter',sans-serif; font-size:2.2rem; font-weight:700; color:#1B2A4A; letter-spacing:-0.5px; }
    .sub-header { font-family:'Inter',sans-serif; font-size:1rem; color:#888; }
    .score-box { padding:1.2rem; border-radius:12px; text-align:center; margin:0.5rem 0; }
    .score-green { background:#E8F5E9; border-left:5px solid #4CAF50; }
    .score-yellow { background:#FFF8E1; border-left:5px solid #FF9800; }
    .score-red { background:#FFEBEE; border-left:5px solid #F44336; }
    .flag-item { background:#FFF3E0; padding:0.5rem 1rem; border-radius:8px; margin:0.3rem 0; border-left:4px solid #FF9800; font-size:0.9rem; }
    .clean-item { background:#E8F5E9; padding:0.5rem 1rem; border-radius:8px; margin:0.3rem 0; border-left:4px solid #4CAF50; font-size:0.9rem; }
    .tier-full { color:#2E7D32; font-weight:700; font-size:1.3rem; }
    .tier-standard { color:#1565C0; font-weight:700; font-size:1.3rem; }
    .tier-half { color:#F57F17; font-weight:700; font-size:1.3rem; }
    .tier-watch { color:#C62828; font-weight:700; font-size:1.3rem; }
    .banking-box { background:#E3F2FD; border-left:4px solid #1565C0; padding:1rem; border-radius:8px; margin:1rem 0; }
    .layer-pass { background:#E8F5E9; padding:0.4rem 0.8rem; border-radius:6px; margin:0.2rem 0; border-left:3px solid #4CAF50; }
    .layer-fail { background:#FFEBEE; padding:0.4rem 0.8rem; border-radius:6px; margin:0.2rem 0; border-left:3px solid #F44336; }
    .layer-warn { background:#FFF8E1; padding:0.4rem 0.8rem; border-radius:6px; margin:0.2rem 0; border-left:3px solid #FF9800; }
    .verdict-section { background:#F8F9FA; padding:1.5rem; border-radius:12px; margin:1rem 0; border:1px solid #E0E0E0; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

BANKING_KEYWORDS = ['bank', 'finance', 'insurance', 'nbfc', 'housing finance', 'credit', 'lending', 'microfinance']
NIFTY_200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"


# ============================================================
# HELPER: NaN-safe check
# ============================================================
def is_valid_number(val):
    if val is None:
        return False
    try:
        return np.isfinite(float(val))
    except (TypeError, ValueError):
        return False


# ============================================================
# ROBUST DATA EXTRACTION
# ============================================================
def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            val = df.loc[label].iloc[col]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return None


def safe_fmt(val, fmt=".1f", suffix="", prefix=""):
    if not is_valid_number(val):
        return "N/A"
    return f"{prefix}{val:{fmt}}{suffix}"


def get_revenue(fin, col=0):
    for field in ['Total Revenue', 'Operating Revenue', 'Revenue', 'Net Revenue']:
        val = safe_get(fin, field, col)
        if val and val > 0:
            return val
    return None


def get_net_income(fin, col=0):
    for field in ['Net Income', 'Net Income Common Stockholders', 'Net Income Continuous Operations']:
        val = safe_get(fin, field, col)
        if val is not None:
            return val
    return None


def get_ebitda(fin, col=0):
    for field in ['EBITDA', 'Normalized EBITDA']:
        val = safe_get(fin, field, col)
        if val and val > 0:
            return val
    return None


def get_equity(bs, col=0):
    for field in ['Stockholders Equity', 'Total Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
        val = safe_get(bs, field, col)
        if val and val > 0:
            return val
    return None


def get_cfo(cf, col=0):
    for field in ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities', 'Free Cash Flow']:
        val = safe_get(cf, field, col)
        if val is not None:
            return val
    return None


def get_shares_outstanding(bs, info):
    """Get shares from balance sheet first (always works), then info."""
    if bs is not None:
        for field in ['Ordinary Shares Number', 'Share Issued', 'Diluted Average Shares']:
            val = safe_get(bs, field, 0)
            if is_valid_number(val) and val > 0:
                return val
    # Fallback to info
    val = info.get('sharesOutstanding')
    if is_valid_number(val) and val > 0:
        return float(val)
    return None


def is_banking_stock(info, name=""):
    sector = (info.get('sector', '') or '').lower()
    industry = (info.get('industry', '') or '').lower()
    stock_name = (info.get('shortName', '') or name or '').lower()
    for kw in BANKING_KEYWORDS:
        if kw in sector or kw in industry or kw in stock_name:
            return True
    return False


# ============================================================
# DATA FETCHING — single cached call per ticker, uses t.history()
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_str):
    """Fetch all data from a single Ticker object to share the session."""
    try:
        t = yf.Ticker(ticker_str)

        # --- info (often fails for Indian stocks) ---
        try:
            info = dict(t.info) if t.info else {}
        except Exception:
            info = {}
        info = {k: v for k, v in info.items() if v is not None}

        # --- fast_info (lighter endpoint, often works when info doesn't) ---
        try:
            fi = t.fast_info
            if fi is not None:
                for attr, key in [('last_price', 'currentPrice'), ('previous_close', 'previousClose'),
                                  ('market_cap', 'marketCap'), ('shares', 'sharesOutstanding')]:
                    if not is_valid_number(info.get(key)):
                        try:
                            v = getattr(fi, attr, None)
                            if is_valid_number(v) and v > 0:
                                info[key] = float(v)
                        except Exception:
                            pass
        except Exception:
            pass

        # --- financial statements ---
        fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None

        # --- price history via t.history() (same session, most reliable) ---
        price_history = None
        try:
            hist = t.history(period="1y", auto_adjust=True)
            if hist is not None and not hist.empty and 'Close' in hist.columns:
                price_history = hist['Close'].dropna()
                # Inject current price into info if missing
                if not is_valid_number(info.get('currentPrice')) and len(price_history) > 0:
                    info['currentPrice'] = float(price_history.iloc[-1])
        except Exception:
            pass

        # If history failed, try shorter period
        if price_history is None or len(price_history) == 0:
            try:
                hist = t.history(period="5d", auto_adjust=True)
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    short_prices = hist['Close'].dropna()
                    if len(short_prices) > 0:
                        if not is_valid_number(info.get('currentPrice')):
                            info['currentPrice'] = float(short_prices.iloc[-1])
                        # Keep short_prices but mark that we don't have 1Y
                        price_history = short_prices
            except Exception:
                pass

        # --- shares from balance sheet (fallback) ---
        if not is_valid_number(info.get('sharesOutstanding')) and bs is not None:
            shares = get_shares_outstanding(bs, {})
            if shares:
                info['sharesOutstanding'] = shares

        return {
            "info": info, "financials": fin, "balance_sheet": bs,
            "cashflow": cf, "quarterly_financials": qfin,
            "price_history": price_history
        }
    except Exception:
        return None


# ============================================================
# DERIVED METRICS (use cached data, no extra API calls)
# ============================================================
def get_price_from_data(sd):
    """Extract current price from fetched data."""
    info = sd['info']
    for field in ['currentPrice', 'regularMarketPrice', 'regularMarketPreviousClose',
                  'previousClose', 'open', 'regularMarketOpen']:
        val = info.get(field)
        if is_valid_number(val) and val > 0:
            return round(float(val), 2)
    # From price history
    ph = sd.get('price_history')
    if ph is not None and len(ph) > 0:
        val = float(ph.iloc[-1])
        if is_valid_number(val) and val > 0:
            return round(val, 2)
    return None


def get_1y_return_from_data(sd):
    """Calculate 1Y return from cached price history."""
    ph = sd.get('price_history')
    if ph is None or len(ph) < 30:
        return None
    try:
        current = float(ph.iloc[-1])
        if not is_valid_number(current) or current <= 0:
            return None
        target = ph.index[-1] - pd.Timedelta(days=365)
        mask = ph.index <= target
        if mask.sum() > 0:
            past = float(ph.loc[mask].iloc[-1])
        else:
            past = float(ph.iloc[0])
        if is_valid_number(past) and past > 0:
            ret = round((current / past - 1) * 100, 1)
            return ret if is_valid_number(ret) else None
    except Exception:
        pass
    return None


def get_pe_from_data(sd, price):
    """Calculate PE from all available sources."""
    info = sd['info']
    # 1. info dict
    for field in ['trailingPE', 'forwardPE']:
        val = info.get(field)
        if is_valid_number(val) and val > 0:
            return round(float(val), 1)
    if not is_valid_number(price) or price <= 0:
        return None
    # 2. price / EPS from info
    for field in ['trailingEps', 'forwardEps']:
        eps = info.get(field)
        if is_valid_number(eps) and eps > 0:
            return round(price / eps, 1)
    # 3. TTM from quarterly financials (sum last 4 quarters net income / shares)
    qfin = sd.get('quarterly_financials')
    if qfin is not None and qfin.shape[1] >= 4:
        ttm_ni = 0
        valid_quarters = 0
        for i in range(min(4, qfin.shape[1])):
            ni = get_net_income(qfin, i)
            if is_valid_number(ni):
                ttm_ni += ni
                valid_quarters += 1
        if valid_quarters == 4 and ttm_ni > 0:
            shares = get_shares_outstanding(sd.get('balance_sheet'), info)
            if is_valid_number(shares) and shares > 0:
                ttm_eps = ttm_ni / shares
                if ttm_eps > 0:
                    return round(price / ttm_eps, 1)
    # 4. Annual net income / shares
    fin = sd.get('financials')
    if fin is not None:
        ni = get_net_income(fin, 0)
        if is_valid_number(ni) and ni > 0:
            shares = get_shares_outstanding(sd.get('balance_sheet'), info)
            if is_valid_number(shares) and shares > 0:
                eps_calc = ni / shares
                if eps_calc > 0:
                    return round(price / eps_calc, 1)
    return None


def get_mcap_from_data(sd, price):
    """Calculate market cap."""
    info = sd['info']
    mcap = info.get('marketCap')
    if is_valid_number(mcap) and mcap > 0:
        return float(mcap)
    if is_valid_number(price) and price > 0:
        shares = get_shares_outstanding(sd.get('balance_sheet'), info)
        if is_valid_number(shares) and shares > 0:
            return price * shares
    return None


def get_de_from_data(sd):
    info = sd['info']
    de = info.get('debtToEquity')
    if is_valid_number(de):
        return round(float(de) / 100, 2)
    bs = sd.get('balance_sheet')
    if bs is not None:
        total_debt = safe_get(bs, 'Total Debt', 0) or safe_get(bs, 'Long Term Debt', 0) or 0
        equity = get_equity(bs, 0)
        if equity and equity > 0:
            return round(total_debt / equity, 2)
    return 0.0


def get_name_from_data(sd, ticker_str):
    info = sd['info']
    for field in ['shortName', 'longName', 'displayName']:
        val = info.get(field)
        if val and str(val).strip() and str(val).strip().lower() != 'none':
            return str(val).strip()
    return ticker_str.replace('.NS', '').replace('.BO', '')


def get_sector_from_data(sd):
    info = sd['info']
    for field in ['sector', 'industry']:
        val = info.get(field)
        if val and str(val).strip() and str(val).strip().lower() != 'none':
            return str(val).strip()
    return "N/A"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nifty200_tickers():
    try:
        df = pd.read_csv(NIFTY_200_URL)
        return [s + ".NS" for s in df['Symbol'].tolist()]
    except Exception:
        return []


def get_multi_year_data(fin, bs, cf):
    if fin is None or bs is None or cf is None:
        return None
    years = min(fin.shape[1], bs.shape[1], cf.shape[1])
    if years < 2:
        return None
    records = []
    for i in range(years):
        year_label = str(fin.columns[i].year) if hasattr(fin.columns[i], 'year') else str(i)
        records.append({
            'year': year_label,
            'revenue': get_revenue(fin, i),
            'net_income': get_net_income(fin, i),
            'ebitda': get_ebitda(fin, i),
            'ebit': safe_get(fin, 'EBIT', i),
            'receivables': safe_get(bs, 'Accounts Receivable', i) or safe_get(bs, 'Net Receivables', i) or safe_get(bs, 'Receivables', i),
            'inventory': safe_get(bs, 'Inventory', i),
            'total_assets': safe_get(bs, 'Total Assets', i),
            'equity': get_equity(bs, i),
            'current_liabilities': safe_get(bs, 'Current Liabilities', i),
            'cfo': get_cfo(cf, i)
        })
    return records


# ============================================================
# LAYER 1: FUNDAMENTALS
# ============================================================
def run_layer1(sd, ticker_str):
    """Run Layer 1 using the single fetched data dict."""
    r = {}
    info = sd['info']
    fin = sd.get('financials')
    bs = sd.get('balance_sheet')

    price = get_price_from_data(sd)
    r['price'] = price

    mcap = get_mcap_from_data(sd, price)
    r['market_cap_cr'] = round(mcap / 1e7, 0) if is_valid_number(mcap) else None
    r['mcap_pass'] = is_valid_number(mcap) and mcap > 150_000_000_000

    r['pe'] = get_pe_from_data(sd, price)

    r['debt_to_equity'] = get_de_from_data(sd)
    r['de_pass'] = r['debt_to_equity'] < 0.5

    # ROE
    ni = get_net_income(fin, 0) if fin is not None else None
    eq = get_equity(bs, 0) if bs is not None else None
    if is_valid_number(ni) and is_valid_number(eq) and eq > 0:
        r['roe'] = round(ni / eq * 100, 1)
        r['roe_pass'] = r['roe'] > 15
    else:
        roe_info = info.get('returnOnEquity')
        if is_valid_number(roe_info):
            r['roe'] = round(float(roe_info) * 100, 1)
            r['roe_pass'] = r['roe'] > 15
        else:
            r['roe'] = None
            r['roe_pass'] = False

    # ROCE
    ebit_val = safe_get(fin, 'EBIT', 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if is_valid_number(ebit_val) and is_valid_number(ta) and is_valid_number(cl) and (ta - cl) > 0:
        r['roce'] = round(ebit_val / (ta - cl) * 100, 1)
        r['roce_pass'] = r['roce'] > 18
    else:
        r['roce'] = None
        r['roce_pass'] = False

    # Growth CAGRs
    r['sales_cagr'] = None
    r['pat_cagr'] = None
    if fin is not None and fin.shape[1] >= 2:
        yrs = fin.shape[1] - 1
        rev_l = get_revenue(fin, 0)
        rev_o = get_revenue(fin, fin.shape[1] - 1)
        ni_l = get_net_income(fin, 0)
        ni_o = get_net_income(fin, fin.shape[1] - 1)
        if is_valid_number(rev_l) and is_valid_number(rev_o) and rev_o > 0 and rev_l > 0 and yrs > 0:
            r['sales_cagr'] = round(((rev_l / rev_o) ** (1 / yrs) - 1) * 100, 1)
        if is_valid_number(ni_l) and is_valid_number(ni_o) and ni_o > 0 and ni_l > 0 and yrs > 0:
            r['pat_cagr'] = round(((ni_l / ni_o) ** (1 / yrs) - 1) * 100, 1)

    if r['sales_cagr'] is None:
        rg = info.get('revenueGrowth')
        if is_valid_number(rg):
            r['sales_cagr'] = round(float(rg) * 100, 1)
    if r['pat_cagr'] is None:
        eg = info.get('earningsGrowth')
        if is_valid_number(eg):
            r['pat_cagr'] = round(float(eg) * 100, 1)

    r['growth_pass'] = ((r.get('sales_cagr') or 0) > 15 and (r.get('pat_cagr') or 0) > 15)
    r['phase1_pass'] = all([r['mcap_pass'], r['de_pass'], r['roe_pass'], r['roce_pass'], r['growth_pass']])

    r['name'] = get_name_from_data(sd, ticker_str)
    r['sector'] = get_sector_from_data(sd)

    return r


# ============================================================
# LAYER 3: ACCOUNTING QUALITY
# ============================================================
def run_accounting_quality(data):
    flags = []
    score = 100
    details = {}
    sd = sorted(data, key=lambda x: x['year'])
    latest = sd[-1]
    prior = sd[-2] if len(sd) >= 2 else None

    recv_pcts = []
    for d in sd:
        if is_valid_number(d.get('revenue')) and is_valid_number(d.get('receivables')) and d['revenue'] > 0:
            recv_pcts.append(round(d['receivables'] / d['revenue'] * 100, 1))
        else:
            recv_pcts.append(None)
    details['recv_pcts'] = recv_pcts
    details['recv_years'] = [d['year'] for d in sd]

    if len(recv_pcts) >= 3:
        valid = [x for x in recv_pcts if x is not None]
        if len(valid) >= 3:
            rising = sum(1 for i in range(1, len(valid)) if valid[i] > valid[i - 1])
            if rising >= 2:
                flags.append(f"Receivables rising: {valid[-3]}% → {valid[-2]}% → {valid[-1]}% of revenue")
                score -= 15

    if prior:
        inv_l, inv_p = latest.get('inventory'), prior.get('inventory')
        rev_l, rev_p = latest.get('revenue'), prior.get('revenue')
        if all(is_valid_number(v) and v > 0 for v in [inv_l, inv_p, rev_l, rev_p]):
            inv_g = round((inv_l / inv_p - 1) * 100, 1)
            rev_g = round((rev_l / rev_p - 1) * 100, 1)
            details['inv_growth'] = inv_g
            details['rev_growth'] = rev_g
            if inv_g > rev_g + 10:
                flags.append(f"Inventory bloat: grew {inv_g}% vs Revenue {rev_g}%")
                score -= 15

    cfo_vals = [d['cfo'] for d in sd if is_valid_number(d.get('cfo'))]
    pat_vals = [d['net_income'] for d in sd if is_valid_number(d.get('net_income'))]
    total_cfo = sum(cfo_vals) if cfo_vals else 0
    total_pat = sum(pat_vals) if pat_vals else 0

    if total_pat > 0:
        details['cum_cfo_pat'] = round(total_cfo / total_pat, 2)
        if details['cum_cfo_pat'] < 0.5:
            flags.append(f"Critical: Cumulative CFO/PAT = {details['cum_cfo_pat']}x over {len(sd)} years")
            score -= 30
        elif details['cum_cfo_pat'] < 0.7:
            flags.append(f"Low cash conversion: Cumulative CFO/PAT = {details['cum_cfo_pat']}x over {len(sd)} years")
            score -= 20
    else:
        details['cum_cfo_pat'] = None

    if is_valid_number(latest.get('cfo')) and is_valid_number(latest.get('net_income')) and latest['net_income'] > 0:
        details['single_yr_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2)
    else:
        details['single_yr_cfo_pat'] = None

    ebitda_vals = [d['ebitda'] for d in sd if is_valid_number(d.get('ebitda')) and d['ebitda'] > 0]
    total_ebitda = sum(ebitda_vals) if ebitda_vals else 0

    if total_ebitda > 0:
        details['cum_cfo_ebitda'] = round(total_cfo / total_ebitda, 2)
        if details['cum_cfo_ebitda'] < 0.5:
            flags.append(f"Critical: Cumulative CFO/EBITDA = {details['cum_cfo_ebitda']}x over {len(sd)} years")
            score -= 20
        elif details['cum_cfo_ebitda'] < 0.7:
            flags.append(f"Weak: Cumulative CFO/EBITDA = {details['cum_cfo_ebitda']}x over {len(sd)} years")
            score -= 15
    else:
        details['cum_cfo_ebitda'] = None

    yr_ratios = []
    for d in sd:
        if is_valid_number(d.get('cfo')) and is_valid_number(d.get('ebitda')) and d['ebitda'] > 0:
            yr_ratios.append(round(d['cfo'] / d['ebitda'], 2))
    details['cfo_ebitda_trend'] = yr_ratios
    if len(yr_ratios) >= 2 and is_valid_number(details.get('cum_cfo_ebitda')):
        if yr_ratios[-1] < yr_ratios[-2] and yr_ratios[-1] < 0.5 and details['cum_cfo_ebitda'] < 0.7:
            flags.append(f"Deteriorating trend: CFO/EBITDA fell {yr_ratios[-2]} → {yr_ratios[-1]} AND cumulative weak")
            score -= 10

    if total_cfo < 0:
        flags.append(f"Negative cumulative CFO over {len(sd)} years")
        score -= 25

    score = max(score, 0)
    details['score'] = score
    details['flags'] = flags
    details['num_flags'] = len(flags)
    return details


# ============================================================
# LAYER 6: CYCLICAL
# ============================================================
def run_cyclical_check(data):
    sd = sorted(data, key=lambda x: x['year'])
    roe_vals, roe_by_year = [], {}
    for d in sd:
        if is_valid_number(d.get('net_income')) and is_valid_number(d.get('equity')) and d['equity'] > 0:
            roe = round(d['net_income'] / d['equity'] * 100, 1)
            roe_vals.append(roe)
            roe_by_year[d['year']] = roe
    if len(roe_vals) < 2:
        return {'cyclical_peak': False, 'roe_values': [], 'roe_by_year': {}}
    latest = roe_vals[-1]
    avg = round(np.mean(roe_vals), 1)
    median = round(np.median(roe_vals), 1)
    return {
        'latest_roe': latest, 'avg_roe': avg, 'median_roe': median,
        'min_roe': round(min(roe_vals), 1), 'normalized_roe': median,
        'cyclical_peak': latest > avg * 2 and latest > 20,
        'roe_values': roe_vals, 'roe_by_year': roe_by_year
    }


# ============================================================
# LAYER 5: MOMENTUM
# ============================================================
def run_momentum_check(qfin):
    if qfin is None or qfin.shape[1] < 3:
        return {'available': False}
    eps_vals, quarters = [], []
    for i in range(min(4, qfin.shape[1])):
        eps = safe_get(qfin, 'Diluted EPS', i) or safe_get(qfin, 'Basic EPS', i)
        if eps is not None:
            eps_vals.append(eps)
            quarters.append(str(qfin.columns[i].strftime('%b %Y')) if hasattr(qfin.columns[i], 'strftime') else str(i))
    if len(eps_vals) < 3:
        return {'available': False}
    latest_qoq = round((eps_vals[0] / eps_vals[1] - 1) * 100, 1) if eps_vals[1] != 0 else None
    prior_qoq = round((eps_vals[1] / eps_vals[2] - 1) * 100, 1) if eps_vals[2] != 0 else None
    if not is_valid_number(latest_qoq): latest_qoq = None
    if not is_valid_number(prior_qoq): prior_qoq = None
    return {
        'available': True, 'latest_qoq': latest_qoq, 'prior_qoq': prior_qoq,
        'eps_values': eps_vals[:4], 'quarters': quarters[:4]
    }


# ============================================================
# LAYER 7: POSITION SIZING v2
# ============================================================
def get_position_size(acct_score, num_flags, cyclical_peak, peg, momentum_1y):
    if acct_score is None or acct_score < 50:
        return 'WATCH', '0% (monitor only)'
    if is_valid_number(peg):
        if peg > 5.0: return 'WATCH', '0% (PEG > 5, extremely overvalued)'
        if peg > 3.0: return 'HALF', '4-6% (PEG > 3, significantly overvalued)'
    if acct_score >= 85 and num_flags == 0 and not cyclical_peak:
        base = 'FULL'
    elif acct_score >= 70 and num_flags <= 1:
        base = 'STANDARD'
    else:
        base = 'HALF'
    if base == 'FULL' and not is_valid_number(peg):
        return 'STANDARD', '8-10% (growth not measurable)'
    if base == 'FULL' and peg < 0.5:
        return 'FULL', '12-15%'
    if base == 'FULL' and is_valid_number(momentum_1y) and momentum_1y < -30:
        return 'HALF', '4-6% (momentum risk)'
    if base == 'FULL' and peg > 1.5:
        return 'STANDARD', '8-10% (valuation full)'
    return {'FULL': ('FULL', '12-15%'), 'STANDARD': ('STANDARD', '8-10%'), 'HALF': ('HALF', '4-6%')}[base]


def calc_peg(pe, pat_cagr):
    if is_valid_number(pe) and is_valid_number(pat_cagr) and pat_cagr > 0:
        return round(pe / pat_cagr, 2)
    return None


# ============================================================
# LAYER BREAKDOWN & VERDICT
# ============================================================
def generate_layer_breakdown(layer1, acct, cyclical, peg, momentum, ret_1y, is_bank):
    layers = []

    if layer1['phase1_pass']:
        layers.append(("Fundamentals", "pass",
                        f"Market Cap ₹{safe_fmt(layer1.get('market_cap_cr'), ',.0f')} Cr · ROE {safe_fmt(layer1.get('roe'), '.1f')}% · "
                        f"ROCE {safe_fmt(layer1.get('roce'), '.1f')}% · D/E {layer1.get('debt_to_equity', 'N/A')} · "
                        f"Sales CAGR {safe_fmt(layer1.get('sales_cagr'), '.1f')}% · PAT CAGR {safe_fmt(layer1.get('pat_cagr'), '.1f')}%"))
    else:
        fails = []
        if not layer1['mcap_pass']: fails.append(f"Market cap ₹{safe_fmt(layer1.get('market_cap_cr'), ',.0f')} Cr below ₹15,000 Cr threshold")
        if not layer1['roe_pass']: fails.append(f"ROE {safe_fmt(layer1.get('roe'), '.1f')}% below 15%")
        if not layer1['roce_pass']: fails.append(f"ROCE {safe_fmt(layer1.get('roce'), '.1f')}% below 18%")
        if not layer1['de_pass']: fails.append(f"D/E {layer1.get('debt_to_equity', 'N/A')} above 0.5")
        if not layer1['growth_pass']: fails.append(f"Growth too slow — Sales CAGR {safe_fmt(layer1.get('sales_cagr'), '.1f')}%, PAT CAGR {safe_fmt(layer1.get('pat_cagr'), '.1f')}%")
        layers.append(("Fundamentals", "fail", " · ".join(fails) if fails else "Multiple criteria not met"))

    score = acct.get('score')
    if score is not None:
        if score >= 85:
            layers.append(("Forensic Accounting", "pass", f"Score {score}/100 · Cumulative CFO/PAT {safe_fmt(acct.get('cum_cfo_pat'), '.2f')}x · CFO/EBITDA {safe_fmt(acct.get('cum_cfo_ebitda'), '.2f')}x · Cash generation validates reported profits"))
        elif score >= 50:
            layers.append(("Forensic Accounting", "warn", f"Score {score}/100 · {acct.get('num_flags', 0)} flag(s) · {' · '.join(acct.get('flags', [])[:2])}"))
        else:
            layers.append(("Forensic Accounting", "fail", f"Score {score}/100 · {acct.get('num_flags', 0)} flag(s) · {' · '.join(acct.get('flags', [])[:2])}"))

    if is_valid_number(peg):
        if peg < 0.5: layers.append(("PEG Valuation", "pass", f"PEG {peg:.2f} — significantly undervalued"))
        elif peg < 1.0: layers.append(("PEG Valuation", "pass", f"PEG {peg:.2f} — attractive. Growth justifies PE"))
        elif peg < 1.5: layers.append(("PEG Valuation", "warn", f"PEG {peg:.2f} — fairly valued"))
        elif peg < 2.0: layers.append(("PEG Valuation", "warn", f"PEG {peg:.2f} — expensive"))
        else: layers.append(("PEG Valuation", "fail", f"PEG {peg:.2f} — overvalued"))
    else:
        layers.append(("PEG Valuation", "warn", "PEG not calculable — either negative/zero earnings growth or PE unavailable"))

    pe = layer1.get('pe')
    if is_valid_number(pe) and pe > 80:
        layers.append(("Adani Filter", "fail", f"PE {pe:.0f} exceeds 80 — extremely high risk"))

    if momentum.get('available'):
        lq, pq = momentum.get('latest_qoq'), momentum.get('prior_qoq')
        if is_valid_number(lq) and is_valid_number(pq):
            if lq > 0 and pq > 0: layers.append(("Earnings Momentum", "pass", f"Both quarters positive — QoQ {lq:+.1f}%, prior {pq:+.1f}%"))
            elif lq > 0 or pq > 0: layers.append(("Earnings Momentum", "warn", f"Mixed — QoQ {lq:+.1f}%, prior {pq:+.1f}%"))
            else: layers.append(("Earnings Momentum", "fail", f"Both declining — QoQ {lq:+.1f}%, prior {pq:+.1f}%"))
        else:
            layers.append(("Earnings Momentum", "warn", "Partial quarterly data"))
    else:
        layers.append(("Earnings Momentum", "warn", "Insufficient quarterly data"))

    if cyclical.get('cyclical_peak'):
        layers.append(("Cyclical ROE", "warn", f"CYCLICAL PEAK — Latest {cyclical['latest_roe']}% vs normalized {cyclical['median_roe']}%"))
    elif cyclical.get('roe_by_year'):
        layers.append(("Cyclical ROE", "pass", f"Not at peak. Latest ROE {cyclical.get('latest_roe', 'N/A')}%, normalized {cyclical.get('median_roe', 'N/A')}%"))

    if is_valid_number(ret_1y):
        if ret_1y > 30: layers.append(("Price Momentum", "pass", f"Up {ret_1y:+.1f}% in 1Y — strong"))
        elif ret_1y > 0: layers.append(("Price Momentum", "pass", f"Up {ret_1y:+.1f}% in 1Y — steady"))
        elif ret_1y > -20: layers.append(("Price Momentum", "warn", f"Down {ret_1y:.1f}% in 1Y — mild weakness"))
        else: layers.append(("Price Momentum", "fail", f"Down {ret_1y:.1f}% in 1Y — significant selling pressure"))
    else:
        layers.append(("Price Momentum", "warn", "1-year price return not available"))

    return layers


def generate_verdict_text(name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank):
    if is_bank:
        return (f"**{name}** is a banking/financial stock. This framework uses metrics designed for non-financial companies. "
                f"Banks need NIM, CASA, Credit Cost, GNPA metrics. Scores shown for reference only.")

    score = acct.get('score', 0) or 0
    pe, roe, cfo_pat = layer1.get('pe'), layer1.get('roe'), acct.get('cum_cfo_pat')
    paras = []

    if tier == 'FULL':
        paras.append(f"**{name}** passes all 7 layers with conviction. Accounting quality is pristine, valuation attractive, momentum supportive. Data supports 12-15% allocation.")
    elif tier == 'STANDARD':
        paras.append(f"**{name}** is solid but one or more factors prevent full conviction. Data supports 8-10% — a good business at a fair price.")
    elif tier == 'HALF':
        paras.append(f"**{name}** shows mixed signals. Genuine positives but flags need resolution. Data supports cautious 4-6%, add after next quarterly confirms improvement.")
    else:
        paras.append(f"**{name}** fails critical checks. Do not deploy capital. Monitor quarterly and re-evaluate when flagged issues improve.")

    positives = []
    if is_valid_number(roe) and roe > 20: positives.append(f"ROE of {roe}% indicates strong capital efficiency")
    if is_valid_number(cfo_pat) and cfo_pat > 1.0: positives.append(f"exceptional cash generation — cumulative CFO/PAT of {cfo_pat}x")
    elif is_valid_number(cfo_pat) and cfo_pat > 0.7: positives.append(f"healthy cash conversion at {cfo_pat}x cumulative CFO/PAT")
    if is_valid_number(peg) and peg < 0.5: positives.append(f"deeply undervalued at PEG {peg}")
    elif is_valid_number(peg) and peg < 1.0: positives.append(f"attractively valued at PEG {peg}")
    if is_valid_number(layer1.get('sales_cagr')) and layer1['sales_cagr'] > 20: positives.append(f"strong revenue growth at {layer1['sales_cagr']}% CAGR")
    if is_valid_number(ret_1y) and ret_1y > 20: positives.append(f"positive sentiment with {ret_1y:+.1f}% 1Y return")
    if positives: paras.append("**What's working:** " + ", ".join(positives) + ".")

    concerns = []
    if score < 70: concerns.append(f"accounting score {score}/100 with {acct.get('num_flags', 0)} flag(s)")
    if is_valid_number(peg) and peg > 2.0: concerns.append(f"PEG {peg} — overvalued")
    if is_valid_number(pe) and pe > 80: concerns.append(f"PE {pe:.0f} triggers Adani Filter")
    if cyclical.get('cyclical_peak'): concerns.append(f"ROE at cyclical peak ({cyclical['latest_roe']}% vs normalized {cyclical['median_roe']}%)")
    if is_valid_number(ret_1y) and ret_1y < -30: concerns.append(f"down {abs(ret_1y):.0f}% in 1Y")
    if momentum.get('available'):
        lq, pq = momentum.get('latest_qoq'), momentum.get('prior_qoq')
        if is_valid_number(lq) and is_valid_number(pq) and lq < 0 and pq < 0:
            concerns.append(f"two quarters declining EPS ({lq:+.1f}%, {pq:+.1f}% QoQ)")
    for flag in acct.get('flags', [])[:2]: concerns.append(flag.lower())
    if concerns: paras.append("**What needs attention:** " + ", ".join(concerns) + ".")

    actions = {'FULL': "Deploy 12-15%. Read latest concall to confirm.", 'STANDARD': "Position at 8-10%. Hold and compound.",
               'HALF': "Half position at 4-6% only. Wait for next results.", 'WATCH': "Do not buy. Monitor quarterly."}
    paras.append(f"**Action:** {actions.get(tier, 'Monitor.')}")

    return "\n\n".join(paras)


# ============================================================
# BATCH SCREENER
# ============================================================
def analyse_quick(ticker_str):
    try:
        sd = fetch_stock_data(ticker_str)
        if not sd:
            return None
        if is_banking_stock(sd['info']):
            return None
        layer1 = run_layer1(sd, ticker_str)
        if not layer1['mcap_pass'] or not layer1['de_pass']:
            return None
        if is_valid_number(layer1['roe']) and layer1['roe'] < 15:
            return None
        if layer1['roe'] is None:
            return None
        multi = get_multi_year_data(sd['financials'], sd['balance_sheet'], sd['cashflow'])
        if not multi:
            return None
        acct = run_accounting_quality(multi)
        cyc = run_cyclical_check(multi)
        peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))
        ret = get_1y_return_from_data(sd)
        tier, size = get_position_size(acct['score'], acct['num_flags'], cyc.get('cyclical_peak', False), peg, ret)
        return {
            'ticker': ticker_str.replace('.NS', ''), 'name': layer1.get('name', ''),
            'sector': layer1.get('sector', 'N/A'), 'price': layer1.get('price'),
            'pe': layer1.get('pe'), 'peg': peg, 'roe': layer1.get('roe'), 'roce': layer1.get('roce'),
            'acct_score': acct['score'], 'num_flags': acct['num_flags'],
            'cum_cfo_pat': acct.get('cum_cfo_pat'), 'cyclical_peak': cyc.get('cyclical_peak', False),
            'ret_1y': ret, 'tier': tier, 'size': size,
            'sales_cagr': layer1.get('sales_cagr'), 'pat_cagr': layer1.get('pat_cagr')
        }
    except Exception:
        return None


# ============================================================
# MAIN APP
# ============================================================
st.markdown('<p class="main-header">📊 High Compounder Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">7-Layer Systematic Indian Equity Screener · Quality + Valuation + Momentum</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Single Stock", "Auto Top 10", "Live Tracker", "How It Works"], label_visibility="collapsed")
st.sidebar.markdown("### Portfolio")
st.sidebar.dataframe(pd.DataFrame({
    "Stock": ["LUPIN", "DIXON", "ENRIN", "BSE", "MCX", "ICICI AMC", "EICHER", "KPIT", "POLYCAB", "HDFC AMC"],
    "Tier": ["FULL", "FULL", "FULL", "STD", "STD", "STD", "STD", "HALF", "HALF", "HALF"],
    "Score": [100, 100, 100, 85, 85, 85, 85, 100, 55, 70],
    "PEG": [0.14, 0.54, 0.91, 0.38, 0.51, 1.46, 1.60, 1.48, 1.67, 1.42]
}), hide_index=True, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Vinayak Nagral · Sep 2026")


# ============================================================
# PAGE 1: SINGLE STOCK
# ============================================================
if page == "Single Stock":
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="LUPIN, BSE, DIXON, TCS, HCLTECH").strip().upper()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔍 Analyse", type="primary", use_container_width=True)

    qcols = st.columns(7)
    for i, qt in enumerate(["LUPIN", "BSE", "DIXON", "KPITTECH", "HDFCAMC", "MAZDOCK", "HCLTECH"]):
        with qcols[i]:
            if st.button(qt, key=f"q_{qt}", use_container_width=True):
                ticker_input = qt
                go = True

    if not ticker_input.endswith(".NS"):
        ticker_input += ".NS"

    if go:
        with st.spinner(f"Analysing {ticker_input.replace('.NS', '')}..."):
            sd = fetch_stock_data(ticker_input)
        if not sd:
            st.error("Could not fetch data. Check ticker.")
            st.stop()

        info = sd['info']
        fin, bs, cf, qfin = sd['financials'], sd['balance_sheet'], sd['cashflow'], sd['quarterly_financials']
        is_bank = is_banking_stock(info, ticker_input)

        with st.spinner("Running all 7 layers..."):
            layer1 = run_layer1(sd, ticker_input)
            multi = get_multi_year_data(fin, bs, cf)
            acct = run_accounting_quality(multi) if multi else {
                'score': None, 'flags': ['Insufficient data'], 'num_flags': 0,
                'cum_cfo_pat': None, 'cum_cfo_ebitda': None, 'single_yr_cfo_pat': None,
                'cfo_ebitda_trend': [], 'recv_pcts': [], 'recv_years': []
            }
            cyclical = run_cyclical_check(multi) if multi else {'cyclical_peak': False, 'roe_by_year': {}}
            peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))
            momentum = run_momentum_check(qfin)
            ret_1y = get_1y_return_from_data(sd)

        tier, size = get_position_size(acct.get('score'), acct.get('num_flags', 0), cyclical.get('cyclical_peak', False), peg, ret_1y)
        if not layer1['phase1_pass'] and tier in ['FULL', 'STANDARD']:
            tier, size = 'HALF', '4-6% (fails Phase I fundamentals)'

        name = layer1.get('name', ticker_input)
        sector = layer1.get('sector', 'N/A')
        price = layer1.get('price')

        st.markdown(f"## {name}")
        price_str = f"₹{price:,.2f}" if is_valid_number(price) else "Price unavailable"
        st.markdown(f"**{ticker_input}** · {sector} · {price_str}")

        if is_bank:
            st.markdown('<div class="banking-box">⚠️ <strong>Banking/Financial Stock.</strong> Framework designed for non-financial companies. Scores for reference only.</div>', unsafe_allow_html=True)

        st.markdown("---")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Score", f"{acct.get('score', 'N/A')}/100" if acct.get('score') is not None else "N/A")
        m2.metric("PE", safe_fmt(layer1.get('pe'), '.1f'))
        m3.metric("PEG", safe_fmt(peg, '.2f'))
        m4.metric("ROE", safe_fmt(layer1.get('roe'), '.1f', '%'))
        m5.metric("1Y Return", safe_fmt(ret_1y, '+.1f', '%'))
        with m6:
            st.markdown("**Tier**")
            tc = f"tier-{tier.lower()}" if tier in ['FULL', 'STANDARD', 'HALF', 'WATCH'] else ""
            st.markdown(f'<p class="{tc}">{tier} — {size}</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Layer-by-Layer Breakdown")
        layers = generate_layer_breakdown(layer1, acct, cyclical, peg, momentum, ret_1y, is_bank)
        for lname, status, detail in layers:
            css = "layer-pass" if status == "pass" else "layer-fail" if status == "fail" else "layer-warn"
            icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
            st.markdown(f'<div class="{css}">{icon} <strong>{lname}:</strong> {detail}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Investment Verdict")
        verdict = generate_verdict_text(name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank)
        st.markdown(f'<div class="verdict-section">{verdict}</div>', unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Forensic Detail", "📊 Cyclical & Valuation", "📈 Momentum", "📋 Raw Data"])

        with tab1:
            if acct.get('score') is not None:
                sc = acct['score']
                css = "score-green" if sc >= 85 else "score-yellow" if sc >= 50 else "score-red"
                label = "CLEAN" if sc >= 85 else "FLAGS DETECTED" if sc >= 50 else "SERIOUS CONCERNS"
                st.markdown(f'<div class="score-box {css}"><h2>{sc}/100</h2><p>{label}</p></div>', unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Cumulative CFO/PAT**")
                    v = acct.get('cum_cfo_pat')
                    if is_valid_number(v): st.markdown(f"{'✅' if v >= 0.7 else '⚠️' if v >= 0.5 else '❌'} **{v}x** {'(healthy)' if v >= 0.7 else '(weak)' if v >= 0.5 else '(critical)'}")
                    else: st.markdown("N/A")
                    sv = acct.get('single_yr_cfo_pat')
                    if is_valid_number(sv): st.caption(f"Latest single year: {sv}x")
                with cb:
                    st.markdown("**Cumulative CFO/EBITDA**")
                    v = acct.get('cum_cfo_ebitda')
                    if is_valid_number(v): st.markdown(f"{'✅' if v >= 0.7 else '⚠️' if v >= 0.5 else '❌'} **{v}x** {'(healthy)' if v >= 0.7 else '(weak)' if v >= 0.5 else '(critical)'}")
                    else: st.markdown("N/A")
                    t = acct.get('cfo_ebitda_trend', [])
                    if t: st.caption(f"Trend: {' → '.join(str(x) for x in t)}")
                if acct.get('flags'):
                    for f in acct['flags']: st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="clean-item">✅ All forensic checks passed</div>', unsafe_allow_html=True)
                recv = acct.get('recv_pcts', [])
                yrs = acct.get('recv_years', [])
                if recv and any(r is not None for r in recv):
                    st.markdown("**Receivables % of Revenue:**")
                    st.dataframe(pd.DataFrame({'Year': yrs, 'Recv%': [f"{r:.1f}%" if r else "N/A" for r in recv]}), hide_index=True, use_container_width=True)
            else:
                st.warning("Insufficient data for forensic analysis")

        with tab2:
            cc, cd = st.columns(2)
            with cc:
                st.subheader("Cyclical ROE")
                if cyclical.get('roe_by_year'):
                    st.dataframe(pd.DataFrame(list(cyclical['roe_by_year'].items()), columns=['Year', 'ROE%']), hide_index=True, use_container_width=True)
                    st.markdown(f"Latest **{cyclical.get('latest_roe')}%** · Avg **{cyclical.get('avg_roe')}%** · Normalized **{cyclical.get('median_roe')}%**")
                    if cyclical.get('cyclical_peak'): st.error("⚠️ CYCLICAL PEAK")
                    else: st.success("✅ Not at peak")
                else: st.warning("Insufficient data")
            with cd:
                st.subheader("Valuation")
                st.markdown(f"**PE:** {safe_fmt(layer1.get('pe'), '.1f')} · **PEG:** {safe_fmt(peg, '.2f')}")
                if is_valid_number(peg):
                    if peg < 0.5: st.success(f"PEG {peg:.2f} — Undervalued")
                    elif peg < 1.0: st.success(f"PEG {peg:.2f} — Attractive")
                    elif peg < 1.5: st.info(f"PEG {peg:.2f} — Fair")
                    elif peg < 2.0: st.warning(f"PEG {peg:.2f} — Expensive")
                    else: st.error(f"PEG {peg:.2f} — Overvalued")
                st.markdown(f"**Sales CAGR:** {safe_fmt(layer1.get('sales_cagr'), '.1f')}% · **PAT CAGR:** {safe_fmt(layer1.get('pat_cagr'), '.1f')}%")

        with tab3:
            if momentum.get('available'):
                mc1, mc2 = st.columns(2)
                mc1.metric("Latest QoQ", safe_fmt(momentum.get('latest_qoq'), '+.1f', '%'))
                mc2.metric("Prior QoQ", safe_fmt(momentum.get('prior_qoq'), '+.1f', '%'))
                if momentum.get('eps_values') and momentum.get('quarters'):
                    st.dataframe(pd.DataFrame({'Quarter': momentum['quarters'], 'EPS': [round(e, 2) for e in momentum['eps_values']]}), hide_index=True, use_container_width=True)
            else:
                st.warning("Insufficient quarterly data")
            if is_valid_number(ret_1y): st.markdown(f"**1Y Price Return:** {ret_1y:+.1f}%")
            else: st.markdown("**1Y Price Return:** N/A")

        with tab4:
            st.subheader("All Layers Summary")
            st.dataframe(pd.DataFrame({
                "Layer": ["Fundamentals", "Cash Flow", "Forensic", "PEG", "Momentum", "Cyclical", "Sizing"],
                "Result": [
                    "PASS ✅" if layer1['phase1_pass'] else "FAIL ❌",
                    f"CFO/PAT: {safe_fmt(acct.get('cum_cfo_pat'), '.2f')}x",
                    f"{acct.get('score', 'N/A')}/100 ({acct.get('num_flags', 0)} flags)",
                    safe_fmt(peg, '.2f'),
                    f"QoQ: {safe_fmt(momentum.get('latest_qoq'), '+.1f')}%" if momentum.get('available') else "N/A",
                    "PEAK ⚠️" if cyclical.get('cyclical_peak') else "OK ✅",
                    f"{tier} — {size}"
                ]
            }), hide_index=True, use_container_width=True)

            with st.expander("🐛 Debug: Data Sources"):
                st.markdown("**Info dict fields:**")
                key_fields = ['currentPrice', 'regularMarketPrice', 'previousClose', 'trailingPE',
                              'forwardPE', 'trailingEps', 'forwardEps', 'marketCap', 'sharesOutstanding',
                              'shortName', 'longName', 'sector', 'industry', 'debtToEquity', 'returnOnEquity']
                debug_rows = [{'Field': k, 'Value': str(info.get(k, '❌ None'))[:50], 'Valid': '✅' if is_valid_number(info.get(k)) or (isinstance(info.get(k), str) and info.get(k).strip()) else '❌'} for k in key_fields]
                st.dataframe(pd.DataFrame(debug_rows), hide_index=True, use_container_width=True)

                st.markdown("**Financial statements:**")
                st.markdown(f"- Financials: {'✅ ' + str(fin.shape) if fin is not None else '❌ None'}")
                st.markdown(f"- Balance Sheet: {'✅ ' + str(bs.shape) if bs is not None else '❌ None'}")
                st.markdown(f"- Cash Flow: {'✅ ' + str(cf.shape) if cf is not None else '❌ None'}")
                st.markdown(f"- Quarterly: {'✅ ' + str(qfin.shape) if qfin is not None else '❌ None'}")

                ph = sd.get('price_history')
                st.markdown(f"- Price History: {'✅ ' + str(len(ph)) + ' days' if ph is not None and len(ph) > 0 else '❌ None/Empty'}")

                st.markdown("**Computed values:**")
                st.markdown(f"- Price: {layer1.get('price')}")
                st.markdown(f"- PE: {layer1.get('pe')}")
                st.markdown(f"- Market Cap Cr: {layer1.get('market_cap_cr')}")
                shares = get_shares_outstanding(bs, info)
                st.markdown(f"- Shares Outstanding: {shares:,.0f}" if is_valid_number(shares) else "- Shares: ❌ Not found")

        st.caption("For research only, not investment advice · Data from Yahoo Finance")


# ============================================================
# PAGE 2: AUTO TOP 10
# ============================================================
elif page == "Auto Top 10":
    st.subheader("🏆 Automatic Top 10 Picker")
    st.markdown("Screens Nifty 200 through all 7 layers. Banking/insurance auto-excluded.")

    c1, c2 = st.columns(2)
    with c1: max_stocks = st.slider("Stocks to screen", 20, 200, 50, 10)
    with c2: top_n = st.slider("Show top N", 5, 20, 10)

    if st.button("🚀 Run Full Screen", type="primary", use_container_width=True):
        tickers = fetch_nifty200_tickers()[:max_stocks]
        if not tickers:
            st.error("Could not fetch Nifty 200.")
            st.stop()
        results = []
        prog = st.progress(0, "Starting...")
        for i, t in enumerate(tickers):
            prog.progress((i + 1) / len(tickers), f"Analysing {t.replace('.NS', '')} ({i + 1}/{len(tickers)})")
            r = analyse_quick(t)
            if r: results.append(r)
        prog.empty()
        if not results:
            st.error("No stocks passed.")
            st.stop()

        df = pd.DataFrame(results)

        def peg_score(x):
            if not is_valid_number(x): return 10
            if x < 0.3: return 100
            if x < 0.5: return 90
            if x < 1.0: return 75
            if x < 1.5: return 55
            if x < 2.0: return 30
            if x < 3.0: return 10
            return 0

        def mom_score(x):
            if not is_valid_number(x): return 40
            if x > 30: return 80
            if x > 0: return 65
            if x > -20: return 40
            if x > -40: return 20
            return 5

        def cfo_s(x):
            if not is_valid_number(x): return 20
            if x > 1.2: return 100
            if x > 0.9: return 85
            if x > 0.7: return 65
            if x > 0.5: return 40
            return 10

        df['rank_score'] = (
            df['acct_score'].fillna(0) * 0.25 + df['peg'].apply(peg_score) * 0.30 +
            df['roe'].fillna(0) * 0.10 + df['ret_1y'].apply(mom_score) * 0.15 +
            df['cum_cfo_pat'].apply(cfo_s) * 0.20
        )
        df = df.sort_values('rank_score', ascending=False)
        top = df.head(top_n)

        st.markdown(f"### Top {top_n} from {len(tickers)} screened ({len(results)} passed)")
        for idx, row in top.iterrows():
            rank = list(top.index).index(idx) + 1
            tc = {"FULL": "🟢", "STANDARD": "🔵", "HALF": "🟡", "WATCH": "🔴"}.get(row['tier'], "⚪")
            with st.expander(f"**#{rank} · {row['ticker']}** — {row['name']} · {tc} {row['tier']} · Score {row['acct_score']}/100 · PEG {safe_fmt(row['peg'], '.2f')}", expanded=(rank <= 3)):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Price", f"₹{row['price']:,.0f}" if is_valid_number(row.get('price')) else "N/A")
                c2.metric("PE", safe_fmt(row.get('pe'), '.1f'))
                c3.metric("PEG", safe_fmt(row.get('peg'), '.2f'))
                c4.metric("ROE", safe_fmt(row.get('roe'), '.1f', '%'))
                c5.metric("1Y Ret", safe_fmt(row.get('ret_1y'), '+.1f', '%'))
                c6.metric("CFO/PAT", safe_fmt(row.get('cum_cfo_pat'), '.2f', 'x'))
                st.markdown(f"**Sector:** {row['sector']} · **Sales CAGR:** {safe_fmt(row.get('sales_cagr'), '.1f')}% · **PAT CAGR:** {safe_fmt(row.get('pat_cagr'), '.1f')}% · **Flags:** {row['num_flags']} · **Cyclical Peak:** {'Yes ⚠️' if row['cyclical_peak'] else 'No'}")
                st.markdown(f"**Position:** {row['tier']} — {row['size']}")

        st.markdown("---")
        st.markdown("### Full Rankings")
        dd = df[['ticker', 'name', 'acct_score', 'pe', 'peg', 'roe', 'cum_cfo_pat', 'ret_1y', 'tier', 'size']].copy()
        dd['peg'] = dd['peg'].apply(lambda x: round(x, 2) if is_valid_number(x) else None)
        dd['cum_cfo_pat'] = dd['cum_cfo_pat'].apply(lambda x: round(x, 2) if is_valid_number(x) else None)
        dd.columns = ['Ticker', 'Name', 'Score', 'PE', 'PEG', 'ROE%', 'CFO/PAT', '1Y Ret%', 'Tier', 'Size']
        dd = dd.reset_index(drop=True)
        dd.index += 1
        st.dataframe(dd, use_container_width=True)

        st.markdown("### Tier Breakdown")
        tc1, tc2, tc3, tc4 = st.columns(4)
        for cw, tn, em in [(tc1, 'FULL', '🟢'), (tc2, 'STANDARD', '🔵'), (tc3, 'HALF', '🟡'), (tc4, 'WATCH', '🔴')]:
            ts = df[df['tier'] == tn]
            with cw:
                st.markdown(f"**{em} {tn}** ({len(ts)})")
                for _, r in ts.iterrows(): st.caption(f"{r['ticker']} · {r['acct_score']}")


# ============================================================
# PAGE 3: LIVE TRACKER
# ============================================================
elif page == "Live Tracker":
    st.subheader("📊 Live Framework Validation")
    st.markdown("Tracking **BUY vs AVOID** from September 1, 2026.")
    baseline_date = "2026-09-01"

    buy_picks = {
        "LUPIN.NS": {"name": "Lupin", "tier": "FULL", "score": 100, "peg": 0.14, "reason": "PEG 0.14, Score 100, pharma compounder"},
        "DIXON.NS": {"name": "Dixon Technologies", "tier": "FULL", "score": 100, "peg": 0.54, "reason": "EMS champion, clean cash flows"},
        "BSE.NS": {"name": "BSE Limited", "tier": "STANDARD", "score": 85, "peg": 0.38, "reason": "Capital market monopoly, PEG 0.38"},
        "EICHERMOT.NS": {"name": "Eicher Motors", "tier": "STANDARD", "score": 85, "peg": 1.60, "reason": "Royal Enfield pricing power"},
        "KPITTECH.NS": {"name": "KPIT Technologies", "tier": "HALF", "score": 100, "peg": 1.48, "reason": "Score 100 but -51% momentum"},
    }
    avoid_picks = {
        "GODFRYPHLP.NS": {"name": "Godfrey Phillips", "tier": "WATCH", "score": 35, "peg": 0.78, "reason": "Structural tax reset"},
        "WAAREEENER.NS": {"name": "Waaree Energies", "tier": "WATCH", "score": 45, "peg": 0.20, "reason": "Cumulative CFO/PAT 0.44x"},
        "MAZDOCK.NS": {"name": "Mazagon Dock", "tier": "WATCH", "score": 25, "peg": 1.16, "reason": "Negative cumulative CFO"},
    }
    benchmark = "^NSEI"

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_tracker_prices(tickers, base_date):
        results = {}
        end = datetime.now()
        start = datetime.strptime(base_date, "%Y-%m-%d") - timedelta(days=5)
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                data = t.history(start=start, end=end, auto_adjust=True)
                if data.empty or 'Close' not in data.columns:
                    continue
                prices = data['Close'].dropna()
                if len(prices) == 0:
                    continue
                base_dt = pd.Timestamp(base_date)
                mask = prices.index <= base_dt
                base_price = float(prices.loc[mask].iloc[-1]) if mask.sum() > 0 else float(prices.iloc[0])
                current_price = float(prices.iloc[-1])
                if not is_valid_number(base_price) or not is_valid_number(current_price) or base_price <= 0:
                    continue
                ret = round((current_price / base_price - 1) * 100, 2)
                history = prices[prices.index >= base_dt].copy()
                if not history.empty:
                    first_val = float(history.iloc[0])
                    if is_valid_number(first_val) and first_val > 0:
                        history = (history / first_val - 1) * 100
                results[ticker] = {'base_price': round(base_price, 2), 'current_price': round(current_price, 2),
                                   'return_pct': ret if is_valid_number(ret) else 0, 'history': history}
            except Exception:
                continue
        return results

    all_tickers = list(buy_picks.keys()) + list(avoid_picks.keys()) + [benchmark]
    with st.spinner("Fetching live prices..."):
        prices = get_tracker_prices(all_tickers, baseline_date)

    if not prices:
        st.error("Could not fetch prices. Try again.")
        st.stop()

    bench_ret = prices.get(benchmark, {}).get('return_pct', 0)
    st.markdown(f"### Performance Since {baseline_date}")
    st.markdown(f"**Nifty 50 (Benchmark):** {bench_ret:+.2f}%")
    st.markdown("---")

    st.markdown("### 🟢 BUY Picks")
    buy_rows = []
    for ticker, binfo in buy_picks.items():
        p = prices.get(ticker, {})
        ret = p.get('return_pct')
        alpha = round(ret - bench_ret, 2) if is_valid_number(ret) else None
        buy_rows.append({'Stock': binfo['name'], 'Tier': binfo['tier'], 'Score': binfo['score'], 'PEG': binfo['peg'],
                         'Entry': f"₹{p.get('base_price', 'N/A')}", 'Current': f"₹{p.get('current_price', 'N/A')}",
                         'Return': f"{ret:+.2f}%" if is_valid_number(ret) else "N/A",
                         'vs Nifty': f"{alpha:+.2f}%" if is_valid_number(alpha) else "N/A", 'Thesis': binfo['reason']})
    st.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)
    buy_returns = [prices[t]['return_pct'] for t in buy_picks if t in prices and is_valid_number(prices[t].get('return_pct'))]
    if buy_returns:
        avg_buy = round(np.mean(buy_returns), 2)
        st.markdown(f"**Average BUY: {avg_buy:+.2f}%** (vs Nifty {bench_ret:+.2f}%)")

    st.markdown("---")
    st.markdown("### 🔴 AVOID Picks")
    avoid_rows = []
    for ticker, ainfo in avoid_picks.items():
        p = prices.get(ticker, {})
        ret = p.get('return_pct')
        alpha = round(ret - bench_ret, 2) if is_valid_number(ret) else None
        avoid_rows.append({'Stock': ainfo['name'], 'Tier': ainfo['tier'], 'Score': ainfo['score'], 'PEG': ainfo['peg'],
                           'Entry': f"₹{p.get('base_price', 'N/A')}", 'Current': f"₹{p.get('current_price', 'N/A')}",
                           'Return': f"{ret:+.2f}%" if is_valid_number(ret) else "N/A",
                           'vs Nifty': f"{alpha:+.2f}%" if is_valid_number(alpha) else "N/A", 'Reason': ainfo['reason']})
    st.dataframe(pd.DataFrame(avoid_rows), hide_index=True, use_container_width=True)
    avoid_returns = [prices[t]['return_pct'] for t in avoid_picks if t in prices and is_valid_number(prices[t].get('return_pct'))]
    if avoid_returns:
        avg_avoid = round(np.mean(avoid_returns), 2)
        st.markdown(f"**Average AVOID: {avg_avoid:+.2f}%** (vs Nifty {bench_ret:+.2f}%)")

    st.markdown("---")
    st.markdown("### Validation Verdict")
    if buy_returns and avoid_returns:
        avg_b, avg_a = round(np.mean(buy_returns), 2), round(np.mean(avoid_returns), 2)
        spread = round(avg_b - avg_a, 2)
        alpha_n = round(avg_b - bench_ret, 2)
        if spread > 0 and alpha_n > 0:
            st.success(f"**✅ Framework working.** BUY {avg_b:+.2f}% · AVOID {avg_a:+.2f}% · Spread {spread:+.2f}% · Alpha vs Nifty {alpha_n:+.2f}%")
        elif spread > 0:
            st.info(f"**📊 Partial.** BUY ({avg_b:+.2f}%) beat AVOID ({avg_a:+.2f}%) by {spread:.1f}pp. {'Outperforming' if alpha_n > 0 else 'Underperforming'} Nifty by {abs(alpha_n):.1f}pp.")
        else:
            st.warning(f"**⚠️ Not validated yet.** BUY {avg_b:+.2f}% vs AVOID {avg_a:+.2f}%. Give it 3-6 months.")

    st.markdown("---")
    st.markdown("### Cumulative Returns")
    chart_data = pd.DataFrame()
    for ticker in list(buy_picks.keys()) + list(avoid_picks.keys()) + [benchmark]:
        p = prices.get(ticker)
        if p and 'history' in p and not p['history'].empty:
            label = buy_picks.get(ticker, avoid_picks.get(ticker, {})).get('name', 'Nifty 50')
            chart_data[label] = p['history']
    if not chart_data.empty:
        st.line_chart(chart_data)
        st.caption("Indexed to 0% on baseline date.")
    st.caption("Tracking started Sep 1, 2026 · Updated live via Yahoo Finance")


# ============================================================
# PAGE 4: HOW IT WORKS
# ============================================================
elif page == "How It Works":
    st.subheader("How This Framework Works")
    st.markdown("""
This dashboard runs a **7-layer systematic analysis** on any NSE-listed stock, designed to find high-quality compounders while avoiding value traps.

**Layer 1 — Quantitative Screen:** Market Cap > ₹15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, 3Y Sales & PAT CAGR > 15%.

**Layer 2 — Cash Flow Quality:** CFO/EBITDA > 70%.

**Layer 3 — Forensic Accounting:** Four checks using **cumulative multi-year** data. Catches receivables stuffing, inventory bloat, fake profits.

**Layer 4 — PEG Valuation:** PE ÷ earnings growth. "Adani Filter" flags PE > 80.

**Layer 5 — Earnings Momentum:** Last 2 quarters QoQ EPS change.

**Layer 6 — Cyclical ROE Normalization:** If ROE > 2x historical average → "cyclical peak."

**Layer 7 — Position Sizing v2:** Clean + cheap = FULL (12-15%). Clean + expensive = STANDARD (8-10%). Flags = HALF (4-6%). Failed = WATCH (0%).

---

### Not Covered
**Banking/NBFC/Insurance** — need NIM, CASA, Credit Cost. **Commodity producers** — CAGR filter creates false signals.

### Framework Score: 90/100
Remaining 10 points need point-in-time backtesting and management quality scoring.
    """)
    st.caption("Built by Vinayak Nagral · September 2026")
