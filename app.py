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
# ROBUST DATA EXTRACTION
# ============================================================
def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            val = df.loc[label].iloc[col]
            if pd.notna(val):
                return float(val)
    except:
        pass
    return None


def safe_fmt(val, fmt=".1f", suffix="", prefix=""):
    """Safely format a number for display. Returns 'N/A' if None/NaN."""
    if val is None:
        return "N/A"
    try:
        if np.isnan(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{prefix}{val:{fmt}}{suffix}"


def get_revenue(fin, col=0):
    """Get revenue with exhaustive fallbacks for Indian stocks."""
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


def is_banking_stock(info, name=""):
    sector = (info.get('sector', '') or '').lower()
    industry = (info.get('industry', '') or '').lower()
    stock_name = (info.get('shortName', '') or name or '').lower()
    for kw in BANKING_KEYWORDS:
        if kw in sector or kw in industry or kw in stock_name:
            return True
    return False


# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_str):
    try:
        t = yf.Ticker(ticker_str)
        info = dict(t.info) if t.info else {}
        fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None
        return {"info": info, "financials": fin, "balance_sheet": bs, "cashflow": cf, "quarterly_financials": qfin}
    except:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_1y_return(ticker_str):
    try:
        end = datetime.now()
        start = end - timedelta(days=400)
        data = yf.download(ticker_str, start=start, end=end, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        current = float(data['Close'].iloc[-1])
        lookback = end - timedelta(days=365)
        mask = data.index <= lookback
        if mask.sum() > 0:
            past = float(data.loc[mask, 'Close'].iloc[-1])
            return round((current / past - 1) * 100, 1)
    except:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nifty200_tickers():
    try:
        df = pd.read_csv(NIFTY_200_URL)
        return [s + ".NS" for s in df['Symbol'].tolist()]
    except:
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
def run_layer1(info, fin, bs):
    r = {}
    mcap = info.get('marketCap', 0) or 0
    r['market_cap_cr'] = round(mcap / 1e7, 0) if mcap else None
    r['mcap_pass'] = mcap > 150_000_000_000

    r['pe'] = info.get('trailingPE')

    de = info.get('debtToEquity', 0) or 0
    r['debt_to_equity'] = round(de / 100, 2) if de else 0
    r['de_pass'] = de < 50

    ni = get_net_income(fin, 0) if fin is not None else None
    eq = get_equity(bs, 0) if bs is not None else None
    if ni and eq and eq > 0:
        r['roe'] = round(ni / eq * 100, 1)
        r['roe_pass'] = r['roe'] > 15
    else:
        r['roe'] = None
        r['roe_pass'] = False

    ebit_val = safe_get(fin, 'EBIT', 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if ebit_val and ta and cl and (ta - cl) > 0:
        r['roce'] = round(ebit_val / (ta - cl) * 100, 1)
        r['roce_pass'] = r['roce'] > 18
    else:
        r['roce'] = None
        r['roce_pass'] = False

    r['sales_cagr'] = None
    r['pat_cagr'] = None
    if fin is not None and fin.shape[1] >= 2:
        yrs = fin.shape[1] - 1
        rev_l = get_revenue(fin, 0)
        rev_o = get_revenue(fin, fin.shape[1] - 1)
        ni_l = get_net_income(fin, 0)
        ni_o = get_net_income(fin, fin.shape[1] - 1)

        if rev_l and rev_o and rev_o > 0 and yrs > 0:
            r['sales_cagr'] = round(((rev_l / rev_o) ** (1 / yrs) - 1) * 100, 1)
        if ni_l and ni_o and ni_o > 0 and yrs > 0:
            r['pat_cagr'] = round(((ni_l / ni_o) ** (1 / yrs) - 1) * 100, 1)

    # Fallback: use info dict growth rates if financials didn't work
    if r['sales_cagr'] is None:
        rg = info.get('revenueGrowth')
        if rg is not None:
            r['sales_cagr'] = round(rg * 100, 1)
    if r['pat_cagr'] is None:
        eg = info.get('earningsGrowth')
        if eg is not None:
            r['pat_cagr'] = round(eg * 100, 1)

    r['growth_pass'] = ((r.get('sales_cagr') or 0) > 15 and (r.get('pat_cagr') or 0) > 15)
    r['phase1_pass'] = all([r['mcap_pass'], r['de_pass'], r['roe_pass'], r['roce_pass'], r['growth_pass']])
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

    # CHECK 1: Receivables
    recv_pcts = []
    for d in sd:
        if d['revenue'] and d['receivables'] and d['revenue'] > 0:
            recv_pcts.append(round(d['receivables'] / d['revenue'] * 100, 1))
        else:
            recv_pcts.append(None)
    details['recv_pcts'] = recv_pcts
    details['recv_years'] = [d['year'] for d in sd]

    if len(recv_pcts) >= 3:
        valid = [x for x in recv_pcts if x is not None]
        if len(valid) >= 3:
            rising = sum(1 for i in range(1, len(valid)) if valid[i] > valid[i-1])
            if rising >= 2:
                flags.append(f"Receivables rising: {valid[-3]}% → {valid[-2]}% → {valid[-1]}% of revenue")
                score -= 15

    # CHECK 2: Inventory
    if prior:
        inv_l, inv_p = latest.get('inventory'), prior.get('inventory')
        rev_l, rev_p = latest.get('revenue'), prior.get('revenue')
        if all(v and v > 0 for v in [inv_l, inv_p, rev_l, rev_p]):
            inv_g = round((inv_l / inv_p - 1) * 100, 1)
            rev_g = round((rev_l / rev_p - 1) * 100, 1)
            details['inv_growth'] = inv_g
            details['rev_growth'] = rev_g
            if inv_g > rev_g + 10:
                flags.append(f"Inventory bloat: grew {inv_g}% vs Revenue {rev_g}%")
                score -= 15

    # CHECK 3: Cumulative CFO/PAT
    total_cfo = sum(d['cfo'] for d in sd if d.get('cfo') is not None)
    total_pat = sum(d['net_income'] for d in sd if d.get('net_income') is not None)
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

    if latest.get('cfo') is not None and latest.get('net_income') and latest['net_income'] > 0:
        details['single_yr_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2)

    # CHECK 4: Cumulative CFO/EBITDA
    total_ebitda = sum(d['ebitda'] for d in sd if d.get('ebitda') is not None and d['ebitda'] > 0)
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
        if d.get('cfo') is not None and d.get('ebitda') and d['ebitda'] > 0:
            yr_ratios.append(round(d['cfo'] / d['ebitda'], 2))
    details['cfo_ebitda_trend'] = yr_ratios
    if len(yr_ratios) >= 2 and details.get('cum_cfo_ebitda'):
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
        if d.get('net_income') and d.get('equity') and d['equity'] > 0:
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
    return {
        'available': True,
        'latest_qoq': round((eps_vals[0] / eps_vals[1] - 1) * 100, 1) if eps_vals[1] != 0 else None,
        'prior_qoq': round((eps_vals[1] / eps_vals[2] - 1) * 100, 1) if eps_vals[2] != 0 else None,
        'eps_values': eps_vals[:4], 'quarters': quarters[:4]
    }


# ============================================================
# LAYER 7: POSITION SIZING v2
# ============================================================
def get_position_size(acct_score, num_flags, cyclical_peak, peg, momentum_1y):
    if acct_score < 50:
        return 'WATCH', '0% (monitor only)'
    if peg is not None and not (isinstance(peg, float) and np.isnan(peg)):
        if peg > 5.0:
            return 'WATCH', '0% (PEG > 5, extremely overvalued)'
        if peg > 3.0:
            return 'HALF', '4-6% (PEG > 3, significantly overvalued)'
    if acct_score >= 85 and num_flags == 0 and not cyclical_peak:
        base = 'FULL'
    elif acct_score >= 70 and num_flags <= 1:
        base = 'STANDARD'
    else:
        base = 'HALF'
    if base == 'FULL' and (peg is None or (isinstance(peg, float) and np.isnan(peg))):
        return 'STANDARD', '8-10% (growth not measurable)'
    if base == 'FULL' and peg < 0.5:
        return 'FULL', '12-15%'
    if base == 'FULL' and momentum_1y is not None and momentum_1y < -30:
        return 'HALF', '4-6% (momentum risk)'
    if base == 'FULL' and peg > 1.5:
        return 'STANDARD', '8-10% (valuation full)'
    return {'FULL': ('FULL', '12-15%'), 'STANDARD': ('STANDARD', '8-10%'), 'HALF': ('HALF', '4-6%')}[base]

def calc_peg(pe, pat_cagr):
    if pe and pat_cagr and pat_cagr > 0:
        return round(pe / pat_cagr, 2)
    return None


# ============================================================
# DETAILED VERDICT GENERATOR
# ============================================================
def generate_layer_breakdown(layer1, acct, cyclical, peg, momentum, ret_1y, is_bank):
    """Returns list of (layer_name, status, detail) tuples."""
    layers = []

    # Layer 1
    if layer1['phase1_pass']:
        layers.append(("Fundamentals", "pass", f"Market Cap ₹{safe_fmt(layer1.get('market_cap_cr'), ',.0f')} Cr · ROE {safe_fmt(layer1.get('roe'), '.1f')}% · ROCE {safe_fmt(layer1.get('roce'), '.1f')}% · D/E {layer1.get('debt_to_equity', 'N/A')} · Sales CAGR {safe_fmt(layer1.get('sales_cagr'), '.1f')}% · PAT CAGR {safe_fmt(layer1.get('pat_cagr'), '.1f')}%"))
    else:
        fails = []
        if not layer1['mcap_pass']: fails.append(f"Market cap ₹{safe_fmt(layer1.get('market_cap_cr'), ',.0f')} Cr below ₹15,000 Cr threshold")
        if not layer1['roe_pass']: fails.append(f"ROE {safe_fmt(layer1.get('roe'), '.1f')}% below 15%")
        if not layer1['roce_pass']: fails.append(f"ROCE {safe_fmt(layer1.get('roce'), '.1f')}% below 18%")
        if not layer1['de_pass']: fails.append(f"D/E {layer1.get('debt_to_equity', 'N/A')} above 0.5")
        if not layer1['growth_pass']: fails.append(f"Growth too slow — Sales CAGR {safe_fmt(layer1.get('sales_cagr'), '.1f')}%, PAT CAGR {safe_fmt(layer1.get('pat_cagr'), '.1f')}%")
        layers.append(("Fundamentals", "fail", " · ".join(fails) if fails else "Multiple criteria not met"))

    # Layer 2-3: Accounting
    score = acct.get('score')
    if score is not None:
        if score >= 85:
            layers.append(("Forensic Accounting", "pass", f"Score {score}/100 · Cumulative CFO/PAT {safe_fmt(acct.get('cum_cfo_pat'), '.2f')}x · CFO/EBITDA {safe_fmt(acct.get('cum_cfo_ebitda'), '.2f')}x · Cash generation validates reported profits"))
        elif score >= 50:
            flag_text = " · ".join(acct.get('flags', [])[:2])
            layers.append(("Forensic Accounting", "warn", f"Score {score}/100 · {acct.get('num_flags', 0)} flag(s) · {flag_text}"))
        else:
            flag_text = " · ".join(acct.get('flags', [])[:2])
            layers.append(("Forensic Accounting", "fail", f"Score {score}/100 · {acct.get('num_flags', 0)} flag(s) · {flag_text}"))

    # Layer 4: PEG
    if peg is not None and not (isinstance(peg, float) and np.isnan(peg)):
        if peg < 0.5:
            layers.append(("PEG Valuation", "pass", f"PEG {peg:.2f} — significantly undervalued. Paying {int(peg*100)} paise per rupee of growth"))
        elif peg < 1.0:
            layers.append(("PEG Valuation", "pass", f"PEG {peg:.2f} — attractive. Growth more than justifies the PE"))
        elif peg < 1.5:
            layers.append(("PEG Valuation", "warn", f"PEG {peg:.2f} — fairly valued. Growth roughly matches PE, no discount"))
        elif peg < 2.0:
            layers.append(("PEG Valuation", "warn", f"PEG {peg:.2f} — expensive. Paying a premium over what growth justifies"))
        else:
            layers.append(("PEG Valuation", "fail", f"PEG {peg:.2f} — overvalued. PE far exceeds what growth justifies"))
    else:
        layers.append(("PEG Valuation", "warn", "PEG not calculable — either negative/zero earnings growth or PE unavailable"))

    pe = layer1.get('pe')
    if pe and pe > 80:
        layers.append(("Adani Filter", "fail", f"PE {pe:.0f} exceeds 80 — extremely high risk unless earnings catch up within 2 years"))

    # Layer 5: Momentum
    if momentum.get('available'):
        lq = momentum.get('latest_qoq')
        pq = momentum.get('prior_qoq')
        if lq is not None and pq is not None:
            if lq > 0 and pq > 0:
                layers.append(("Earnings Momentum", "pass", f"Both quarters positive — latest QoQ {lq:+.1f}%, prior {pq:+.1f}%. Trend confirmed"))
            elif lq > 0 or pq > 0:
                layers.append(("Earnings Momentum", "warn", f"Mixed — latest QoQ {lq:+.1f}%, prior {pq:+.1f}%. One quarter positive, one negative"))
            else:
                layers.append(("Earnings Momentum", "fail", f"Both quarters declining — latest QoQ {lq:+.1f}%, prior {pq:+.1f}%. Momentum has turned negative"))
    else:
        layers.append(("Earnings Momentum", "warn", "Insufficient quarterly data to assess momentum"))

    # Layer 6: Cyclical
    if cyclical.get('cyclical_peak'):
        layers.append(("Cyclical ROE", "warn", f"CYCLICAL PEAK — Latest ROE {cyclical['latest_roe']}% vs normalized {cyclical['median_roe']}%. Use {cyclical['median_roe']}% for valuation, not {cyclical['latest_roe']}%"))
    elif cyclical.get('roe_by_year'):
        layers.append(("Cyclical ROE", "pass", f"Not at cyclical peak. Latest ROE {cyclical.get('latest_roe', 'N/A')}%, normalized {cyclical.get('median_roe', 'N/A')}%"))

    # 1Y Return context
    if ret_1y is not None:
        if ret_1y > 30:
            layers.append(("Price Momentum", "pass", f"Stock up {ret_1y:+.1f}% in 1 year — strong market sentiment"))
        elif ret_1y > 0:
            layers.append(("Price Momentum", "pass", f"Stock up {ret_1y:+.1f}% in 1 year — steady"))
        elif ret_1y > -20:
            layers.append(("Price Momentum", "warn", f"Stock down {ret_1y:.1f}% in 1 year — mild weakness"))
        else:
            layers.append(("Price Momentum", "fail", f"Stock down {ret_1y:.1f}% in 1 year — significant selling pressure"))

    return layers


def generate_verdict_text(name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank):
    """Generate a multi-paragraph investment verdict."""
    if is_bank:
        return f"**{name}** is a banking or financial services company. This framework uses ROE, D/E, and CFO/EBITDA metrics designed for non-financial companies. Banks require a separate framework using NIM (Net Interest Margin), CASA ratio, Credit Cost, GNPA/NNPA trends, and Provision Coverage Ratio. The scores above are shown for reference only and should not drive investment decisions on this stock."

    score = acct.get('score', 0) or 0
    pe = layer1.get('pe')
    roe = layer1.get('roe')
    cfo_pat = acct.get('cum_cfo_pat')
    paras = []

    # Paragraph 1: Overall assessment
    if tier == 'FULL':
        paras.append(f"**{name}** passes all 7 layers of the framework with conviction. This is a stock where accounting quality is pristine, valuation is attractive relative to growth, and earnings momentum is supportive. The data supports a full 12-15% portfolio allocation.")
    elif tier == 'STANDARD':
        paras.append(f"**{name}** is a solid business with good fundamentals but one or more factors prevent full conviction. This could be slightly rich valuation, a cyclical earnings peak, or growth that's harder to measure. The data supports an 8-10% position — a good business at a fair price.")
    elif tier == 'HALF':
        paras.append(f"**{name}** shows mixed signals across the framework. There are genuine positives here, but also flags that need resolution before committing significant capital. The data supports a cautious 4-6% position, with additional capital deployed only after the next quarterly results confirm improvement in the flagged areas.")
    else:
        paras.append(f"**{name}** fails critical checks in this framework. The data does not support deploying capital at this time, regardless of how attractive the stock may appear on other metrics. Monitor quarterly and re-evaluate when the specific issues flagged below improve.")

    # Paragraph 2: What's working
    positives = []
    if roe and roe > 20:
        positives.append(f"ROE of {roe}% indicates strong capital efficiency")
    if cfo_pat and cfo_pat > 1.0:
        positives.append(f"exceptional cash generation — cumulative CFO/PAT of {cfo_pat}x means the company generates more cash than it reports as profit")
    elif cfo_pat and cfo_pat > 0.7:
        positives.append(f"healthy cash conversion at {cfo_pat}x cumulative CFO/PAT")
    if peg and not np.isnan(peg) and peg < 0.5:
        positives.append(f"deeply undervalued on a growth-adjusted basis at PEG {peg}")
    elif peg and not np.isnan(peg) and peg < 1.0:
        positives.append(f"attractively valued at PEG {peg}")
    if layer1.get('sales_cagr') and layer1['sales_cagr'] > 20:
        positives.append(f"strong revenue growth at {layer1['sales_cagr']}% CAGR")
    if ret_1y and ret_1y > 20:
        positives.append(f"positive market sentiment with {ret_1y:+.1f}% 1Y return")

    if positives:
        paras.append("**What's working:** " + ", ".join(positives) + ".")

    # Paragraph 3: What's concerning
    concerns = []
    if score < 70:
        concerns.append(f"accounting quality score of {score}/100 with {acct.get('num_flags', 0)} flag(s)")
    if peg and not np.isnan(peg) and peg > 2.0:
        concerns.append(f"PEG of {peg} means the stock is overvalued relative to its growth")
    if pe and pe > 80:
        concerns.append(f"PE of {pe:.0f} triggers the Adani Filter — extremely high risk")
    if cyclical.get('cyclical_peak'):
        concerns.append(f"ROE at cyclical peak ({cyclical['latest_roe']}% vs normalized {cyclical['median_roe']}%) — current earnings may not be sustainable")
    if ret_1y and ret_1y < -30:
        concerns.append(f"stock down {abs(ret_1y):.0f}% in 1 year — the market is pricing in something negative")
    if momentum.get('available'):
        lq = momentum.get('latest_qoq', 0) or 0
        pq = momentum.get('prior_qoq', 0) or 0
        if lq < 0 and pq < 0:
            concerns.append(f"two consecutive quarters of declining EPS ({lq:+.1f}% and {pq:+.1f}% QoQ)")
    for flag in acct.get('flags', [])[:2]:
        concerns.append(flag.lower())

    if concerns:
        paras.append("**What needs attention:** " + ", ".join(concerns) + ".")

    # Paragraph 4: Action
    if tier == 'FULL':
        paras.append(f"**Action:** Deploy 12-15% of portfolio. Read the latest concall transcript to confirm management guidance aligns with what the numbers show.")
    elif tier == 'STANDARD':
        paras.append(f"**Action:** Position at 8-10%. This is a hold-and-compound stock, not a trade. Re-evaluate sizing after next quarterly results.")
    elif tier == 'HALF':
        paras.append(f"**Action:** Half position at 4-6% only. Wait for Q2 results to see if the flagged issues improve before adding more capital.")
    else:
        paras.append(f"**Action:** Do not buy. Add to quarterly watchlist and re-screen when the next results are published. The specific flags above need to improve before this stock deserves capital.")

    return "\n\n".join(paras)


# ============================================================
# BATCH SCREENER
# ============================================================
def analyse_quick(ticker_str):
    try:
        data = fetch_stock_data(ticker_str)
        if not data or not data['info']:
            return None
        info = data['info']
        if is_banking_stock(info):
            return None

        layer1 = run_layer1(info, data['financials'], data['balance_sheet'])
        if not layer1['mcap_pass'] or not layer1['de_pass']:
            return None
        if layer1['roe'] is not None and layer1['roe'] < 15:
            return None

        multi = get_multi_year_data(data['financials'], data['balance_sheet'], data['cashflow'])
        if not multi:
            return None
        acct = run_accounting_quality(multi)
        cyc = run_cyclical_check(multi)
        peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))
        ret = get_1y_return(ticker_str)
        tier, size = get_position_size(acct['score'], acct['num_flags'], cyc.get('cyclical_peak', False), peg, ret)

        return {
            'ticker': ticker_str.replace('.NS', ''), 'name': info.get('shortName', ''),
            'sector': info.get('sector', 'N/A'), 'price': info.get('currentPrice', info.get('regularMarketPrice')),
            'pe': layer1.get('pe'), 'peg': peg, 'roe': layer1.get('roe'), 'roce': layer1.get('roce'),
            'acct_score': acct['score'], 'num_flags': acct['num_flags'],
            'cum_cfo_pat': acct.get('cum_cfo_pat'), 'cyclical_peak': cyc.get('cyclical_peak', False),
            'ret_1y': ret, 'tier': tier, 'size': size,
            'sales_cagr': layer1.get('sales_cagr'), 'pat_cagr': layer1.get('pat_cagr')
        }
    except:
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
    "Stock": ["LUPIN","DIXON","ENRIN","BSE","MCX","ICICI AMC","EICHER","KPIT","POLYCAB","HDFC AMC"],
    "Tier": ["FULL","FULL","FULL","STD","STD","STD","STD","HALF","HALF","HALF"],
    "Score": [100,100,100,85,85,85,85,100,55,70],
    "PEG": [0.14,0.54,0.91,0.38,0.51,1.46,1.60,1.48,1.67,1.42]
}), hide_index=True, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Vinayak Nagral · Sep 2026")


# ============================================================
# PAGE 1: SINGLE STOCK
# ============================================================
if page == "Single Stock":
    c1, c2 = st.columns([3,1])
    with c1:
        ticker_input = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="LUPIN, BSE, DIXON, TCS, HCLTECH").strip().upper()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔍 Analyse", type="primary", use_container_width=True)

    qcols = st.columns(7)
    for i, qt in enumerate(["LUPIN","BSE","DIXON","KPITTECH","HDFCAMC","MAZDOCK","HCLTECH"]):
        with qcols[i]:
            if st.button(qt, key=f"q_{qt}", use_container_width=True):
                ticker_input = qt
                go = True

    if not ticker_input.endswith(".NS"):
        ticker_input += ".NS"

    if go:
        with st.spinner(f"Analysing {ticker_input.replace('.NS','')}..."):
            sd = fetch_stock_data(ticker_input)
        if not sd:
            st.error("Could not fetch data. Check ticker.")
            st.stop()

        info = sd['info']
        fin, bs, cf, qfin = sd['financials'], sd['balance_sheet'], sd['cashflow'], sd['quarterly_financials']
        name = info.get('shortName', ticker_input)
        sector = info.get('sector', 'N/A')
        price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        is_bank = is_banking_stock(info, name)

        st.markdown(f"## {name}")
        st.markdown(f"**{ticker_input}** · {sector} · ₹{price}")

        if is_bank:
            st.markdown('<div class="banking-box">⚠️ <strong>Banking/Financial Stock.</strong> This framework is designed for non-financial companies. Banks need NIM, CASA, Credit Cost, GNPA metrics. Scores shown for reference only.</div>', unsafe_allow_html=True)

        st.markdown("---")

        with st.spinner("Running all 7 layers..."):
            layer1 = run_layer1(info, fin, bs)
            multi = get_multi_year_data(fin, bs, cf)
            acct = run_accounting_quality(multi) if multi else {'score': None, 'flags': ['Insufficient data'], 'num_flags': 0, 'cum_cfo_pat': None, 'cum_cfo_ebitda': None}
            cyclical = run_cyclical_check(multi) if multi else {'cyclical_peak': False, 'roe_by_year': {}}
            peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))
            momentum = run_momentum_check(qfin)
            ret_1y = get_1y_return(ticker_input)
        tier, size = get_position_size(acct['score'], acct['num_flags'], cyclical.get('cyclical_peak', False), peg, ret_1y)
            if not layer1['phase1_pass'] and tier in ['FULL', 'STANDARD']:
                tier, size = 'HALF', '4-6% (fails Phase I fundamentals)' if acct.get('score') is not None else ('N/A', 'Insufficient data')

        # METRICS ROW
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Score", f"{acct.get('score', 'N/A')}/100")
        m2.metric("PE", safe_fmt(layer1.get('pe'), '.1f'))
        m3.metric("PEG", safe_fmt(peg, '.2f'))
        m4.metric("ROE", safe_fmt(layer1.get('roe'), '.1f', '%'))
        m5.metric("1Y Return", safe_fmt(ret_1y, '+.1f', '%'))
        with m6:
            st.markdown("**Tier**")
            tc = f"tier-{tier.lower()}" if tier in ['FULL','STANDARD','HALF','WATCH'] else ""
            st.markdown(f'<p class="{tc}">{tier} — {size}</p>', unsafe_allow_html=True)

        st.markdown("---")

        # LAYER-BY-LAYER BREAKDOWN
        st.subheader("Layer-by-Layer Breakdown")
        layers = generate_layer_breakdown(layer1, acct, cyclical, peg, momentum, ret_1y, is_bank)
        for lname, status, detail in layers:
            css = "layer-pass" if status == "pass" else "layer-fail" if status == "fail" else "layer-warn"
            icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
            st.markdown(f'<div class="{css}">{icon} <strong>{lname}:</strong> {detail}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # VERDICT
        st.subheader("Investment Verdict")
        verdict = generate_verdict_text(name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank)
        st.markdown(f'<div class="verdict-section">{verdict}</div>', unsafe_allow_html=True)

        # DETAILED TABS
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
                    if v is not None:
                        st.markdown(f"{'✅' if v >= 0.7 else '⚠️' if v >= 0.5 else '❌'} **{v}x** {'(healthy)' if v >= 0.7 else '(weak)' if v >= 0.5 else '(critical)'}")
                    sv = acct.get('single_yr_cfo_pat')
                    if sv is not None:
                        st.caption(f"Latest single year: {sv}x")
                with cb:
                    st.markdown("**Cumulative CFO/EBITDA**")
                    v = acct.get('cum_cfo_ebitda')
                    if v is not None:
                        st.markdown(f"{'✅' if v >= 0.7 else '⚠️' if v >= 0.5 else '❌'} **{v}x** {'(healthy)' if v >= 0.7 else '(weak)' if v >= 0.5 else '(critical)'}")
                    t = acct.get('cfo_ebitda_trend', [])
                    if t: st.caption(f"Trend: {' → '.join(str(x) for x in t)}")

                if acct['flags']:
                    for f in acct['flags']:
                        st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="clean-item">✅ All forensic checks passed</div>', unsafe_allow_html=True)

                recv = acct.get('recv_pcts', [])
                yrs = acct.get('recv_years', [])
                if recv and any(r is not None for r in recv):
                    st.markdown("**Receivables % of Revenue:**")
                    st.dataframe(pd.DataFrame({'Year': yrs, 'Recv%': [f"{r:.1f}%" if r else "N/A" for r in recv]}), hide_index=True, use_container_width=True)

        with tab2:
            cc, cd = st.columns(2)
            with cc:
                st.subheader("Cyclical ROE")
                if cyclical.get('roe_by_year'):
                    st.dataframe(pd.DataFrame(list(cyclical['roe_by_year'].items()), columns=['Year','ROE%']), hide_index=True, use_container_width=True)
                    st.markdown(f"Latest **{cyclical.get('latest_roe')}%** · Avg **{cyclical.get('avg_roe')}%** · Normalized **{cyclical.get('median_roe')}%**")
                    if cyclical.get('cyclical_peak'): st.error("⚠️ CYCLICAL PEAK")
                    else: st.success("✅ Not at peak")
            with cd:
                st.subheader("Valuation")
                st.markdown(f"**PE:** {safe_fmt(layer1.get('pe'), '.1f')} · **PEG:** {safe_fmt(peg, '.2f')}")
                if peg and not np.isnan(peg):
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
                    st.dataframe(pd.DataFrame({'Quarter': momentum['quarters'], 'EPS': [round(e,2) for e in momentum['eps_values']]}), hide_index=True, use_container_width=True)
            else:
                st.warning("Insufficient quarterly data")
            if ret_1y is not None:
                st.markdown(f"**1Y Price Return:** {ret_1y:+.1f}%")

        with tab4:
            st.subheader("All Layers Summary")
            st.dataframe(pd.DataFrame({
                "Layer": ["Fundamentals","Cash Flow","Forensic","PEG","Momentum","Cyclical","Sizing"],
                "Result": [
                    "PASS ✅" if layer1['phase1_pass'] else "FAIL ❌",
                    f"CFO/PAT: {safe_fmt(acct.get('cum_cfo_pat'),'.2f')}x",
                    f"{acct.get('score','N/A')}/100 ({acct.get('num_flags',0)} flags)",
                    safe_fmt(peg, '.2f'),
                    f"QoQ: {safe_fmt(momentum.get('latest_qoq'),'+.1f')}%" if momentum.get('available') else "N/A",
                    "PEAK ⚠️" if cyclical.get('cyclical_peak') else "OK ✅",
                    f"{tier} — {size}"
                ]
            }), hide_index=True, use_container_width=True)

        st.caption("For research only, not investment advice · Data from Yahoo Finance")


# ============================================================
# PAGE 2: AUTO TOP 10
# ============================================================
elif page == "Auto Top 10":
    st.subheader("🏆 Automatic Top 10 Picker")
    st.markdown("Screens Nifty 200 through all 7 layers. Banking/insurance auto-excluded.")

    c1, c2 = st.columns(2)
    with c1: max_stocks = st.slider("Stocks to screen", 20, 200, 50, 10, help="50 ≈ 5 min, 200 ≈ 20 min")
    with c2: top_n = st.slider("Show top N", 5, 20, 10)

    if st.button("🚀 Run Full Screen", type="primary", use_container_width=True):
        tickers = fetch_nifty200_tickers()[:max_stocks]
        if not tickers:
            st.error("Could not fetch Nifty 200.")
            st.stop()

        results = []
        prog = st.progress(0, "Starting...")
        for i, t in enumerate(tickers):
            prog.progress((i+1)/len(tickers), f"Analysing {t.replace('.NS','')} ({i+1}/{len(tickers)})")
            r = analyse_quick(t)
            if r: results.append(r)
        prog.empty()

        if not results:
            st.error("No stocks passed.")
            st.stop()

        df = pd.DataFrame(results)

        def peg_score(x):
            if x is None: return 10
            try:
                if np.isnan(x): return 10
            except: pass
            if x < 0.3: return 100
            if x < 0.5: return 90
            if x < 1.0: return 75
            if x < 1.5: return 55
            if x < 2.0: return 30
            if x < 3.0: return 10
            return 0

        def mom_score(x):
            if x is None: return 40
            try:
                if np.isnan(x): return 40
            except: pass
            if x > 30: return 80
            if x > 0: return 65
            if x > -20: return 40
            if x > -40: return 20
            return 5

        def cfo_s(x):
            if x is None: return 20
            try:
                if np.isnan(x): return 20
            except: pass
            if x > 1.2: return 100
            if x > 0.9: return 85
            if x > 0.7: return 65
            if x > 0.5: return 40
            return 10

        df['rank_score'] = (
            df['acct_score'].fillna(0) * 0.25 +
            df['peg'].apply(peg_score) * 0.30 +
            df['roe'].fillna(0) * 0.10 +
            df['ret_1y'].apply(mom_score) * 0.15 +
            df['cum_cfo_pat'].apply(cfo_s) * 0.20
        )
        df = df.sort_values('rank_score', ascending=False)
        top = df.head(top_n)

        st.markdown(f"### Top {top_n} from {len(tickers)} screened ({len(results)} passed filters)")

        for idx, row in top.iterrows():
            rank = list(top.index).index(idx) + 1
            tc = {"FULL":"🟢","STANDARD":"🔵","HALF":"🟡","WATCH":"🔴"}.get(row['tier'],"⚪")
            peg_d = safe_fmt(row['peg'], '.2f')

            with st.expander(f"**#{rank} · {row['ticker']}** — {row['name']} · {tc} {row['tier']} · Score {row['acct_score']}/100 · PEG {peg_d}", expanded=(rank <= 3)):
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                c1.metric("Price", f"₹{row['price']:,.0f}" if row['price'] else "N/A")
                c2.metric("PE", safe_fmt(row['pe'], '.1f'))
                c3.metric("PEG", safe_fmt(row['peg'], '.2f'))
                c4.metric("ROE", safe_fmt(row['roe'], '.1f', '%'))
                c5.metric("1Y Ret", safe_fmt(row['ret_1y'], '+.1f', '%'))
                c6.metric("CFO/PAT", safe_fmt(row['cum_cfo_pat'], '.2f', 'x'))
                st.markdown(f"**Sector:** {row['sector']} · **Sales CAGR:** {safe_fmt(row['sales_cagr'], '.1f')}% · **PAT CAGR:** {safe_fmt(row['pat_cagr'], '.1f')}% · **Flags:** {row['num_flags']} · **Cyclical Peak:** {'Yes ⚠️' if row['cyclical_peak'] else 'No'}")
                st.markdown(f"**Position:** {row['tier']} — {row['size']}")

        st.markdown("---")
        st.markdown("### Full Rankings")
        dd = df[['ticker','name','acct_score','pe','peg','roe','cum_cfo_pat','ret_1y','tier','size']].copy()
        dd['peg'] = dd['peg'].apply(lambda x: round(x,2) if pd.notna(x) else None)
        dd['cum_cfo_pat'] = dd['cum_cfo_pat'].apply(lambda x: round(x,2) if pd.notna(x) else None)
        dd.columns = ['Ticker','Name','Score','PE','PEG','ROE%','CFO/PAT','1Y Ret%','Tier','Size']
        dd = dd.reset_index(drop=True)
        dd.index += 1
        st.dataframe(dd, use_container_width=True)

        st.markdown("### Tier Breakdown")
        tc1,tc2,tc3,tc4 = st.columns(4)
        for cw, tn, em in [(tc1,'FULL','🟢'),(tc2,'STANDARD','🔵'),(tc3,'HALF','🟡'),(tc4,'WATCH','🔴')]:
            ts = df[df['tier']==tn]
            with cw:
                st.markdown(f"**{em} {tn}** ({len(ts)})")
                for _,r in ts.iterrows():
                    st.caption(f"{r['ticker']} · {r['acct_score']}")


# ============================================================
# PAGE 3: HOW IT WORKS
# ============================================================
# ============================================================
# PAGE 3: LIVE VALIDATION TRACKER
# ============================================================
elif page == "Live Tracker":
    st.subheader("📊 Live Framework Validation")
    st.markdown("Tracking **BUY picks vs AVOID picks** from September 1, 2026 to prove the framework works forward, not just backward.")

    # Baseline: prices on Sep 1, 2026 (the day framework was built)
    # UPDATE THESE with actual closing prices on your start date
    baseline_date = "2026-09-01"

    buy_picks = {
        "LUPIN.NS":    {"name": "Lupin", "tier": "FULL", "score": 100, "peg": 0.14, "reason": "PEG 0.14, Score 100, pharma compounder with 30% EBITDA margins"},
        "DIXON.NS":    {"name": "Dixon Technologies", "tier": "FULL", "score": 100, "peg": 0.54, "reason": "India's EMS champion, PLI 2.0 beneficiary, clean cash flows"},
        "BSE.NS":      {"name": "BSE Limited", "tier": "STANDARD", "score": 85, "peg": 0.38, "reason": "Capital market monopoly, PEG 0.38, 64% revenue growth"},
        "EICHERMOT.NS":{"name": "Eicher Motors", "tier": "STANDARD", "score": 85, "peg": 1.60, "reason": "Royal Enfield pricing power, defensive anchor"},
        "KPITTECH.NS": {"name": "KPIT Technologies", "tier": "HALF", "score": 100, "peg": 1.48, "reason": "Score 100 but -51% momentum, contrarian half position"},
    }

    avoid_picks = {
        "GODFRYPHLP.NS": {"name": "Godfrey Phillips", "tier": "WATCH", "score": 35, "peg": 0.78, "reason": "Structural tax reset, EBITDA margin collapsed 18.6% to 4.75%"},
        "WAAREEENER.NS": {"name": "Waaree Energies", "tier": "WATCH", "score": 45, "peg": 0.20, "reason": "Cumulative CFO/PAT 0.44x — profits not converting to cash"},
        "MAZDOCK.NS":    {"name": "Mazagon Dock", "tier": "WATCH", "score": 25, "peg": 1.16, "reason": "Negative cumulative CFO, lowest score in universe"},
    }

    benchmark = "^NSEI"

    all_tickers = list(buy_picks.keys()) + list(avoid_picks.keys()) + [benchmark]

    # Fetch baseline prices
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_tracker_prices(tickers, base_date):
        results = {}
        end = datetime.now()
        start = datetime.strptime(base_date, "%Y-%m-%d") - timedelta(days=5)

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Baseline price (closest to baseline date)
                base_dt = pd.Timestamp(base_date)
                mask = data.index <= base_dt
                if mask.sum() > 0:
                    base_price = float(data.loc[mask, 'Close'].iloc[-1])
                else:
                    base_price = float(data['Close'].iloc[0])

                current_price = float(data['Close'].iloc[-1])
                ret = round((current_price / base_price - 1) * 100, 2)

                # Get price history for chart
                history = data[data.index >= base_dt]['Close'].copy()
                if not history.empty:
                    history = (history / history.iloc[0] - 1) * 100

                results[ticker] = {
                    'base_price': round(base_price, 2),
                    'current_price': round(current_price, 2),
                    'return_pct': ret,
                    'history': history
                }
            except:
                continue
        return results

    with st.spinner("Fetching live prices..."):
        prices = get_tracker_prices(all_tickers, baseline_date)

    if not prices:
        st.error("Could not fetch prices. Try again.")
        st.stop()

    # Benchmark return
    bench_ret = prices.get(benchmark, {}).get('return_pct', 0)

    # Build comparison table
    st.markdown(f"### Performance Since {baseline_date}")
    st.markdown(f"**Nifty 50 (Benchmark):** {bench_ret:+.2f}%")
    st.markdown("---")

    # BUY PICKS
    st.markdown("### 🟢 BUY Picks (Framework said deploy capital)")
    buy_rows = []
    for ticker, info in buy_picks.items():
        p = prices.get(ticker, {})
        ret = p.get('return_pct', None)
        alpha = round(ret - bench_ret, 2) if ret is not None else None
        buy_rows.append({
            'Stock': info['name'],
            'Tier': info['tier'],
            'Score': info['score'],
            'PEG': info['peg'],
            'Entry Price': f"₹{p.get('base_price', 'N/A')}",
            'Current Price': f"₹{p.get('current_price', 'N/A')}",
            'Return': f"{ret:+.2f}%" if ret is not None else "N/A",
            'vs Nifty': f"{alpha:+.2f}%" if alpha is not None else "N/A",
            'Thesis': info['reason']
        })

    buy_df = pd.DataFrame(buy_rows)
    st.dataframe(buy_df, hide_index=True, use_container_width=True)

    buy_returns = [prices[t]['return_pct'] for t in buy_picks if t in prices and prices[t].get('return_pct') is not None]
    if buy_returns:
        avg_buy = round(np.mean(buy_returns), 2)
        st.markdown(f"**Average BUY return: {avg_buy:+.2f}%** (vs Nifty {bench_ret:+.2f}%)")

    st.markdown("---")

    # AVOID PICKS
    st.markdown("### 🔴 AVOID Picks (Framework said do not deploy)")
    avoid_rows = []
    for ticker, info in avoid_picks.items():
        p = prices.get(ticker, {})
        ret = p.get('return_pct', None)
        alpha = round(ret - bench_ret, 2) if ret is not None else None
        avoid_rows.append({
            'Stock': info['name'],
            'Tier': info['tier'],
            'Score': info['score'],
            'PEG': info['peg'],
            'Entry Price': f"₹{p.get('base_price', 'N/A')}",
            'Current Price': f"₹{p.get('current_price', 'N/A')}",
            'Return': f"{ret:+.2f}%" if ret is not None else "N/A",
            'vs Nifty': f"{alpha:+.2f}%" if alpha is not None else "N/A",
            'Reason Avoided': info['reason']
        })

    avoid_df = pd.DataFrame(avoid_rows)
    st.dataframe(avoid_df, hide_index=True, use_container_width=True)

    avoid_returns = [prices[t]['return_pct'] for t in avoid_picks if t in prices and prices[t].get('return_pct') is not None]
    if avoid_returns:
        avg_avoid = round(np.mean(avoid_returns), 2)
        st.markdown(f"**Average AVOID return: {avg_avoid:+.2f}%** (vs Nifty {bench_ret:+.2f}%)")

    # VERDICT
    st.markdown("---")
    st.markdown("### Framework Validation Verdict")

    if buy_returns and avoid_returns:
        spread = round(avg_buy - avg_avoid, 2)
        alpha_vs_nifty = round(avg_buy - bench_ret, 2)

        if spread > 0 and alpha_vs_nifty > 0:
            st.success(f"""
            **✅ Framework is working.**

            BUY picks: **{avg_buy:+.2f}%** · AVOID picks: **{avg_avoid:+.2f}%** · Spread: **{spread:+.2f}%**

            BUY picks outperformed AVOID picks by {spread:.1f} percentage points and beat the Nifty by {alpha_vs_nifty:.1f} percentage points.
            The quality + valuation + momentum framework is generating alpha in live, forward-looking tracking.
            """)
        elif spread > 0:
            st.info(f"""
            **📊 Partial validation.**

            BUY picks ({avg_buy:+.2f}%) outperformed AVOID picks ({avg_avoid:+.2f}%) by {spread:.1f}pp.
            However, BUY picks are {'outperforming' if alpha_vs_nifty > 0 else 'underperforming'} the Nifty by {abs(alpha_vs_nifty):.1f}pp.
            The framework separates good from bad but {'is' if alpha_vs_nifty > 0 else 'is not yet'} generating absolute alpha.
            """)
        else:
            st.warning(f"""
            **⚠️ Framework not validated yet.**

            BUY picks ({avg_buy:+.2f}%) are currently underperforming AVOID picks ({avg_avoid:+.2f}%).
            Spread: {spread:+.2f}pp. This could be early-stage noise (give it 3-6 months)
            or could indicate the framework needs recalibration.
            Continue monitoring — don't change the methodology based on less than one quarter of data.
            """)

    # CHART: Cumulative returns
    st.markdown("---")
    st.markdown("### Cumulative Return Chart")

    chart_data = pd.DataFrame()
    for ticker in list(buy_picks.keys()) + list(avoid_picks.keys()) + [benchmark]:
        p = prices.get(ticker)
        if p and 'history' in p and not p['history'].empty:
            label = buy_picks.get(ticker, avoid_picks.get(ticker, {})).get('name', 'Nifty 50')
            chart_data[label] = p['history']

    if not chart_data.empty:
        st.line_chart(chart_data)
        st.caption("Returns indexed to 0% on baseline date. Lines above 0 = gain, below = loss.")

    st.markdown("---")
    st.markdown("""
    **How to read this page:**

    This is a forward-looking validation of the framework. On September 1, 2026, the framework identified
    specific stocks as BUY (clean accounting, cheap valuation, positive momentum) and AVOID (poor cash conversion,
    structural problems, or broken thesis). This page tracks whether those calls were right.

    If BUY picks consistently outperform AVOID picks by a meaningful margin over 3-6 months, the framework
    has predictive power — it's not just a nice-looking table, it actually identifies future winners and losers.

    Updated live every time you visit this page.
    """)
    st.caption("Tracking started September 1, 2026 · Updated live via Yahoo Finance")
elif page == "How It Works":
    st.subheader("How This Framework Works")
    st.markdown("""
This dashboard runs a **7-layer systematic analysis** on any NSE-listed stock, designed to find high-quality compounders while avoiding value traps.

**Layer 1 — Quantitative Screen:** Market Cap > ₹15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, 3Y Sales & PAT CAGR > 15%. Filters the Nifty 200 to a manageable shortlist.

**Layer 2 — Cash Flow Quality:** CFO/EBITDA > 70%. Ensures reported profits generate actual cash.

**Layer 3 — Forensic Accounting:** Four checks using **cumulative multi-year** data (not single-year). Catches receivables stuffing, inventory bloat, and fake profits that single-year ratios miss.

**Layer 4 — PEG Valuation:** PE ÷ earnings growth. A stock can have a high PE and still be cheap if growth justifies it. "Adani Filter" flags PE > 80 as high risk.

**Layer 5 — Earnings Momentum:** Last 2 quarters QoQ EPS change. Catches stocks that pass backward-looking tests but show fresh deterioration.

**Layer 6 — Cyclical ROE Normalization:** If current ROE > 2x historical average, the stock is at "cyclical peak." Use median ROE for valuation, not the inflated current figure.

**Layer 7 — Position Sizing v2:** Integrates quality + PEG + momentum. Clean + cheap = FULL (12-15%). Clean + expensive = STANDARD (8-10%). Flags present = HALF (4-6%). Failed quality = WATCH (0%).

---

### What This Framework Does NOT Cover

**Banking, NBFC, Insurance stocks** need NIM, CASA, Credit Cost, GNPA metrics. Auto-detected and flagged.

**Commodity producers** where revenue is price-driven. CAGR filter creates false signals across cycles.

---

### Framework Score: 90/100

The remaining 10 points require point-in-time backtesting (paid data), management quality scoring, and a separate financials framework.
    """)
    st.caption("Built by Vinayak Nagral · September 2026")
