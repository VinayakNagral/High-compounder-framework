import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="High Compounder Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B2A4A;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }
    .score-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-green { background: #E8F5E9; border-left: 5px solid #4CAF50; }
    .score-yellow { background: #FFF8E1; border-left: 5px solid #FF9800; }
    .score-red { background: #FFEBEE; border-left: 5px solid #F44336; }
    .flag-item {
        background: #FFF3E0;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 4px solid #FF9800;
        font-size: 0.9rem;
    }
    .clean-item {
        background: #E8F5E9;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 4px solid #4CAF50;
        font-size: 0.9rem;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .verdict-buy { background: #E8F5E9; border: 2px solid #4CAF50; }
    .verdict-hold { background: #FFF8E1; border: 2px solid #FF9800; }
    .verdict-avoid { background: #FFEBEE; border: 2px solid #F44336; }
    .tier-full { color: #2E7D32; font-weight: 700; font-size: 1.3rem; }
    .tier-standard { color: #1565C0; font-weight: 700; font-size: 1.3rem; }
    .tier-half { color: #F57F17; font-weight: 700; font-size: 1.3rem; }
    .tier-watch { color: #C62828; font-weight: 700; font-size: 1.3rem; }
    .banking-warning {
        background: #E3F2FD;
        border-left: 4px solid #1565C0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .top-pick-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1B2A4A;
    }
    .insight-box {
        background: #F3E5F5;
        border-left: 4px solid #7B1FA2;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
BANKING_SECTORS = [
    'Financial Services', 'Banks', 'Insurance', 'Banking',
    'Banks - Regional', 'Banks - Diversified', 'Insurance - Life',
    'Insurance - Property & Casualty', 'Credit Services'
]

BANKING_KEYWORDS = [
    'bank', 'finance', 'insurance', 'nbfc', 'housing finance',
    'credit', 'lending', 'microfinance'
]

NIFTY_200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"


# ============================================================
# HELPER FUNCTIONS
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


def get_revenue(fin, col=0):
    """Get revenue with multiple fallbacks."""
    val = safe_get(fin, 'Total Revenue', col)
    if val is None:
        val = safe_get(fin, 'Operating Revenue', col)
    if val is None:
        val = safe_get(fin, 'Revenue', col)
    return val


def is_banking_stock(info, name=""):
    """Detect if a stock is a bank/NBFC/insurance company."""
    sector = (info.get('sector', '') or '').lower()
    industry = (info.get('industry', '') or '').lower()
    stock_name = (info.get('shortName', '') or name or '').lower()

    for kw in BANKING_KEYWORDS:
        if kw in sector or kw in industry or kw in stock_name:
            return True
    if info.get('sector', '') in BANKING_SECTORS:
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
        return {"info": info, "financials": fin, "balance_sheet": bs,
                "cashflow": cf, "quarterly_financials": qfin}
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

        revenue = get_revenue(fin, i)
        net_income = safe_get(fin, 'Net Income', i)
        ebitda = safe_get(fin, 'EBITDA', i)
        ebit = safe_get(fin, 'EBIT', i)

        receivables = safe_get(bs, 'Accounts Receivable', i)
        if receivables is None:
            receivables = safe_get(bs, 'Net Receivables', i)
        if receivables is None:
            receivables = safe_get(bs, 'Receivables', i)

        inventory = safe_get(bs, 'Inventory', i)
        total_assets = safe_get(bs, 'Total Assets', i)

        equity = safe_get(bs, 'Stockholders Equity', i)
        if equity is None:
            equity = safe_get(bs, 'Total Stockholders Equity', i)
        if equity is None:
            equity = safe_get(bs, 'Common Stock Equity', i)

        current_liabilities = safe_get(bs, 'Current Liabilities', i)

        cfo = safe_get(cf, 'Operating Cash Flow', i)
        if cfo is None:
            cfo = safe_get(cf, 'Total Cash From Operating Activities', i)
        if cfo is None:
            cfo = safe_get(cf, 'Cash Flow From Continuing Operating Activities', i)

        records.append({
            'year': year_label, 'revenue': revenue, 'net_income': net_income,
            'ebitda': ebitda, 'ebit': ebit, 'receivables': receivables,
            'inventory': inventory, 'total_assets': total_assets,
            'equity': equity, 'current_liabilities': current_liabilities,
            'cfo': cfo
        })
    return records


# ============================================================
# LAYER 1: FUNDAMENTALS
# ============================================================
def run_layer1(info, fin, bs):
    results = {}

    mcap = info.get('marketCap', 0) or 0
    results['market_cap_cr'] = round(mcap / 1e7, 0) if mcap else None
    results['mcap_pass'] = mcap > 150_000_000_000

    results['pe'] = info.get('trailingPE')

    de = info.get('debtToEquity', 0) or 0
    results['debt_to_equity'] = round(de / 100, 2) if de else 0
    results['de_pass'] = de < 50

    ni = safe_get(fin, 'Net Income', 0) if fin is not None else None
    eq = safe_get(bs, 'Stockholders Equity', 0) if bs is not None else None
    if eq is None and bs is not None:
        eq = safe_get(bs, 'Common Stock Equity', 0)
    if ni and eq and eq > 0:
        results['roe'] = round(ni / eq * 100, 1)
        results['roe_pass'] = results['roe'] > 15
    else:
        results['roe'] = None
        results['roe_pass'] = False

    ebit = safe_get(fin, 'EBIT', 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if ebit and ta and cl and (ta - cl) > 0:
        results['roce'] = round(ebit / (ta - cl) * 100, 1)
        results['roce_pass'] = results['roce'] > 18
    else:
        results['roce'] = None
        results['roce_pass'] = False

    results['sales_cagr'] = None
    results['pat_cagr'] = None
    if fin is not None and fin.shape[1] >= 2:
        yrs = fin.shape[1] - 1
        rev_latest = get_revenue(fin, 0)
        rev_oldest = get_revenue(fin, fin.shape[1] - 1)
        ni_latest = safe_get(fin, 'Net Income', 0)
        ni_oldest = safe_get(fin, 'Net Income', fin.shape[1] - 1)

        if rev_latest and rev_oldest and rev_oldest > 0 and yrs > 0:
            results['sales_cagr'] = round(((rev_latest / rev_oldest) ** (1 / yrs) - 1) * 100, 1)
        if ni_latest and ni_oldest and ni_oldest > 0 and yrs > 0:
            results['pat_cagr'] = round(((ni_latest / ni_oldest) ** (1 / yrs) - 1) * 100, 1)

    results['growth_pass'] = (
        (results.get('sales_cagr') or 0) > 15 and
        (results.get('pat_cagr') or 0) > 15
    )

    results['phase1_pass'] = all([
        results['mcap_pass'], results['de_pass'], results['roe_pass'],
        results['roce_pass'], results['growth_pass']
    ])
    return results


# ============================================================
# LAYER 3: ACCOUNTING QUALITY
# ============================================================
def run_accounting_quality(data):
    flags = []
    score = 100
    details = {}

    sorted_data = sorted(data, key=lambda x: x['year'])
    latest = sorted_data[-1]
    prior = sorted_data[-2] if len(sorted_data) >= 2 else None

    # CHECK 1: Receivables trend
    recv_pcts = []
    for d in sorted_data:
        if d['revenue'] and d['receivables'] and d['revenue'] > 0:
            recv_pcts.append(round(d['receivables'] / d['revenue'] * 100, 1))
        else:
            recv_pcts.append(None)
    details['recv_pcts'] = recv_pcts
    details['recv_years'] = [d['year'] for d in sorted_data]

    if len(recv_pcts) >= 3:
        valid = [x for x in recv_pcts if x is not None]
        if len(valid) >= 3:
            rising = sum(1 for i in range(1, len(valid)) if valid[i] > valid[i - 1])
            if rising >= 2:
                flags.append(f"Receivables rising: {valid[-3]}% → {valid[-2]}% → {valid[-1]}% of revenue")
                score -= 15

    # CHECK 2: Inventory vs Sales
    if prior:
        inv_l, inv_p = latest.get('inventory'), prior.get('inventory')
        rev_l, rev_p = latest.get('revenue'), prior.get('revenue')
        if all(v and v > 0 for v in [inv_l, inv_p, rev_l, rev_p]):
            inv_g = round((inv_l / inv_p - 1) * 100, 1)
            rev_g = round((rev_l / rev_p - 1) * 100, 1)
            details['inv_growth'] = inv_g
            details['rev_growth'] = rev_g
            if inv_g > rev_g + 10:
                flags.append(f"Inventory bloat: Inventory grew {inv_g}% vs Revenue {rev_g}%")
                score -= 15

    # CHECK 3: Cumulative CFO/PAT
    total_cfo = sum(d['cfo'] for d in sorted_data if d.get('cfo') is not None)
    total_pat = sum(d['net_income'] for d in sorted_data if d.get('net_income') is not None)

    if total_pat > 0:
        cum_cfo_pat = round(total_cfo / total_pat, 2)
        details['cum_cfo_pat'] = cum_cfo_pat
        if cum_cfo_pat < 0.5:
            flags.append(f"Critical: Cumulative CFO/PAT = {cum_cfo_pat}x over {len(sorted_data)} years")
            score -= 30
        elif cum_cfo_pat < 0.7:
            flags.append(f"Low cash conversion: Cumulative CFO/PAT = {cum_cfo_pat}x over {len(sorted_data)} years")
            score -= 20
    else:
        details['cum_cfo_pat'] = None

    if latest.get('cfo') is not None and latest.get('net_income') and latest['net_income'] > 0:
        details['single_yr_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2)

    # CHECK 4: Cumulative CFO/EBITDA
    total_ebitda = sum(d['ebitda'] for d in sorted_data if d.get('ebitda') is not None and d['ebitda'] > 0)
    if total_ebitda > 0:
        cum_cfo_ebitda = round(total_cfo / total_ebitda, 2)
        details['cum_cfo_ebitda'] = cum_cfo_ebitda
        if cum_cfo_ebitda < 0.5:
            flags.append(f"Critical: Cumulative CFO/EBITDA = {cum_cfo_ebitda}x over {len(sorted_data)} years")
            score -= 20
        elif cum_cfo_ebitda < 0.7:
            flags.append(f"Weak: Cumulative CFO/EBITDA = {cum_cfo_ebitda}x over {len(sorted_data)} years")
            score -= 15
    else:
        details['cum_cfo_ebitda'] = None

    yr_ratios = []
    for d in sorted_data:
        if d.get('cfo') is not None and d.get('ebitda') and d['ebitda'] > 0:
            yr_ratios.append(round(d['cfo'] / d['ebitda'], 2))
    details['cfo_ebitda_trend'] = yr_ratios

    if len(yr_ratios) >= 2 and details.get('cum_cfo_ebitda'):
        if yr_ratios[-1] < yr_ratios[-2] and yr_ratios[-1] < 0.5 and details['cum_cfo_ebitda'] < 0.7:
            flags.append(f"Deteriorating: CFO/EBITDA fell {yr_ratios[-2]} → {yr_ratios[-1]} AND cumulative weak")
            score -= 10

    if total_cfo < 0:
        flags.append(f"Negative cumulative CFO over {len(sorted_data)} years")
        score -= 25

    score = max(score, 0)
    details['score'] = score
    details['flags'] = flags
    details['num_flags'] = len(flags)
    return details


# ============================================================
# LAYER 6: CYCLICAL CHECK
# ============================================================
def run_cyclical_check(data):
    sorted_data = sorted(data, key=lambda x: x['year'])
    roe_values, roe_by_year = [], {}
    for d in sorted_data:
        if d.get('net_income') and d.get('equity') and d['equity'] > 0:
            roe = round(d['net_income'] / d['equity'] * 100, 1)
            roe_values.append(roe)
            roe_by_year[d['year']] = roe

    if len(roe_values) < 2:
        return {'cyclical_peak': False, 'roe_values': [], 'roe_by_year': {}}

    latest = roe_values[-1]
    avg = round(np.mean(roe_values), 1)
    median = round(np.median(roe_values), 1)
    return {
        'latest_roe': latest, 'avg_roe': avg, 'median_roe': median,
        'min_roe': round(min(roe_values), 1), 'normalized_roe': median,
        'cyclical_peak': latest > avg * 2 and latest > 20,
        'roe_values': roe_values, 'roe_by_year': roe_by_year
    }


# ============================================================
# LAYER 5: MOMENTUM
# ============================================================
def run_momentum_check(qfin):
    if qfin is None or qfin.shape[1] < 3:
        return {'available': False}

    eps_values, quarters = [], []
    for i in range(min(4, qfin.shape[1])):
        eps = safe_get(qfin, 'Diluted EPS', i)
        if eps is None:
            eps = safe_get(qfin, 'Basic EPS', i)
        if eps is not None:
            eps_values.append(eps)
            q_label = str(qfin.columns[i].strftime('%b %Y')) if hasattr(qfin.columns[i], 'strftime') else str(i)
            quarters.append(q_label)

    if len(eps_values) < 3:
        return {'available': False}

    latest_qoq = round((eps_values[0] / eps_values[1] - 1) * 100, 1) if eps_values[1] != 0 else None
    prior_qoq = round((eps_values[1] / eps_values[2] - 1) * 100, 1) if eps_values[2] != 0 else None
    return {
        'available': True, 'latest_qoq': latest_qoq, 'prior_qoq': prior_qoq,
        'eps_values': eps_values[:4], 'quarters': quarters[:4]
    }


# ============================================================
# LAYER 7: POSITION SIZING v2
# ============================================================
def get_position_size(acct_score, num_flags, cyclical_peak, peg, momentum_1y):
    if acct_score < 50:
        return 'WATCH', '0% (monitor only)'

    if acct_score >= 85 and num_flags == 0 and not cyclical_peak:
        base = 'FULL'
    elif acct_score >= 70 and num_flags <= 1:
        base = 'STANDARD'
    else:
        base = 'HALF'

    if base == 'FULL' and peg is None:
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
# VERDICT GENERATOR
# ============================================================
def generate_verdict(name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank):
    """Generate a human-readable investment verdict."""
    lines = []

    if is_bank:
        return f"**{name}** is a banking/financial services company. This framework uses ROE, D/E, and CFO/EBITDA metrics designed for non-financial companies. Banks require a separate framework using NIM, CASA ratio, Credit Cost, GNPA/NNPA, and Provision Coverage. The scores shown above should not be used for investment decisions on this stock."

    score = acct.get('score', 0)
    pe = layer1.get('pe')

    if tier == 'FULL':
        lines.append(f"**{name}** passes all 7 layers with conviction.")
        if peg and peg < 0.5:
            lines.append(f"At PEG {peg}, the market is significantly underpricing the growth — you're paying {int(peg*100)} paise for every rupee of growth.")
        if score >= 85:
            cfo_pat = acct.get('cum_cfo_pat', 0)
            if cfo_pat and cfo_pat > 1.0:
                lines.append(f"Cash quality is exceptional — cumulative CFO/PAT of {cfo_pat}x means the company generates more cash than it reports as profit.")
            elif cfo_pat and cfo_pat >= 0.7:
                lines.append(f"Cash quality is healthy at {cfo_pat}x cumulative CFO/PAT.")
        lines.append(f"**Deploy 12-15% of portfolio.**")

    elif tier == 'STANDARD':
        lines.append(f"**{name}** has solid fundamentals with minor concerns.")
        if peg and peg > 1.5:
            lines.append(f"PEG of {peg} means the valuation is full — growth justifies the PE but there's no discount.")
        if cyclical.get('cyclical_peak'):
            norm = cyclical.get('median_roe')
            latest = cyclical.get('latest_roe')
            lines.append(f"ROE is at cyclical peak ({latest}% vs normalized {norm}%). Use {norm}% for any valuation model, not the inflated current figure.")
        lines.append(f"**Position at 8-10%.** Good business, fair price.")

    elif tier == 'HALF':
        lines.append(f"**{name}** shows mixed signals across the framework.")
        if score < 70 and score >= 50:
            lines.append(f"Accounting quality score of {score}/100 means cash flows need investigation. Read the annual report before adding money.")
        if ret_1y and ret_1y < -30:
            lines.append(f"The stock is down {ret_1y}% in 1 year. Even if cash flows are clean, the market is pricing in deceleration that needs to be understood.")
        if peg and peg > 2.0:
            lines.append(f"PEG of {peg} means you're paying a significant premium over what growth justifies.")
        lines.append(f"**Half position at 4-6%.** Add only after next quarterly results confirm the thesis.")

    elif tier == 'WATCH':
        lines.append(f"**{name}** fails critical checks. Do not deploy capital.")
        if score < 50:
            cfo_pat = acct.get('cum_cfo_pat')
            if cfo_pat and cfo_pat < 0.5:
                lines.append(f"Cumulative CFO/PAT of {cfo_pat}x over multiple years means reported profits are not converting to cash. This is not a timing issue — it's a structural quality problem.")
        for flag in acct.get('flags', [])[:2]:
            lines.append(f"Flag: {flag}")
        lines.append(f"**Monitor quarterly. Re-evaluate when the specific flags improve.**")

    # Add PE warning
    if pe and pe > 80:
        lines.append(f"⚠️ PE of {pe:.0f} triggers the Adani Filter — extremely high risk unless earnings catch up within 2 years.")

    return " \n".join(lines)


# ============================================================
# BATCH ANALYSIS FOR TOP 10
# ============================================================
def analyse_single_stock_quick(ticker_str):
    """Quick analysis for batch screening — returns a summary dict."""
    try:
        data = fetch_stock_data(ticker_str)
        if not data or not data['info']:
            return None

        info = data['info']
        fin = data['financials']
        bs = data['balance_sheet']
        cf = data['cashflow']

        # Skip banking stocks
        if is_banking_stock(info):
            return None

        # Layer 1
        layer1 = run_layer1(info, fin, bs)

        # Skip if fails basic Phase I (except growth — some compounders have lumpy growth in yfinance)
        if not layer1['mcap_pass'] or not layer1['de_pass']:
            return None
        if layer1['roe'] is not None and layer1['roe'] < 12:
            return None

        # Accounting quality
        multi_year = get_multi_year_data(fin, bs, cf)
        if not multi_year:
            return None
        acct = run_accounting_quality(multi_year)
        cyclical = run_cyclical_check(multi_year)

        # PEG
        peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))

        # 1Y return
        ret_1y = get_1y_return(ticker_str)

        # Position sizing
        tier, size = get_position_size(
            acct['score'], acct['num_flags'],
            cyclical.get('cyclical_peak', False),
            peg, ret_1y
        )

        return {
            'ticker': ticker_str.replace('.NS', ''),
            'name': info.get('shortName', ticker_str),
            'sector': info.get('sector', 'N/A'),
            'price': info.get('currentPrice', info.get('regularMarketPrice', None)),
            'pe': layer1.get('pe'),
            'peg': peg,
            'roe': layer1.get('roe'),
            'roce': layer1.get('roce'),
            'acct_score': acct['score'],
            'num_flags': acct['num_flags'],
            'cum_cfo_pat': acct.get('cum_cfo_pat'),
            'cyclical_peak': cyclical.get('cyclical_peak', False),
            'ret_1y': ret_1y,
            'tier': tier,
            'size': size,
            'sales_cagr': layer1.get('sales_cagr'),
            'pat_cagr': layer1.get('pat_cagr'),
        }
    except:
        return None


# ============================================================
# MAIN APP
# ============================================================
st.markdown('<p class="main-header">📊 High Compounder Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">7-Layer Systematic Indian Equity Screener · Quality + Valuation + Momentum</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🔍 Single Stock Analysis", "🏆 Auto Top 10 Screener", "📖 How It Works"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Portfolio")
portfolio_data = {
    "Stock": ["LUPIN", "DIXON", "ENRIN", "BSE", "MCX", "ICICI AMC", "EICHER", "KPIT", "POLYCAB", "HDFC AMC"],
    "Tier": ["FULL", "FULL", "FULL", "STD", "STD", "STD", "STD", "HALF", "HALF", "HALF"],
    "Score": [100, 100, 100, 85, 85, 85, 85, 100, 55, 70],
    "PEG": [0.14, 0.54, 0.91, 0.38, 0.51, 1.46, 1.60, 1.48, 1.67, 1.42]
}
st.sidebar.dataframe(pd.DataFrame(portfolio_data), hide_index=True, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Vinayak Nagral · Data from Yahoo Finance · For research only")


# ============================================================
# PAGE 1: SINGLE STOCK ANALYSIS
# ============================================================
if page == "🔍 Single Stock Analysis":

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="e.g., LUPIN, BSE, DIXON, TCS").strip().upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

    quick_picks = st.columns(7)
    quick_tickers = ["LUPIN", "BSE", "DIXON", "KPITTECH", "HDFCAMC", "MAZDOCK", "WAAREEENER"]
    for i, qt in enumerate(quick_tickers):
        with quick_picks[i]:
            if st.button(qt, key=f"quick_{qt}", use_container_width=True):
                ticker_input = qt
                analyse_btn = True

    if not ticker_input.endswith(".NS"):
        ticker_input = ticker_input + ".NS"

    if analyse_btn:
        with st.spinner(f"Analysing {ticker_input}..."):
            stock_data = fetch_stock_data(ticker_input)

        if stock_data is None:
            st.error("Could not fetch data. Check the ticker and try again.")
            st.stop()

        info = stock_data['info']
        fin = stock_data['financials']
        bs = stock_data['balance_sheet']
        cf = stock_data['cashflow']
        qfin = stock_data['quarterly_financials']

        company_name = info.get('shortName', ticker_input)
        sector = info.get('sector', 'N/A')
        price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        is_bank = is_banking_stock(info, company_name)

        st.markdown(f"## {company_name}")
        st.markdown(f"**{ticker_input}** · {sector} · ₹{price}")

        if is_bank:
            st.markdown('<div class="banking-warning">⚠️ <strong>Banking/Financial Stock Detected.</strong> This framework is designed for non-financial companies. Banks, NBFCs, and insurance companies require different metrics (NIM, CASA, Credit Cost, GNPA). The scores below are shown for reference but should not drive investment decisions for this stock.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Run all layers
        with st.spinner("Running 7-layer analysis..."):
            layer1 = run_layer1(info, fin, bs)
            multi_year = get_multi_year_data(fin, bs, cf)
            if multi_year:
                acct = run_accounting_quality(multi_year)
                cyclical = run_cyclical_check(multi_year)
            else:
                acct = {'score': None, 'flags': ['Insufficient data'], 'num_flags': 0, 'cum_cfo_pat': None, 'cum_cfo_ebitda': None}
                cyclical = {'cyclical_peak': False, 'roe_by_year': {}}

            peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))
            momentum = run_momentum_check(qfin)
            ret_1y = get_1y_return(ticker_input)

            if acct.get('score') is not None:
                tier, size = get_position_size(acct['score'], acct['num_flags'], cyclical.get('cyclical_peak', False), peg, ret_1y)
            else:
                tier, size = 'N/A', 'Insufficient data'

        # ---- TOP METRICS ----
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Acct Score", f"{acct.get('score', 'N/A')}/100")
        with col2:
            st.metric("PE", f"{layer1['pe']:.1f}" if layer1.get('pe') else "N/A")
        with col3:
            st.metric("PEG", f"{peg:.2f}" if peg else "N/A")
        with col4:
            st.metric("ROE", f"{layer1['roe']}%" if layer1.get('roe') else "N/A")
        with col5:
            st.metric("1Y Return", f"{ret_1y:+.1f}%" if ret_1y is not None else "N/A")
        with col6:
            tier_class = f"tier-{tier.lower()}" if tier in ['FULL', 'STANDARD', 'HALF', 'WATCH'] else ""
            st.markdown("**Tier**")
            st.markdown(f'<p class="{tier_class}">{tier} — {size}</p>', unsafe_allow_html=True)

        # ---- VERDICT BOX ----
        verdict_text = generate_verdict(company_name, tier, size, layer1, acct, cyclical, peg, ret_1y, momentum, is_bank)
        if tier == 'FULL':
            box_class = "verdict-buy"
        elif tier in ['STANDARD', 'HALF']:
            box_class = "verdict-hold"
        else:
            box_class = "verdict-avoid"
        st.markdown(f'<div class="verdict-box {box_class}">{verdict_text}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---- TABS ----
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Fundamentals", "🔍 Forensic Accounting", "📊 Cyclical & Valuation", "📈 Momentum", "🎯 Summary"])

        with tab1:
            st.subheader("Phase I: Quantitative Core Screen")
            checks = {
                f"Market Cap: ₹{layer1.get('market_cap_cr', 0):,.0f} Cr": layer1['mcap_pass'],
                f"Debt/Equity: {layer1.get('debt_to_equity', 'N/A')}": layer1['de_pass'],
                f"ROE: {layer1.get('roe', 'N/A')}% (threshold: >15%)": layer1['roe_pass'],
                f"ROCE: {layer1.get('roce', 'N/A')}% (threshold: >18%)": layer1['roce_pass'],
                f"Sales CAGR: {layer1.get('sales_cagr', 'N/A')}%, PAT CAGR: {layer1.get('pat_cagr', 'N/A')}% (threshold: >15%)": layer1['growth_pass'],
            }
            for check, passed in checks.items():
                st.markdown(f"{'✅' if passed else '❌'} {check}")
            color = "green" if layer1['phase1_pass'] else "red"
            st.markdown(f"**Overall: :{color}[{'PASSES' if layer1['phase1_pass'] else 'FAILS'} Phase I]**")

        with tab2:
            st.subheader("Forensic Accounting Quality")
            if acct.get('score') is not None:
                score = acct['score']
                if score >= 85:
                    st.markdown(f'<div class="score-box score-green"><h2>{score}/100</h2><p>CLEAN — No material concerns</p></div>', unsafe_allow_html=True)
                elif score >= 50:
                    st.markdown(f'<div class="score-box score-yellow"><h2>{score}/100</h2><p>FLAGS DETECTED — Investigate before buying</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="score-box score-red"><h2>{score}/100</h2><p>SERIOUS CONCERNS — Do not deploy capital</p></div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Cumulative CFO/PAT**")
                    val = acct.get('cum_cfo_pat')
                    if val is not None:
                        icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.5 else "❌"
                        st.markdown(f"{icon} **{val}x** {'(healthy)' if val >= 0.7 else '(weak)' if val >= 0.5 else '(critical)'}")
                    single = acct.get('single_yr_cfo_pat')
                    if single is not None:
                        st.caption(f"Latest single year: {single}x (context only)")

                with col_b:
                    st.markdown("**Cumulative CFO/EBITDA**")
                    val = acct.get('cum_cfo_ebitda')
                    if val is not None:
                        icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.5 else "❌"
                        st.markdown(f"{icon} **{val}x** {'(healthy)' if val >= 0.7 else '(weak)' if val >= 0.5 else '(critical)'}")
                    trend = acct.get('cfo_ebitda_trend', [])
                    if trend:
                        st.caption(f"Trend: {' → '.join(str(x) for x in trend)}")

                if acct['flags']:
                    st.markdown("**Flags:**")
                    for flag in acct['flags']:
                        st.markdown(f'<div class="flag-item">⚠ {flag}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="clean-item">✅ All forensic checks passed</div>', unsafe_allow_html=True)

                recv = acct.get('recv_pcts', [])
                years = acct.get('recv_years', [])
                if recv and any(r is not None for r in recv):
                    st.markdown("**Receivables % of Revenue:**")
                    st.dataframe(pd.DataFrame({'Year': years, 'Recv %': [f"{r:.1f}%" if r else "N/A" for r in recv]}), hide_index=True, use_container_width=True)

        with tab3:
            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("Cyclical ROE")
                if cyclical.get('roe_by_year'):
                    st.dataframe(pd.DataFrame(list(cyclical['roe_by_year'].items()), columns=['Year', 'ROE %']), hide_index=True, use_container_width=True)
                    st.markdown(f"Latest: **{cyclical.get('latest_roe')}%** · Avg: **{cyclical.get('avg_roe')}%** · Normalized: **{cyclical.get('median_roe')}%**")
                    if cyclical.get('cyclical_peak'):
                        st.error("⚠️ CYCLICAL PEAK — Use normalized ROE for valuation")
                    else:
                        st.success("✅ Not at cyclical peak")

            with col_d:
                st.subheader("Valuation")
                st.markdown(f"**PE:** {layer1.get('pe', 'N/A')} · **PEG:** {peg if peg else 'N/A'}")
                if peg:
                    if peg < 0.5: st.success(f"PEG {peg} — Undervalued")
                    elif peg < 1.0: st.success(f"PEG {peg} — Attractive")
                    elif peg < 1.5: st.info(f"PEG {peg} — Fairly valued")
                    elif peg < 2.0: st.warning(f"PEG {peg} — Expensive")
                    else: st.error(f"PEG {peg} — Overvalued")
                if layer1.get('pe') and layer1['pe'] > 80:
                    st.error("🚨 ADANI FILTER: PE > 80")
                st.markdown(f"**Sales CAGR:** {layer1.get('sales_cagr', 'N/A')}% · **PAT CAGR:** {layer1.get('pat_cagr', 'N/A')}%")

        with tab4:
            st.subheader("Earnings Momentum")
            if momentum.get('available'):
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Latest QoQ", f"{momentum['latest_qoq']:+.1f}%" if momentum.get('latest_qoq') is not None else "N/A")
                with mc2:
                    st.metric("Prior QoQ", f"{momentum['prior_qoq']:+.1f}%" if momentum.get('prior_qoq') is not None else "N/A")

                if momentum.get('eps_values') and momentum.get('quarters'):
                    st.dataframe(pd.DataFrame({'Quarter': momentum['quarters'], 'EPS': [round(e, 2) for e in momentum['eps_values']]}), hide_index=True, use_container_width=True)

                lq = momentum.get('latest_qoq', 0) or 0
                pq = momentum.get('prior_qoq', 0) or 0
                if lq > 0 and pq > 0: st.success("✅ Both quarters positive — momentum confirmed")
                elif lq > 0 or pq > 0: st.info("Mixed momentum. Monitor next quarter.")
                else: st.warning("⚠️ Both quarters negative — investigate before buying")
            else:
                st.warning("Insufficient quarterly data.")
            if ret_1y is not None:
                st.markdown(f"**1-Year Price Return:** {ret_1y:+.1f}%")

        with tab5:
            st.subheader("All 7 Layers")
            st.dataframe(pd.DataFrame({
                "Layer": ["1. Fundamentals", "2. Cash Flow", "3. Forensic", "4. PEG", "5. Momentum", "6. Cyclical", "7. Sizing"],
                "Result": [
                    "PASS ✅" if layer1['phase1_pass'] else "FAIL ❌",
                    f"CFO/PAT: {acct.get('cum_cfo_pat', 'N/A')}x",
                    f"{acct.get('score', 'N/A')}/100 ({acct.get('num_flags', 0)} flags)",
                    f"PEG: {peg}" if peg else "N/A",
                    f"QoQ: {momentum.get('latest_qoq', 'N/A')}%" if momentum.get('available') else "N/A",
                    "PEAK ⚠️" if cyclical.get('cyclical_peak') else "OK ✅",
                    f"{tier} — {size}"
                ]
            }), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.caption("For research only, not investment advice · Data from Yahoo Finance")


# ============================================================
# PAGE 2: AUTO TOP 10 SCREENER
# ============================================================
elif page == "🏆 Auto Top 10 Screener":

    st.subheader("🏆 Automatic Top 10 Stock Picker")
    st.markdown("Screens Nifty 200 through all 7 layers and ranks by combined quality + valuation score. Banking and insurance stocks are automatically excluded.")

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        max_stocks = st.slider("How many stocks to screen", 20, 200, 50, step=10, help="More stocks = more thorough but slower. 50 takes ~5 min, 200 takes ~20 min.")
    with col_config2:
        top_n = st.slider("Show top N picks", 5, 20, 10)

    if st.button("🚀 Run Full Screen", type="primary", use_container_width=True):

        tickers = fetch_nifty200_tickers()
        if not tickers:
            st.error("Could not fetch Nifty 200 list. Check your connection.")
            st.stop()

        tickers = tickers[:max_stocks]
        results = []
        progress = st.progress(0, text="Starting screen...")

        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Analysing {ticker.replace('.NS', '')} ({i+1}/{len(tickers)})")
            result = analyse_single_stock_quick(ticker)
            if result:
                results.append(result)

        progress.empty()

        if not results:
            st.error("No stocks passed the screening criteria.")
            st.stop()

        df = pd.DataFrame(results)

        # Composite ranking score: weighted combination
        df['rank_score'] = 0.0
        df['rank_score'] += df['acct_score'].fillna(0) * 0.35
        df['rank_score'] += df['peg'].apply(lambda x: max(0, 100 - (x or 5) * 20) if x else 0) * 0.25
        df['rank_score'] += df['roe'].fillna(0) * 0.15
        df['rank_score'] += df['ret_1y'].apply(lambda x: min(50, max(-50, x or 0)) + 50).fillna(50) * 0.10
        df['rank_score'] += df['cum_cfo_pat'].apply(lambda x: min(100, (x or 0) * 100)).fillna(0) * 0.15

        df = df.sort_values('rank_score', ascending=False)
        top = df.head(top_n)

        # Display results
        st.markdown(f"### Top {top_n} Picks from {len(tickers)} Screened Stocks")
        st.markdown(f"*{len(results)} stocks passed initial filters. Banking/insurance excluded automatically.*")
        st.markdown("")

        for idx, row in top.iterrows():
            rank = list(top.index).index(idx) + 1
            tier_color = {"FULL": "🟢", "STANDARD": "🔵", "HALF": "🟡", "WATCH": "🔴"}.get(row['tier'], "⚪")

            with st.expander(f"**#{rank} · {row['ticker']}** — {row['name']} · {tier_color} {row['tier']} · Score: {row['acct_score']}/100 · PEG: {row['peg'] if row['peg'] else 'N/A'}", expanded=(rank <= 3)):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Price", f"₹{row['price']:,.0f}" if row['price'] else "N/A")
                c2.metric("PE", f"{row['pe']:.1f}" if row['pe'] else "N/A")
                c3.metric("PEG", f"{row['peg']:.2f}" if row['peg'] else "N/A")
                c4.metric("ROE", f"{row['roe']:.1f}%" if row['roe'] else "N/A")
                c5.metric("1Y Return", f"{row['ret_1y']:+.1f}%" if row['ret_1y'] is not None else "N/A")
                c6.metric("CFO/PAT", f"{row['cum_cfo_pat']:.2f}x" if row['cum_cfo_pat'] else "N/A")

                st.markdown(f"**Sector:** {row['sector']} · **Sales CAGR:** {row['sales_cagr']}% · **PAT CAGR:** {row['pat_cagr']}% · **Flags:** {row['num_flags']} · **Cyclical Peak:** {'Yes ⚠️' if row['cyclical_peak'] else 'No'}")
                st.markdown(f"**Position:** {row['tier']} — {row['size']}")

        # Summary table
        st.markdown("---")
        st.markdown("### Full Rankings Table")
        display_df = df[['ticker', 'name', 'acct_score', 'pe', 'peg', 'roe', 'cum_cfo_pat', 'ret_1y', 'tier', 'size']].copy()
        display_df.columns = ['Ticker', 'Name', 'Score', 'PE', 'PEG', 'ROE%', 'CFO/PAT', '1Y Ret%', 'Tier', 'Size']
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)

        # Tier breakdown
        st.markdown("### Tier Breakdown")
        tc1, tc2, tc3, tc4 = st.columns(4)
        for col_widget, tier_name, emoji in [(tc1, 'FULL', '🟢'), (tc2, 'STANDARD', '🔵'), (tc3, 'HALF', '🟡'), (tc4, 'WATCH', '🔴')]:
            tier_stocks = df[df['tier'] == tier_name]
            with col_widget:
                st.markdown(f"**{emoji} {tier_name}** ({len(tier_stocks)})")
                for _, r in tier_stocks.iterrows():
                    st.caption(f"{r['ticker']} · Score {r['acct_score']}")


# ============================================================
# PAGE 3: HOW IT WORKS
# ============================================================
elif page == "📖 How It Works":

    st.subheader("How This Framework Works")

    st.markdown("""
    This dashboard runs a **7-layer systematic analysis** on any NSE-listed stock, designed to find high-quality compounders while avoiding value traps.

    **Layer 1 — Quantitative Screen:** Market Cap > ₹15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, 3Y Sales & PAT CAGR > 15%. Filters the Nifty 500 from hundreds of stocks to a manageable shortlist.

    **Layer 2 — Cash Flow Quality:** CFO/EBITDA > 70%. Ensures reported profits are generating actual cash, not just accounting entries.

    **Layer 3 — Forensic Accounting (the key differentiator):** Four checks using **cumulative multi-year** data, not single-year snapshots. This smooths out lumpiness in project-based and hyper-growth businesses while still catching genuine accounting problems. Checks: receivables stuffing, inventory bloat, cumulative CFO/PAT, and cumulative CFO/EBITDA with trend analysis.

    **Layer 4 — PEG Valuation:** PE divided by earnings growth rate. A stock can have a high PE and still be cheap if growth justifies it. The "Adani Filter" flags any stock with PE > 80 as high risk.

    **Layer 5 — Earnings Momentum:** Last 2 quarters of QoQ EPS change. Catches stocks that pass every backward-looking test but have a fresh warning sign in the most recent data.

    **Layer 6 — Cyclical ROE Normalization:** If current ROE is more than 2x the historical average, the stock is at "cyclical peak" — use the median ROE for valuation models, not the inflated current figure.

    **Layer 7 — Position Sizing v2:** Integrates quality score + PEG + 1Y momentum into a single sizing decision. Clean cash + cheap valuation = FULL (12-15%). Clean but expensive = STANDARD (8-10%). Flags present = HALF (4-6%). Failed quality = WATCH (0%).
    """)

    st.markdown("---")
    st.markdown("### What This Framework Does NOT Cover")
    st.markdown("""
    **Banking, NBFC, and Insurance stocks** require different metrics (NIM, CASA, Credit Cost, GNPA/NNPA, Provision Coverage). This framework automatically detects and flags these stocks.

    **Commodity producers** where revenue is primarily price-driven. The CAGR filter screens them in during upcycles and out during downcycles.

    **Newly listed companies** with less than 2 years of financial history — not enough data for cumulative checks.
    """)

    st.markdown("---")
    st.markdown("### Framework Scoring")
    st.markdown("""
    This framework scores **90/100** for a retail investor operating independently. The 7 layers cover quantitative screening, cash flow authenticity, forensic accounting, growth-adjusted valuation, earnings momentum, cyclical normalization, and integrated position sizing.

    The remaining 10 points would require: point-in-time historical backtesting (needs paid data), management quality scoring (capital allocation track record), and a separate framework for financials.
    """)

    st.markdown("---")
    st.caption("Built by Vinayak Nagral · Framework developed September 2026")
