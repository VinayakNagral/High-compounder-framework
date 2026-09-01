import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="High Compounder Framework",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B2A4A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
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
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    .flag-item {
        background: #FFF3E0;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #FF9800;
    }
    .clean-item {
        background: #E8F5E9;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #4CAF50;
    }
    .tier-full { color: #2E7D32; font-weight: 700; font-size: 1.3rem; }
    .tier-standard { color: #1565C0; font-weight: 700; font-size: 1.3rem; }
    .tier-half { color: #F57F17; font-weight: 700; font-size: 1.3rem; }
    .tier-watch { color: #C62828; font-weight: 700; font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_str):
    """Fetch all required data for a stock."""
    try:
        t = yf.Ticker(ticker_str)
        info = dict(t.info) if t.info else {}
        fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None

        return {
            "info": info,
            "financials": fin,
            "balance_sheet": bs,
            "cashflow": cf,
            "quarterly_financials": qfin
        }
    except Exception as e:
        return None


def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            val = df.loc[label].iloc[col]
            if pd.notna(val):
                return float(val)
    except:
        pass
    return None


def get_multi_year_data(fin, bs, cf):
    """Extract multi-year records for accounting checks."""
    years = min(fin.shape[1], bs.shape[1], cf.shape[1])
    if years < 2:
        return None

    records = []
    for i in range(years):
        year_label = str(fin.columns[i].year) if hasattr(fin.columns[i], 'year') else str(i)

        revenue = safe_get(fin, 'Total Revenue', i)
        if revenue is None:
            revenue = safe_get(fin, 'Operating Revenue', i)
        net_income = safe_get(fin, 'Net Income', i)
        ebitda = safe_get(fin, 'EBITDA', i)

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

        ebit = safe_get(fin, 'EBIT', i)

        records.append({
            'year': year_label, 'revenue': revenue, 'net_income': net_income,
            'ebitda': ebitda, 'ebit': ebit, 'receivables': receivables,
            'inventory': inventory, 'total_assets': total_assets,
            'equity': equity, 'current_liabilities': current_liabilities,
            'cfo': cfo
        })

    return records


# ============================================================
# LAYER 1: BASIC FUNDAMENTALS
# ============================================================
def run_layer1(info, fin, bs):
    """Phase I: Quantitative Core Screener."""
    results = {}

    # Market Cap
    mcap = info.get('marketCap', 0)
    results['market_cap_cr'] = round(mcap / 1e7, 0) if mcap else None
    results['mcap_pass'] = mcap > 150_000_000_000 if mcap else False

    # PE
    results['pe'] = info.get('trailingPE')

    # Debt to Equity
    de = info.get('debtToEquity', 0)
    results['debt_to_equity'] = round(de / 100, 2) if de else 0
    results['de_pass'] = (de or 0) < 50

    # ROE (calculated)
    ni = safe_get(fin, 'Net Income', 0)
    eq = safe_get(bs, 'Stockholders Equity', 0) or safe_get(bs, 'Common Stock Equity', 0)
    if ni and eq and eq > 0:
        results['roe'] = round(ni / eq * 100, 1)
        results['roe_pass'] = results['roe'] > 15
    else:
        results['roe'] = None
        results['roe_pass'] = False

    # ROCE
    ebit = safe_get(fin, 'EBIT', 0)
    ta = safe_get(bs, 'Total Assets', 0)
    cl = safe_get(bs, 'Current Liabilities', 0)
    if ebit and ta and cl and (ta - cl) > 0:
        results['roce'] = round(ebit / (ta - cl) * 100, 1)
        results['roce_pass'] = results['roce'] > 18
    else:
        results['roce'] = None
        results['roce_pass'] = False

    # Growth (CAGR)
    if fin is not None and fin.shape[1] >= 2:
        rev_latest = safe_get(fin, 'Total Revenue', 0)
        rev_oldest = safe_get(fin, 'Total Revenue', fin.shape[1] - 1)
        ni_latest = safe_get(fin, 'Net Income', 0)
        ni_oldest = safe_get(fin, 'Net Income', fin.shape[1] - 1)
        yrs = fin.shape[1] - 1

             # Fallback to Operating Revenue if Total Revenue is None
        if rev_latest is None:
            rev_latest = safe_get(fin, 'Operating Revenue', 0)
        if rev_oldest is None:
            rev_oldest = safe_get(fin, 'Operating Revenue', fin.shape[1] - 1)

        if rev_latest and rev_oldest and rev_oldest > 0 and yrs > 0:
            results['sales_cagr'] = round(((rev_latest / rev_oldest) ** (1 / yrs) - 1) * 100, 1)
        else:
            results['sales_cagr'] = None

        if ni_latest and ni_oldest and ni_oldest > 0 and yrs > 0:
            results['pat_cagr'] = round(((ni_latest / ni_oldest) ** (1 / yrs) - 1) * 100, 1)
        else:
            results['pat_cagr'] = None
    else:
        results['sales_cagr'] = None
        results['pat_cagr'] = None

    results['growth_pass'] = (
        (results.get('sales_cagr') or 0) > 15 and
        (results.get('pat_cagr') or 0) > 15
    )

    # Overall Phase I
    results['phase1_pass'] = all([
        results['mcap_pass'], results['de_pass'], results['roe_pass'],
        results['roce_pass'], results['growth_pass']
    ])

    return results


# ============================================================
# LAYER 3: ACCOUNTING QUALITY (FORENSIC)
# ============================================================
def run_accounting_quality(data):
    """Phase IIB: Forensic accounting checks with cumulative smoothing."""
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
        inv_l = latest.get('inventory')
        inv_p = prior.get('inventory')
        rev_l = latest.get('revenue')
        rev_p = prior.get('revenue')
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

    # Single year for display
    if latest.get('cfo') is not None and latest.get('net_income') and latest['net_income'] > 0:
        details['single_yr_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2)

    # CHECK 4: Cumulative CFO/EBITDA
    total_cfo_e = sum(d['cfo'] for d in sorted_data if d.get('cfo') is not None)
    total_ebitda = sum(d['ebitda'] for d in sorted_data if d.get('ebitda') is not None and d['ebitda'] > 0)

    if total_ebitda > 0:
        cum_cfo_ebitda = round(total_cfo_e / total_ebitda, 2)
        details['cum_cfo_ebitda'] = cum_cfo_ebitda
        if cum_cfo_ebitda < 0.5:
            flags.append(f"Critical: Cumulative CFO/EBITDA = {cum_cfo_ebitda}x over {len(sorted_data)} years")
            score -= 20
        elif cum_cfo_ebitda < 0.7:
            flags.append(f"Weak: Cumulative CFO/EBITDA = {cum_cfo_ebitda}x over {len(sorted_data)} years")
            score -= 15
    else:
        details['cum_cfo_ebitda'] = None

    # Trend check
    yr_ratios = []
    for d in sorted_data:
        if d.get('cfo') is not None and d.get('ebitda') and d['ebitda'] > 0:
            yr_ratios.append(round(d['cfo'] / d['ebitda'], 2))
    details['cfo_ebitda_trend'] = yr_ratios

    if len(yr_ratios) >= 2 and details.get('cum_cfo_ebitda'):
        if yr_ratios[-1] < yr_ratios[-2] and yr_ratios[-1] < 0.5 and details['cum_cfo_ebitda'] < 0.7:
            flags.append(f"Deteriorating: CFO/EBITDA fell {yr_ratios[-2]} → {yr_ratios[-1]} AND cumulative weak")
            score -= 10

    # BONUS: Negative cumulative CFO
    if total_cfo < 0:
        flags.append(f"Negative cumulative CFO over {len(sorted_data)} years")
        score -= 25

    score = max(score, 0)
    details['score'] = score
    details['flags'] = flags
    details['num_flags'] = len(flags)

    return details


# ============================================================
# LAYER 6: CYCLICAL ROE NORMALIZATION
# ============================================================
def run_cyclical_check(data):
    """Check if current ROE is at cyclical peak."""
    sorted_data = sorted(data, key=lambda x: x['year'])
    roe_values = []
    roe_by_year = {}

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
    minimum = round(min(roe_values), 1)

    is_peak = latest > avg * 2 and latest > 20

    return {
        'latest_roe': latest,
        'avg_roe': avg,
        'median_roe': median,
        'min_roe': minimum,
        'normalized_roe': median,
        'cyclical_peak': is_peak,
        'roe_values': roe_values,
        'roe_by_year': roe_by_year
    }


# ============================================================
# LAYER 5: EARNINGS MOMENTUM
# ============================================================
def run_momentum_check(qfin):
    """Check last 2 quarters QoQ EPS change."""
    if qfin is None or qfin.shape[1] < 3:
        return {'available': False}

    eps_values = []
    quarters = []
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
        'available': True,
        'latest_qoq': latest_qoq,
        'prior_qoq': prior_qoq,
        'eps_values': eps_values[:4],
        'quarters': quarters[:4]
    }


# ============================================================
# LAYER 7: POSITION SIZING v2
# ============================================================
def get_position_size(acct_score, num_flags, cyclical_peak, peg, momentum_1y):
    """Quality + Valuation + Momentum integrated sizing."""
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

    sizes = {
        'FULL': ('FULL', '12-15%'),
        'STANDARD': ('STANDARD', '8-10%'),
        'HALF': ('HALF', '4-6%'),
    }
    return sizes[base]


# ============================================================
# 1Y RETURN
# ============================================================
@st.cache_data(ttl=3600)
def get_1y_return(ticker_str):
    try:
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=395)
        data = yf.download(ticker_str, start=start, end=end, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        current = data['Close'].iloc[-1]
        lookback = end - timedelta(days=365)
        mask = data.index <= lookback
        if mask.sum() > 0:
            past = data.loc[mask, 'Close'].iloc[-1]
            return round(float((current / past - 1) * 100), 1)
    except:
        pass
    return None


# ============================================================
# PEG CALCULATION
# ============================================================
def calc_peg(pe, pat_cagr):
    if pe and pat_cagr and pat_cagr > 0:
        return round(pe / pat_cagr, 2)
    return None


# ============================================================
# MAIN APP
# ============================================================
st.markdown('<p class="main-header">📊 High Compounder Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">7-Layer Systematic Indian Equity Screener — Quality + Valuation + Momentum</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.title("Analyse a Stock")
ticker_input = st.sidebar.text_input(
    "Enter NSE ticker (e.g., LUPIN, BSE, DIXON)",
    value="LUPIN"
).strip().upper()

if not ticker_input.endswith(".NS"):
    ticker_input = ticker_input + ".NS"

analyse_btn = st.sidebar.button("🔍 Run Full Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Framework Layers")
st.sidebar.markdown("""
1. Quantitative Screen
2. Cash Flow Quality
3. **Forensic Accounting**
4. PEG Valuation
5. Earnings Momentum
6. Cyclical Normalization
7. Position Sizing v2
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Portfolio")

portfolio_data = {
    "Stock": ["LUPIN", "DIXON", "ENRIN", "BSE", "MCX", "ICICI AMC", "EICHER", "KPIT", "POLYCAB", "HDFC AMC"],
    "Tier": ["FULL", "FULL", "FULL", "STD", "STD", "STD", "STD", "HALF", "HALF", "HALF"],
    "Score": [100, 100, 100, 85, 85, 85, 85, 100, 55, 70],
    "PEG": [0.14, 0.54, 0.91, 0.38, 0.51, 1.46, 1.60, 1.48, 1.67, 1.42]
}
st.sidebar.dataframe(pd.DataFrame(portfolio_data), hide_index=True, use_container_width=True)


# ============================================================
# ANALYSIS
# ============================================================
if analyse_btn:

    with st.spinner(f"Analysing {ticker_input}... fetching data from Yahoo Finance"):
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

    # Header
    st.markdown(f"## {company_name}")
    st.markdown(f"**{ticker_input}** · {sector} · ₹{price}")

    st.markdown("---")

    # ---- RUN ALL LAYERS ----
    with st.spinner("Running 7-layer analysis..."):
        # Layer 1
        layer1 = run_layer1(info, fin, bs)

        # Layer 3: Accounting Quality
        multi_year = get_multi_year_data(fin, bs, cf)
        if multi_year:
            acct = run_accounting_quality(multi_year)
            cyclical = run_cyclical_check(multi_year)
        else:
            acct = {'score': None, 'flags': ['Insufficient data'], 'num_flags': 0}
            cyclical = {'cyclical_peak': False}

        # Layer 4: PEG
        peg = calc_peg(layer1.get('pe'), layer1.get('pat_cagr'))

        # Layer 5: Momentum
        momentum = run_momentum_check(qfin)

        # 1Y Return
        ret_1y = get_1y_return(ticker_input)

        # Layer 7: Position Sizing
        if acct.get('score') is not None:
            tier, size = get_position_size(
                acct['score'], acct['num_flags'],
                cyclical.get('cyclical_peak', False),
                peg, ret_1y
            )
        else:
            tier, size = 'N/A', 'Insufficient data'

    # ============================================================
    # DISPLAY: TOP METRICS ROW
    # ============================================================
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        score_val = acct.get('score', 'N/A')
        if isinstance(score_val, (int, float)):
            color = "green" if score_val >= 85 else "orange" if score_val >= 50 else "red"
        else:
            color = "gray"
        st.metric("Accounting Score", f"{score_val}/100")

    with col2:
        st.metric("PE Ratio", f"{layer1.get('pe', 'N/A'):.1f}" if layer1.get('pe') else "N/A")

    with col3:
        st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")

    with col4:
        st.metric("ROE", f"{layer1.get('roe', 'N/A')}%")

    with col5:
        st.metric("1Y Return", f"{ret_1y:+.1f}%" if ret_1y else "N/A")

    with col6:
        tier_class = f"tier-{tier.lower()}" if tier in ['FULL', 'STANDARD', 'HALF', 'WATCH'] else ""
        st.markdown(f"**Position Tier**")
        st.markdown(f'<p class="{tier_class}">{tier} — {size}</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # DISPLAY: DETAILED LAYERS
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Phase I: Fundamentals",
        "🔍 Forensic Accounting",
        "📊 Cyclical & Valuation",
        "📈 Momentum",
        "🎯 Final Verdict"
    ])

    # ---- TAB 1: Fundamentals ----
    with tab1:
        st.subheader("Phase I: Quantitative Core Screen")

        checks = {
            f"Market Cap: ₹{layer1.get('market_cap_cr', 'N/A'):,.0f} Cr": layer1['mcap_pass'],
            f"Debt/Equity: {layer1.get('debt_to_equity', 'N/A')}": layer1['de_pass'],
            f"ROE: {layer1.get('roe', 'N/A')}% (threshold: >15%)": layer1['roe_pass'],
            f"ROCE: {layer1.get('roce', 'N/A')}% (threshold: >18%)": layer1['roce_pass'],
            f"Sales CAGR: {layer1.get('sales_cagr', 'N/A')}%, PAT CAGR: {layer1.get('pat_cagr', 'N/A')}% (threshold: >15%)": layer1['growth_pass'],
        }

        for check, passed in checks.items():
            icon = "✅" if passed else "❌"
            st.markdown(f"{icon} {check}")

        overall = "PASSES Phase I" if layer1['phase1_pass'] else "FAILS Phase I"
        color = "green" if layer1['phase1_pass'] else "red"
        st.markdown(f"**Overall: :{color}[{overall}]**")

    # ---- TAB 2: Forensic ----
    with tab2:
        st.subheader("Phase IIB: Forensic Accounting Quality")

        if acct.get('score') is not None:
            # Score display
            score = acct['score']
            if score >= 85:
                st.markdown(f'<div class="score-box score-green"><h2>{score}/100</h2><p>CLEAN — No material accounting concerns</p></div>', unsafe_allow_html=True)
            elif score >= 50:
                st.markdown(f'<div class="score-box score-yellow"><h2>{score}/100</h2><p>FLAGS DETECTED — Investigate before buying</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="score-box score-red"><h2>{score}/100</h2><p>SERIOUS CONCERNS — Do not deploy capital</p></div>', unsafe_allow_html=True)

            st.markdown("")

            # Cash metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Cumulative CFO/PAT (multi-year)**")
                val = acct.get('cum_cfo_pat')
                if val is not None:
                    icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.5 else "❌"
                    st.markdown(f"{icon} **{val}x** {'(healthy)' if val >= 0.7 else '(weak)' if val >= 0.5 else '(critical)'}")
                single = acct.get('single_yr_cfo_pat')
                if single is not None:
                    st.caption(f"Latest single year: {single}x (context only, not scored)")

            with col_b:
                st.markdown("**Cumulative CFO/EBITDA (multi-year)**")
                val = acct.get('cum_cfo_ebitda')
                if val is not None:
                    icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.5 else "❌"
                    st.markdown(f"{icon} **{val}x** {'(healthy)' if val >= 0.7 else '(weak)' if val >= 0.5 else '(critical)'}")
                trend = acct.get('cfo_ebitda_trend', [])
                if trend:
                    st.caption(f"Year-by-year trend: {' → '.join(str(x) for x in trend)}")

            st.markdown("")

            # Flags
            if acct['flags']:
                st.markdown("**⚠️ Flags Detected:**")
                for flag in acct['flags']:
                    st.markdown(f'<div class="flag-item">⚠ {flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="clean-item">✅ All 4 forensic checks passed — no flags</div>', unsafe_allow_html=True)

            # Receivables trend
            recv = acct.get('recv_pcts', [])
            years = acct.get('recv_years', [])
            if recv and any(r is not None for r in recv):
                st.markdown("")
                st.markdown("**Receivables as % of Revenue (year-by-year):**")
                recv_df = pd.DataFrame({
                    'Year': years,
                    'Recv % of Revenue': [f"{r:.1f}%" if r else "N/A" for r in recv]
                })
                st.dataframe(recv_df, hide_index=True, use_container_width=True)
        else:
            st.warning("Insufficient financial data for forensic analysis.")

    # ---- TAB 3: Cyclical & Valuation ----
    with tab3:
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Cyclical ROE Check")
            if cyclical.get('roe_by_year'):
                roe_df = pd.DataFrame(
                    list(cyclical['roe_by_year'].items()),
                    columns=['Year', 'ROE %']
                )
                st.dataframe(roe_df, hide_index=True, use_container_width=True)

                st.markdown(f"**Latest ROE:** {cyclical.get('latest_roe', 'N/A')}%")
                st.markdown(f"**Average ROE:** {cyclical.get('avg_roe', 'N/A')}%")
                st.markdown(f"**Median (Normalized) ROE:** {cyclical.get('median_roe', 'N/A')}%")

                if cyclical.get('cyclical_peak'):
                    st.error("⚠️ CYCLICAL PEAK — Use normalized ROE for valuation, not latest")
                else:
                    st.success("✅ Not at cyclical peak")
            else:
                st.warning("Insufficient data")

        with col_d:
            st.subheader("Valuation")
            st.markdown(f"**PE Ratio:** {layer1.get('pe', 'N/A')}")
            st.markdown(f"**PEG Ratio:** {peg if peg else 'N/A'}")

            if peg:
                if peg < 0.5:
                    st.success(f"PEG {peg} — Undervalued (growth far exceeds PE)")
                elif peg < 1.0:
                    st.success(f"PEG {peg} — Attractive")
                elif peg < 1.5:
                    st.info(f"PEG {peg} — Fairly valued")
                elif peg < 2.0:
                    st.warning(f"PEG {peg} — Expensive")
                else:
                    st.error(f"PEG {peg} — Overvalued")

            pe = layer1.get('pe')
            if pe and pe > 80:
                st.error("🚨 ADANI FILTER: PE > 80 — High Risk unless earnings justify within 2 years")

            st.markdown(f"**Sales CAGR:** {layer1.get('sales_cagr', 'N/A')}%")
            st.markdown(f"**PAT CAGR:** {layer1.get('pat_cagr', 'N/A')}%")

    # ---- TAB 4: Momentum ----
    with tab4:
        st.subheader("Earnings Momentum (QoQ)")

        if momentum.get('available'):
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                val = momentum.get('latest_qoq')
                if val is not None:
                    icon = "✅" if val > 0 else "⚠️" if val > -10 else "❌"
                    st.metric("Latest Quarter QoQ", f"{val:+.1f}%")
            with mcol2:
                val = momentum.get('prior_qoq')
                if val is not None:
                    icon = "✅" if val > 0 else "⚠️" if val > -10 else "❌"
                    st.metric("Prior Quarter QoQ", f"{val:+.1f}%")

            if momentum.get('eps_values') and momentum.get('quarters'):
                eps_df = pd.DataFrame({
                    'Quarter': momentum['quarters'],
                    'EPS': [round(e, 2) for e in momentum['eps_values']]
                })
                st.dataframe(eps_df, hide_index=True, use_container_width=True)

            lq = momentum.get('latest_qoq', 0) or 0
            pq = momentum.get('prior_qoq', 0) or 0
            if lq > 0 and pq > 0:
                st.success("✅ Both quarters positive — momentum confirmed")
            elif lq > 0 or pq > 0:
                st.info("Mixed — one quarter positive, one negative. Monitor.")
            else:
                st.warning("⚠️ Both quarters negative — momentum concern. Investigate WHY before buying.")
        else:
            st.warning("Insufficient quarterly data for momentum check.")

        st.markdown("")
        st.markdown(f"**1-Year Price Return:** {ret_1y:+.1f}%" if ret_1y else "**1-Year Return:** N/A")

    # ---- TAB 5: Final Verdict ----
    with tab5:
        st.subheader("Final Verdict: All 7 Layers Combined")

        # Summary table
        verdict_data = {
            "Layer": [
                "1. Phase I Fundamentals",
                "2. Cash Flow Quality",
                "3. Forensic Accounting",
                "4. PEG Valuation",
                "5. Earnings Momentum",
                "6. Cyclical Check",
                "7. Position Sizing"
            ],
            "Result": [
                "PASS ✅" if layer1['phase1_pass'] else "FAIL ❌",
                f"CFO/PAT: {acct.get('cum_cfo_pat', 'N/A')}x",
                f"Score: {acct.get('score', 'N/A')}/100 ({acct.get('num_flags', 0)} flags)",
                f"PEG: {peg}" if peg else "N/A",
                f"Latest QoQ: {momentum.get('latest_qoq', 'N/A')}%" if momentum.get('available') else "N/A",
                "CYCLICAL PEAK ⚠️" if cyclical.get('cyclical_peak') else "OK ✅",
                f"{tier} — {size}"
            ]
        }

        st.dataframe(pd.DataFrame(verdict_data), hide_index=True, use_container_width=True)

        st.markdown("---")

        # Final recommendation
        st.markdown("### Recommendation")

        tier_html = {
            'FULL': '<p class="tier-full">✅ FULL POSITION (12-15%) — Deploy with conviction</p>',
            'STANDARD': '<p class="tier-standard">📊 STANDARD POSITION (8-10%) — Quality confirmed, valuation reasonable</p>',
            'HALF': '<p class="tier-half">⚠️ HALF POSITION (4-6%) — Flags present, add only on confirmation</p>',
            'WATCH': '<p class="tier-watch">🛑 WATCH ONLY (0%) — Do not deploy capital until flags clear</p>'
        }

        st.markdown(tier_html.get(tier, f"<p>{tier} — {size}</p>"), unsafe_allow_html=True)

        # Key reasons
        reasons = []
        if acct.get('score') is not None and acct['score'] >= 85:
            reasons.append("Clean accounting — cumulative cash flows validate reported profits")
        if acct.get('score') is not None and acct['score'] < 50:
            reasons.append("Poor accounting quality — profits not converting to cash")
        if peg and peg < 0.5:
            reasons.append(f"Undervalued — PEG {peg} means growth far exceeds what PE implies")
        if peg and peg > 2.0:
            reasons.append(f"Overvalued — PEG {peg} means paying premium above growth")
        if cyclical.get('cyclical_peak'):
            reasons.append(f"Cyclical peak — use normalized ROE {cyclical.get('median_roe')}% for valuation, not {cyclical.get('latest_roe')}%")
        if ret_1y and ret_1y < -30:
            reasons.append(f"Momentum risk — stock down {ret_1y}% in 1 year")
        if layer1.get('pe') and layer1['pe'] > 80:
            reasons.append("Adani Filter triggered — PE > 80, high risk")

        if reasons:
            st.markdown("**Key factors:**")
            for r in reasons:
                st.markdown(f"• {r}")

        st.markdown("---")
        st.caption("Framework: 7-Layer High Compounder Screen | For research only, not investment advice | Data from Yahoo Finance")


# ============================================================
# DEFAULT STATE (no analysis running)
# ============================================================
else:
    st.info("👈 Enter a ticker in the sidebar and click 'Run Full Analysis' to begin.")

    st.markdown("### How this works")
    st.markdown("""
    This dashboard runs a **7-layer systematic analysis** on any NSE-listed stock:

    **Layer 1** — Fundamental screen (Market Cap, ROE, ROCE, D/E, Growth)

    **Layer 2** — Cash flow quality (CFO/EBITDA > 70%)

    **Layer 3** — Forensic accounting with cumulative multi-year checks (catches revenue stuffing, inventory bloat, and fake profits)

    **Layer 4** — PEG valuation (growth-adjusted price assessment)

    **Layer 5** — Quarterly earnings momentum (QoQ EPS trajectory)

    **Layer 6** — Cyclical ROE normalization (detects peak-earnings traps)

    **Layer 7** — Integrated position sizing combining quality + valuation + momentum

    Each stock gets a final tier: **FULL** (12-15%), **STANDARD** (8-10%), **HALF** (4-6%), or **WATCH** (0%).
    """)

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("Try: `LUPIN` · `BSE` · `DIXON` · `KPITTECH` · `HDFCAMC` · `MAZDOCK` · `WAAREEENER`")
