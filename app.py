import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="High Compounder Framework", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 1. STYLING — clean, professional, minimal
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg: #FAFAFA; --card: #FFFFFF; --text: #1A1A2E; --muted: #6B7280;
    --green: #059669; --green-bg: #ECFDF5; --yellow: #D97706; --yellow-bg: #FFFBEB;
    --red: #DC2626; --red-bg: #FEF2F2; --blue: #2563EB; --blue-bg: #EFF6FF;
    --border: #E5E7EB; --accent: #4F46E5;
}
.main-title { font-family:'DM Sans',sans-serif; font-size:1.75rem; font-weight:700; color:var(--text); margin:0; }
.sub-title { font-family:'DM Sans',sans-serif; font-size:0.875rem; color:var(--muted); margin-top:2px; }
.stock-name { font-family:'DM Sans',sans-serif; font-size:1.5rem; font-weight:700; color:var(--text); }
.stock-meta { font-family:'DM Sans',sans-serif; font-size:0.875rem; color:var(--muted); }
.metric-card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem; text-align:center; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:var(--text); }
.metric-label { font-size:0.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
.tier-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.tier-full { background:var(--green-bg); color:var(--green); border:1px solid var(--green); }
.tier-standard { background:var(--blue-bg); color:var(--blue); border:1px solid var(--blue); }
.tier-half { background:var(--yellow-bg); color:var(--yellow); border:1px solid var(--yellow); }
.tier-watch { background:var(--red-bg); color:var(--red); border:1px solid var(--red); }
.layer-row { padding:0.6rem 1rem; border-radius:6px; margin:4px 0; font-size:0.9rem; border-left:3px solid; }
.layer-pass { background:var(--green-bg); border-color:var(--green); }
.layer-fail { background:var(--red-bg); border-color:var(--red); }
.layer-warn { background:var(--yellow-bg); border-color:var(--yellow); }
.verdict-box { background:var(--card); padding:1.5rem; border-radius:10px; border:1px solid var(--border); line-height:1.8; }
.flag-item { background:var(--yellow-bg); padding:0.5rem 1rem; border-radius:6px; margin:3px 0; border-left:3px solid var(--yellow); font-size:0.85rem; }
.clean-item { background:var(--green-bg); padding:0.5rem 1rem; border-radius:6px; margin:3px 0; border-left:3px solid var(--green); font-size:0.85rem; }
.score-card { padding:1.2rem; border-radius:10px; text-align:center; }
.score-green { background:var(--green-bg); border:1px solid var(--green); }
.score-yellow { background:var(--yellow-bg); border:1px solid var(--yellow); }
.score-red { background:var(--red-bg); border:1px solid var(--red); }
.data-quality { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:600; }
.dq-high { background:var(--green-bg); color:var(--green); }
.dq-med { background:var(--yellow-bg); color:var(--yellow); }
.dq-low { background:var(--red-bg); color:var(--red); }
.banking-box { background:var(--blue-bg); border-left:4px solid var(--blue); padding:1rem; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

BANKING_KW = ['bank','finance','insurance','nbfc','housing finance','credit','lending','microfinance']

# ============================================================
# 2. UTILITIES
# ============================================================
def is_valid(val):
    if val is None: return False
    try: return np.isfinite(float(val))
    except: return False

def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            v = df.loc[label].iloc[col]
            if pd.notna(v): return float(v)
    except: pass
    return None

def fmt(val, f=".1f", suffix="", prefix=""):
    if not is_valid(val): return "N/A"
    return f"{prefix}{val:{f}}{suffix}"

def get_revenue(fin, col=0):
    for f in ['Total Revenue','Operating Revenue','Revenue','Net Revenue']:
        v = safe_get(fin, f, col)
        if v and v > 0: return v
    return None

def get_ni(fin, col=0):
    for f in ['Net Income','Net Income Common Stockholders','Net Income Continuous Operations']:
        v = safe_get(fin, f, col)
        if v is not None: return v
    return None

def get_ebitda(fin, col=0):
    for f in ['EBITDA','Normalized EBITDA']:
        v = safe_get(fin, f, col)
        if v and v > 0: return v
    return None

def get_ebit(fin, col=0):
    return safe_get(fin, 'EBIT', col)

def get_equity(bs, col=0):
    for f in ['Stockholders Equity','Total Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest']:
        v = safe_get(bs, f, col)
        if v and v > 0: return v
    return None

def get_cfo(cf, col=0):
    for f in ['Operating Cash Flow','Total Cash From Operating Activities','Cash Flow From Continuing Operating Activities']:
        v = safe_get(cf, f, col)
        if v is not None: return v
    return None

def get_fcf(cf, col=0):
    v = safe_get(cf, 'Free Cash Flow', col)
    if v is not None: return v
    cfo = get_cfo(cf, col)
    capex = safe_get(cf, 'Capital Expenditure', col)
    if is_valid(cfo) and is_valid(capex): return cfo + capex  # capex is negative
    return cfo

def get_shares(fin, bs, qfin, info):
    """Get shares from most reliable source. Order: quarterly income stmt > annual > balance sheet > info."""
    for src in [qfin, fin]:
        if src is not None:
            for f in ['Diluted Average Shares','Basic Average Shares']:
                v = safe_get(src, f, 0)
                if is_valid(v) and v > 1000: return v
    if bs is not None:
        for f in ['Ordinary Shares Number','Share Issued']:
            v = safe_get(bs, f, 0)
            if is_valid(v) and v > 1000: return v
    v = info.get('sharesOutstanding')
    if is_valid(v) and v > 1000: return float(v)
    return None

def is_banking(info, name=""):
    s = (info.get('sector','') or '').lower()
    ind = (info.get('industry','') or '').lower()
    n = (info.get('shortName','') or name or '').lower()
    return any(k in s or k in ind or k in n for k in BANKING_KW)

# ============================================================
# 3. DATA FETCHING — single Ticker session, all sources
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker):
    try:
        t = yf.Ticker(ticker)
        # Info
        try: info = {k:v for k,v in (dict(t.info) if t.info else {}).items() if v is not None}
        except: info = {}
        # Fast info fallback
        try:
            fi = t.fast_info
            if fi:
                for a,k in [('last_price','currentPrice'),('previous_close','previousClose'),
                             ('market_cap','marketCap'),('shares','sharesOutstanding')]:
                    if not is_valid(info.get(k)):
                        try:
                            v = getattr(fi, a, None)
                            if is_valid(v) and v > 0: info[k] = float(v)
                        except: pass
        except: pass
        # Statements
        fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None
        qbs = t.quarterly_balance_sheet.copy() if t.quarterly_balance_sheet is not None and not t.quarterly_balance_sheet.empty else None
        # Price history via t.history (same session — reliable)
        ph = None
        try:
            h = t.history(period="1y", auto_adjust=True)
            if h is not None and not h.empty and 'Close' in h.columns:
                ph = h['Close'].dropna()
                if len(ph) > 0 and not is_valid(info.get('currentPrice')):
                    info['currentPrice'] = float(ph.iloc[-1])
        except: pass
        if ph is None or len(ph) == 0:
            try:
                h = t.history(period="5d", auto_adjust=True)
                if h is not None and not h.empty and 'Close' in h.columns:
                    ph = h['Close'].dropna()
                    if len(ph) > 0 and not is_valid(info.get('currentPrice')):
                        info['currentPrice'] = float(ph.iloc[-1])
            except: pass
        return {"info":info,"fin":fin,"bs":bs,"cf":cf,"qfin":qfin,"qbs":qbs,"ph":ph}
    except: return None

# ============================================================
# 4. METRIC EXTRACTORS — with sanity checks
# ============================================================
def ext_price(sd):
    for f in ['currentPrice','regularMarketPrice','regularMarketPreviousClose','previousClose']:
        v = sd['info'].get(f)
        if is_valid(v) and v > 0: return round(float(v), 2)
    ph = sd.get('ph')
    if ph is not None and len(ph) > 0:
        v = float(ph.iloc[-1])
        if is_valid(v) and v > 0: return round(v, 2)
    return None

def ext_name(sd, ticker):
    for f in ['shortName','longName','displayName']:
        v = sd['info'].get(f)
        if v and str(v).strip() and str(v).strip().lower() != 'none': return str(v).strip()
    return ticker.replace('.NS','').replace('.BO','')

def ext_sector(sd):
    for f in ['sector','industry']:
        v = sd['info'].get(f)
        if v and str(v).strip() and str(v).strip().lower() != 'none': return str(v).strip()
    return "—"

def ext_pe(sd, price):
    """PE with sanity check: must be 1-500."""
    info = sd['info']
    # 1. Direct from info
    for f in ['trailingPE','forwardPE']:
        v = info.get(f)
        if is_valid(v) and 1 < v < 500: return round(float(v), 1)
    if not is_valid(price) or price <= 0: return None
    # 2. From EPS in info
    for f in ['trailingEps','forwardEps']:
        eps = info.get(f)
        if is_valid(eps) and eps > 0:
            pe = price / eps
            if 1 < pe < 500: return round(pe, 1)
    # 3. TTM from quarterly financials
    qfin = sd.get('qfin')
    if qfin is not None and qfin.shape[1] >= 4:
        ttm = sum(get_ni(qfin, i) or 0 for i in range(4))
        if ttm > 0:
            shares = get_shares(sd.get('fin'), sd.get('bs'), qfin, info)
            if is_valid(shares) and shares > 0:
                eps = ttm / shares
                if eps > 0:
                    pe = price / eps
                    if 1 < pe < 500: return round(pe, 1)
    # 4. Annual
    fin = sd.get('fin')
    if fin is not None:
        ni = get_ni(fin, 0)
        if is_valid(ni) and ni > 0:
            shares = get_shares(fin, sd.get('bs'), sd.get('qfin'), info)
            if is_valid(shares) and shares > 0:
                eps = ni / shares
                if eps > 0:
                    pe = price / eps
                    if 1 < pe < 500: return round(pe, 1)
    return None

def ext_mcap(sd, price):
    v = sd['info'].get('marketCap')
    if is_valid(v) and v > 0: return float(v)
    if is_valid(price) and price > 0:
        shares = get_shares(sd.get('fin'), sd.get('bs'), sd.get('qfin'), sd['info'])
        if is_valid(shares) and shares > 0: return price * shares
    return None

def ext_de(sd):
    v = sd['info'].get('debtToEquity')
    if is_valid(v): return round(float(v)/100, 2)
    bs = sd.get('bs')
    if bs:
        debt = safe_get(bs,'Total Debt',0) or safe_get(bs,'Long Term Debt',0) or 0
        eq = get_equity(bs, 0)
        if eq and eq > 0: return round(debt/eq, 2)
    return 0.0

def ext_cagr(fin, getter, min_years=1.5):
    """Date-aware CAGR from any financial statement. Scans for valid endpoints."""
    if fin is None or fin.shape[1] < 2: return None
    latest_v, latest_d, oldest_v, oldest_d = None, None, None, None
    for i in range(fin.shape[1]):
        v = getter(fin, i)
        if is_valid(v) and v > 0:
            d = fin.columns[i]
            if latest_v is None: latest_v, latest_d = v, d
            oldest_v, oldest_d = v, d
    if latest_v and oldest_v and latest_d != oldest_d:
        try: yrs = abs((latest_d - oldest_d).days) / 365.25
        except: yrs = abs(fin.columns.get_loc(latest_d) - fin.columns.get_loc(oldest_d))
        if yrs >= min_years:
            return round(((latest_v / oldest_v) ** (1/yrs) - 1) * 100, 1)
    return None

def ext_1y_ret(sd):
    ph = sd.get('ph')
    if ph is None or len(ph) < 30: return None
    try:
        cur = float(ph.iloc[-1])
        if not is_valid(cur) or cur <= 0: return None
        tgt = ph.index[-1] - pd.Timedelta(days=365)
        mask = ph.index <= tgt
        past = float(ph.loc[mask].iloc[-1]) if mask.sum() > 0 else float(ph.iloc[0])
        if is_valid(past) and past > 0:
            r = round((cur/past - 1)*100, 1)
            return r if is_valid(r) else None
    except: pass
    return None

def ext_roe(fin, bs, info):
    ni = get_ni(fin, 0) if fin is not None else None
    eq = get_equity(bs, 0) if bs is not None else None
    if is_valid(ni) and is_valid(eq) and eq > 0: return round(ni/eq*100, 1)
    v = info.get('returnOnEquity')
    if is_valid(v): return round(float(v)*100, 1)
    return None

def ext_roce(fin, bs):
    ebit = get_ebit(fin, 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if is_valid(ebit) and is_valid(ta) and is_valid(cl) and (ta-cl) > 0:
        return round(ebit/(ta-cl)*100, 1)
    return None

# ============================================================
# 5. FRAMEWORK LAYERS
# ============================================================
def run_layer1(sd, ticker):
    """Fundamentals screen."""
    r = {}
    info = sd['info']; fin = sd.get('fin'); bs = sd.get('bs')
    r['price'] = ext_price(sd)
    mcap = ext_mcap(sd, r['price'])
    r['mcap_cr'] = round(mcap/1e7) if is_valid(mcap) else None
    r['mcap_pass'] = is_valid(mcap) and mcap > 150e9
    r['pe'] = ext_pe(sd, r['price'])
    r['de'] = ext_de(sd)
    r['de_pass'] = r['de'] < 0.5
    r['roe'] = ext_roe(fin, bs, info)
    r['roe_pass'] = is_valid(r['roe']) and r['roe'] > 15
    r['roce'] = ext_roce(fin, bs)
    r['roce_pass'] = is_valid(r['roce']) and r['roce'] > 18
    r['sales_cagr'] = ext_cagr(fin, get_revenue) or (round(float(info['revenueGrowth'])*100,1) if is_valid(info.get('revenueGrowth')) else None)
    r['pat_cagr'] = ext_cagr(fin, get_ni) or (round(float(info['earningsGrowth'])*100,1) if is_valid(info.get('earningsGrowth')) else None)
    r['growth_pass'] = (r.get('sales_cagr') or 0) > 15 and (r.get('pat_cagr') or 0) > 15
    r['pass'] = all([r['mcap_pass'], r['de_pass'], r['roe_pass'], r['roce_pass'], r['growth_pass']])
    r['name'] = ext_name(sd, ticker)
    r['sector'] = ext_sector(sd)
    return r

def run_forensic(data):
    """Forensic accounting — 4 checks + margin trend."""
    flags, score, det = [], 100, {}
    sd = sorted(data, key=lambda x: x['year'])
    latest, prior = sd[-1], sd[-2] if len(sd) >= 2 else None
    # Check 1: Receivables trend
    rp = [(round(d['receivables']/d['revenue']*100,1) if is_valid(d.get('receivables')) and is_valid(d.get('revenue')) and d['revenue']>0 else None) for d in sd]
    det['recv_pcts'], det['recv_years'] = rp, [d['year'] for d in sd]
    valid_rp = [x for x in rp if x is not None]
    if len(valid_rp) >= 3:
        rising = sum(1 for i in range(1,len(valid_rp)) if valid_rp[i]>valid_rp[i-1])
        if rising >= 2:
            flags.append(f"Receivables rising: {valid_rp[-3]}% → {valid_rp[-2]}% → {valid_rp[-1]}% of revenue")
            score -= 15
    # Check 2: Inventory vs revenue growth
    if prior:
        il, ip = latest.get('inventory'), prior.get('inventory')
        rl, rp2 = latest.get('revenue'), prior.get('revenue')
        if all(is_valid(v) and v > 0 for v in [il,ip,rl,rp2]):
            ig, rg = round((il/ip-1)*100,1), round((rl/rp2-1)*100,1)
            det['inv_growth'], det['rev_growth'] = ig, rg
            if ig > rg + 10:
                flags.append(f"Inventory bloat: grew {ig}% vs Revenue {rg}%"); score -= 15
    # Check 3: Cumulative CFO/PAT
    tc = sum(d['cfo'] for d in sd if is_valid(d.get('cfo')))
    tp = sum(d['net_income'] for d in sd if is_valid(d.get('net_income')))
    if tp > 0:
        det['cum_cfo_pat'] = round(tc/tp, 2)
        if det['cum_cfo_pat'] < 0.5: flags.append(f"Critical: Cum CFO/PAT {det['cum_cfo_pat']}x"); score -= 30
        elif det['cum_cfo_pat'] < 0.7: flags.append(f"Low cash: Cum CFO/PAT {det['cum_cfo_pat']}x"); score -= 20
    else: det['cum_cfo_pat'] = None
    det['single_cfo_pat'] = round(latest['cfo']/latest['net_income'],2) if is_valid(latest.get('cfo')) and is_valid(latest.get('net_income')) and latest['net_income']>0 else None
    # Check 4: Cumulative CFO/EBITDA
    te = sum(d['ebitda'] for d in sd if is_valid(d.get('ebitda')) and d['ebitda']>0)
    if te > 0:
        det['cum_cfo_ebitda'] = round(tc/te, 2)
        if det['cum_cfo_ebitda'] < 0.5: flags.append(f"Critical: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x"); score -= 20
        elif det['cum_cfo_ebitda'] < 0.7: flags.append(f"Weak: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x"); score -= 15
    else: det['cum_cfo_ebitda'] = None
    # Trend check
    yr = [round(d['cfo']/d['ebitda'],2) for d in sd if is_valid(d.get('cfo')) and is_valid(d.get('ebitda')) and d['ebitda']>0]
    det['cfo_ebitda_trend'] = yr
    if len(yr)>=2 and is_valid(det.get('cum_cfo_ebitda')):
        if yr[-1]<yr[-2] and yr[-1]<0.5 and det['cum_cfo_ebitda']<0.7:
            flags.append(f"Deteriorating: CFO/EBITDA {yr[-2]}→{yr[-1]}"); score -= 10
    if tc < 0: flags.append(f"Negative cumulative CFO"); score -= 25
    # NEW: Operating margin trend
    margins = [round(d['ebitda']/d['revenue']*100,1) for d in sd if is_valid(d.get('ebitda')) and is_valid(d.get('revenue')) and d['revenue']>0]
    det['margins'] = margins
    if len(margins) >= 3:
        if margins[-1] < margins[0] - 3:
            det['margin_trend'] = 'contracting'
        elif margins[-1] > margins[0] + 2:
            det['margin_trend'] = 'expanding'
        else:
            det['margin_trend'] = 'stable'
    score = max(score, 0)
    det.update({'score':score, 'flags':flags, 'num_flags':len(flags)})
    return det

def run_moat(data):
    """Moat durability — consecutive years of ROE > 15%."""
    sd = sorted(data, key=lambda x: x['year'])
    roe_history = []
    for d in sd:
        if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity']>0:
            roe_history.append(round(d['net_income']/d['equity']*100,1))
    if not roe_history: return {'years_above_15':0, 'consistency':'insufficient data', 'roe_history':[]}
    above = sum(1 for r in roe_history if r > 15)
    total = len(roe_history)
    pct = round(above/total*100) if total > 0 else 0
    if pct >= 80: cons = 'strong moat'
    elif pct >= 60: cons = 'moderate moat'
    else: cons = 'weak/no moat'
    return {'years_above_15': above, 'total_years': total, 'pct': pct, 'consistency': cons, 'roe_history': roe_history}

def run_cyclical(data):
    sd = sorted(data, key=lambda x: x['year'])
    rv, rby = [], {}
    for d in sd:
        if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity']>0:
            r = round(d['net_income']/d['equity']*100,1)
            rv.append(r); rby[d['year']] = r
    if len(rv) < 2: return {'peak':False, 'roe_by_year':{}}
    l, a, m = rv[-1], round(np.mean(rv),1), round(np.median(rv),1)
    return {'latest':l, 'avg':a, 'median':m, 'min':round(min(rv),1),
            'peak': l > a*2 and l > 20, 'values':rv, 'roe_by_year':rby}

def run_momentum(qfin):
    if qfin is None or qfin.shape[1] < 3: return {'available':False}
    ev, qs = [], []
    for i in range(min(4, qfin.shape[1])):
        e = safe_get(qfin,'Diluted EPS',i) or safe_get(qfin,'Basic EPS',i)
        if e is not None:
            ev.append(e)
            qs.append(str(qfin.columns[i].strftime('%b %Y')) if hasattr(qfin.columns[i],'strftime') else str(i))
    if len(ev) < 3: return {'available':False}
    lq = round((ev[0]/ev[1]-1)*100,1) if ev[1]!=0 else None
    pq = round((ev[1]/ev[2]-1)*100,1) if ev[2]!=0 else None
    if not is_valid(lq): lq = None
    if not is_valid(pq): pq = None
    return {'available':True, 'latest_qoq':lq, 'prior_qoq':pq, 'eps':ev[:4], 'quarters':qs[:4]}

def run_valuation(pe, pat_cagr, price, mcap, sd):
    """PEG + FCF Yield dual valuation."""
    peg = round(pe/pat_cagr, 2) if is_valid(pe) and is_valid(pat_cagr) and pat_cagr > 0 else None
    # FCF Yield
    fcf_yield = None
    cf = sd.get('cf')
    if cf is not None and is_valid(mcap) and mcap > 0:
        fcf = get_fcf(cf, 0)
        if is_valid(fcf): fcf_yield = round(fcf/mcap*100, 2)
    return {'peg':peg, 'fcf_yield':fcf_yield}

def get_tier(score, nf, peak, peg, mom_1y, moat_pct):
    if score is None or score < 50: return 'WATCH', '0%'
    if is_valid(peg) and peg > 5: return 'WATCH', '0%'
    if is_valid(peg) and peg > 3: return 'HALF', '4-6%'
    # Base from quality + moat
    if score >= 85 and nf == 0 and not peak and moat_pct >= 60:
        base = 'FULL'
    elif score >= 85 and nf == 0 and not peak:
        base = 'STANDARD'  # good quality but unproven moat
    elif score >= 70 and nf <= 1:
        base = 'STANDARD'
    else:
        base = 'HALF'
    # Adjustments
    if base == 'FULL' and not is_valid(peg): return 'STANDARD', '8-10%'
    if base == 'FULL' and peg < 0.5: return 'FULL', '12-15%'
    if base == 'FULL' and is_valid(mom_1y) and mom_1y < -30: return 'HALF', '4-6%'
    if base == 'FULL' and peg > 1.5: return 'STANDARD', '8-10%'
    return {'FULL':('FULL','12-15%'), 'STANDARD':('STANDARD','8-10%'), 'HALF':('HALF','4-6%')}[base]

# ============================================================
# 6. MULTI-YEAR DATA BUILDER
# ============================================================
def build_multi(fin, bs, cf):
    if fin is None or bs is None or cf is None: return None
    yrs = min(fin.shape[1], bs.shape[1], cf.shape[1])
    if yrs < 2: return None
    recs = []
    for i in range(yrs):
        yl = str(fin.columns[i].year) if hasattr(fin.columns[i],'year') else str(i)
        recs.append({
            'year':yl, 'revenue':get_revenue(fin,i), 'net_income':get_ni(fin,i),
            'ebitda':get_ebitda(fin,i), 'ebit':get_ebit(fin,i),
            'receivables':safe_get(bs,'Accounts Receivable',i) or safe_get(bs,'Net Receivables',i) or safe_get(bs,'Receivables',i),
            'inventory':safe_get(bs,'Inventory',i), 'total_assets':safe_get(bs,'Total Assets',i),
            'equity':get_equity(bs,i), 'current_liabilities':safe_get(bs,'Current Liabilities',i),
            'cfo':get_cfo(cf,i), 'fcf':get_fcf(cf,i)
        })
    return recs

# ============================================================
# 7. LAYER DISPLAY + VERDICT
# ============================================================
def gen_layers(l1, acct, cyc, val, mom, ret, moat, is_bank):
    L = []
    # Fundamentals
    if l1['pass']:
        L.append(("Fundamentals","pass",f"MCap ₹{fmt(l1.get('mcap_cr'),',.0f')} Cr · ROE {fmt(l1.get('roe'),'.1f')}% · ROCE {fmt(l1.get('roce'),'.1f')}% · D/E {l1.get('de','—')} · Sales CAGR {fmt(l1.get('sales_cagr'),'.1f')}% · PAT CAGR {fmt(l1.get('pat_cagr'),'.1f')}%"))
    else:
        fails = []
        if not l1['mcap_pass']: fails.append(f"MCap ₹{fmt(l1.get('mcap_cr'),',.0f')} Cr < ₹15,000 Cr")
        if not l1['roe_pass']: fails.append(f"ROE {fmt(l1.get('roe'),'.1f')}% < 15%")
        if not l1['roce_pass']: fails.append(f"ROCE {fmt(l1.get('roce'),'.1f')}% < 18%")
        if not l1['de_pass']: fails.append(f"D/E {l1.get('de','—')} > 0.5")
        if not l1['growth_pass']: fails.append(f"Growth: Sales {fmt(l1.get('sales_cagr'),'.1f')}%, PAT {fmt(l1.get('pat_cagr'),'.1f')}%")
        L.append(("Fundamentals","fail"," · ".join(fails) if fails else "Criteria not met"))
    # Forensic
    s = acct.get('score')
    if s is not None:
        if s >= 85: L.append(("Forensic Accounting","pass",f"Score {s}/100 · CFO/PAT {fmt(acct.get('cum_cfo_pat'),'.2f')}x · CFO/EBITDA {fmt(acct.get('cum_cfo_ebitda'),'.2f')}x"))
        elif s >= 50: L.append(("Forensic Accounting","warn",f"Score {s}/100 · {acct.get('num_flags',0)} flags · {' · '.join(acct.get('flags',[])[:2])}"))
        else: L.append(("Forensic Accounting","fail",f"Score {s}/100 · {acct.get('num_flags',0)} flags · {' · '.join(acct.get('flags',[])[:2])}"))
    # Moat
    if moat.get('total_years',0) >= 2:
        mc = moat['consistency']
        if mc == 'strong moat': L.append(("Moat Durability","pass",f"ROE > 15% in {moat['years_above_15']}/{moat['total_years']} years ({moat['pct']}%) — {mc}"))
        elif mc == 'moderate moat': L.append(("Moat Durability","warn",f"ROE > 15% in {moat['years_above_15']}/{moat['total_years']} years ({moat['pct']}%) — {mc}"))
        else: L.append(("Moat Durability","fail",f"ROE > 15% in {moat['years_above_15']}/{moat['total_years']} years ({moat['pct']}%) — {mc}"))
    # Valuation
    peg = val.get('peg')
    fy = val.get('fcf_yield')
    if is_valid(peg):
        if peg < 0.5: L.append(("PEG Valuation","pass",f"PEG {peg:.2f} — undervalued"))
        elif peg < 1.0: L.append(("PEG Valuation","pass",f"PEG {peg:.2f} — attractive"))
        elif peg < 1.5: L.append(("PEG Valuation","warn",f"PEG {peg:.2f} — fair"))
        elif peg < 2.0: L.append(("PEG Valuation","warn",f"PEG {peg:.2f} — expensive"))
        else: L.append(("PEG Valuation","fail",f"PEG {peg:.2f} — overvalued"))
    else: L.append(("PEG Valuation","warn","PEG not calculable"))
    if is_valid(fy):
        if fy > 5: L.append(("FCF Yield","pass",f"{fy:.1f}% — strong free cash generation relative to price"))
        elif fy > 2: L.append(("FCF Yield","pass",f"{fy:.1f}% — adequate"))
        elif fy > 0: L.append(("FCF Yield","warn",f"{fy:.1f}% — thin"))
        else: L.append(("FCF Yield","fail",f"{fy:.1f}% — negative free cash flow"))
    pe = l1.get('pe')
    if is_valid(pe) and pe > 80: L.append(("Adani Filter","fail",f"PE {pe:.0f} > 80"))
    # Momentum
    if mom.get('available'):
        lq, pq = mom.get('latest_qoq'), mom.get('prior_qoq')
        if is_valid(lq) and is_valid(pq):
            if lq > 0 and pq > 0: L.append(("Earnings Momentum","pass",f"Both +ve: {lq:+.1f}%, {pq:+.1f}%"))
            elif lq > 0 or pq > 0: L.append(("Earnings Momentum","warn",f"Mixed: {lq:+.1f}%, {pq:+.1f}%"))
            else: L.append(("Earnings Momentum","fail",f"Both -ve: {lq:+.1f}%, {pq:+.1f}%"))
        else: L.append(("Earnings Momentum","warn","Partial data"))
    else: L.append(("Earnings Momentum","warn","Insufficient data"))
    # Cyclical
    if cyc.get('peak'): L.append(("Cyclical ROE","warn",f"PEAK — {cyc['latest']}% vs norm {cyc['median']}%"))
    elif cyc.get('roe_by_year'): L.append(("Cyclical ROE","pass",f"Not at peak. {cyc.get('latest','—')}% vs norm {cyc.get('median','—')}%"))
    # Price
    if is_valid(ret):
        if ret > 30: L.append(("1Y Price","pass",f"{ret:+.1f}%"))
        elif ret > 0: L.append(("1Y Price","pass",f"{ret:+.1f}%"))
        elif ret > -20: L.append(("1Y Price","warn",f"{ret:+.1f}%"))
        else: L.append(("1Y Price","fail",f"{ret:+.1f}% — heavy selling"))
    else: L.append(("1Y Price","warn","Unavailable"))
    return L

def gen_verdict(name, tier, size, l1, acct, cyc, val, ret, mom, moat, is_bank):
    if is_bank:
        return f"**{name}** is a banking/financial stock. This framework uses metrics for non-financials. Banks need NIM, CASA, Credit Cost, GNPA. Scores for reference only."
    s = acct.get('score',0) or 0
    peg, fy = val.get('peg'), val.get('fcf_yield')
    roe, cfo = l1.get('roe'), acct.get('cum_cfo_pat')
    p = []
    intro = {'FULL':f"**{name}** passes all layers with conviction. Data supports 12-15% allocation.",
             'STANDARD':f"**{name}** is solid but factors prevent full conviction. Data supports 8-10%.",
             'HALF':f"**{name}** shows mixed signals. Cautious 4-6% — add after next quarter confirms improvement.",
             'WATCH':f"**{name}** fails critical checks. Do not deploy capital until flagged issues resolve."}
    p.append(intro.get(tier, f"**{name}**: monitor."))
    pos = []
    if is_valid(roe) and roe > 20: pos.append(f"ROE {roe}%")
    if is_valid(cfo) and cfo > 0.7: pos.append(f"cash conversion {cfo}x")
    if is_valid(peg) and peg < 1.0: pos.append(f"PEG {peg}")
    if is_valid(fy) and fy > 3: pos.append(f"FCF yield {fy}%")
    if moat.get('consistency') == 'strong moat': pos.append(f"durable moat ({moat['pct']}% years ROE>15%)")
    if is_valid(l1.get('sales_cagr')) and l1['sales_cagr'] > 20: pos.append(f"{l1['sales_cagr']}% revenue CAGR")
    if pos: p.append("**Strengths:** " + ", ".join(pos) + ".")
    con = []
    if s < 70: con.append(f"forensic score {s}/100")
    if is_valid(peg) and peg > 2: con.append(f"PEG {peg}")
    if is_valid(ret) and ret < -30: con.append(f"{abs(ret):.0f}% drawdown")
    if cyc.get('peak'): con.append(f"cyclical peak ROE")
    for f in acct.get('flags',[])[:2]: con.append(f.lower())
    if con: p.append("**Risks:** " + ", ".join(con) + ".")
    acts = {'FULL':"Deploy 12-15%. Confirm with latest concall.",'STANDARD':"Position 8-10%. Hold and compound.",
            'HALF':"4-6% only. Wait for next results.",'WATCH':"Do not buy. Monitor quarterly."}
    p.append(f"**Action:** {acts.get(tier,'Monitor.')}")
    return "\n\n".join(p)

# ============================================================
# 8. DATA QUALITY SCORE
# ============================================================
def data_quality(sd, l1):
    """Score how much data we actually got: 0-5."""
    pts = 0
    if is_valid(l1.get('price')): pts += 1
    if is_valid(l1.get('pe')): pts += 1
    if is_valid(l1.get('sales_cagr')): pts += 1
    if sd.get('ph') is not None and len(sd.get('ph',[])) > 100: pts += 1
    if sd.get('fin') is not None: pts += 1
    return pts

# ============================================================
# 9. BATCH SCREENER
# ============================================================
def analyse_quick(ticker):
    try:
        sd = fetch(ticker)
        if not sd: return None
        if is_banking(sd['info']): return None
        l1 = run_layer1(sd, ticker)
        if not l1['mcap_pass'] or not l1['de_pass']: return None
        if is_valid(l1['roe']) and l1['roe'] < 15: return None
        if l1['roe'] is None: return None
        multi = build_multi(sd['fin'], sd['bs'], sd['cf'])
        if not multi: return None
        acct = run_forensic(multi)
        cyc = run_cyclical(multi)
        moat = run_moat(multi)
        val = run_valuation(l1['pe'], l1['pat_cagr'], l1['price'], ext_mcap(sd, l1['price']), sd)
        ret = ext_1y_ret(sd)
        tier, size = get_tier(acct['score'], acct['num_flags'], cyc.get('peak',False), val.get('peg'), ret, moat.get('pct',0))
        return {'ticker':ticker.replace('.NS',''), 'name':l1['name'], 'sector':l1['sector'], 'price':l1['price'],
                'pe':l1['pe'], 'peg':val.get('peg'), 'roe':l1['roe'], 'roce':l1['roce'],
                'score':acct['score'], 'nf':acct['num_flags'], 'cum_cfo':acct.get('cum_cfo_pat'),
                'peak':cyc.get('peak',False), 'ret':ret, 'tier':tier, 'size':size,
                'sc':l1.get('sales_cagr'), 'pc':l1.get('pat_cagr'), 'moat':moat.get('pct',0),
                'fcf_yield':val.get('fcf_yield')}
    except: return None

# ============================================================
# APP LAYOUT
# ============================================================
st.markdown('<p class="main-title">📊 High Compounder Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">7-Layer Systematic Indian Equity Screener — Quality · Valuation · Momentum</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Single Stock","Auto Top 10","Live Tracker","How It Works"], label_visibility="collapsed")
st.sidebar.markdown("### Current Portfolio")
st.sidebar.dataframe(pd.DataFrame({
    "Stock":["LUPIN","DIXON","ENRIN","BSE","MCX","ICICI AMC","EICHER","KPIT","POLYCAB","HDFC AMC"],
    "Tier":["FULL","FULL","FULL","STD","STD","STD","STD","HALF","HALF","HALF"],
    "Score":[100,100,100,85,85,85,85,100,55,70],
    "PEG":[0.14,0.54,0.91,0.38,0.51,1.46,1.60,1.48,1.67,1.42]
}), hide_index=True, use_container_width=True)
st.sidebar.caption("Built by Vinayak Nagral · Sep 2026")

# ============================================================
# PAGE: SINGLE STOCK
# ============================================================
if page == "Single Stock":
    c1, c2 = st.columns([4,1])
    with c1: ticker_in = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="LUPIN, INFY, TCS, DIXON").strip().upper()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Analyse", type="primary", use_container_width=True)
    qc = st.columns(7)
    for i, q in enumerate(["LUPIN","BSE","DIXON","INFY","HDFCAMC","MAZDOCK","HCLTECH"]):
        with qc[i]:
            if st.button(q, key=f"q_{q}", use_container_width=True): ticker_in = q; go = True
    if not ticker_in.endswith(".NS"): ticker_in += ".NS"

    if go:
        with st.spinner(f"Fetching {ticker_in.replace('.NS','')}..."):
            sd = fetch(ticker_in)
        if not sd: st.error("Could not fetch data. Check ticker."); st.stop()

        info = sd['info']
        fin, bs, cf, qfin = sd['fin'], sd['bs'], sd['cf'], sd['qfin']
        is_bank = is_banking(info, ticker_in)

        with st.spinner("Running 7-layer analysis..."):
            l1 = run_layer1(sd, ticker_in)
            multi = build_multi(fin, bs, cf)
            acct = run_forensic(multi) if multi else {'score':None,'flags':['No data'],'num_flags':0,'cum_cfo_pat':None,'cum_cfo_ebitda':None,'single_cfo_pat':None,'cfo_ebitda_trend':[],'recv_pcts':[],'recv_years':[],'margins':[]}
            cyc = run_cyclical(multi) if multi else {'peak':False,'roe_by_year':{}}
            moat = run_moat(multi) if multi else {'pct':0,'consistency':'no data','years_above_15':0,'total_years':0}
            mom = run_momentum(qfin)
            ret = ext_1y_ret(sd)
            mcap = ext_mcap(sd, l1['price'])
            val = run_valuation(l1['pe'], l1['pat_cagr'], l1['price'], mcap, sd)

        tier, size = get_tier(acct.get('score'), acct.get('num_flags',0), cyc.get('peak',False), val.get('peg'), ret, moat.get('pct',0))
        if not l1['pass'] and tier in ['FULL','STANDARD']: tier, size = 'HALF', '4-6%'
        dq = data_quality(sd, l1)

        name, sector, price = l1['name'], l1['sector'], l1['price']

        # HEADER
        st.markdown(f'<p class="stock-name">{name}</p>', unsafe_allow_html=True)
        dq_cls = 'dq-high' if dq >= 4 else 'dq-med' if dq >= 2 else 'dq-low'
        dq_lbl = 'High' if dq >= 4 else 'Medium' if dq >= 2 else 'Low'
        price_s = f"₹{price:,.2f}" if is_valid(price) else "—"
        st.markdown(f'<p class="stock-meta">{ticker_in} · {sector} · {price_s} &nbsp; <span class="data-quality {dq_cls}">Data: {dq_lbl}</span></p>', unsafe_allow_html=True)
        if is_bank: st.markdown('<div class="banking-box">⚠️ Banking/Financial — framework designed for non-financials. Scores for reference.</div>', unsafe_allow_html=True)
        st.markdown("---")

        # METRICS ROW
        cols = st.columns(7)
        labels = ["Score","PE","PEG","ROE","1Y Return","FCF Yield","Tier"]
        values = [f"{acct.get('score','—')}/100" if acct.get('score') is not None else "—",
                  fmt(l1['pe'],'.1f'), fmt(val.get('peg'),'.2f'), fmt(l1['roe'],'.1f','%'),
                  fmt(ret,'+.1f','%'), fmt(val.get('fcf_yield'),'.1f','%'), ""]
        for i, (lb, vl) in enumerate(zip(labels, values)):
            with cols[i]:
                if lb == "Tier":
                    tc = tier.lower()
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Tier</div><span class="tier-badge tier-{tc}">{tier} · {size}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{lb}</div><div class="metric-value">{vl}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # LAYERS
        st.subheader("Layer-by-Layer Breakdown")
        layers = gen_layers(l1, acct, cyc, val, mom, ret, moat, is_bank)
        for ln, status, detail in layers:
            icon = "✅" if status=="pass" else "❌" if status=="fail" else "⚠️"
            st.markdown(f'<div class="layer-row layer-{status}">{icon} <strong>{ln}:</strong> {detail}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Investment Verdict")
        v = gen_verdict(name, tier, size, l1, acct, cyc, val, ret, mom, moat, is_bank)
        st.markdown(f'<div class="verdict-box">{v}</div>', unsafe_allow_html=True)

        # TABS
        st.markdown("---")
        t1,t2,t3,t4 = st.tabs(["Forensic","Cyclical & Valuation","Momentum","Debug"])

        with t1:
            if acct.get('score') is not None:
                sc = acct['score']
                css = "score-green" if sc>=85 else "score-yellow" if sc>=50 else "score-red"
                lbl = "CLEAN" if sc>=85 else "FLAGS" if sc>=50 else "CONCERN"
                st.markdown(f'<div class="score-card {css}"><h2>{sc}/100</h2><p>{lbl}</p></div>', unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Cumulative CFO/PAT**")
                    v2 = acct.get('cum_cfo_pat')
                    if is_valid(v2): st.markdown(f"{'✅' if v2>=0.7 else '⚠️' if v2>=0.5 else '❌'} **{v2}x**")
                    sv = acct.get('single_cfo_pat')
                    if is_valid(sv): st.caption(f"Latest year: {sv}x")
                with cb:
                    st.markdown("**Cumulative CFO/EBITDA**")
                    v3 = acct.get('cum_cfo_ebitda')
                    if is_valid(v3): st.markdown(f"{'✅' if v3>=0.7 else '⚠️' if v3>=0.5 else '❌'} **{v3}x**")
                    tr = acct.get('cfo_ebitda_trend',[])
                    if tr: st.caption(f"Trend: {' → '.join(str(x) for x in tr)}")
                # Margin trend
                margins = acct.get('margins',[])
                mt = acct.get('margin_trend','')
                if margins:
                    st.markdown(f"**EBITDA Margins:** {' → '.join(str(m)+'%' for m in margins)} {'📈' if mt=='expanding' else '📉' if mt=='contracting' else '➡️'}")
                if acct.get('flags'):
                    for f in acct['flags']: st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="clean-item">✅ All checks passed</div>', unsafe_allow_html=True)
            else: st.warning("Insufficient data")

        with t2:
            cc, cd = st.columns(2)
            with cc:
                st.markdown("**Cyclical ROE**")
                if cyc.get('roe_by_year'):
                    st.dataframe(pd.DataFrame(list(cyc['roe_by_year'].items()), columns=['Year','ROE%']), hide_index=True, use_container_width=True)
                    st.markdown(f"Latest **{cyc.get('latest','—')}%** · Norm **{cyc.get('median','—')}%**")
                    st.success("Not at peak ✅") if not cyc.get('peak') else st.error("CYCLICAL PEAK ⚠️")
                st.markdown("**Moat Durability**")
                st.markdown(f"ROE > 15% in **{moat.get('years_above_15',0)}/{moat.get('total_years',0)}** years → **{moat.get('consistency','—')}**")
            with cd:
                st.markdown("**Valuation**")
                st.markdown(f"PE: {fmt(l1['pe'],'.1f')} · PEG: {fmt(val.get('peg'),'.2f')} · FCF Yield: {fmt(val.get('fcf_yield'),'.1f','%')}")
                st.markdown(f"Sales CAGR: {fmt(l1.get('sales_cagr'),'.1f')}% · PAT CAGR: {fmt(l1.get('pat_cagr'),'.1f')}%")

        with t3:
            if mom.get('available'):
                mc1, mc2 = st.columns(2)
                mc1.metric("Latest QoQ", fmt(mom.get('latest_qoq'),'+.1f','%'))
                mc2.metric("Prior QoQ", fmt(mom.get('prior_qoq'),'+.1f','%'))
                if mom.get('eps') and mom.get('quarters'):
                    st.dataframe(pd.DataFrame({'Quarter':mom['quarters'],'EPS':[round(e,2) for e in mom['eps']]}), hide_index=True, use_container_width=True)
            else: st.warning("Insufficient quarterly data")
            st.markdown(f"**1Y Return:** {fmt(ret,'+.1f','%')}")

        with t4:
            st.markdown("**Data sources received:**")
            key_f = ['currentPrice','trailingPE','trailingEps','marketCap','sharesOutstanding','shortName','sector','debtToEquity']
            rows = [{'Field':k,'Value':str(info.get(k,'—'))[:40],'OK':'✅' if is_valid(info.get(k)) or (isinstance(info.get(k),str) and info.get(k).strip()) else '❌'} for k in key_f]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.markdown(f"Financials: {'✅' if fin is not None else '❌'} · BS: {'✅' if bs is not None else '❌'} · CF: {'✅' if cf is not None else '❌'} · Quarterly: {'✅' if qfin is not None else '❌'}")
            ph = sd.get('ph')
            st.markdown(f"Price History: {'✅ '+str(len(ph))+' days' if ph is not None and len(ph)>0 else '❌'}")
            shares = get_shares(fin, bs, qfin, info)
            st.markdown(f"Shares: {shares:,.0f}" if is_valid(shares) else "Shares: ❌")
            st.markdown(f"**Computed:** Price={l1['price']} · PE={l1['pe']} · MCap Cr={l1.get('mcap_cr')}")

        st.caption("Research only, not investment advice · Data from Yahoo Finance")

# ============================================================
# PAGE: AUTO TOP 10
# ============================================================
elif page == "Auto Top 10":
    st.subheader("Automatic Top 10 Picker")
    st.markdown("Screens Nifty 200 through all 7 layers.")
    c1, c2 = st.columns(2)
    with c1: mx = st.slider("Stocks to screen", 20, 200, 50, 10)
    with c2: tn = st.slider("Show top N", 5, 20, 10)
    if st.button("Run Screen", type="primary", use_container_width=True):
        try: tickers = [s+".NS" for s in pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty200list.csv")['Symbol'].tolist()][:mx]
        except: st.error("Could not fetch Nifty 200."); st.stop()
        results = []
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            prog.progress((i+1)/len(tickers), f"{t.replace('.NS','')} ({i+1}/{len(tickers)})")
            r = analyse_quick(t)
            if r: results.append(r)
        prog.empty()
        if not results: st.error("No stocks passed."); st.stop()
        df = pd.DataFrame(results)
        ps = lambda x: (90 if x<0.5 else 75 if x<1 else 55 if x<1.5 else 30 if x<2 else 10) if is_valid(x) else 10
        ms = lambda x: (80 if x>30 else 65 if x>0 else 40 if x>-20 else 20 if x>-40 else 5) if is_valid(x) else 40
        cs = lambda x: (100 if x>1.2 else 85 if x>.9 else 65 if x>.7 else 40 if x>.5 else 10) if is_valid(x) else 20
        df['rank'] = df['score'].fillna(0)*.20 + df['peg'].apply(ps)*.25 + df['roe'].fillna(0)*.10 + df['ret'].apply(ms)*.15 + df['cum_cfo'].apply(cs)*.15 + df['moat']*.15
        df = df.sort_values('rank', ascending=False)
        top = df.head(tn)
        st.markdown(f"### Top {tn} from {len(tickers)} screened ({len(results)} passed)")
        for idx, row in top.iterrows():
            rk = list(top.index).index(idx)+1
            ic = {"FULL":"🟢","STANDARD":"🔵","HALF":"🟡","WATCH":"🔴"}.get(row['tier'],"⚪")
            with st.expander(f"#{rk} · {row['ticker']} — {row['name']} · {ic} {row['tier']} · Score {row['score']}/100 · PEG {fmt(row['peg'],'.2f')}", expanded=rk<=3):
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                c1.metric("Price", f"₹{row['price']:,.0f}" if is_valid(row.get('price')) else "—")
                c2.metric("PE", fmt(row.get('pe'),'.1f'))
                c3.metric("ROE", fmt(row.get('roe'),'.1f','%'))
                c4.metric("1Y", fmt(row.get('ret'),'+.1f','%'))
                c5.metric("FCF Yield", fmt(row.get('fcf_yield'),'.1f','%'))
                c6.metric("Moat", f"{row.get('moat',0)}%")
                st.markdown(f"**{row['sector']}** · Sales {fmt(row.get('sc'),'.1f')}% · PAT {fmt(row.get('pc'),'.1f')}% · {row['nf']} flags · Position: **{row['tier']} {row['size']}**")
        st.markdown("---")
        dd = df[['ticker','name','score','pe','peg','roe','cum_cfo','ret','moat','tier','size']].copy()
        dd.columns = ['Ticker','Name','Score','PE','PEG','ROE%','CFO/PAT','1Y%','Moat%','Tier','Size']
        st.dataframe(dd.reset_index(drop=True), use_container_width=True)

# ============================================================
# PAGE: LIVE TRACKER
# ============================================================
elif page == "Live Tracker":
    st.subheader("Live Framework Validation")
    bd = "2026-09-01"
    buys = {"LUPIN.NS":{"n":"Lupin","t":"FULL","s":100,"p":0.14,"r":"PEG 0.14, Score 100"},
            "DIXON.NS":{"n":"Dixon","t":"FULL","s":100,"p":0.54,"r":"EMS champion"},
            "BSE.NS":{"n":"BSE","t":"STD","s":85,"p":0.38,"r":"Capital market monopoly"},
            "EICHERMOT.NS":{"n":"Eicher","t":"STD","s":85,"p":1.60,"r":"Royal Enfield pricing power"},
            "KPITTECH.NS":{"n":"KPIT","t":"HALF","s":100,"p":1.48,"r":"Score 100, -51% momentum"}}
    avoids = {"GODFRYPHLP.NS":{"n":"Godfrey Phillips","t":"WATCH","s":35,"p":0.78,"r":"Tax reset"},
              "WAAREEENER.NS":{"n":"Waaree","t":"WATCH","s":45,"p":0.20,"r":"CFO/PAT 0.44x"},
              "MAZDOCK.NS":{"n":"Mazagon Dock","t":"WATCH","s":25,"p":1.16,"r":"Negative CFO"}}
    bench = "^NSEI"
    @st.cache_data(ttl=3600, show_spinner=False)
    def tracker_prices(tickers, bd):
        res = {}
        start = datetime.strptime(bd,"%Y-%m-%d") - timedelta(days=5)
        for tk in tickers:
            try:
                t = yf.Ticker(tk)
                h = t.history(start=start, end=datetime.now(), auto_adjust=True)
                if h.empty or 'Close' not in h.columns: continue
                pr = h['Close'].dropna()
                if len(pr)==0: continue
                bdt = pd.Timestamp(bd)
                m = pr.index <= bdt
                bp = float(pr.loc[m].iloc[-1]) if m.sum()>0 else float(pr.iloc[0])
                cp = float(pr.iloc[-1])
                if not is_valid(bp) or bp<=0: continue
                ret = round((cp/bp-1)*100,2)
                hist = pr[pr.index >= bdt].copy()
                if not hist.empty:
                    fv = float(hist.iloc[0])
                    if is_valid(fv) and fv>0: hist = (hist/fv-1)*100
                res[tk] = {'bp':round(bp,2),'cp':round(cp,2),'ret':ret if is_valid(ret) else 0,'hist':hist}
            except: continue
        return res

    with st.spinner("Fetching..."): prices = tracker_prices(list(buys)+list(avoids)+[bench], bd)
    if not prices: st.error("No prices."); st.stop()
    br = prices.get(bench,{}).get('ret',0)
    st.markdown(f"**Nifty 50:** {br:+.2f}%")
    st.markdown("---")
    for label, picks, color in [("BUY", buys, "🟢"), ("AVOID", avoids, "🔴")]:
        st.markdown(f"### {color} {label} Picks")
        rows = []
        for tk, inf in picks.items():
            p = prices.get(tk,{})
            r = p.get('ret')
            a = round(r-br,2) if is_valid(r) else None
            rows.append({'Stock':inf['n'],'Tier':inf['t'],'Entry':f"₹{p.get('bp','—')}",
                         'Current':f"₹{p.get('cp','—')}",'Return':f"{r:+.2f}%" if is_valid(r) else "—",
                         'Alpha':f"{a:+.2f}%" if is_valid(a) else "—",'Why':inf['r']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        rets = [prices[t]['ret'] for t in picks if t in prices and is_valid(prices[t].get('ret'))]
        if rets: st.markdown(f"**Avg {label}: {np.mean(rets):+.2f}%**")
    st.markdown("---")
    buy_r = [prices[t]['ret'] for t in buys if t in prices and is_valid(prices[t].get('ret'))]
    avoid_r = [prices[t]['ret'] for t in avoids if t in prices and is_valid(prices[t].get('ret'))]
    if buy_r and avoid_r:
        sp = round(np.mean(buy_r)-np.mean(avoid_r),2)
        al = round(np.mean(buy_r)-br,2)
        if sp>0 and al>0: st.success(f"✅ Framework working. Spread {sp:+.2f}pp, Alpha {al:+.2f}pp")
        elif sp>0: st.info(f"📊 Partial. Spread {sp:+.2f}pp, Alpha {al:+.2f}pp")
        else: st.warning(f"⚠️ Not validated. Spread {sp:+.2f}pp. Give 3-6 months.")
    cd = pd.DataFrame()
    for tk in list(buys)+list(avoids)+[bench]:
        p = prices.get(tk)
        if p and 'hist' in p and not p['hist'].empty:
            cd[buys.get(tk,avoids.get(tk,{})).get('n','Nifty 50')] = p['hist']
    if not cd.empty: st.line_chart(cd)
    st.caption("Tracking from Sep 1, 2026")

# ============================================================
# PAGE: HOW IT WORKS
# ============================================================
elif page == "How It Works":
    st.subheader("Framework Architecture")
    st.markdown("""
**Layer 1 — Quantitative Screen:** MCap > ₹15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, 3Y Sales & PAT CAGR > 15%.

**Layer 2 — Forensic Accounting:** Four cumulative multi-year checks catch receivables stuffing, inventory bloat, and fake profits. Single-year ratios miss lump-sum project businesses — cumulative doesn't. Plus operating margin trajectory analysis.

**Layer 3 — Moat Durability:** Measures what percentage of available years the company sustained ROE > 15%. A company that held high returns for 4/4 years likely has structural protection. One that managed 2/4 may be riding a cycle.

**Layer 4 — Dual Valuation:** PEG (PE ÷ growth) identifies if you're overpaying for growth. FCF Yield (Free Cash Flow ÷ Market Cap) confirms the company generates real cash relative to its price. "Adani Filter" flags PE > 80.

**Layer 5 — Earnings Momentum:** Last 2 quarters QoQ EPS change catches fresh deterioration that backward-looking annual tests miss.

**Layer 6 — Cyclical ROE:** If ROE > 2× historical average → "cyclical peak." Use median ROE for valuation, not the inflated figure.

**Layer 7 — Position Sizing v3:** Integrates quality + moat + valuation + momentum. Clean + durable moat + cheap = FULL (12-15%). Clean + expensive = STANDARD (8-10%). Flags = HALF (4-6%). Failed = WATCH (0%).

---

**Structural exclusions:** Banking/NBFC (need NIM, CASA, GNPA), commodity producers (price-driven CAGR), newly listed (<2 years).

**What this framework doesn't do:** Predict short-term price moves, assess management character, or replace reading annual reports and concall transcripts.
    """)
    st.caption("Built by Vinayak Nagral · September 2026")
