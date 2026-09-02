import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.utils import simpleSplit
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="High Compounder Framework", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

NAVY = '#1B2A4A'
BANKING_KW = ['bank','finance','insurance','nbfc','housing finance','credit','lending','microfinance']

# ============================================================
# STYLING
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.main-title {{ font-family:'DM Sans',sans-serif; font-size:1.75rem; font-weight:700; color:{NAVY}; }}
.sub-title {{ font-size:0.85rem; color:#6B7280; }}
.stock-name {{ font-size:1.5rem; font-weight:700; color:{NAVY}; }}
.stock-meta {{ font-size:0.85rem; color:#6B7280; }}
.metric-card {{ background:#fff; border:1px solid #E5E7EB; border-radius:8px; padding:0.8rem; text-align:center; }}
.metric-val {{ font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600; color:{NAVY}; }}
.metric-lbl {{ font-size:0.7rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:3px; }}
.tier-badge {{ display:inline-block; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.85rem; }}
.tier-full {{ background:#ECFDF5; color:#059669; border:1px solid #059669; }}
.tier-standard {{ background:#EFF6FF; color:#2563EB; border:1px solid #2563EB; }}
.tier-half {{ background:#FFFBEB; color:#D97706; border:1px solid #D97706; }}
.tier-watch {{ background:#FEF2F2; color:#DC2626; border:1px solid #DC2626; }}
.layer-row {{ padding:0.5rem 0.8rem; border-radius:4px; margin:3px 0; font-size:0.85rem; border-left:3px solid; }}
.layer-pass {{ background:#ECFDF5; border-color:#059669; }}
.layer-fail {{ background:#FEF2F2; border-color:#DC2626; }}
.layer-warn {{ background:#FFFBEB; border-color:#D97706; }}
.verdict-box {{ background:#F9FAFB; padding:1.2rem; border-radius:8px; border:1px solid #E5E7EB; line-height:1.8; font-size:0.9rem; }}
.flag-item {{ background:#FFFBEB; padding:0.4rem 0.8rem; border-radius:4px; margin:3px 0; border-left:3px solid #D97706; font-size:0.85rem; }}
.clean-item {{ background:#ECFDF5; padding:0.4rem 0.8rem; border-radius:4px; margin:3px 0; border-left:3px solid #059669; font-size:0.85rem; }}
.score-card {{ padding:1rem; border-radius:8px; text-align:center; }}
.score-green {{ background:#ECFDF5; border:1px solid #059669; }}
.score-yellow {{ background:#FFFBEB; border:1px solid #D97706; }}
.score-red {{ background:#FEF2F2; border:1px solid #DC2626; }}
.funnel-row {{ display:flex; align-items:center; gap:10px; margin:4px 0; font-size:0.85rem; }}
.funnel-bar {{ height:24px; background:{NAVY}; border-radius:4px; display:flex; align-items:center; padding:0 8px; color:white; font-size:0.75rem; font-weight:500; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITIES
# ============================================================
def is_valid(val):
    if val is None: return False
    try: return bool(np.isfinite(float(val)))
    except (TypeError, ValueError, OverflowError): return False

def safe_get(df, label, col=0):
    try:
        if df is not None and label in df.index:
            v = df.loc[label].iloc[col]
            if pd.notna(v): return float(v)
    except Exception: pass
    return None

def fmt(val, f=".1f", suffix="", prefix=""):
    if not is_valid(val): return "N/A"
    return f"{prefix}{val:{f}}{suffix}"

def get_revenue(fin, col=0):
    for field in ['Total Revenue','Operating Revenue','Revenue','Net Revenue']:
        v = safe_get(fin, field, col)
        if v and v > 0: return v
    return None

def get_ni(fin, col=0):
    for field in ['Net Income','Net Income Common Stockholders','Net Income Continuous Operations']:
        v = safe_get(fin, field, col)
        if v is not None: return v
    return None

def get_ebitda(fin, col=0):
    for field in ['EBITDA','Normalized EBITDA']:
        v = safe_get(fin, field, col)
        if v and v > 0: return v
    return None

def get_equity(bs, col=0):
    for field in ['Stockholders Equity','Total Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest']:
        v = safe_get(bs, field, col)
        if v and v > 0: return v
    return None

def get_cfo(cf, col=0):
    for field in ['Operating Cash Flow','Total Cash From Operating Activities','Cash Flow From Continuing Operating Activities']:
        v = safe_get(cf, field, col)
        if v is not None: return v
    return None

def get_fcf(cf, col=0):
    v = safe_get(cf, 'Free Cash Flow', col)
    if v is not None: return v
    cfo = get_cfo(cf, col)
    capex = safe_get(cf, 'Capital Expenditure', col)
    if is_valid(cfo) and is_valid(capex): return cfo + capex
    return cfo

def get_shares(fin, bs, qfin, info):
    for src in [qfin, fin]:
        if src is not None:
            for field in ['Diluted Average Shares','Basic Average Shares']:
                v = safe_get(src, field, 0)
                if is_valid(v) and v > 1000: return v
    if bs is not None:
        for field in ['Ordinary Shares Number','Share Issued']:
            v = safe_get(bs, field, 0)
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
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker):
    try:
        t = yf.Ticker(ticker)
        try: info = {k:v for k,v in (dict(t.info) if t.info else {}).items() if v is not None}
        except Exception: info = {}
        try:
            fi = t.fast_info
            if fi:
                for a, k in [('last_price','currentPrice'),('previous_close','previousClose'),('market_cap','marketCap'),('shares','sharesOutstanding')]:
                    if not is_valid(info.get(k)):
                        try:
                            v = getattr(fi, a, None)
                            if is_valid(v) and v > 0: info[k] = float(v)
                        except Exception: pass
        except Exception: pass
        fin = t.financials.copy() if t.financials is not None and not t.financials.empty else None
        bs = t.balance_sheet.copy() if t.balance_sheet is not None and not t.balance_sheet.empty else None
        cf = t.cashflow.copy() if t.cashflow is not None and not t.cashflow.empty else None
        qfin = t.quarterly_financials.copy() if t.quarterly_financials is not None and not t.quarterly_financials.empty else None
        ph = None
        try:
            h = t.history(period="1y", auto_adjust=True)
            if h is not None and not h.empty and 'Close' in h.columns:
                ph = h['Close'].dropna()
                if len(ph) > 0 and not is_valid(info.get('currentPrice')): info['currentPrice'] = float(ph.iloc[-1])
        except Exception: pass
        if ph is None or len(ph) == 0:
            try:
                h = t.history(period="5d", auto_adjust=True)
                if h is not None and not h.empty and 'Close' in h.columns:
                    ph = h['Close'].dropna()
                    if len(ph) > 0 and not is_valid(info.get('currentPrice')): info['currentPrice'] = float(ph.iloc[-1])
            except Exception: pass
        if not is_valid(info.get('sharesOutstanding')):
            sh = get_shares(fin, bs, qfin, {})
            if sh: info['sharesOutstanding'] = sh
        return {"info":info,"fin":fin,"bs":bs,"cf":cf,"qfin":qfin,"ph":ph}
    except Exception: return None

# ============================================================
# METRIC EXTRACTORS
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
    info = sd['info']
    for f in ['trailingPE','forwardPE']:
        v = info.get(f)
        if is_valid(v) and 1 < v < 500: return round(float(v), 1)
    if not is_valid(price) or price <= 0: return None
    for f in ['trailingEps','forwardEps']:
        eps = info.get(f)
        if is_valid(eps) and eps > 0:
            pe = price / eps
            if 1 < pe < 500: return round(pe, 1)
    qfin = sd.get('qfin')
    if qfin is not None and qfin.shape[1] >= 4:
        ttm = sum(filter(None, [get_ni(qfin, i) for i in range(4)]))
        if ttm and ttm > 0:
            sh = get_shares(sd.get('fin'), sd.get('bs'), qfin, info)
            if is_valid(sh) and sh > 0:
                pe = price / (ttm / sh)
                if 1 < pe < 500: return round(pe, 1)
    fin = sd.get('fin')
    if fin is not None:
        ni = get_ni(fin, 0)
        if is_valid(ni) and ni > 0:
            sh = get_shares(fin, sd.get('bs'), sd.get('qfin'), info)
            if is_valid(sh) and sh > 0:
                pe = price / (ni / sh)
                if 1 < pe < 500: return round(pe, 1)
    return None

def ext_mcap(sd, price):
    v = sd['info'].get('marketCap')
    if is_valid(v) and v > 0: return float(v)
    if is_valid(price) and price > 0:
        sh = get_shares(sd.get('fin'), sd.get('bs'), sd.get('qfin'), sd['info'])
        if is_valid(sh) and sh > 0: return price * sh
    return None

def ext_de(sd):
    v = sd['info'].get('debtToEquity')
    if is_valid(v): return round(float(v) / 100, 2)
    bs = sd.get('bs')
    if bs:
        debt = safe_get(bs, 'Total Debt', 0) or safe_get(bs, 'Long Term Debt', 0) or 0
        eq = get_equity(bs, 0)
        if eq and eq > 0: return round(debt / eq, 2)
    return 0.0

def ext_cagr(fin, getter, min_years=0.8):
    if fin is None or fin.shape[1] < 2: return None
    lv, ld, ov, od = None, None, None, None
    for i in range(fin.shape[1]):
        v = getter(fin, i)
        if is_valid(v) and v > 0:
            d = fin.columns[i]
            if lv is None: lv, ld = v, d
            ov, od = v, d
    if lv is None or ov is None or ld == od: return None
    try: yrs = abs((ld - od).days) / 365.25
    except Exception: yrs = float(abs(fin.columns.tolist().index(ld) - fin.columns.tolist().index(od)))
    if yrs < min_years: return None
    try:
        c = ((lv / ov) ** (1.0 / yrs) - 1) * 100
        return round(c, 1) if is_valid(c) and -99 < c < 500 else None
    except Exception: return None

def ext_cagr_q(qfin, getter):
    if qfin is None or qfin.shape[1] < 5: return None
    n = qfin.shape[1]
    lt = sum(filter(None, [getter(qfin, i) for i in range(min(4, n))]))
    if n >= 8:
        ot = sum(filter(None, [getter(qfin, i) for i in range(4, 8)]))
        try: yrs = abs((qfin.columns[0] - qfin.columns[4]).days) / 365.25
        except Exception: yrs = 1.0
    else:
        ot = sum(filter(None, [getter(qfin, i) for i in range(n - 4, n)]))
        try: yrs = abs((qfin.columns[0] - qfin.columns[n - 4]).days) / 365.25
        except Exception: yrs = 1.0
    if lt and lt > 0 and ot and ot > 0 and yrs > 0.5:
        try:
            c = ((lt / ot) ** (1.0 / yrs) - 1) * 100
            return round(c, 1) if is_valid(c) and -99 < c < 500 else None
        except Exception: return None
    return None

def ext_roe(fin, bs, info):
    ni = get_ni(fin, 0) if fin is not None else None
    eq = get_equity(bs, 0) if bs is not None else None
    if is_valid(ni) and is_valid(eq) and eq > 0: return round(ni / eq * 100, 1)
    v = info.get('returnOnEquity')
    if is_valid(v): return round(float(v) * 100, 1)
    return None

def ext_roce(fin, bs):
    ebit = safe_get(fin, 'EBIT', 0) if fin is not None else None
    ta = safe_get(bs, 'Total Assets', 0) if bs is not None else None
    cl = safe_get(bs, 'Current Liabilities', 0) if bs is not None else None
    if is_valid(ebit) and is_valid(ta) and is_valid(cl) and (ta - cl) > 0: return round(ebit / (ta - cl) * 100, 1)
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
            r = round((cur / past - 1) * 100, 1)
            return r if is_valid(r) else None
    except Exception: pass
    return None

# ============================================================
# FRAMEWORK LAYERS
# ============================================================
def build_multi(fin, bs, cf):
    if fin is None or bs is None or cf is None: return None
    yrs = min(fin.shape[1], bs.shape[1], cf.shape[1])
    if yrs < 2: return None
    return [{'year': str(fin.columns[i].year) if hasattr(fin.columns[i], 'year') else str(i),
             'revenue': get_revenue(fin, i), 'net_income': get_ni(fin, i), 'ebitda': get_ebitda(fin, i),
             'ebit': safe_get(fin, 'EBIT', i),
             'receivables': safe_get(bs, 'Accounts Receivable', i) or safe_get(bs, 'Net Receivables', i) or safe_get(bs, 'Receivables', i),
             'inventory': safe_get(bs, 'Inventory', i), 'equity': get_equity(bs, i),
             'current_liabilities': safe_get(bs, 'Current Liabilities', i),
             'total_assets': safe_get(bs, 'Total Assets', i), 'cfo': get_cfo(cf, i)} for i in range(yrs)]

def run_layer1(sd, ticker):
    r = {}; info = sd['info']; fin = sd.get('fin'); bs = sd.get('bs')
    r['price'] = ext_price(sd)
    mcap = ext_mcap(sd, r['price'])
    r['mcap_cr'] = round(mcap / 1e7) if is_valid(mcap) else None
    r['mcap_pass'] = is_valid(mcap) and mcap > 150e9
    r['pe'] = ext_pe(sd, r['price'])
    r['de'] = ext_de(sd); r['de_pass'] = r['de'] < 0.5
    r['roe'] = ext_roe(fin, bs, info); r['roe_pass'] = is_valid(r['roe']) and r['roe'] > 15
    r['roce'] = ext_roce(fin, bs); r['roce_pass'] = is_valid(r['roce']) and r['roce'] > 18
    r['sales_cagr'] = ext_cagr(fin, get_revenue) or ext_cagr_q(sd.get('qfin'), get_revenue) or (round(float(info['revenueGrowth']) * 100, 1) if is_valid(info.get('revenueGrowth')) else None)
    r['pat_cagr'] = ext_cagr(fin, get_ni) or ext_cagr_q(sd.get('qfin'), get_ni) or (round(float(info['earningsGrowth']) * 100, 1) if is_valid(info.get('earningsGrowth')) else None)
    r['growth_pass'] = (r.get('sales_cagr') or 0) > 15 and (r.get('pat_cagr') or 0) > 15
    r['pass'] = all([r['mcap_pass'], r['de_pass'], r['roe_pass'], r['roce_pass'], r['growth_pass']])
    r['name'] = ext_name(sd, ticker); r['sector'] = ext_sector(sd)
    return r

def run_forensic(data):
    flags, score, det = [], 100, {}
    sd = sorted(data, key=lambda x: x['year']); latest = sd[-1]; prior = sd[-2] if len(sd) >= 2 else None
    rp = [(round(d['receivables'] / d['revenue'] * 100, 1) if is_valid(d.get('receivables')) and is_valid(d.get('revenue')) and d['revenue'] > 0 else None) for d in sd]
    det['recv_pcts'] = rp; det['recv_years'] = [d['year'] for d in sd]
    vr = [x for x in rp if x is not None]
    if len(vr) >= 3 and sum(1 for i in range(1, len(vr)) if vr[i] > vr[i - 1]) >= 2:
        flags.append(f"Receivables rising: {vr[-3]}% → {vr[-2]}% → {vr[-1]}% of revenue"); score -= 15
    if prior:
        il, ip, rl, rp2 = latest.get('inventory'), prior.get('inventory'), latest.get('revenue'), prior.get('revenue')
        if all(is_valid(v) and v > 0 for v in [il, ip, rl, rp2]):
            ig, rg = round((il / ip - 1) * 100, 1), round((rl / rp2 - 1) * 100, 1)
            det['inv_growth'] = ig; det['rev_growth'] = rg
            if ig > rg + 10: flags.append(f"Inventory bloat: {ig}% vs Revenue {rg}%"); score -= 15
    tc = sum(d['cfo'] for d in sd if is_valid(d.get('cfo'))); tp = sum(d['net_income'] for d in sd if is_valid(d.get('net_income')))
    if tp > 0:
        det['cum_cfo_pat'] = round(tc / tp, 2)
        if det['cum_cfo_pat'] < 0.5: flags.append(f"Critical: Cum CFO/PAT {det['cum_cfo_pat']}x"); score -= 30
        elif det['cum_cfo_pat'] < 0.7: flags.append(f"Low cash: Cum CFO/PAT {det['cum_cfo_pat']}x"); score -= 20
    else: det['cum_cfo_pat'] = None
    det['single_cfo_pat'] = round(latest['cfo'] / latest['net_income'], 2) if is_valid(latest.get('cfo')) and is_valid(latest.get('net_income')) and latest['net_income'] > 0 else None
    te = sum(d['ebitda'] for d in sd if is_valid(d.get('ebitda')) and d['ebitda'] > 0)
    if te > 0:
        det['cum_cfo_ebitda'] = round(tc / te, 2)
        if det['cum_cfo_ebitda'] < 0.5: flags.append(f"Critical: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x"); score -= 20
        elif det['cum_cfo_ebitda'] < 0.7: flags.append(f"Weak: Cum CFO/EBITDA {det['cum_cfo_ebitda']}x"); score -= 15
    else: det['cum_cfo_ebitda'] = None
    yr = [round(d['cfo'] / d['ebitda'], 2) for d in sd if is_valid(d.get('cfo')) and is_valid(d.get('ebitda')) and d['ebitda'] > 0]
    det['cfo_ebitda_trend'] = yr
    if len(yr) >= 2 and is_valid(det.get('cum_cfo_ebitda')) and yr[-1] < yr[-2] and yr[-1] < 0.5 and det['cum_cfo_ebitda'] < 0.7:
        flags.append(f"Deteriorating: CFO/EBITDA {yr[-2]} → {yr[-1]}"); score -= 10
    if tc < 0: flags.append("Negative cumulative CFO"); score -= 25
    margins = [round(d['ebitda'] / d['revenue'] * 100, 1) for d in sd if is_valid(d.get('ebitda')) and is_valid(d.get('revenue')) and d['revenue'] > 0]
    det['margins'] = margins
    det['margin_trend'] = ('expanding' if len(margins) >= 2 and margins[-1] > margins[0] + 2 else 'contracting' if len(margins) >= 2 and margins[-1] < margins[0] - 3 else 'stable')
    det.update({'score': max(score, 0), 'flags': flags, 'num_flags': len(flags)}); return det

def run_moat(data):
    sd = sorted(data, key=lambda x: x['year'])
    rh = [round(d['net_income'] / d['equity'] * 100, 1) for d in sd if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity'] > 0]
    if not rh: return {'years_above_15': 0, 'total_years': 0, 'pct': 0, 'consistency': 'no data', 'roe_history': []}
    above = sum(1 for r in rh if r > 15); total = len(rh); pct = round(above / total * 100) if total > 0 else 0
    cons = 'strong moat' if pct >= 80 else 'moderate moat' if pct >= 60 else 'weak/no moat'
    return {'years_above_15': above, 'total_years': total, 'pct': pct, 'consistency': cons, 'roe_history': rh}

def run_cyclical(data):
    sd = sorted(data, key=lambda x: x['year']); rv = []; rby = {}
    for d in sd:
        if is_valid(d.get('net_income')) and is_valid(d.get('equity')) and d['equity'] > 0:
            r = round(d['net_income'] / d['equity'] * 100, 1); rv.append(r); rby[d['year']] = r
    if len(rv) < 2: return {'peak': False, 'roe_by_year': {}}
    l, a, m = rv[-1], round(np.mean(rv), 1), round(np.median(rv), 1)
    return {'latest': l, 'avg': a, 'median': m, 'peak': l > a * 2 and l > 20, 'values': rv, 'roe_by_year': rby}

def run_momentum(qfin):
    if qfin is None or qfin.shape[1] < 3: return {'available': False}
    ev, qs = [], []
    for i in range(min(4, qfin.shape[1])):
        e = safe_get(qfin, 'Diluted EPS', i) or safe_get(qfin, 'Basic EPS', i)
        if e is not None: ev.append(e); qs.append(str(qfin.columns[i].strftime('%b %Y')) if hasattr(qfin.columns[i], 'strftime') else str(i))
    if len(ev) < 3: return {'available': False}
    lq = round((ev[0] / ev[1] - 1) * 100, 1) if ev[1] != 0 else None
    pq = round((ev[1] / ev[2] - 1) * 100, 1) if ev[2] != 0 else None
    if not is_valid(lq): lq = None
    if not is_valid(pq): pq = None
    return {'available': True, 'latest_qoq': lq, 'prior_qoq': pq, 'eps': ev[:4], 'quarters': qs[:4]}

def run_valuation(pe, pat_cagr, price, mcap, sd):
    peg = round(pe / pat_cagr, 2) if is_valid(pe) and is_valid(pat_cagr) and pat_cagr > 0 else None
    fcf_yield = None
    cf = sd.get('cf')
    if cf is not None and is_valid(mcap) and mcap > 0:
        fcf = get_fcf(cf, 0)
        if is_valid(fcf): fcf_yield = round(fcf / mcap * 100, 2)
    return {'peg': peg, 'fcf_yield': fcf_yield}

def get_tier(score, nf, peak, peg, mom_1y, moat_pct):
    if score is None or score < 50: return 'WATCH', '0%'
    if is_valid(peg) and peg > 5: return 'WATCH', '0%'
    if is_valid(peg) and peg > 3: return 'HALF', '4-6%'
    if score >= 85 and nf == 0 and not peak and moat_pct >= 60: base = 'FULL'
    elif score >= 85 and nf == 0 and not peak: base = 'STANDARD'
    elif score >= 70 and nf <= 1: base = 'STANDARD'
    else: base = 'HALF'
    if base == 'FULL' and not is_valid(peg): return 'STANDARD', '8-10%'
    if base == 'FULL' and peg < 0.5: return 'FULL', '12-15%'
    if base == 'FULL' and is_valid(mom_1y) and mom_1y < -30: return 'HALF', '4-6%'
    if base == 'FULL' and peg > 1.5: return 'STANDARD', '8-10%'
    return {'FULL': ('FULL', '12-15%'), 'STANDARD': ('STANDARD', '8-10%'), 'HALF': ('HALF', '4-6%')}[base]

# ============================================================
# VERDICT PARAGRAPH
# ============================================================
def gen_verdict(name, tier, l1, acct, cyc, val, ret, moat, is_bank):
    if is_bank: return f"{name} is a banking/financial stock. This framework uses ROE, D/E, and CFO/EBITDA metrics for non-financials. Banks need NIM, CASA, Credit Cost, GNPA. Scores for reference only."
    s = acct.get('score', 0) or 0; peg = val.get('peg'); fy = val.get('fcf_yield'); roe = l1.get('roe'); de = l1.get('de')
    cfo = acct.get('cum_cfo_pat'); margins = acct.get('margins', []); mt = acct.get('margin_trend', ''); mp = moat.get('pct', 0)
    sc, pc = l1.get('sales_cagr'), l1.get('pat_cagr')
    sents = []
    if tier == 'FULL': sents.append(f"{name} passes all seven layers with conviction.")
    elif tier == 'STANDARD': sents.append(f"{name} clears the quality bar but one or more factors prevent full conviction.")
    elif tier == 'HALF': sents.append(f"{name} presents a mixed profile — genuine strengths alongside flags that need resolution.")
    else: sents.append(f"{name} fails critical checks and does not warrant capital deployment at current readings.")
    q = []
    if is_valid(roe) and roe > 15: q.append(f"ROE at {roe}%{' on ' + str(de) + 'x D/E' if is_valid(de) else ''}")
    if is_valid(cfo) and cfo > 1.0: q.append(f"cumulative cash conversion at {cfo}x — the company generates more cash than reported profit")
    elif is_valid(cfo) and cfo > 0.7: q.append(f"cumulative cash conversion at {cfo}x")
    elif is_valid(cfo) and cfo < 0.5: q.append(f"cumulative cash conversion at only {cfo}x — a structural gap between reported profit and actual cash")
    if margins and len(margins) >= 2 and mt == 'expanding': q.append(f"EBITDA margins expanding from {margins[0]}% to {margins[-1]}%")
    elif margins and len(margins) >= 2 and mt == 'contracting': q.append(f"EBITDA margins contracting from {margins[0]}% to {margins[-1]}%")
    if q: sents.append(", ".join(q) + " confirm the earnings quality.")
    if is_valid(sc) and is_valid(pc): sents.append(f"Revenue compounded at {sc}% with PAT at {pc}% over the measurement period.")
    if is_valid(peg):
        if peg < 0.5: sents.append(f"At PEG {peg}, the market prices a fraction of earnings growth into the multiple — the widest valuation gap in the screening universe.")
        elif peg < 1.0: sents.append(f"PEG at {peg} indicates growth more than justifies the PE.")
        elif peg < 1.5: sents.append(f"PEG at {peg} — fairly valued, growth roughly matches what the PE implies.")
        elif peg < 2.0: sents.append(f"PEG at {peg} embeds a valuation premium over what current growth rates justify.")
        else: sents.append(f"PEG at {peg} signals overvaluation.")
    if is_valid(fy) and fy > 5: sents.append(f"FCF yield at {fy}% provides additional margin of safety.")
    elif is_valid(fy) and fy < 0: sents.append(f"Negative FCF yield at {fy}% — the company consumes cash despite reported profitability.")
    if mp >= 80: sents.append(f"Moat durability at {mp}% confirms the competitive advantage is structural, not cyclical.")
    elif mp >= 60: sents.append(f"Moat durability at {mp}% is moderate — above-threshold returns most years but not all.")
    elif moat.get('total_years', 0) >= 2: sents.append(f"Moat durability at only {mp}% raises questions about whether high returns are sustainable.")
    if cyc.get('peak'): sents.append(f"Current ROE at {cyc.get('latest')}% vs median {cyc.get('median')}% indicates a cyclical peak — use the normalized figure for valuation.")
    if is_valid(ret) and ret < -30: sents.append(f"The {abs(ret):.0f}% drawdown signals the market is repricing something negative.")
    if tier in ['HALF', 'WATCH'] and acct.get('flags'): sents.append("Specific flags: " + "; ".join(acct['flags'][:2]).lower() + ".")
    actions = {'FULL': "Position at 12-15% of portfolio.", 'STANDARD': "Position at 8-10%.", 'HALF': "Half position at 4-6% only — add after next results confirm improvement.", 'WATCH': "Do not deploy capital. Monitor quarterly."}
    sents.append(actions.get(tier, "Monitor."))
    return " ".join(sents)

# ============================================================
# LAYER DISPLAY
# ============================================================
def gen_layers(l1, acct, cyc, val, mom, ret, moat):
    L = []
    if l1['pass']: L.append(("Fundamentals", "pass", f"MCap ₹{fmt(l1.get('mcap_cr'), ',.0f')} Cr · ROE {fmt(l1.get('roe'), '.1f')}% · ROCE {fmt(l1.get('roce'), '.1f')}% · D/E {l1.get('de', '—')} · Sales CAGR {fmt(l1.get('sales_cagr'), '.1f')}% · PAT CAGR {fmt(l1.get('pat_cagr'), '.1f')}%"))
    else:
        fails = []
        if not l1['mcap_pass']: fails.append(f"MCap ₹{fmt(l1.get('mcap_cr'), ',.0f')} Cr < ₹15,000 Cr")
        if not l1['roe_pass']: fails.append(f"ROE {fmt(l1.get('roe'), '.1f')}% < 15%")
        if not l1['roce_pass']: fails.append(f"ROCE {fmt(l1.get('roce'), '.1f')}% < 18%")
        if not l1['de_pass']: fails.append(f"D/E {l1.get('de', '—')} > 0.5")
        if not l1['growth_pass']: fails.append(f"Growth: Sales {fmt(l1.get('sales_cagr'), '.1f')}%, PAT {fmt(l1.get('pat_cagr'), '.1f')}%")
        L.append(("Fundamentals", "fail", " · ".join(fails) if fails else "Criteria not met"))
    s = acct.get('score')
    if s is not None:
        if s >= 85: L.append(("Forensic accounting", "pass", f"Score {s}/100 · CFO/PAT {fmt(acct.get('cum_cfo_pat'), '.2f')}x · CFO/EBITDA {fmt(acct.get('cum_cfo_ebitda'), '.2f')}x"))
        elif s >= 50: L.append(("Forensic accounting", "warn", f"Score {s}/100 · {acct.get('num_flags', 0)} flags · {' · '.join(acct.get('flags', [])[:2])}"))
        else: L.append(("Forensic accounting", "fail", f"Score {s}/100 · {acct.get('num_flags', 0)} flags · {' · '.join(acct.get('flags', [])[:2])}"))
    mc = moat.get('consistency', '')
    if moat.get('total_years', 0) >= 2:
        st2 = 'pass' if mc == 'strong moat' else 'warn' if mc == 'moderate moat' else 'fail'
        L.append(("Moat durability", st2, f"ROE>15% in {moat['years_above_15']}/{moat['total_years']} yrs ({moat['pct']}%) — {mc}"))
    peg = val.get('peg')
    if is_valid(peg):
        if peg < 1.0: L.append(("PEG valuation", "pass", f"PEG {peg:.2f}"))
        elif peg < 1.5: L.append(("PEG valuation", "warn", f"PEG {peg:.2f} — fair"))
        elif peg < 2.0: L.append(("PEG valuation", "warn", f"PEG {peg:.2f} — expensive"))
        else: L.append(("PEG valuation", "fail", f"PEG {peg:.2f} — overvalued"))
    else: L.append(("PEG valuation", "warn", "Not calculable"))
    fy = val.get('fcf_yield')
    if is_valid(fy):
        if fy > 3: L.append(("FCF yield", "pass", f"{fy:.1f}%"))
        elif fy > 0: L.append(("FCF yield", "warn", f"{fy:.1f}% — thin"))
        else: L.append(("FCF yield", "fail", f"{fy:.1f}% — negative"))
    pe = l1.get('pe')
    if is_valid(pe) and pe > 80: L.append(("Adani filter", "fail", f"PE {pe:.0f} > 80"))
    if mom.get('available'):
        lq, pq = mom.get('latest_qoq'), mom.get('prior_qoq')
        if is_valid(lq) and is_valid(pq):
            if lq > 0 and pq > 0: L.append(("Earnings momentum", "pass", f"+{lq:.1f}%, +{pq:.1f}% QoQ"))
            elif lq > 0 or pq > 0: L.append(("Earnings momentum", "warn", f"{lq:+.1f}%, {pq:+.1f}% QoQ"))
            else: L.append(("Earnings momentum", "fail", f"{lq:+.1f}%, {pq:+.1f}% QoQ — declining"))
        else: L.append(("Earnings momentum", "warn", "Partial data"))
    else: L.append(("Earnings momentum", "warn", "Insufficient data"))
    if cyc.get('peak'): L.append(("Cyclical ROE", "warn", f"PEAK: {cyc['latest']}% vs norm {cyc['median']}%"))
    elif cyc.get('roe_by_year'): L.append(("Cyclical ROE", "pass", f"Not at peak: {cyc.get('latest', '—')}% vs norm {cyc.get('median', '—')}%"))
    if is_valid(ret):
        if ret > 0: L.append(("1Y price", "pass", f"{ret:+.1f}%"))
        elif ret > -20: L.append(("1Y price", "warn", f"{ret:+.1f}%"))
        else: L.append(("1Y price", "fail", f"{ret:+.1f}%"))
    else: L.append(("1Y price", "warn", "Unavailable"))
    return L

# ============================================================
# PDF SCORECARD
# ============================================================
def gen_scorecard_html(name, ticker, sector, price, tier, size, l1, acct, val, ret, moat, layers, verdict):
    navy = NAVY
    pr = f"₹{price:,.2f}" if is_valid(price) else "—"
    sc = acct.get('score', '—')
    metrics = [('PE', fmt(l1['pe'], '.1f')), ('PEG', fmt(val.get('peg'), '.2f')), ('ROE', fmt(l1['roe'], '.1f', '%')), ('ROCE', fmt(l1['roce'], '.1f', '%')),
               ('D/E', fmt(l1['de'], '.2f')), ('FCF Yield', fmt(val.get('fcf_yield'), '.1f', '%')), ('1Y Return', fmt(ret, '+.1f', '%'))]
    met_cells = "".join(f'<td style="padding:8px 4px;text-align:center;border-right:1px solid #ddd;font-size:10px;"><div style="color:#888;font-size:8px;margin-bottom:2px;">{lb}</div><div style="font-family:Courier,monospace;font-weight:bold;">{vl}</div></td>' for lb, vl in metrics)
    layer_rows = ""
    for ln, st2, dt in layers:
        ic = '✓' if st2 == 'pass' else '✗' if st2 == 'fail' else '⚠'
        cl = '#059669' if st2 == 'pass' else '#DC2626' if st2 == 'fail' else '#D97706'
        layer_rows += f'<tr><td style="padding:3px 6px;color:{cl};font-size:10px;width:14px;">{ic}</td><td style="padding:3px 6px;font-size:10px;font-weight:bold;width:110px;">{ln}</td><td style="padding:3px 6px;font-size:9px;color:#555;">{dt}</td></tr>'
    growth_rows = ""
    for lb, vl in [('Sales CAGR', fmt(l1.get('sales_cagr'), '.1f', '%')), ('PAT CAGR', fmt(l1.get('pat_cagr'), '.1f', '%')),
                   ('Cum CFO/PAT', fmt(acct.get('cum_cfo_pat'), '.2f', 'x')), ('Cum CFO/EBITDA', fmt(acct.get('cum_cfo_ebitda'), '.2f', 'x')),
                   ('Moat durability', f"{moat.get('pct', 0)}%")]:
        growth_rows += f'<tr><td style="padding:3px 6px;font-size:9px;color:#888;">{lb}</td><td style="padding:3px 6px;font-size:10px;font-family:Courier,monospace;text-align:right;">{vl}</td></tr>'
    margins = acct.get('margins', [])
    if margins and len(margins) >= 2:
        arr = '↑' if acct.get('margin_trend') == 'expanding' else '↓' if acct.get('margin_trend') == 'contracting' else '→'
        growth_rows += f'<tr><td style="padding:3px 6px;font-size:9px;color:#888;">EBITDA margins</td><td style="padding:3px 6px;font-size:10px;font-family:Courier,monospace;text-align:right;">{margins[0]}% → {margins[-1]}% {arr}</td></tr>'
    tier_bg = {'FULL': '#059669', 'STANDARD': '#2563EB', 'HALF': '#D97706', 'WATCH': '#DC2626'}.get(tier, '#888')
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>{name} — Scorecard</title>
<style>@page{{size:A4;margin:12mm;}}body{{font-family:Segoe UI,Arial,sans-serif;margin:0;padding:0;color:#1a1a1a;font-size:11px;}}
table{{border-collapse:collapse;width:100%;}}</style></head><body>
<div style="background:{navy};color:white;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;">
<div><div style="font-size:9px;letter-spacing:1px;opacity:0.8;">HIGH COMPOUNDER FRAMEWORK</div><div style="font-size:8px;opacity:0.6;margin-top:2px;">Vinayak Nagral · {datetime.now().strftime("%d %b %Y")}</div></div>
<div style="font-size:8px;opacity:0.6;">Quantitative screening · Not investment advice</div></div>
<div style="padding:12px 16px;border-bottom:2px solid {navy};display:flex;justify-content:space-between;align-items:flex-end;">
<div><div style="font-size:20px;font-weight:bold;color:{navy};">{name}</div><div style="font-size:10px;color:#888;margin-top:2px;">{ticker} · {sector} · {pr}</div></div>
<div style="text-align:right;"><div style="background:{tier_bg};color:white;padding:4px 14px;border-radius:14px;font-size:11px;font-weight:bold;display:inline-block;">{tier} · {size}</div>
<div style="font-size:9px;color:#888;margin-top:3px;">Score {sc}/100</div></div></div>
<table style="border-bottom:1px solid #ddd;"><tr>{met_cells}</tr></table>
<div style="display:flex;border-bottom:1px solid #ddd;">
<div style="flex:1;padding:10px 16px;border-right:1px solid #ddd;"><div style="font-size:9px;font-weight:bold;color:{navy};margin-bottom:6px;">LAYER RESULTS</div><table>{layer_rows}</table></div>
<div style="width:220px;padding:10px 16px;"><div style="font-size:9px;font-weight:bold;color:{navy};margin-bottom:6px;">GROWTH & CASH</div><table>{growth_rows}</table></div></div>
<div style="padding:10px 16px;border-bottom:1px solid #ddd;"><div style="font-size:9px;font-weight:bold;color:{navy};margin-bottom:6px;">INVESTMENT VERDICT</div>
<div style="font-size:10px;line-height:1.7;">{verdict}</div></div>
<div style="padding:6px 16px;font-size:7px;color:#aaa;">Quantitative framework output. Not a research recommendation. Data from Yahoo Finance. For personal use only.</div>
</body></html>'''

def gen_pdf_bytes(name, ticker, sector, price, tier, size, l1, acct, val, ret, moat, layers, verdict):
    buf = BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    w, h = A4
    navy = HexColor(NAVY); gray = HexColor('#6B7280'); lgray = HexColor('#E5E7EB')
    green = HexColor('#059669'); red = HexColor('#DC2626'); amber = HexColor('#D97706'); blue = HexColor('#2563EB')
    pr = f"\u20B9{price:,.2f}" if is_valid(price) else "—"
    # Header band
    c.setFillColor(navy); c.rect(0, h - 50, w, 50, fill=1, stroke=0)
    c.setFillColor(white); c.setFont('Helvetica-Bold', 10)
    c.drawString(20, h - 20, 'HIGH COMPOUNDER FRAMEWORK')
    c.setFont('Helvetica-Bold', 8)
    c.drawString(20, h - 34, f'Vinayak Nagral  |  {datetime.now().strftime("%d %b %Y")}')
    c.setFont('Helvetica', 7.5)
    c.drawRightString(w - 20, h - 20, 'Quantitative screening')
    c.drawRightString(w - 20, h - 34, 'Not investment advice')
    y = h - 65
    # Company name
    c.setFillColor(navy); c.setFont('Helvetica-Bold', 22); c.drawString(20, y, name)
    # Tier badge
    tier_c = {'FULL': green, 'STANDARD': blue, 'HALF': amber, 'WATCH': red}.get(tier, gray)
    bt = f'{tier}  {size}'; bw = c.stringWidth(bt, 'Helvetica-Bold', 10) + 24
    c.setFillColor(tier_c); c.roundRect(w - 20 - bw, y - 3, bw, 22, 11, fill=1, stroke=0)
    c.setFillColor(white); c.setFont('Helvetica-Bold', 10); c.drawCentredString(w - 20 - bw / 2, y + 3, bt)
    y -= 18
    c.setFillColor(gray); c.setFont('Helvetica', 9); c.drawString(20, y, f'{ticker}  |  {sector}  |  {pr}')
    c.drawRightString(w - 20, y, f'Score {acct.get("score", "---")}/100')
    y -= 12; c.setStrokeColor(navy); c.setLineWidth(2); c.line(20, y, w - 20, y)
    y -= 25
    # Metrics row
    metrics = [('PE', fmt(l1['pe'], '.1f')), ('PEG', fmt(val.get('peg'), '.2f')), ('ROE', fmt(l1['roe'], '.1f', '%')),
               ('ROCE', fmt(l1['roce'], '.1f', '%')), ('D/E', fmt(l1['de'], '.2f')), ('FCF Yield', fmt(val.get('fcf_yield'), '.1f', '%')),
               ('1Y Return', fmt(ret, '+.1f', '%'))]
    cw = (w - 40) / len(metrics)
    for i, (lb, vl) in enumerate(metrics):
        x = 20 + i * cw
        c.setFillColor(gray); c.setFont('Helvetica', 7); c.drawCentredString(x + cw / 2, y + 2, lb)
        c.setFillColor(navy); c.setFont('Helvetica-Bold', 13); c.drawCentredString(x + cw / 2, y - 14, vl)
        if i < len(metrics) - 1: c.setStrokeColor(lgray); c.setLineWidth(0.5); c.line(x + cw, y + 8, x + cw, y - 20)
    y -= 32; c.setStrokeColor(lgray); c.setLineWidth(0.5); c.line(20, y, w - 20, y)
    y -= 18; mid = w * 0.6
    # Section headers
    c.setFillColor(navy); c.setFont('Helvetica-Bold', 8.5)
    c.drawString(24, y, 'LAYER RESULTS'); c.drawString(mid + 10, y, 'GROWTH & CASH')
    c.setStrokeColor(lgray); c.line(mid, y + 12, mid, y - 160)
    y -= 16
    # Layers
    ly = y
    for ln, st2, dt in layers:
        ic_char = '\u2713' if st2 == 'pass' else '\u2717' if st2 == 'fail' else '\u26A0'
        cl = green if st2 == 'pass' else red if st2 == 'fail' else amber
        c.setFillColor(cl); c.setFont('Helvetica-Bold', 9); c.drawString(28, ly, ic_char)
        c.setFillColor(navy); c.setFont('Helvetica-Bold', 8); c.drawString(44, ly, ln)
        c.setFillColor(gray); c.setFont('Helvetica', 7)
        detail = dt[:58] + '...' if len(dt) > 58 else dt
        c.drawString(148, ly, detail)
        ly -= 15
    # Growth table
    gy = y
    growth = [('Sales CAGR', fmt(l1.get('sales_cagr'), '.1f', '%')), ('PAT CAGR', fmt(l1.get('pat_cagr'), '.1f', '%')),
              ('Cum CFO/PAT', fmt(acct.get('cum_cfo_pat'), '.2f', 'x')), ('Cum CFO/EBITDA', fmt(acct.get('cum_cfo_ebitda'), '.2f', 'x')),
              ('Moat durability', f"{moat.get('pct', 0)}%")]
    margins = acct.get('margins', [])
    if margins and len(margins) >= 2:
        arr = '\u2191' if acct.get('margin_trend') == 'expanding' else '\u2193' if acct.get('margin_trend') == 'contracting' else '\u2192'
        growth.append(('EBITDA margins', f"{margins[0]}% \u2192 {margins[-1]}% {arr}"))
    for lb, vl in growth:
        c.setFillColor(gray); c.setFont('Helvetica', 8); c.drawString(mid + 14, gy, lb)
        c.setFillColor(navy); c.setFont('Courier-Bold', 9); c.drawRightString(w - 28, gy, vl)
        gy -= 15
    y = min(ly, gy) - 12; c.setStrokeColor(lgray); c.line(20, y, w - 20, y)
    y -= 18
    # Verdict
    c.setFillColor(navy); c.setFont('Helvetica-Bold', 8.5); c.drawString(24, y, 'INVESTMENT VERDICT')
    y -= 14; c.setFillColor(black); c.setFont('Helvetica', 8.5)
    for line in simpleSplit(verdict, 'Helvetica', 8.5, w - 55):
        c.drawString(28, y, line); y -= 12
    y -= 8; c.setStrokeColor(lgray); c.line(20, y, w - 20, y)
    y -= 12; c.setFillColor(HexColor('#AAAAAA')); c.setFont('Helvetica', 6)
    c.drawString(20, y, 'Quantitative framework output. Not a research recommendation. Data from Yahoo Finance. For personal use only.')
    c.save(); buf.seek(0); return buf.getvalue()

# ============================================================
# BATCH SCREENER WITH FUNNEL
# ============================================================
def analyse_with_funnel(ticker):
    sd = fetch(ticker)
    if not sd: return None, 'fetch_fail'
    if is_banking(sd['info']): return None, 'banking'
    l1 = run_layer1(sd, ticker)
    if not l1['mcap_pass']: return None, 'mcap'
    if not l1['de_pass']: return None, 'debt'
    if not l1['roe_pass']: return None, 'roe'
    if not l1['roce_pass']: return None, 'roce'
    if not l1['growth_pass']: return None, 'growth'
    multi = build_multi(sd['fin'], sd['bs'], sd['cf'])
    if not multi: return None, 'data'
    acct = run_forensic(multi); cyc = run_cyclical(multi); moat = run_moat(multi)
    val = run_valuation(l1['pe'], l1['pat_cagr'], l1['price'], ext_mcap(sd, l1['price']), sd)
    ret = ext_1y_ret(sd)
    tier, size = get_tier(acct['score'], acct['num_flags'], cyc.get('peak', False), val.get('peg'), ret, moat.get('pct', 0))
    return {'ticker': ticker.replace('.NS', ''), 'name': l1['name'], 'sector': l1['sector'], 'price': l1['price'],
            'pe': l1['pe'], 'peg': val.get('peg'), 'roe': l1['roe'], 'roce': l1['roce'], 'score': acct['score'],
            'nf': acct['num_flags'], 'cum_cfo': acct.get('cum_cfo_pat'), 'peak': cyc.get('peak', False),
            'ret': ret, 'tier': tier, 'size': size, 'sc': l1.get('sales_cagr'), 'pc': l1.get('pat_cagr'),
            'moat': moat.get('pct', 0), 'fcf_yield': val.get('fcf_yield')}, 'pass'

# ============================================================
# FULL ANALYSIS (for single stock & compare)
# ============================================================
def full_analysis(ticker):
    sd = fetch(ticker)
    if not sd: return None
    info = sd['info']; fin = sd['fin']; bs = sd['bs']; cf = sd['cf']; qfin = sd['qfin']
    bank = is_banking(info, ticker)
    l1 = run_layer1(sd, ticker)
    multi = build_multi(fin, bs, cf)
    acct = run_forensic(multi) if multi else {'score': None, 'flags': [], 'num_flags': 0, 'cum_cfo_pat': None, 'cum_cfo_ebitda': None, 'single_cfo_pat': None, 'cfo_ebitda_trend': [], 'recv_pcts': [], 'recv_years': [], 'margins': [], 'margin_trend': ''}
    cyc = run_cyclical(multi) if multi else {'peak': False, 'roe_by_year': {}}
    moat = run_moat(multi) if multi else {'pct': 0, 'consistency': 'no data', 'years_above_15': 0, 'total_years': 0}
    mom = run_momentum(qfin)
    ret = ext_1y_ret(sd); mcap = ext_mcap(sd, l1['price'])
    val = run_valuation(l1['pe'], l1['pat_cagr'], l1['price'], mcap, sd)
    tier, size = get_tier(acct.get('score'), acct.get('num_flags', 0), cyc.get('peak', False), val.get('peg'), ret, moat.get('pct', 0))
    if not l1['pass'] and tier in ['FULL', 'STANDARD']: tier, size = 'HALF', '4-6%'
    layers = gen_layers(l1, acct, cyc, val, mom, ret, moat)
    verdict = gen_verdict(l1['name'], tier, l1, acct, cyc, val, ret, moat, bank)
    html = gen_scorecard_html(l1['name'], ticker, l1['sector'], l1['price'], tier, size, l1, acct, val, ret, moat, layers, verdict)
    try: pdf = gen_pdf_bytes(l1['name'], ticker, l1['sector'], l1['price'], tier, size, l1, acct, val, ret, moat, layers, verdict)
    except Exception: pdf = None
    return {'sd': sd, 'l1': l1, 'acct': acct, 'cyc': cyc, 'moat': moat, 'mom': mom, 'ret': ret, 'val': val, 'tier': tier, 'size': size, 'layers': layers, 'verdict': verdict, 'html': html, 'pdf': pdf, 'bank': bank, 'info': info, 'fin': fin, 'bs': bs, 'cf': cf, 'qfin': qfin}

# ============================================================
# APP
# ============================================================
st.markdown(f'<p class="main-title">📊 High Compounder Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">7-Layer Systematic Indian Equity Screener — Quality · Valuation · Momentum</p>', unsafe_allow_html=True)
st.markdown("---")
st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Single Stock", "Compare", "Auto Top 10", "How It Works"], label_visibility="collapsed")
st.sidebar.markdown("### Portfolio")
st.sidebar.dataframe(pd.DataFrame({"Stock": ["LUPIN", "DIXON", "BSE", "EICHER", "KPIT"], "Tier": ["FULL", "FULL", "STD", "STD", "HALF"], "Score": [100, 100, 85, 85, 100], "PEG": [0.14, 0.54, 0.38, 1.60, 1.48]}), hide_index=True, use_container_width=True)
st.sidebar.caption("Vinayak Nagral · Sep 2026")

# ---- SINGLE STOCK ----
if page == "Single Stock":
    c1, c2 = st.columns([4, 1])
    with c1: ti = st.text_input("Enter NSE ticker", value="LUPIN", placeholder="LUPIN, INFY, TCS, DIXON").strip().upper()
    with c2: st.markdown("<br>", unsafe_allow_html=True); go = st.button("Analyse", type="primary", use_container_width=True)
    qc = st.columns(7)
    for i, q in enumerate(["LUPIN", "BSE", "DIXON", "INFY", "HDFCAMC", "MAZDOCK", "HCLTECH"]):
        with qc[i]:
            if st.button(q, key=f"q_{q}", use_container_width=True): ti = q; go = True
    if not ti.endswith(".NS"): ti += ".NS"
    if go:
        with st.spinner(f"Analysing {ti.replace('.NS', '')}..."): res = full_analysis(ti)
        if not res: st.error("Could not fetch data."); st.stop()
        l1, acct, val, mom, ret, moat, cyc = res['l1'], res['acct'], res['val'], res['mom'], res['ret'], res['moat'], res['cyc']
        tier, size, layers, verdict, bank = res['tier'], res['size'], res['layers'], res['verdict'], res['bank']
        name, sector, price = l1['name'], l1['sector'], l1['price']
        st.markdown(f'<p class="stock-name">{name}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="stock-meta">{ti} · {sector} · {"₹" + f"{price:,.2f}" if is_valid(price) else "—"}</p>', unsafe_allow_html=True)
        if bank: st.info("Banking/Financial — framework for non-financials. Scores for reference.")
        st.markdown("---")
        # PDF Export
        if res.get('pdf'):
            st.download_button("📄 Download PDF scorecard", data=res['pdf'], file_name=f"{name.replace(' ', '_')}_Scorecard.pdf", mime="application/pdf")
        else:
            st.download_button("📄 Download scorecard (HTML)", data=res['html'], file_name=f"{name.replace(' ', '_')}_Scorecard.html", mime="text/html")
        # Metrics
        cols = st.columns(7)
        vals = [f"{acct.get('score', '—')}/100" if acct.get('score') is not None else "—", fmt(l1['pe'], '.1f'), fmt(val.get('peg'), '.2f'), fmt(l1['roe'], '.1f', '%'), fmt(val.get('fcf_yield'), '.1f', '%'), fmt(ret, '+.1f', '%'), ""]
        for i, (lb, vl) in enumerate(zip(["Score", "PE", "PEG", "ROE", "FCF Yield", "1Y Return", "Tier"], vals)):
            with cols[i]:
                if lb == "Tier": st.markdown(f'<div class="metric-card"><div class="metric-lbl">Tier</div><span class="tier-badge tier-{tier.lower()}">{tier} · {size}</span></div>', unsafe_allow_html=True)
                else: st.markdown(f'<div class="metric-card"><div class="metric-lbl">{lb}</div><div class="metric-val">{vl}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Layer-by-layer breakdown")
        for ln, st2, dt in layers:
            ic = "✅" if st2 == "pass" else "❌" if st2 == "fail" else "⚠️"
            st.markdown(f'<div class="layer-row layer-{st2}">{ic} <strong>{ln}:</strong> {dt}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Investment verdict")
        st.markdown(f'<div class="verdict-box">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("---")
        t1, t2, t3, t4 = st.tabs(["Forensic", "Cyclical & valuation", "Momentum", "Debug"])
        with t1:
            if acct.get('score') is not None:
                sc2 = acct['score']; css = "score-green" if sc2 >= 85 else "score-yellow" if sc2 >= 50 else "score-red"
                st.markdown(f'<div class="score-card {css}"><h2>{sc2}/100</h2></div>', unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Cumulative CFO/PAT**"); v2 = acct.get('cum_cfo_pat')
                    if is_valid(v2): st.markdown(f"{'✅' if v2 >= 0.7 else '⚠️' if v2 >= 0.5 else '❌'} **{v2}x**")
                with cb:
                    st.markdown("**Cumulative CFO/EBITDA**"); v3 = acct.get('cum_cfo_ebitda')
                    if is_valid(v3): st.markdown(f"{'✅' if v3 >= 0.7 else '⚠️' if v3 >= 0.5 else '❌'} **{v3}x**")
                margins = acct.get('margins', [])
                if margins and len(margins) >= 2: st.markdown(f"**Margins:** {' → '.join(str(m) + '%' for m in margins)} {'📈' if acct.get('margin_trend') == 'expanding' else '📉' if acct.get('margin_trend') == 'contracting' else '➡️'}")
                if acct.get('flags'):
                    for f in acct['flags']: st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="clean-item">✅ All checks passed</div>', unsafe_allow_html=True)
        with t2:
            cc, cd = st.columns(2)
            with cc:
                if cyc.get('roe_by_year'): st.dataframe(pd.DataFrame(list(cyc['roe_by_year'].items()), columns=['Year', 'ROE%']), hide_index=True, use_container_width=True)
                st.markdown(f"**Moat:** {moat.get('years_above_15', 0)}/{moat.get('total_years', 0)} yrs ROE>15% → **{moat.get('consistency', '—')}**")
            with cd:
                st.markdown(f"**PE:** {fmt(l1['pe'], '.1f')} · **PEG:** {fmt(val.get('peg'), '.2f')} · **FCF Yield:** {fmt(val.get('fcf_yield'), '.1f', '%')}")
                st.markdown(f"**Sales CAGR:** {fmt(l1.get('sales_cagr'), '.1f')}% · **PAT CAGR:** {fmt(l1.get('pat_cagr'), '.1f')}%")
        with t3:
            if mom.get('available'):
                mc1, mc2 = st.columns(2); mc1.metric("Latest QoQ", fmt(mom.get('latest_qoq'), '+.1f', '%')); mc2.metric("Prior QoQ", fmt(mom.get('prior_qoq'), '+.1f', '%'))
                if mom.get('eps'): st.dataframe(pd.DataFrame({'Quarter': mom['quarters'], 'EPS': [round(e, 2) for e in mom['eps']]}), hide_index=True, use_container_width=True)
            st.markdown(f"**1Y Return:** {fmt(ret, '+.1f', '%')}")
        with t4:
            info = res['info']; fin = res['fin']; bs = res['bs']
            for k in ['currentPrice', 'trailingPE', 'trailingEps', 'marketCap', 'sharesOutstanding', 'shortName', 'sector']:
                v = info.get(k, '—'); st.markdown(f"`{k}`: {str(v)[:40]} {'✅' if is_valid(v) or (isinstance(v, str) and v.strip()) else '❌'}")
            st.markdown(f"Fin: {'✅' if fin is not None else '❌'} · BS: {'✅' if bs is not None else '❌'} · Price History: {'✅ ' + str(len(res['sd'].get('ph', []))) + 'd' if res['sd'].get('ph') is not None and len(res['sd'].get('ph', [])) > 0 else '❌'}")
        st.caption("Research only, not investment advice · Yahoo Finance")

# ---- COMPARE ----
elif page == "Compare":
    st.subheader("Compare stocks")
    c1, c2, c3 = st.columns(3)
    with c1: t1 = st.text_input("Stock 1", value="LUPIN").strip().upper()
    with c2: t2 = st.text_input("Stock 2", value="INFY").strip().upper()
    with c3: t3 = st.text_input("Stock 3 (optional)", value="").strip().upper()
    if st.button("Compare", type="primary", use_container_width=True):
        tickers = [t + ".NS" if not t.endswith(".NS") else t for t in [t1, t2, t3] if t]
        results = {}
        for t in tickers:
            with st.spinner(f"Analysing {t.replace('.NS', '')}..."): r = full_analysis(t)
            if r: results[t] = r
        if len(results) < 2: st.error("Need at least 2 valid stocks."); st.stop()
        rows_data = []
        metric_labels = ['Tier', 'Score', 'PE', 'PEG', 'ROE', 'ROCE', 'D/E', 'Sales CAGR', 'PAT CAGR', 'CFO/PAT', 'FCF Yield', 'Moat', '1Y Return']
        for t, r in results.items():
            rows_data.append({'Stock': r['l1']['name'],
                'Tier': r['tier'], 'Score': f"{r['acct'].get('score', '—')}/100",
                'PE': fmt(r['l1']['pe'], '.1f'), 'PEG': fmt(r['val'].get('peg'), '.2f'),
                'ROE': fmt(r['l1']['roe'], '.1f', '%'), 'ROCE': fmt(r['l1']['roce'], '.1f', '%'),
                'D/E': fmt(r['l1']['de'], '.2f'), 'Sales CAGR': fmt(r['l1'].get('sales_cagr'), '.1f', '%'),
                'PAT CAGR': fmt(r['l1'].get('pat_cagr'), '.1f', '%'), 'CFO/PAT': fmt(r['acct'].get('cum_cfo_pat'), '.2f', 'x'),
                'FCF Yield': fmt(r['val'].get('fcf_yield'), '.1f', '%'), 'Moat': f"{r['moat'].get('pct', 0)}%",
                '1Y Return': fmt(r['ret'], '+.1f', '%')})
        df = pd.DataFrame(rows_data).set_index('Stock').T
        df.index.name = 'Metric'
        st.dataframe(df, use_container_width=True)
        st.markdown("---")
        for t, r in results.items():
            with st.expander(f"**{r['l1']['name']}** — {r['tier']} · {r['size']}", expanded=True):
                st.markdown(f'<div class="verdict-box">{r["verdict"]}</div>', unsafe_allow_html=True)

# ---- AUTO TOP 10 ----
elif page == "Auto Top 10":
    st.subheader("Automatic Top 10 Picker")
    c1, c2 = st.columns(2)
    with c1: mx = st.slider("Stocks to screen", 20, 200, 50, 10)
    with c2: tn = st.slider("Show top N", 5, 20, 10)
    if st.button("Run screen", type="primary", use_container_width=True):
        try: tickers = [s + ".NS" for s in pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty200list.csv")['Symbol'].tolist()][:mx]
        except Exception: st.error("Could not fetch Nifty 200."); st.stop()
        results, funnel = [], {'total': len(tickers), 'banking': 0, 'mcap': 0, 'debt': 0, 'roe': 0, 'roce': 0, 'growth': 0, 'data': 0, 'fetch_fail': 0, 'pass': 0}
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            prog.progress((i + 1) / len(tickers), f"{t.replace('.NS', '')} ({i + 1}/{len(tickers)})")
            r, reason = analyse_with_funnel(t)
            funnel[reason] = funnel.get(reason, 0) + 1
            if r: results.append(r)
        prog.empty()
        # REJECTION FUNNEL
        st.markdown("### Screening funnel")
        total = funnel['total']; passed = funnel['pass']
        stages = [('Universe', total), ('After banking exclusion', total - funnel['banking']),
                  ('After MCap > ₹15,000 Cr', total - funnel['banking'] - funnel['mcap']),
                  ('After D/E < 0.5', total - funnel['banking'] - funnel['mcap'] - funnel['debt']),
                  ('After ROE > 15%', total - funnel['banking'] - funnel['mcap'] - funnel['debt'] - funnel['roe']),
                  ('After growth > 15%', total - funnel['banking'] - funnel['mcap'] - funnel['debt'] - funnel['roe'] - funnel['growth'] - funnel['roce']),
                  ('Final (passed all)', passed)]
        for label, count in stages:
            pct = count / total * 100 if total > 0 else 0
            st.markdown(f'<div class="funnel-row"><div style="width:140px;font-size:12px;color:#6B7280;">{label}</div><div class="funnel-bar" style="width:{max(pct, 3)}%;">{count}</div></div>', unsafe_allow_html=True)
        st.markdown(f"**{passed} of {total} stocks passed** ({round(passed / total * 100, 1) if total > 0 else 0}% pass rate)")
        if not results: st.error("No stocks passed."); st.stop()
        st.markdown("---")
        df = pd.DataFrame(results)
        ps = lambda x: (90 if x < 0.5 else 75 if x < 1 else 55 if x < 1.5 else 30 if x < 2 else 10) if is_valid(x) else 10
        ms = lambda x: (80 if x > 30 else 65 if x > 0 else 40 if x > -20 else 20) if is_valid(x) else 40
        cs = lambda x: (100 if x > 1.2 else 85 if x > .9 else 65 if x > .7 else 40) if is_valid(x) else 20
        df['rank'] = df['score'].fillna(0) * .20 + df['peg'].apply(ps) * .25 + df['roe'].fillna(0) * .10 + df['ret'].apply(ms) * .15 + df['cum_cfo'].apply(cs) * .15 + df['moat'] * .15
        df = df.sort_values('rank', ascending=False).head(tn)
        for idx, row in df.iterrows():
            rk = list(df.index).index(idx) + 1
            ic = {"FULL": "🟢", "STANDARD": "🔵", "HALF": "🟡", "WATCH": "🔴"}.get(row['tier'], "⚪")
            with st.expander(f"#{rk} · {row['ticker']} — {ic} {row['tier']} · Score {row['score']} · PEG {fmt(row['peg'], '.2f')}", expanded=rk <= 3):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ROE", fmt(row.get('roe'), '.1f', '%')); c2.metric("PE", fmt(row.get('pe'), '.1f'))
                c3.metric("FCF Yield", fmt(row.get('fcf_yield'), '.1f', '%')); c4.metric("Moat", f"{row.get('moat', 0)}%")
                st.markdown(f"**{row['sector']}** · Sales {fmt(row.get('sc'), '.1f')}% · PAT {fmt(row.get('pc'), '.1f')}% · **{row['tier']} {row['size']}**")

# ---- HOW IT WORKS ----
elif page == "How It Works":
    st.subheader("Framework architecture")
    st.markdown("""
**Layer 1 — Quantitative screen:** MCap > ₹15,000 Cr, ROE > 15%, ROCE > 18%, D/E < 0.5, 3Y Sales & PAT CAGR > 15%.

**Layer 2 — Forensic accounting:** Four cumulative multi-year checks catch receivables stuffing, inventory bloat, and fake profits. Single-year ratios miss project businesses — cumulative doesn't. Plus operating margin trajectory.

**Layer 3 — Moat durability:** Percentage of available years the company sustained ROE > 15%. A company at 4/4 (100%) has structural protection. One at 2/4 (50%) is riding a cycle.

**Layer 4 — Dual valuation:** PEG confirms growth justifies the PE. FCF Yield confirms the company generates real cash relative to price. PE > 80 triggers the Adani Filter.

**Layer 5 — Earnings momentum:** Last 2 quarters QoQ EPS change catches fresh deterioration that annual tests miss.

**Layer 6 — Cyclical ROE:** If ROE > 2x historical average → cyclical peak. Use median for valuation.

**Layer 7 — Position sizing v3:** Quality + moat + valuation + momentum integrated. FULL requires score ≥ 85, zero flags, moat ≥ 60%, PEG < 1.5. STANDARD for quality without full conviction. HALF for mixed signals. WATCH for failures.

---
**Structural exclusions:** Banking/NBFC (need NIM, CASA, GNPA), commodity producers (price-driven CAGR), newly listed (<2 years).
    """)
    st.caption("Vinayak Nagral · September 2026")
