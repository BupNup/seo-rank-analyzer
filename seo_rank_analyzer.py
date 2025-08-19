
from __future__ import annotations
import io
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import chardet
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_PAGERANK_ALPHA = 0.85

URL_COL_CANDIDATES_PAGES = ["Address", "URL", "Url", "address", "Final URL", "Final Address"]
INLINKS_SRC_CANDIDATES = ["Source", "From", "From URL", "From Address", "source"]
INLINKS_DST_CANDIDATES = ["Destination", "To", "Target", "Target URL", "To URL", "To Address"]
AHREFS_TARGET_CANDIDATES = ["Target URL", "Target url", "URL", "Address"]
AHREFS_REFERRING_PAGE_CANDS = ["Referring page URL", "Referring Page", "Source URL", "Backlink URL"]
AHREFS_REFERRING_DOMAIN_CANDS = ["Referring domain", "Referring Domain", "Domain"]

GSC_PAGE_CANDS = ["Page","Landing page","Landing Page","URL","Address"]
GSC_QUERY_CANDS = ["Query","Search query","Search Query"]
GSC_IMPR_CANDS = ["Impressions","Impr","Impressions Total"]

PAGE_TITLE_CANDS = ["Title 1","Title","Meta Title"]
PAGE_DESC_CANDS = ["Meta Description 1","Meta Description"]
PAGE_H1_CANDS   = ["H1-1","H1"]

@st.cache_data(show_spinner=False)
def sniff_encoding(file_bytes: bytes) -> str:
    res = chardet.detect(file_bytes)
    enc = res.get('encoding') or 'utf-8'
    return enc

@st.cache_data(show_spinner=False)
def read_table(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    enc = sniff_encoding(raw)
    buf = io.BytesIO(raw)
    try:
        df = pd.read_csv(buf, sep=None, engine='python', encoding=enc)
    except Exception:
        buf.seek(0)
        df = pd.read_excel(buf)
    return df

@st.cache_data(show_spinner=False)
def read_gsc(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    enc = sniff_encoding(raw)
    buf = io.BytesIO(raw)
    try:
        df = pd.read_csv(buf, encoding=enc)
    except Exception:
        buf.seek(0)
        df = pd.read_excel(buf)
    return df

def _strip_fragment(url: str) -> str:
    return url.split('#', 1)[0]

def _strip_query(url: str, keep_query: bool) -> str:
    if keep_query: return url
    return url.split('?', 1)[0]

def normalize_url(url: str, keep_query: bool = False) -> str:
    if not isinstance(url, str): return ""
    u = url.strip().lower()
    if not u: return ""
    u = _strip_fragment(u); u = _strip_query(u, keep_query)
    if u.endswith('/'):
        m = re.match(r"^(https?://[^/]+)/$", u)
        if not m: u = u[:-1]
    return u

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns: return cand
        if cand.lower() in cols_lower: return cols_lower[cand.lower()]
    return None

def get_host(u: str) -> str:
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

def canonical_home_root(homepage_url: str) -> str:
    try:
        host = urlparse(homepage_url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def is_internal_host(host: str, home_root: str) -> bool:
    if not home_root: return True
    h = host.lower(); base = home_root.lower()
    return h == base or h.endswith("." + base) or h == "www." + base

def detect_link_position_columns(df: pd.DataFrame):
    pos = pick_column(df, ["Link Position","Position","Placement"])
    path = pick_column(df, ["Link Path","DOM Path","XPath"])
    elem = pick_column(df, ["Link Element","Element"])
    anch = pick_column(df, ["Anchor","Anchor Text","Anchor text"])
    return pos, path, elem, anch

def classify_inlink_row(row: pd.Series, pos_col: Optional[str], path_col: Optional[str], elem_col: Optional[str], anchor_col: Optional[str]) -> str:
    def s(v): return str(v).lower() if pd.notna(v) else ""
    pos = s(row.get(pos_col)) if pos_col else ""
    path = s(row.get(path_col)) if path_col else ""
    elem = s(row.get(elem_col)) if elem_col else ""
    anch = s(row.get(anchor_col)) if anchor_col else ""

    if any(tok in pos for tok in ["header","navigation","nav","footer","breadcrumbs","breadcrumb","sidebar","pagination","menu","widget"]):
        return "menu_footer"
    if "content" in pos or "main" in pos:
        return "contextual"
    if any(tok in path for tok in ["/header","/nav","/footer","/aside","breadcrumb","sidebar","menu","mega-menu","pagination","pager"]):
        return "menu_footer"
    if any(tok in elem for tok in ["nav","header","footer","aside","breadcrumb"]):
        return "menu_footer"
    if any(tok in path for tok in ["/main","/article","/section","/content"]) or any(tok in elem for tok in ["main","article","section"]):
        return "contextual"
    generic = {"home","about","contact","login","register","sign in","sign up","terms","privacy","blog","categories","slots","casinos"}
    if anch in generic or len(anch) <= 2: return "menu_footer"
    return "other"

def filter_inlinks_by_scope(inlinks: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == 'all': return inlinks
    pos_col, path_col, elem_col, anch_col = detect_link_position_columns(inlinks)
    if not (pos_col or path_col or elem_col or anch_col):
        return inlinks if scope == 'all' else inlinks.iloc[0:0]
    tmp = inlinks.copy()
    tmp['_cls'] = tmp.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
    if scope == 'contextual':
        return tmp[tmp['_cls'].isin(['contextual','other'])].drop(columns=['_cls'])
    if scope == 'menu_footer':
        return tmp[tmp['_cls'] == 'menu_footer'].drop(columns=['_cls'])
    return inlinks

def build_internal_graph(pages: pd.DataFrame, inlinks: pd.DataFrame, keep_query: bool, link_scope: str = 'all', home_root: str = "") -> Tuple[nx.DiGraph, pd.DataFrame]:
    url_col = pick_column(pages, URL_COL_CANDIDATES_PAGES)
    if not url_col: raise ValueError("Pages file needs a URL column like 'Address' or 'URL'.")
    pages = pages.copy()
    pages['url_norm'] = pages[url_col].map(lambda x: normalize_url(x, keep_query))
    pages['host'] = pages['url_norm'].map(get_host)
    if home_root:
        pages = pages[pages['host'].map(lambda h: is_internal_host(h, home_root))]
    pages_nodes = pages['url_norm'].dropna().drop_duplicates()
    pages_nodes = pages_nodes[pages_nodes != ""]

    src_col = pick_column(inlinks, INLINKS_SRC_CANDIDATES)
    dst_col = pick_column(inlinks, INLINKS_DST_CANDIDATES)
    if not src_col or not dst_col: raise ValueError("Inlinks file needs columns like 'Source' and 'Destination'.")

    inlinks = inlinks.copy()
    inlinks['src'] = inlinks[src_col].map(lambda x: normalize_url(x, keep_query))
    inlinks['dst'] = inlinks[dst_col].map(lambda x: normalize_url(x, keep_query))
    inlinks['src_host'] = inlinks['src'].map(get_host)
    inlinks['dst_host'] = inlinks['dst'].map(get_host)

    inlinks = filter_inlinks_by_scope(inlinks, link_scope)
    if home_root:
        inlinks = inlinks[inlinks['src_host'].map(lambda h: is_internal_host(h, home_root)) & inlinks['dst_host'].map(lambda h: is_internal_host(h, home_root))]

    valid_nodes = set(pages_nodes.tolist())
    df_edges = inlinks[['src','dst']].dropna()
    df_edges = df_edges[(df_edges['src'] != "") & (df_edges['dst'] != "")]
    df_edges = df_edges[df_edges['dst'].isin(valid_nodes) & df_edges['src'].isin(valid_nodes)]
    df_edges['weight'] = 1.0
    df_edges = df_edges.groupby(['src','dst'], as_index=False)['weight'].sum()

    G = nx.DiGraph(); G.add_nodes_from(valid_nodes)
    for row in df_edges.itertuples(index=False):
        G.add_edge(row.src, row.dst, weight=float(row.weight))
    return G, pages

def aggregate_backlinks(backlinks: pd.DataFrame, keep_query: bool) -> pd.Series:
    if backlinks is None or backlinks.empty: return pd.Series(dtype=float)
    tgt_col = pick_column(backlinks, AHREFS_TARGET_CANDIDATES)
    if not tgt_col: raise ValueError("Backlinks file needs a 'Target URL' or similar column.")
    df = backlinks.copy()
    df['target'] = df[tgt_col].map(lambda x: normalize_url(x, keep_query))

    group_col = pick_column(df, ["Links in group","links in group"])
    if group_col:
        df['strength'] = pd.to_numeric(df[group_col], errors="coerce").fillna(1.0)
        agg = df.dropna(subset=['target','strength'])
        agg = agg[(agg['target'] != "")]
        return agg.groupby('target')['strength'].sum().astype(float)

    dom_col = pick_column(df, AHREFS_REFERRING_DOMAIN_CANDS)
    if dom_col:
        df['ref_key'] = df[dom_col].astype(str).str.lower().str.strip()
    else:
        ref_col = pick_column(df, AHREFS_REFERRING_PAGE_CANDS)
        if ref_col:
            df['ref_key'] = df[ref_col].map(lambda x: normalize_url(x, keep_query))
        else:
            df['ref_key'] = np.arange(len(df)).astype(str)

    agg = df.dropna(subset=['target','ref_key'])
    agg = agg[(agg['target'] != "") & (agg['ref_key'] != "")]
    counts = agg.groupby('target')['ref_key'].nunique().astype(float)
    return counts

def compute_pagerank(G: nx.DiGraph, alpha: float, personalization: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        if n == 0: return {}
        uni = 1.0 / n; return {node: uni for node in G.nodes}
    try:
        return nx.pagerank(G, alpha=alpha, weight='weight', personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, alpha=alpha, weight='weight', max_iter=200, personalization=personalization)

def compute_cheirank(G: nx.DiGraph, alpha: float) -> Dict[str, float]:
    GR = G.reverse(copy=True)
    return compute_pagerank(GR, alpha=alpha, personalization=None)

def build_gsc_semantics(gsc: pd.DataFrame, keep_query: bool) -> pd.Series:
    if gsc is None or gsc.empty: return pd.Series(dtype=str)
    pg_col = pick_column(gsc, GSC_PAGE_CANDS)
    q_col  = pick_column(gsc, GSC_QUERY_CANDS)
    i_col  = pick_column(gsc, GSC_IMPR_CANDS)
    if not (pg_col and q_col and i_col): return pd.Series(dtype=str)
    tmp = gsc[[pg_col, q_col, i_col]].rename(columns={pg_col:"page", q_col:"query", i_col:"impr"}).copy()
    tmp['page'] = tmp['page'].map(lambda x: normalize_url(x, keep_query))
    tmp['query'] = tmp['query'].astype(str).str.lower().str.replace(r"[^a-z0-9\-\s]"," ", regex=True)
    tmp['w'] = (np.log2(pd.to_numeric(tmp['impr'], errors='coerce').fillna(0).clip(lower=1.0)) + 1.0).round().astype(int)
    tmp['repeat'] = tmp.apply(lambda r: (r['query'] + ' ') * int(max(1, r['w'])), axis=1)
    agg = tmp.groupby('page')['repeat'].apply(lambda x: ' '.join(x)).astype(str)
    return agg

def build_page_texts(df_pages: pd.DataFrame, gsc_sem: pd.Series) -> Dict[str, str]:
    url_col = pick_column(df_pages, URL_COL_CANDIDATES_PAGES)
    t_col = pick_column(df_pages, PAGE_TITLE_CANDS)
    d_col = pick_column(df_pages, PAGE_DESC_CANDS)
    h1_col = pick_column(df_pages, PAGE_H1_CANDS)
    h2_cols = [c for c in df_pages.columns if str(c).startswith('H2')]

    texts = {}
    for _, row in df_pages.iterrows():
        url = row.get(url_col, "") if url_col else ""
        u = normalize_url(url, keep_query=False)
        if not u: continue
        parts = []
        if t_col: parts.append(str(row.get(t_col, "")))
        if d_col: parts.append(str(row.get(d_col, "")))
        if h1_col: parts.append(str(row.get(h1_col, "")))
        for c in h2_cols[:6]: parts.append(str(row.get(c, "")))
        if isinstance(gsc_sem, pd.Series) and u in gsc_sem.index: parts.append(str(gsc_sem[u]))
        texts[u] = ' '.join([p for p in parts if isinstance(p, str)]).lower()
    return texts

def build_page_texts_for_category(df_pages: pd.DataFrame) -> Dict[str, str]:
    url_col = pick_column(df_pages, URL_COL_CANDIDATES_PAGES)
    t_col = pick_column(df_pages, PAGE_TITLE_CANDS)
    d_col = pick_column(df_pages, PAGE_DESC_CANDS)
    out = {}
    for _, row in df_pages.iterrows():
        url = row.get(url_col, "") if url_col else ""
        u = normalize_url(url, keep_query=False)
        if not u: continue
        path = urlparse(u).path.replace("/", " ").replace("-", " ")
        parts = [path]
        if t_col: parts.append(str(row.get(t_col, "")))
        if d_col: parts.append(str(row.get(d_col, "")))
        out[u] = " ".join(parts).lower()
    return out

def assign_categories_semantic(df_pages: pd.DataFrame, categories: List[str]) -> Tuple[Dict[str,str], Dict[str,float]]:
    if not categories: return {}, {}
    pages_text = build_page_texts_for_category(df_pages)
    urls = list(pages_text.keys())
    cat_labels = [c.strip() for c in categories if c.strip()]
    if not urls or not cat_labels: return {u: "" for u in urls}, {u: 0.0 for u in urls}
    corpus = list(pages_text.values()) + cat_labels
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50000)
    X = vec.fit_transform(corpus)
    X_pages = X[:len(urls)]
    X_cats = X[len(urls):]
    sims = cosine_similarity(X_pages, X_cats)
    best_idx = sims.argmax(axis=1)
    best_scores = sims[np.arange(len(urls)), best_idx]
    mapping = {urls[i]: cat_labels[best_idx[i]] for i in range(len(urls))}
    scores  = {urls[i]: float(best_scores[i]) for i in range(len(urls))}
    return mapping, scores

def tfidf_similarity(texts: Dict[str,str], urls: List[str]):
    corpus = [texts.get(u, '') for u in urls]
    vec = TfidfVectorizer(stop_words='english', min_df=1, max_features=50000)
    X = vec.fit_transform(corpus)
    index = {u:i for i,u in enumerate(urls)}
    return vec, X, index

def suggest_internal_links(df_metrics: pd.DataFrame, texts: Dict[str,str], category_map: Dict[str,str],
                           prefer_same_category: bool, same_category_only: bool,
                           homepage_url: str,
                           backlink_weight: float = 0.15, pr_weight: float = 0.35,
                           ch_weight: float = 0.15, sem_weight: float = 0.35,
                           low_pr_q: float = 0.20, top_k_sources: int = 5) -> pd.DataFrame:
    df = df_metrics.copy()
    low_pr_thresh = df['pagerank'].quantile(low_pr_q)
    df['reason'] = ''
    df.loc[df['inlinks'] == 0, 'reason'] = df['reason'] + 'Orphan; '
    df.loc[df['pagerank'] <= low_pr_thresh, 'reason'] = df['reason'] + 'Low PR; '
    df.loc[(df['backlinks_refcnt'] > 0) & (df['pagerank'] <= low_pr_thresh), 'reason'] = df['reason'] + 'Has backlinks, needs internal links; '
    df['reason'] = df['reason'].str.strip()
    targets = df[df['reason'] != ''].copy()
    if targets.empty:
        return pd.DataFrame(columns=['target','category','reason','pagerank','cheirank','backlinks','inlinks','anchor_hint','suggestions'])

    urls = [u for u in df['url'] if u in texts]
    if not urls:
        return pd.DataFrame(columns=['target','category','reason','pagerank','cheirank','backlinks','inlinks','anchor_hint','suggestions'])

    homepage_norm = normalize_url(homepage_url, keep_query=False)

    _, X, idx = tfidf_similarity(texts, urls)
    pr_norm = (df.set_index('url')['pagerank_norm']).reindex(urls).fillna(0.0).to_numpy()
    ch_norm = (df.set_index('url')['cheirank_norm']).reindex(urls).fillna(0.0).to_numpy()
    bln = (df.set_index('url')['backlinks_refcnt']).reindex(urls).fillna(0.0)
    bl_norm = (bln / (bln.max() if bln.max() > 0 else 1.0)).to_numpy()
    out_deg = df.set_index('url')['outlinks'].reindex(urls).fillna(0).to_numpy()

    rows = []
    from collections import Counter

    for t in targets.itertuples(index=False):
        t_url = t.url
        if t_url not in idx: continue
        t_vec = X[idx[t_url]]
        sims = cosine_similarity(t_vec, X).ravel()

        score = sem_weight * sims + pr_weight * pr_norm + ch_weight * ch_norm + backlink_weight * bl_norm

        allow = np.array([True]*len(urls))
        allow[idx[t_url]] = False
        if homepage_norm in urls:
            allow[urls.index(homepage_norm)] = False

        t_cat = category_map.get(t_url, "")
        if same_category_only and t_cat:
            allow = allow & np.array([category_map.get(u, "") == t_cat for u in urls])
        elif prefer_same_category and t_cat:
            same_mask = np.array([category_map.get(u, "") == t_cat for u in urls])
            score = np.where(same_mask, score * 1.15, score)

        score = np.where(out_deg > 0, score, score * 0.5)
        score = np.where(allow, score, -1.0)

        top_idx = score.argsort()[::-1][:top_k_sources]
        cands = [f"{urls[j]} | score={score[j]:.3f} | PR={pr_norm[j]:.3f} | CH={ch_norm[j]:.3f} | sim={sims[j]:.3f} | cat={category_map.get(urls[j],'')}" for j in top_idx]

        tokens = [w for w in re.findall(r"[a-z0-9\-]+", texts.get(t_url, "")) if len(w) > 2]
        common = [w for w,_ in Counter(tokens).most_common(8)]
        anchor = ' '.join(common[:4])

        rows.append({
            'target': t_url,
            'category': t_cat,
            'reason': t.reason,
            'pagerank': t.pagerank,
            'cheirank': t.cheirank,
            'backlinks': t.backlinks_refcnt,
            'inlinks': t.inlinks,
            'anchor_hint': anchor,
            'suggestions': "\n".join(cands)
        })

    return pd.DataFrame(rows)

def flag_pages(df: pd.DataFrame, pr_col: str, ch_col: str, backlink_col: str, low_pr_q: float, high_ch_q: float) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0: return df
    pr_thresh = df[pr_col].quantile(low_pr_q) if df[pr_col].notna().any() else 0.0
    ch_thresh = df[ch_col].quantile(1 - high_ch_q) if df[ch_col].notna().any() else 1.0
    flags = []
    for r in df.itertuples(index=False):
        f = []
        if getattr(r, 'inlinks', 0) == 0: f.append('Orphan')
        if getattr(r, pr_col) <= pr_thresh: f.append('Low internal PR')
        if getattr(r, backlink_col, 0.0) > 0 and getattr(r, pr_col) <= pr_thresh: f.append('Has backlinks, needs internal links')
        if getattr(r, ch_col) >= ch_thresh: f.append('Link hub (high CheiRank)')
        flags.append(', '.join(f))
    df['flags'] = flags
    return df

def parse_suggestion_line(line: str) -> Dict[str, object]:
    out = {"source_url": line.strip(), "score": np.nan, "PR": np.nan, "CH": np.nan, "sim": np.nan, "source_category": ""}
    if not line or '|' not in line:
        return out
    parts = [p.strip() for p in line.split('|')]
    if not parts:
        return out
    out["source_url"] = parts[0]
    for tok in parts[1:]:
        if '=' in tok:
            k, v = tok.split('=', 1)
            k = k.strip().lower()
            v = v.strip()
            if k == 'score':
                try: out['score'] = float(v)
                except: pass
            elif k == 'pr':
                try: out['PR'] = float(v)
                except: pass
            elif k == 'ch':
                try: out['CH'] = float(v)
                except: pass
            elif k == 'sim':
                try: out['sim'] = float(v)
                except: pass
            elif k == 'cat':
                out['source_category'] = v
    return out

def explode_suggestions(sugg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sugg_df.itertuples(index=False):
        lines = str(getattr(r, 'suggestions', '')).split('\n')
        for rank, line in enumerate(lines, start=1):
            parsed = parse_suggestion_line(line)
            rows.append({
                "target": r.target,
                "target_category": r.category,
                "reason": r.reason,
                "anchor_hint": r.anchor_hint,
                "source_rank": rank,
                **parsed
            })
    return pd.DataFrame(rows)

# UI
st.set_page_config(page_title="SEO PageRank & CheiRank Analyzer", layout="wide")
st.title("SEO PageRank & CheiRank Analyzer")

st.markdown("""
By: Andre Neves https://www.linkedin.com/in/andreneves/
**What this tool does**
- Ingests **4 exports** (Screaming Frog Pages + All Inlinks, Ahrefs Backlinks, GSC).
- Builds an **internal-only link graph** using your homepage to filter out external links and excludes homepage as a source.
- Computes **PageRank** (importance) and **CheiRank** (outlinking hubs).
- Uses **Ahrefs** as PageRank personalization (weighted by *Links in group* or unique referring entities).
- Derives **semantics** from GSC queries + Title/H1/H2/Meta Description.
- Assigns **categories** to pages using TF‑IDF similarity of **URL path + Title + Description** vs your category list (we also expose a per-page **category_score**).
- Generates **multi-URL internal linking suggestions** per low-PR/orphan page, ranked by **semantic similarity + PR + CheiRank + backlinks**.
""")

with st.sidebar:
    st.header("Settings")
    homepage_input = st.text_input("Full homepage URL", value="", placeholder="https://www.example.com/", help="Used to: (1) keep ONLY internal links; (2) exclude homepage from suggestions.")
    home_root = canonical_home_root(homepage_input)

    categories_s = st.text_area("Main categories (one per line or comma-separated)", value="")
    categories = [c.strip() for c in re.split(r"[\n,]+", categories_s) if c.strip()]
    prefer_same_category = st.checkbox("Prefer same-category sources", value=True)
    same_category_only = st.checkbox("Restrict to same category only", value=False)

    link_scope = st.selectbox("Which links should count for PR?", options=["all","contextual","menu_footer"], index=1)
    keep_query = st.checkbox("Keep URL query parameters", value=False)
    alpha = st.slider("PageRank damping (alpha)", 0.50, 0.99, DEFAULT_PAGERANK_ALPHA, 0.01)
    low_pr_q = st.slider("Low PR threshold (quantile)", 0.05, 0.5, 0.20, 0.05)
    high_ch_q = st.slider("High CheiRank threshold (top quantile)", 0.05, 0.5, 0.10, 0.05)

st.caption("Upload CSV or Excel exports. All four files are **required**.")
c1, c2, c3, c4 = st.columns(4)
with c1: pages_file = st.file_uploader("PAGES export (Screaming Frog HTML)", type=["csv","xlsx","xls"], key="pages")
with c2: inlinks_file = st.file_uploader("INLINKS export (All Inlinks)", type=["csv","xlsx","xls"], key="inlinks")
with c3: backlinks_file = st.file_uploader("BACKLINKS export (Ahrefs)", type=["csv","xlsx","xls"], key="backlinks", help="Required")
with c4: gsc_file = st.file_uploader("GSC export (for semantics)", type=["csv","xlsx","xls"], key="gsc", help="Required")

if not homepage_input:
    st.warning("Enter your full homepage URL first.")

if pages_file and inlinks_file and backlinks_file and gsc_file and homepage_input:
    try:
        df_pages = read_table(pages_file)
        df_inlinks = read_table(inlinks_file)
        df_backlinks = read_table(backlinks_file)
        df_gsc = read_gsc(gsc_file)

        G, df_pages = build_internal_graph(df_pages, df_inlinks, keep_query, link_scope=link_scope, home_root=home_root)

        backlink_counts = aggregate_backlinks(df_backlinks, keep_query)
        p_vec = backlink_counts[backlink_counts.index.isin(G.nodes())].copy()
        personalization = (p_vec / p_vec.sum()).to_dict() if p_vec.sum() > 0 else None

        pr = compute_pagerank(G, alpha=alpha, personalization=personalization)
        ch = compute_cheirank(G, alpha=alpha)

        indeg = dict(G.in_degree()); outdeg = dict(G.out_degree())
        nodes = list(G.nodes())
        df = pd.DataFrame({
            'url': nodes,
            'inlinks': [indeg.get(n, 0) for n in nodes],
            'outlinks': [outdeg.get(n, 0) for n in nodes],
            'pagerank': [pr.get(n, 0.0) for n in nodes],
            'cheirank': [ch.get(n, 0.0) for n in nodes],
            'backlinks_refcnt': [float(backlink_counts.get(n, 0.0)) for n in nodes],
        })
        df['pagerank_norm'] = df['pagerank'] / (df['pagerank'].sum() if df['pagerank'].sum() > 0 else 1.0)
        df['cheirank_norm'] = df['cheirank'] / (df['cheirank'].sum() if df['cheirank'].sum() > 0 else 1.0)

        gsc_sem = build_gsc_semantics(df_gsc, keep_query=keep_query)
        page_texts = build_page_texts(df_pages, gsc_sem)
        category_map, category_scores = assign_categories_semantic(df_pages, categories) if categories else ({u: "" for u in df['url']}, {u: 0.0 for u in df['url']})
        df['category'] = df['url'].map(lambda u: category_map.get(u, ""))
        df['category_score'] = df['url'].map(lambda u: float(category_scores.get(u, 0.0)))

        df = flag_pages(df, 'pagerank', 'cheirank', 'backlinks_refcnt', low_pr_q=low_pr_q, high_ch_q=high_ch_q)

        st.success("Analysis complete.")

        tabs = st.tabs(["Overview","Low PR candidates","Orphans","Backlinks but low PR","Link hubs (high CheiRank)","All pages","Menu/Footer links (debug)","Suggestions"])

        with tabs[0]:
            st.subheader("Top pages by PageRank")
            st.dataframe(df.sort_values('pagerank', ascending=False).head(25), use_container_width=True)
            st.subheader("Top pages by CheiRank")
            st.dataframe(df.sort_values('cheirank', ascending=False).head(25), use_container_width=True)

        with tabs[1]:
            low_pr_thresh = df['pagerank'].quantile(low_pr_q)
            st.caption(f"Low PR threshold: pagerank <= {low_pr_thresh:.4g} (quantile {low_pr_q})")
            st.dataframe(df[df['pagerank'] <= low_pr_thresh].sort_values('pagerank'), use_container_width=True)

        with tabs[2]:
            st.caption("Pages with zero internal inlinks")
            st.dataframe(df[df['inlinks'] == 0].sort_values('pagerank', ascending=False), use_container_width=True)

        with tabs[3]:
            low_pr_thresh = df['pagerank'].quantile(low_pr_q)
            sel = df[(df['backlinks_refcnt'] > 0) & (df['pagerank'] <= low_pr_thresh)]
            st.caption("Pages that have external backlinks but low internal PageRank, add internal links to them.")
            st.dataframe(sel.sort_values(['backlinks_refcnt','pagerank'], ascending=[False, True]), use_container_width=True)

        with tabs[4]:
            high_ch_thresh = df['cheirank'].quantile(1 - high_ch_q)
            st.caption(f"High CheiRank threshold: cheirank >= {high_ch_thresh:.4g} (top {int(high_ch_q*100)}%)")
            st.dataframe(df[df['cheirank'] >= high_ch_thresh].sort_values('cheirank', ascending=False), use_container_width=True)

        with tabs[5]:
            st.dataframe(df.sort_values('pagerank', ascending=False), use_container_width=True)

        with tabs[6]:
            st.caption("Rows classified as header/footer/navigation, for transparency")
            pos_col, path_col, elem_col, anch_col = detect_link_position_columns(df_inlinks)
            if pos_col or path_col or elem_col or anch_col:
                classified = df_inlinks.copy()
                classified['_cls'] = classified.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
                st.dataframe(classified[classified['_cls'] == 'menu_footer'].head(300), use_container_width=True)
            else:
                st.info("No link position/path columns found in Inlinks to classify.")

        with tabs[7]:
            st.subheader("Internal link suggestions")
            sugg_df = suggest_internal_links(df, page_texts, category_map,
                                             prefer_same_category=prefer_same_category,
                                             same_category_only=same_category_only,
                                             homepage_url=homepage_input,
                                             low_pr_q=low_pr_q,
                                             top_k_sources=5)
            if sugg_df.empty:
                st.info("No targets met the criteria for suggestions.")
            else:
                st.markdown("**Compact view (one row per target)**")
                st.dataframe(sugg_df, use_container_width=True)
                csv_compact = sugg_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download compact CSV", data=csv_compact, file_name="internal_link_suggestions_compact.csv", mime="text/csv")

                st.markdown("**Detailed view (one source per row)**")
                long_df = explode_suggestions(sugg_df)
                st.dataframe(long_df, use_container_width=True, height=min(900, 80 + 28*len(long_df)))
                csv_long = long_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download detailed CSV", data=csv_long, file_name="internal_link_suggestions_detailed.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Download results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download metrics CSV", data=csv, file_name="seo_rank_results.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Quick recommendations")
        orphan_cnt = int((df['inlinks'] == 0).sum())
        low_pr_thresh = df['pagerank'].quantile(low_pr_q)
        low_pr_cnt = int((df['pagerank'] <= low_pr_thresh).sum())
        with_backlinks_low_pr = int(((df['backlinks_refcnt'] > 0) & (df['pagerank'] <= low_pr_thresh)).sum())
        hubs_cnt = int((df['cheirank'] >= df['cheirank'].quantile(1 - high_ch_q)).sum())
        rec = pd.DataFrame({
            "Item": ["Orphans","Low internal PR","Backlinked but low PR","Link hubs"],
            "Count": [orphan_cnt, low_pr_cnt, with_backlinks_low_pr, hubs_cnt],
            "Action": [
                "Add contextual links from relevant categories/hubs",
                f"Promote with internal links from high PR & semantically related pages (<= {low_pr_q:.0%} quantile)",
                "Add internal links from topically related hubs and evergreen guides",
                "Audit outlinks; reduce noise; keep contextual anchors"
            ]
        })
        st.dataframe(rec, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    st.info("Upload ALL FOUR files to begin: PAGES, INLINKS, BACKLINKS (Ahrefs), and GSC. Also enter your homepage URL.")
