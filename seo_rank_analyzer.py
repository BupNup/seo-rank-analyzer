"""
SEO PageRank & CheiRank Analyzer – Streamlit App
Single-file Streamlit app to upload Screaming Frog + Ahrefs exports, compute PageRank and CheiRank,
flag orphan pages and pages needing internal links.

How to run locally:
1) Save this file as `seo_rank_analyzer.py`
2) Install deps:  
   pip install streamlit pandas numpy networkx python-dateutil chardet
3) Start the app:  
   streamlit run seo_rank_analyzer.py

Expected uploads:
- PAGES export (Screaming Frog Internal > HTML): must include a URL column, usually named "Address".
- INLINKS export (Screaming Frog All Inlinks): must include source and target columns, usually "Source" and "Destination".
- BACKLINKS export (Ahrefs, optional): should include at least a target URL column, typically "Target URL" and either
  "Referring page URL" or "Referring domain". We will deduplicate to domains if possible.

Notes:
- We normalize URLs by lowercasing, stripping fragments, and trimming trailing slashes except root. Set options in the sidebar.
- PageRank can optionally use a personalization vector based on backlink strength.
- CheiRank is PageRank computed on the reversed internal link graph (no personalization).
- We flag:
  * Orphans: pages with zero internal inlinks.
  * Low internal PR: bottom quantile threshold (configurable, default 20%).
  * High external opportunity: pages with backlinks but low internal PR.
  * Link hubs: top cheirank quantile (configurable, default top 10%). Consider reducing outlinks or making them more strategic.

Author: ChatGPT
"""

from __future__ import annotations
import io
import re
import sys
import math
from typing import Dict, List, Optional, Tuple

import chardet
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

# ---------------------------
# Helpers
# ---------------------------

DEFAULT_PAGERANK_ALPHA = 0.85

URL_COL_CANDIDATES_PAGES = [
    "Address", "URL", "Url", "address", "Final URL", "Final Address",
]

INLINKS_SRC_CANDIDATES = ["Source", "From", "source", "From URL", "From Address"]
INLINKS_DST_CANDIDATES = ["Destination", "To", "Target", "Target URL", "To URL", "To Address"]

AHREFS_TARGET_CANDIDATES = ["Target URL", "Target url", "URL", "Address"]
AHREFS_REFERRING_PAGE_CANDS = ["Referring page URL", "Referring Page", "Source URL", "Backlink URL"]
AHREFS_REFERRING_DOMAIN_CANDS = ["Referring domain", "Referring Domain", "Domain"]

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

# URL normalization

def _strip_fragment(url: str) -> str:
    return url.split('#', 1)[0]

def _strip_query(url: str, keep_query: bool) -> str:
    if keep_query:
        return url
    return url.split('?', 1)[0]

def normalize_url(url: str, keep_query: bool = False) -> str:
    if not isinstance(url, str):
        return ""
    u = url.strip().lower()
    if not u:
        return ""
    u = _strip_fragment(u)
    u = _strip_query(u, keep_query)
    # Trim trailing slash except root
    if u.endswith('/'):
        # Keep protocol and host only slash (e.g., https://site.com/)
        m = re.match(r"^(https?://[^/]+)/$", u)
        if not m:
            u = u[:-1]
    return u

# Column resolution

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# Build internal graph

def build_internal_graph(pages: pd.DataFrame, inlinks: pd.DataFrame, keep_query: bool, link_scope: str = 'all') -> Tuple[nx.DiGraph, pd.DataFrame]:
    """Build an internal link graph. link_scope: 'all' | 'contextual' | 'menu_footer'"""
    url_col = pick_column(pages, URL_COL_CANDIDATES_PAGES)
    if not url_col:
        raise ValueError("Pages file needs a URL column like 'Address' or 'URL'.")
    pages['url_norm'] = pages[url_col].map(lambda x: normalize_url(x, keep_query))
    pages_nodes = pages['url_norm'].dropna().drop_duplicates()
    pages_nodes = pages_nodes[pages_nodes != ""]

    src_col = pick_column(inlinks, INLINKS_SRC_CANDIDATES)
    dst_col = pick_column(inlinks, INLINKS_DST_CANDIDATES)
    if not src_col or not dst_col:
        raise ValueError("Inlinks file needs columns like 'Source' and 'Destination'.")

    inlinks = inlinks.copy()
    inlinks['src'] = inlinks[src_col].map(lambda x: normalize_url(x, keep_query))
    inlinks['dst'] = inlinks[dst_col].map(lambda x: normalize_url(x, keep_query))

    # Filter by scope (e.g., drop header/footer links when 'contextual')
    inlinks = filter_inlinks_by_scope(inlinks, link_scope)

    # Keep edges that point to known pages
    valid_nodes = set(pages_nodes.tolist())
    df_edges = inlinks[['src', 'dst']].dropna()
    df_edges = df_edges[(df_edges['src'] != "") & (df_edges['dst'] != "")]
    df_edges = df_edges[df_edges['dst'].isin(valid_nodes)]

    # Aggregate multiple links as weight
    df_edges['weight'] = 1.0
    df_edges = df_edges.groupby(['src', 'dst'], as_index=False)['weight'].sum()

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(valid_nodes)

    # Only keep edges where src is in our node set too
    df_edges = df_edges[df_edges['src'].isin(valid_nodes)]

    for row in df_edges.itertuples(index=False):
        G.add_edge(row.src, row.dst, weight=float(row.weight))

    return G, pages

# Backlink strength aggregator

def aggregate_backlinks(backlinks: pd.DataFrame, keep_query: bool) -> pd.Series:
    if backlinks is None or backlinks.empty:
        return pd.Series(dtype=float)

    tgt_col = pick_column(backlinks, AHREFS_TARGET_CANDIDATES)
    if not tgt_col:
        raise ValueError("Backlinks file needs a 'Target URL' or similar column.")

    backlinks = backlinks.copy()
    backlinks['target'] = backlinks[tgt_col].map(lambda x: normalize_url(x, keep_query))

    # Prefer referring domain if present, else dedupe referring pages
    dom_col = pick_column(backlinks, AHREFS_REFERRING_DOMAIN_CANDS)
    ref_col = pick_column(backlinks, AHREFS_REFERRING_PAGE_CANDS)

    if dom_col:
        backlinks['ref_key'] = backlinks[dom_col].astype(str).str.lower().str.strip()
    elif ref_col:
        backlinks['ref_key'] = backlinks[ref_col].map(lambda x: normalize_url(x, keep_query))
    else:
        # Fallback: count rows per target
        backlinks['ref_key'] = np.arange(len(backlinks))

    agg = backlinks.dropna(subset=['target', 'ref_key'])
    agg = agg[(agg['target'] != "") & (agg['ref_key'] != "")]
    counts = agg.groupby('target')['ref_key'].nunique()
    counts = counts.astype(float)
    return counts

# Compute ranks

def compute_pagerank(G: nx.DiGraph, alpha: float, personalization: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    # Ensure graph has at least one edge, otherwise PR is uniform
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        if n == 0:
            return {}
        uni = 1.0 / n
        return {node: uni for node in G.nodes}
    try:
        pr = nx.pagerank(G, alpha=alpha, weight='weight', personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=alpha, weight='weight', max_iter=200, personalization=personalization)
    return pr

def compute_cheirank(G: nx.DiGraph, alpha: float) -> Dict[str, float]:
    GR = G.reverse(copy=True)
    return compute_pagerank(GR, alpha=alpha, personalization=None)

# Flags

def flag_pages(df: pd.DataFrame, pr_col: str, ch_col: str, backlink_col: str,
               low_pr_q: float, high_ch_q: float) -> pd.DataFrame:
    df = df.copy()
    # Quantile thresholds
    if len(df) == 0:
        return df
    pr_thresh = df[pr_col].quantile(low_pr_q) if df[pr_col].notna().any() else 0.0
    ch_thresh = df[ch_col].quantile(1 - high_ch_q) if df[ch_col].notna().any() else 1.0

    flags = []
    for r in df.itertuples(index=False):
        f = []
        if getattr(r, 'inlinks', 0) == 0:
            f.append('Orphan')
        if getattr(r, pr_col) <= pr_thresh:
            f.append('Low internal PR')
        if getattr(r, backlink_col, 0.0) > 0 and getattr(r, pr_col) <= pr_thresh:
            f.append('Has backlinks, needs internal links')
        if getattr(r, ch_col) >= ch_thresh:
            f.append('Link hub (high CheiRank)')
        flags.append(', '.join(f))
    df['flags'] = flags
    return df

# ---------------------------
# UI
# ---------------------------

def detect_link_position_columns(df: pd.DataFrame):
    """Return best-effort columns for position/path/element/anchor."""
    pos = pick_column(df, ["Link Position","Position","Placement"])  # Screaming Frog usually "Link Position"
    path = pick_column(df, ["Link Path","DOM Path","XPath"])        # Screaming Frog usually "Link Path"
    elem = pick_column(df, ["Link Element","Element"])                # sometimes present
    anch = pick_column(df, ["Anchor","Anchor Text","Anchor text"])  # anchor text
    return pos, path, elem, anch


def classify_inlink_row(row: pd.Series, pos_col: Optional[str], path_col: Optional[str], elem_col: Optional[str], anchor_col: Optional[str]) -> str:
    """Classify an inlink as 'contextual', 'menu_footer', or 'other' based on heuristics.
    We favor Screaming Frog's 'Link Position' when present.
    """
    def s(v):
        return str(v).lower() if pd.notna(v) else ""

    pos = s(row.get(pos_col)) if pos_col else ""
    path = s(row.get(path_col)) if path_col else ""
    elem = s(row.get(elem_col)) if elem_col else ""
    anch = s(row.get(anchor_col)) if anchor_col else ""

    # If SF already labels it clearly, trust that first
    if any(tok in pos for tok in ["header","navigation","nav","footer","breadcrumbs","breadcrumb","sidebar","pagination","menu","filter","widget"]):
        return "menu_footer"
    if "content" in pos or "main" in pos:
        return "contextual"

    # Heuristics via DOM path / element
    if any(tok in path for tok in ["/header","/nav","/footer","/aside","breadcrumb","sidebar","menu","mega-menu","pagination","pager"]):
        return "menu_footer"
    if any(tok in elem for tok in ["nav","header","footer","aside","breadcrumb"]):
        return "menu_footer"
    if any(tok in path for tok in ["/main","/article","/section","/content"]) or any(tok in elem for tok in ["main","article","section"]):
        return "contextual"

    # Anchor text heuristic: very short generic anchors are likely nav
    generic = {"home","about","contact","login","register","sign in","sign up","terms","privacy","blog","categories","slots","casinos"}
    if anch in generic or len(anch) <= 2:
        return "menu_footer"

    return "other"


def filter_inlinks_by_scope(inlinks: pd.DataFrame, scope: str) -> pd.DataFrame:
    """scope in {'all','contextual','menu_footer'}"""
    if scope == 'all':
        return inlinks
    pos_col, path_col, elem_col, anch_col = detect_link_position_columns(inlinks)
    if not (pos_col or path_col or elem_col or anch_col):
        # No metadata to classify; return unchanged for 'all', empty for scopes
        return inlinks if scope == 'all' else inlinks.iloc[0:0]
    tmp = inlinks.copy()
    tmp['_cls'] = tmp.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
    if scope == 'contextual':
        return tmp[tmp['_cls'].isin(['contextual','other'])].drop(columns=['_cls'])
    if scope == 'menu_footer':
        return tmp[tmp['_cls'] == 'menu_footer'].drop(columns=['_cls'])
    return inlinks


st.set_page_config(page_title="SEO PageRank & CheiRank Analyzer", layout="wide")

st.title("SEO PageRank & CheiRank Analyzer")

with st.sidebar:
    st.header("Settings")
    link_scope = st.selectbox(
        "Which links should count for PR?",
        options=["all","contextual","menu_footer"],
        index=1,
        help="Choose 'contextual' to downweight/remove header/footer/nav links."
    )
    keep_query = st.checkbox("Keep URL query parameters", value=False, help="If off, query strings are removed before matching.")
    alpha = st.slider("PageRank damping (alpha)", 0.50, 0.99, DEFAULT_PAGERANK_ALPHA, 0.01)
    use_backlinks = st.checkbox("Use backlinks as personalization", value=True,
                               help="Weights teleport toward pages with more referring domains or pages.")
    low_pr_q = st.slider("Low PR threshold (quantile)", 0.05, 0.5, 0.20, 0.05)
    high_ch_q = st.slider("High CheiRank threshold (top quantile)", 0.05, 0.5, 0.10, 0.05)

    st.markdown("---")
    st.caption("Upload CSV or Excel. Column names are auto-detected.")

col1, col2, col3 = st.columns(3)
with col1:
    pages_file = st.file_uploader("PAGES export (Screaming Frog HTML)", type=["csv", "xlsx", "xls"], key="pages")
with col2:
    inlinks_file = st.file_uploader("INLINKS export (All Inlinks)", type=["csv", "xlsx", "xls"], key="inlinks")
with col3:
    backlinks_file = st.file_uploader("BACKLINKS export (Ahrefs, optional)", type=["csv", "xlsx", "xls"], key="backlinks")

if pages_file and inlinks_file:
    try:
        df_pages = read_table(pages_file)
        df_inlinks = read_table(inlinks_file)
        df_backlinks = read_table(backlinks_file) if backlinks_file else pd.DataFrame()

        # Build graph
        G, df_pages = build_internal_graph(df_pages, df_inlinks, keep_query, link_scope=link_scope)

        # Compute backlink counts for personalization
        backlink_counts = aggregate_backlinks(df_backlinks, keep_query) if use_backlinks else pd.Series(dtype=float)

        # Personalization vector: proportional to backlink counts over our nodes
        personalization = None
        if use_backlinks and len(backlink_counts) > 0:
            # Restrict to nodes present in graph
            p_vec = backlink_counts[backlink_counts.index.isin(G.nodes())].copy()
            if p_vec.sum() == 0:
                personalization = None
            else:
                p_vec = p_vec / p_vec.sum()
                personalization = p_vec.to_dict()

        pr = compute_pagerank(G, alpha=alpha, personalization=personalization)
        ch = compute_cheirank(G, alpha=alpha)

        # Degrees
        indeg = dict(G.in_degree())
        outdeg = dict(G.out_degree())

        # Assemble output
        nodes = list(G.nodes())
        df = pd.DataFrame({
            'url': nodes,
            'inlinks': [indeg.get(n, 0) for n in nodes],
            'outlinks': [outdeg.get(n, 0) for n in nodes],
            'pagerank': [pr.get(n, 0.0) for n in nodes],
            'cheirank': [ch.get(n, 0.0) for n in nodes],
            'backlinks_refcnt': [float(backlink_counts.get(n, 0.0)) if len(backlink_counts) else 0.0 for n in nodes],
        })

        # Normalize PR for readability
        if df['pagerank'].sum() > 0:
            df['pagerank_norm'] = df['pagerank'] / df['pagerank'].sum()
        else:
            df['pagerank_norm'] = df['pagerank']
        if df['cheirank'].sum() > 0:
            df['cheirank_norm'] = df['cheirank'] / df['cheirank'].sum()
        else:
            df['cheirank_norm'] = df['cheirank']

        df = flag_pages(df, 'pagerank', 'cheirank', 'backlinks_refcnt', low_pr_q=low_pr_q, high_ch_q=high_ch_q)

        # Rankings and insights
        st.success("Analysis complete.")

        tabs = st.tabs([
            "Overview", "Low PR candidates", "Orphans", "Backlinks but low PR", "Link hubs (high CheiRank)", "All pages", "Menu/Footer links (debug)"
        ])

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
            st.dataframe(sel.sort_values(['backlinks_refcnt', 'pagerank'], ascending=[False, True]), use_container_width=True)

        with tabs[4]:
            high_ch_thresh = df['cheirank'].quantile(1 - high_ch_q)
            st.caption(f"High CheiRank threshold: cheirank >= {high_ch_thresh:.4g} (top {int(high_ch_q*100)}%)")
            st.dataframe(df[df['cheirank'] >= high_ch_thresh].sort_values('cheirank', ascending=False), use_container_width=True)

        with tabs[5]:
            st.dataframe(df.sort_values('pagerank', ascending=False), use_container_width=True)

        with tabs[6]:
            st.caption("Rows classified as header/footer/navigation, for transparency")
            pos_col, path_col, elem_col, anch_col = detect_link_position_columns(df_inlinks)
            if pos_col or path_col:
                classified = df_inlinks.copy()
                if pos_col or path_col or elem_col or anch_col:
                    classified['_cls'] = classified.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
                st.dataframe(classified[classified.get('_cls','other')=='menu_footer'].head(200), use_container_width=True)
            else:
                st.info("No link position/path columns found in Inlinks to classify.")

        # Downloads
        st.markdown("---")
        st.subheader("Download results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="seo_rank_results.csv", mime="text/csv")

        # Simple recommendations
        st.markdown("---")
        st.subheader("Quick recommendations")
        orphan_cnt = int((df['inlinks'] == 0).sum())
        low_pr_thresh = df['pagerank'].quantile(low_pr_q)
        low_pr_cnt = int((df['pagerank'] <= low_pr_thresh).sum())
        with_backlinks_low_pr = int(((df['backlinks_refcnt'] > 0) & (df['pagerank'] <= low_pr_thresh)).sum())
        hubs_cnt = int((df['cheirank'] >= df['cheirank'].quantile(1 - high_ch_q)).sum())

        st.write(
            f"• Orphans: **{orphan_cnt}**. Connect them from relevant hubs and category pages.\n"
            f"• Low internal PR: **{low_pr_cnt}** pages at or below the {low_pr_q:.0%} quantile.\n"
            f"• Backlinked but low PR: **{with_backlinks_low_pr}**. Add contextual internal links to these.\n"
            f"• Link hubs: **{hubs_cnt}** high CheiRank pages. Audit outlinks and anchor relevance."
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    st.info("Upload at least the PAGES and INLINKS exports to begin. Backlinks are optional but recommended.")
