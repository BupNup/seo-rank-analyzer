from __future__ import annotations
import io, re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import chardet, numpy as np, pandas as pd, networkx as nx, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional embeddings (falls back to TF-IDF if unavailable)
_EMBED_READY = False
try:
    from sentence_transformers import SentenceTransformer
    import torch
    _EMBED_READY = True
except Exception:
    _EMBED_READY = False

DEFAULT_PAGERANK_ALPHA = 0.85

# Column name candidates
URL_COL_CANDIDATES_PAGES = ["Address","URL","Url","address","Final URL","Final Address"]
INLINKS_SRC_CANDIDATES   = ["Source","From","From URL","From Address","source"]
INLINKS_DST_CANDIDATES   = ["Destination","To","Target","Target URL","To URL","To Address"]
AHREFS_TARGET_CANDIDATES = ["Target URL","Target url","URL","Address"]
AHREFS_REFERRING_PAGE_CANDS   = ["Referring page URL","Referring Page","Source URL","Backlink URL"]
AHREFS_REFERRING_DOMAIN_CANDS = ["Referring domain","Referring Domain","Domain"]
AHREFS_DR_CANDS = ["DR","Domain Rating","Domain rating"]
AHREFS_UR_CANDS = ["UR","URL Rating","Url Rating","URL rating"]

GSC_PAGE_CANDS = ["Page","Landing page","Landing Page","URL","Address"]
GSC_QUERY_CANDS= ["Query","Search query","Search Query"]
GSC_IMPR_CANDS = ["Impressions","Impr","Impressions Total"]

PAGE_TITLE_CANDS = ["Title 1","Title","Meta Title"]
PAGE_H1_CANDS    = ["H1-1","H1"]

STATUS_CODE_CANDS = ["Status Code","Status code","Status","HTTP Status","StatusCode","Response Codes"]

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def sniff_encoding(file_bytes: bytes) -> str:
    return chardet.detect(file_bytes).get("encoding") or "utf-8"

@st.cache_data(show_spinner=False)
def read_table(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read(); enc = sniff_encoding(raw); buf = io.BytesIO(raw)
    try: return pd.read_csv(buf, sep=None, engine="python", encoding=enc)
    except Exception: buf.seek(0); return pd.read_excel(buf)

@st.cache_data(show_spinner=False)
def read_gsc(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read(); enc = sniff_encoding(raw); buf = io.BytesIO(raw)
    try: return pd.read_csv(buf, encoding=enc)
    except Exception: buf.seek(0); return pd.read_excel(buf)

def _strip_fragment(u:str) -> str: return u.split("#",1)[0]
def _strip_query(u:str, keep:bool) -> str: return u if keep else u.split("?",1)[0]

def normalize_url(url: str, keep_query: bool=False) -> str:
    if not isinstance(url, str): return ""
    u = url.strip().lower()
    if not u: return ""
    u = _strip_fragment(u); u = _strip_query(u, keep_query)
    if u.endswith("/"):
        m = re.match(r"^(https?://[^/]+)/$", u)
        if not m: u = u[:-1]
    return u

def pick_column(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

def get_host(u:str) -> str:
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

def canonical_home_root(homepage_url:str) -> str:
    try:
        host = urlparse(homepage_url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""

def is_internal_host(host:str, home_root:str) -> bool:
    if not home_root: return True
    h, base = host.lower(), home_root.lower()
    return h == base or h.endswith("." + base) or h == "www." + base

def detect_link_position_columns(df: pd.DataFrame):
    pos = pick_column(df, ["Link Position","Position","Placement"])
    path= pick_column(df, ["Link Path","DOM Path","XPath"])
    elem= pick_column(df, ["Link Element","Element"])
    anch= pick_column(df, ["Anchor","Anchor Text","Anchor text"])
    return pos, path, elem, anch

def classify_inlink_row(row: pd.Series, pos_col, path_col, elem_col, anchor_col) -> str:
    def s(v): return str(v).lower() if pd.notna(v) else ""
    pos=s(row.get(pos_col)) if pos_col else ""; path=s(row.get(path_col)) if path_col else ""
    elem=s(row.get(elem_col)) if elem_col else ""; anch=s(row.get(anchor_col)) if anchor_col else ""
    if any(t in pos for t in ["header","navigation","nav","footer","breadcrumbs","breadcrumb","sidebar","pagination","menu","widget"]): return "menu_footer"
    if "content" in pos or "main" in pos: return "contextual"
    if any(t in path for t in ["/header","/nav","/footer","/aside","breadcrumb","sidebar","menu","mega-menu","pagination","pager"]): return "menu_footer"
    if any(t in elem for t in ["nav","header","footer","aside","breadcrumb"]): return "menu_footer"
    if any(t in path for t in ["/main","/article","/section","/content"]) or any(t in elem for t in ["main","article","section"]): return "contextual"
    if anch in {"home","about","contact","login","register","sign in","sign up","terms","privacy","blog","categories","slots","casinos"} or len(anch)<=2: return "menu_footer"
    return "other"

def filter_inlinks_by_scope(inlinks: pd.DataFrame, scope:str) -> pd.DataFrame:
    if scope == "all": return inlinks
    pos_col, path_col, elem_col, anch_col = detect_link_position_columns(inlinks)
    if not (pos_col or path_col or elem_col or anch_col):
        return inlinks if scope == "all" else inlinks.iloc[0:0]
    tmp = inlinks.copy()
    tmp["_cls"] = tmp.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
    if scope == "contextual":   return tmp[tmp["_cls"].isin(["contextual","other"])].drop(columns=["_cls"])
    if scope == "menu_footer":  return tmp[tmp["_cls"] == "menu_footer"].drop(columns=["_cls"])
    return inlinks

# ---------- Graph build (HTTP 200 only) ----------
def build_internal_graph(
    pages: pd.DataFrame,
    inlinks: pd.DataFrame,
    keep_query: bool,
    link_scope: str,
    home_root: str
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    url_col = pick_column(pages, URL_COL_CANDIDATES_PAGES)
    if not url_col:
        raise ValueError("Pages file needs a URL column like 'Address' or 'URL'.")

    pages = pages.copy()
    pages["url_norm"] = pages[url_col].map(lambda x: normalize_url(x, keep_query))
    pages["host"]     = pages["url_norm"].map(get_host)

    if home_root:
        pages = pages[pages["host"].map(lambda h: is_internal_host(h, home_root))]

    status_col = pick_column(pages, STATUS_CODE_CANDS)
    if status_col:
        pages["status_code"] = pd.to_numeric(pages[status_col], errors="coerce")
        pages_200    = pages[(pages["status_code"] == 200) & (pages["url_norm"] != "")]
        pages_non200 = pages[(pages["status_code"] != 200) & (pages["url_norm"] != "")]
    else:
        pages["status_code"] = np.nan
        pages_200    = pages[pages["url_norm"] != ""]
        pages_non200 = pages.iloc[0:0]

    nodes_series = pages_200["url_norm"].dropna().drop_duplicates()
    nodes_series = nodes_series[nodes_series != ""]
    valid_nodes  = set(nodes_series.tolist())

    src_col = pick_column(inlinks, INLINKS_SRC_CANDIDATES)
    dst_col = pick_column(inlinks, INLINKS_DST_CANDIDATES)
    if not src_col or not dst_col:
        raise ValueError("Inlinks file needs columns like 'Source' and 'Destination'.")

    inlinks = inlinks.copy()
    inlinks["src"] = inlinks[src_col].map(lambda x: normalize_url(x, keep_query))
    inlinks["dst"] = inlinks[dst_col].map(lambda x: normalize_url(x, keep_query))
    inlinks["src_host"] = inlinks["src"].map(get_host)
    inlinks["dst_host"] = inlinks["dst"].map(get_host)

    inlinks = filter_inlinks_by_scope(inlinks, link_scope)

    if home_root:
        inlinks = inlinks[
            inlinks["src_host"].map(lambda h: is_internal_host(h, home_root)) &
            inlinks["dst_host"].map(lambda h: is_internal_host(h, home_root))
        ]

    edges = inlinks[["src","dst"]].dropna()
    edges = edges[(edges["src"]!="") & (edges["dst"]!="")]
    edges = edges[edges["dst"].isin(valid_nodes) & edges["src"].isin(valid_nodes)]
    edges["weight"] = 1.0
    edges = edges.groupby(["src","dst"], as_index=False)["weight"].sum()

    G = nx.DiGraph(); G.add_nodes_from(valid_nodes)
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, weight=float(r.weight))
    return G, pages_200, pages_non200

# ---------- Ahrefs aggregation (optional DR/UR quality) ----------
def _pick_numeric(df, cands): 
    col = pick_column(df, cands)
    return pd.to_numeric(df[col], errors="coerce") if col else None

def aggregate_backlinks(backlinks: pd.DataFrame, keep_query: bool, use_quality: bool=False) -> pd.Series:
    if backlinks is None or backlinks.empty: return pd.Series(dtype=float)
    tgt_col = pick_column(backlinks, AHREFS_TARGET_CANDIDATES)
    if not tgt_col: raise ValueError("Backlinks file needs a 'Target URL' or similar column.")
    df = backlinks.copy()
    df["target"] = df[tgt_col].map(lambda x: normalize_url(x, keep_query))

    grp_col = pick_column(df, ["Links in group","links in group"])
    base = pd.to_numeric(df[grp_col], errors="coerce").fillna(1.0) if grp_col else pd.Series(1.0, index=df.index)

    if use_quality:
        dr = _pick_numeric(df, AHREFS_DR_CANDS)  # 0–100
        ur = _pick_numeric(df, AHREFS_UR_CANDS)  # 0–100
        dr_w = np.sqrt(np.clip((dr or 0).fillna(0.0), 0, 100)/100.0) if dr is not None else None
        ur_w = np.sqrt(np.clip((ur or 0).fillna(0.0), 0, 100)/100.0) if ur is not None else None
        if dr_w is not None and ur_w is not None:
            qual = (dr_w * ur_w).replace(0, 1e-6)
        elif dr_w is not None:
            qual = dr_w.replace(0, 1e-6)
        elif ur_w is not None:
            qual = ur_w.replace(0, 1e-6)
        else:
            qual = pd.Series(1.0, index=df.index)
    else:
        qual = pd.Series(1.0, index=df.index)

    df["row_weight"] = base * qual

    dom_col = pick_column(df, AHREFS_REFERRING_DOMAIN_CANDS)
    if dom_col:
        df["ref_key"] = df[dom_col].astype(str).str.lower().str.strip()
    else:
        ref_col = pick_column(df, AHREFS_REFERRING_PAGE_CANDS)
        if ref_col: df["ref_key"] = df[ref_col].map(lambda x: normalize_url(x, keep_query))
        else:       df["ref_key"] = np.arange(len(df)).astype(str)

    agg = (df.dropna(subset=["target","ref_key"])
             .query("target != '' and ref_key != ''")
             .groupby(["target","ref_key"], as_index=False)["row_weight"].max())
    strength = agg.groupby("target")["row_weight"].sum().astype(float)
    return strength

# ---------- PR / CH ----------
def compute_pagerank(G: nx.DiGraph, alpha: float, personalization: Optional[Dict[str, float]]=None) -> Dict[str,float]:
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        return {} if n == 0 else {node: 1.0/n for node in G.nodes}
    try:
        return nx.pagerank(G, alpha=alpha, weight="weight", personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, alpha=alpha, weight="weight", personalization=personalization, max_iter=200)

def compute_cheirank(G: nx.DiGraph, alpha: float) -> Dict[str,float]:
    return compute_pagerank(G.reverse(copy=True), alpha=alpha, personalization=None)

# ---------- Semantics (NEW: only slug + Title + H1 + GSC, with weights) ----------
def build_gsc_semantics(gsc: pd.DataFrame, keep_query: bool) -> pd.Series:
    if gsc is None or gsc.empty: return pd.Series(dtype=str)
    pg_col = pick_column(gsc, GSC_PAGE_CANDS); q_col = pick_column(gsc, GSC_QUERY_CANDS); i_col = pick_column(gsc, GSC_IMPR_CANDS)
    if not (pg_col and q_col and i_col): return pd.Series(dtype=str)
    tmp = gsc[[pg_col,q_col,i_col]].rename(columns={pg_col:"page", q_col:"query", i_col:"impr"}).copy()
    tmp["page"] = tmp["page"].map(lambda x: normalize_url(x, keep_query))
    tmp["query"]= tmp["query"].astype(str).str.lower().str.replace(r"[^a-z0-9\-\s]"," ", regex=True)
    tmp["w"]    = (np.log2(pd.to_numeric(tmp["impr"], errors="coerce").fillna(0).clip(lower=1.0)) + 1.0).round().astype(int)
    tmp["rep"]  = tmp.apply(lambda r: (r["query"] + " ")*int(max(1, r["w"])), axis=1)
    return tmp.groupby("page")["rep"].apply(lambda x: " ".join(x)).astype(str)

def build_component_texts(df_pages: pd.DataFrame, gsc_sem: pd.Series) -> Dict[str, Dict[str,str]]:
    """Return per-url component texts: slug, title, h1, gsc (lowercased)."""
    url_col = pick_column(df_pages, URL_COL_CANDIDATES_PAGES)
    t_col   = pick_column(df_pages, PAGE_TITLE_CANDS)
    h1_col  = pick_column(df_pages, PAGE_H1_CANDS)
    out = {}
    for _, r in df_pages.iterrows():
        u = normalize_url(r.get(url_col,""), keep_query=False) if url_col else ""
        if not u: continue
        path = urlparse(u).path
        slug = path.replace("/", " ").replace("-", " ").strip().lower()
        title = (str(r.get(t_col,"")) if t_col else "").lower()
        h1    = (str(r.get(h1_col,"")) if h1_col else "").lower()
        gsc   = (str(gsc_sem.get(u,"")) if isinstance(gsc_sem, pd.Series) else "").lower()
        out[u] = {"slug":slug, "title":title, "h1":h1, "gsc":gsc}
    return out

@st.cache_resource(show_spinner=False)
def load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not _EMBED_READY: return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None

def _weights_ok(slug_w, title_w, h1_w, gsc_w):
    s = float(slug_w) + float(title_w) + float(h1_w) + float(gsc_w)
    if s <= 1e-9:  # avoid all-zero
        return 1.0, 1.0, 1.0, 1.0
    return float(slug_w), float(title_w), float(h1_w), float(gsc_w)

def semantic_backend_weighted(
    components: Dict[str, Dict[str,str]],
    urls: List[str],
    slug_w: float, title_w: float, h1_w: float, gsc_w: float,
    mode: str = "TF-IDF"
):
    """Return (kind, obj, matrix, index_map) where matrix is page vectors."""
    slug_w, title_w, h1_w, gsc_w = _weights_ok(slug_w, title_w, h1_w, gsc_w)

    # Build component corpora aligned by urls
    slug_c = [components.get(u,{}).get("slug","")  for u in urls]
    titl_c = [components.get(u,{}).get("title","") for u in urls]
    h1_c   = [components.get(u,{}).get("h1","")    for u in urls]
    gsc_c  = [components.get(u,{}).get("gsc","")   for u in urls]

    if mode.startswith("Emb"):
        model = load_embedder()
        if model is None:
            st.warning("Embeddings unavailable; falling back to TF-IDF.")
            mode = "TF-IDF"
        else:
            Es = model.encode(slug_c, normalize_embeddings=True, show_progress_bar=False)
            Et = model.encode(titl_c, normalize_embeddings=True, show_progress_bar=False)
            Eh = model.encode(h1_c,   normalize_embeddings=True, show_progress_bar=False)
            Eg = model.encode(gsc_c,  normalize_embeddings=True, show_progress_bar=False)
            M  = slug_w*Es + title_w*Et + h1_w*Eh + gsc_w*Eg
            # Row-normalize
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            M = M / norms
            return ("emb", None, M, {u:i for i,u in enumerate(urls)})

    # TF-IDF backend (fit on combined text for stable vocab)
    combined = [" ".join([slug_c[i], titl_c[i], h1_c[i], gsc_c[i]]) for i in range(len(urls))]
    vec = TfidfVectorizer(stop_words="english", min_df=1, max_features=80000)
    vec.fit(combined)
    Xs = vec.transform(slug_c)
    Xt = vec.transform(titl_c)
    Xh = vec.transform(h1_c)
    Xg = vec.transform(gsc_c)
    X  = slug_w*Xs + title_w*Xt + h1_w*Xh + gsc_w*Xg
    return ("tfidf", vec, X, {u:i for i,u in enumerate(urls)})

def semantic_sim_vector(kind, obj, mat, i_idx:int):
    if kind == "emb":
        sims = (mat[i_idx] @ mat.T).astype(np.float64)
        return np.clip(sims, -1.0, 1.0)
    return cosine_similarity(mat[i_idx], mat).ravel()

def assign_categories_semantic_weighted(
    df_pages: pd.DataFrame,
    categories: List[str],
    components: Dict[str, Dict[str,str]],
    slug_w: float, title_w: float, h1_w: float, gsc_w: float,
    mode: str = "TF-IDF"
) -> Tuple[Dict[str,str], Dict[str,float]]:
    if not categories:
        return {}, {}
    urls = [u for u in components.keys()]
    if not urls:
        return {}, {}

    # Page vectors (weighted)
    kind, obj, Xp, idx = semantic_backend_weighted(components, urls, slug_w, title_w, h1_w, gsc_w, mode)

    # Category vectors
    cat_labels = [c.strip() for c in categories if c.strip()]
    if not cat_labels:
        return {u:"" for u in urls}, {u:0.0 for u in urls}

    if kind == "emb":
        model = load_embedder()
        if model is None:
            # Fallback to TF-IDF for categories if embeddings unavailable
            mode = "TF-IDF"
    if kind == "emb":
        Xc = load_embedder().encode(cat_labels, normalize_embeddings=True, show_progress_bar=False)
        sims = Xp @ Xc.T
    else:
        vec = obj
        Xc = vec.transform(cat_labels)
        sims = cosine_similarity(Xp, Xc)

    best = sims.argmax(axis=1)
    scores = sims[np.arange(len(urls)), best]
    return (
        {urls[i]: cat_labels[best[i]] for i in range(len(urls))},
        {urls[i]: float(scores[i])    for i in range(len(urls))}
    )

# ---------- Flags & Suggestions ----------
def flag_pages(df: pd.DataFrame, low_pr_q: float, high_ch_q: float) -> pd.DataFrame:
    df = df.copy()
    pr_th = df["pagerank"].quantile(low_pr_q) if df["pagerank"].notna().any() else 0.0
    ch_th = df["cheirank"].quantile(1 - high_ch_q) if df["cheirank"].notna().any() else 1.0
    flags=[]
    for r in df.itertuples(index=False):
        f=[]
        if r.inlinks == 0: f.append("Orphan")
        if r.pagerank <= pr_th: f.append("Low internal PR")
        if r.backlinks_refcnt > 0 and r.pagerank <= pr_th: f.append("Has backlinks, needs internal links")
        if r.cheirank >= ch_th: f.append("Link hub (high CheiRank)")
        flags.append(", ".join(f))
    df["flags"] = flags; return df

def parse_suggestion_line(line:str) -> Dict[str,object]:
    out={"source_url":line.strip(),"score":np.nan,"PR":np.nan,"CH":np.nan,"sim":np.nan,"cap":np.nan,"source_category":""}
    if "|" not in line: return out
    parts=[p.strip() for p in line.split("|")]; out["source_url"]=parts[0]
    for tok in parts[1:]:
        if "=" in tok:
            k,v=[t.strip() for t in tok.split("=",1)]
            def f(x): 
                return float(x) if re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", x, flags=re.I) else np.nan
            kl=k.lower()
            if   kl=="score": out["score"]=f(v)
            elif kl=="pr":    out["PR"]=f(v)
            elif kl=="ch":    out["CH"]=f(v)
            elif kl=="sim":   out["sim"]=f(v)
            elif kl=="cap":   out["cap"]=f(v)
            elif kl=="cat":   out["source_category"]=v
    return out

def explode_suggestions(sugg_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for r in sugg_df.itertuples(index=False):
        for rank,line in enumerate(str(getattr(r,"suggestions","")).split("\n"), start=1):
            rows.append({"target":r.target,"target_category":r.category,"reason":r.reason,
                         "anchor_hint":r.anchor_hint,"source_rank":rank, **parse_suggestion_line(line)})
    return pd.DataFrame(rows)

def suggest_internal_links(
    df_metrics: pd.DataFrame,
    components: Dict[str, Dict[str,str]],
    combined_texts: Dict[str,str],
    category_map: Dict[str,str],
    prefer_same_category: bool,
    same_category_only: bool,
    homepage_url: str,
    exclude_homepage_source: bool = True,
    backlink_weight: float=0.15, pr_weight: float=0.35, ch_weight: float=0.15, sem_weight: float=0.35,
    use_link_budget: bool=False, cap_weight: float=0.20,
    low_pr_q: float=0.20, top_k_sources:int=5,
    semantic_mode:str="TF-IDF",
    slug_w: float=1.0, title_w: float=1.0, h1_w: float=1.0, gsc_w: float=1.0
) -> pd.DataFrame:
    df=df_metrics.copy()
    pr_th = df["pagerank"].quantile(low_pr_q)
    df["reason"]=""
    df.loc[df["inlinks"]==0,"reason"] += "Orphan; "
    df.loc[df["pagerank"]<=pr_th,"reason"] += "Low PR; "
    df.loc[(df["backlinks_refcnt"]>0)&(df["pagerank"]<=pr_th),"reason"] += "Has backlinks, needs internal links; "
    df["reason"]=df["reason"].str.strip()
    targets=df[df["reason"]!=""].copy()
    if targets.empty:
        return pd.DataFrame(columns=["target","category","reason","pagerank","cheirank","backlinks","inlinks","anchor_hint","suggestions"])

    urls=[u for u in df["url"] if u in components]
    if not urls:
        return pd.DataFrame(columns=["target","category","reason","pagerank","cheirank","backlinks","inlinks","anchor_hint","suggestions"])

    homepage_norm = normalize_url(homepage_url, keep_query=False)

    # Weighted semantic matrix
    kind,obj,M,idx = semantic_backend_weighted(components, urls, slug_w, title_w, h1_w, gsc_w, semantic_mode)

    pr_norm = df.set_index("url")["pagerank_norm"].reindex(urls).fillna(0.0).to_numpy()
    ch_norm = df.set_index("url")["cheirank_norm"].reindex(urls).fillna(0.0).to_numpy()
    bl = df.set_index("url")["backlinks_refcnt"].reindex(urls).fillna(0.0); bl_norm = (bl / (bl.max() if bl.max()>0 else 1.0)).to_numpy()
    out_deg = df.set_index("url")["outlinks"].reindex(urls).fillna(0).to_numpy()

    cap = (pr_norm / (out_deg + 1.0)) if use_link_budget else np.zeros_like(pr_norm)
    if use_link_budget and cap.max()>0: cap = cap / cap.max()

    rows=[]; from collections import Counter
    for t in targets.itertuples(index=False):
        if t.url not in idx: continue
        i = idx[t.url]; sims = semantic_sim_vector(kind, obj, M, i)

        score = (sem_weight*sims + pr_weight*pr_norm + ch_weight*ch_norm + backlink_weight*bl_norm + (cap_weight*cap if use_link_budget else 0))
        allow = np.ones(len(urls), dtype=bool); allow[i]=False
        if exclude_homepage_source and homepage_norm in urls:
            allow[urls.index(homepage_norm)] = False

        t_cat = category_map.get(t.url,"")
        if same_category_only and t_cat:
            allow &= np.array([category_map.get(u,"")==t_cat for u in urls])
        elif prefer_same_category and t_cat:
            same_mask = np.array([category_map.get(u,"")==t_cat for u in urls])
            score = np.where(same_mask, score*1.15, score)  # +15% boost

        score = np.where(out_deg>0, score, score*0.5)  # soft penalty for sources with 0 outlinks
        score = np.where(allow, score, -1.0)

        top_idx = score.argsort()[::-1][:top_k_sources]
        cands = [f"{urls[j]} | score={score[j]:.3f} | PR={pr_norm[j]:.3f} | CH={ch_norm[j]:.3f} | sim={sims[j]:.3f} | cap={cap[j]:.3f} | cat={category_map.get(urls[j],'')}" for j in top_idx]

        toks=[w for w in re.findall(r"[a-z0-9\-]+", combined_texts.get(t.url,"")) if len(w)>2]
        anchor=" ".join([w for w,_ in Counter(toks).most_common(8)][:4])

        rows.append({"target":t.url,"category":t_cat,"reason":t.reason,"pagerank":t.pagerank,
                     "cheirank":t.cheirank,"backlinks":t.backlinks_refcnt,"inlinks":t.inlinks,
                     "anchor_hint":anchor,"suggestions":"\n".join(cands)})
    return pd.DataFrame(rows)

# ---------- UI ----------
st.set_page_config(page_title="SEO PageRank & CheiRank Analyzer", layout="wide")
st.title("SEO PageRank & CheiRank Analyzer (v13)")

st.markdown("""

**What this tool does:**
- Reads **4 files**: Screaming Frog **Pages** + **All Inlinks**, **Ahrefs Backlinks**, and **GSC**.
- Builds an **internal link graph** (homepage sets domain scope;).
- Calculates **PageRank** (importance) and **CheiRank** (hubness).
- Understands topics from **Full URL Slug + GSC queries + Meta Title + H1**. Default **TF-IDF**; **Embeddings** optional.
- Assigns **categories** to pages from your category list.

**How suggestions are ranked:**
- Score = **35%** semantic similarity + **35%** PageRank + **15%** CheiRank + **15%** Ahrefs  
- Optional: **Link budget** adds `PR / (outlinks + 1)`.

**Outputs:**
- Problem sets: **orphans**, **low PR**, **backlinks but low PR**, **high CheiRank** hubs.
- **Suggestions** per weak page: best source URLs (score, PR, CH, sim, capacity, category, anchor hint).

➡️ **Need more details?** Read the full guide: **[Readme.txt](https://github.com/BupNup/seo-rank-analyzer/blob/main/Readme.txt)**

""")

with st.sidebar:
    st.header("Settings")
    homepage_input = st.text_input("Full homepage URL", value="", placeholder="https://www.example.com/", help="Sets domain scope so links are internal.")
    home_root = canonical_home_root(homepage_input)
    exclude_home_src = st.checkbox("Exclude homepage as a source", value=True)

    cats_text = st.text_area("Main categories (one per line or comma-separated)", value="")
    categories  = [c.strip() for c in re.split(r"[\n,]+", cats_text) if c.strip()]
    prefer_same = st.checkbox("Prefer same-category sources (+15%)", value=True)
    same_only   = st.checkbox("Restrict to same category only", value=False)

    link_scope  = st.selectbox("Which links should count for PR?", ["all","contextual","menu_footer"], index=1)
    keep_query  = st.checkbox("Keep URL query parameters", value=False)
    alpha       = st.slider("PageRank damping (alpha)", 0.50, 0.99, DEFAULT_PAGERANK_ALPHA, 0.01)
    low_pr_q    = st.slider("Low PR threshold (quantile)", 0.05, 0.50, 0.20, 0.05)
    high_ch_q   = st.slider("High CheiRank threshold (top quantile)", 0.05, 0.50, 0.10, 0.05)

    st.markdown("---")
    st.subheader("Semantics engine")
    sem_choice  = st.radio("Choose engine", ["TF-IDF (fast)","Embeddings (better, needs model)"], index=0)
    if sem_choice.startswith("Embeddings") and not _EMBED_READY:
        st.warning("Embeddings packages not available; falling back to TF-IDF unless installed (pip install sentence-transformers).")

    # NEW: component weights
    st.caption("**Weights** for similarity & categories (default = 1.0 each)")
    slug_w  = st.slider("Weight: URL slug", 0.0, 3.0, 1.0, 0.1)
    title_w = st.slider("Weight: Meta Title", 0.0, 3.0, 1.0, 0.1)
    h1_w    = st.slider("Weight: H1",        0.0, 3.0, 1.0, 0.1)
    gsc_w   = st.slider("Weight: GSC queries",0.0, 3.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Link budget")
    use_cap = st.checkbox("Include link budget (PR_norm / (outlinks+1))", value=False)
    cap_w   = st.slider("Link budget weight", 0.0, 0.50, 0.20, 0.01)

    st.markdown("---")
    st.subheader("Ahrefs quality weighting")
    use_quality = st.checkbox("Weight Ahrefs by DR/UR (quality)", value=False, help="Uses DR/UR from Ahrefs export to weight referring sources. Affects PR personalization and the Ahrefs term in suggestion scoring.")

st.caption("Upload CSV or Excel exports. All **four** files are required.")
c1,c2,c3,c4 = st.columns(4)
with c1: pages_file    = st.file_uploader("PAGES export (Screaming Frog HTML)", type=["csv","xlsx","xls"], key="pages")
with c2: inlinks_file  = st.file_uploader("INLINKS export (All Inlinks)",       type=["csv","xlsx","xls"], key="inlinks")
with c3: backlinks_file= st.file_uploader("BACKLINKS export (Ahrefs)",          type=["csv","xlsx","xls"], key="backlinks")
with c4: gsc_file      = st.file_uploader("GSC export (for semantics)",         type=["csv","xlsx","xls"], key="gsc")

if not homepage_input:
    st.warning("Enter your full homepage URL first.")

if pages_file and inlinks_file and backlinks_file and gsc_file and homepage_input:
    try:
        df_pages_raw = read_table(pages_file)
        df_in        = read_table(inlinks_file)
        df_bl        = read_table(backlinks_file)
        df_gsc       = read_gsc(gsc_file)

        G, df_pages, df_pages_non200 = build_internal_graph(df_pages_raw, df_in, keep_query, link_scope, home_root)
        if df_pages_non200.shape[0] > 0:
            st.info(f"Excluded {df_pages_non200.shape[0]} non-200 pages from analysis.")

        bl_strength = aggregate_backlinks(df_bl, keep_query, use_quality=use_quality)
        p_vec = bl_strength[bl_strength.index.isin(G.nodes())]
        personalization = (p_vec / p_vec.sum()).to_dict() if p_vec.sum() > 0 else None

        pr = compute_pagerank(G, alpha=alpha, personalization=personalization)
        ch = compute_cheirank(G, alpha=alpha)

        indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
        nodes = list(G.nodes())
        df = pd.DataFrame({
            "url": nodes,
            "inlinks": [indeg.get(n,0) for n in nodes],
            "outlinks":[outdeg.get(n,0) for n in nodes],
            "pagerank":[pr.get(n,0.0)   for n in nodes],
            "cheirank":[ch.get(n,0.0)   for n in nodes],
            "backlinks_refcnt":[float(bl_strength.get(n,0.0)) for n in nodes],
        })
        df["pagerank_norm"] = df["pagerank"] / (df["pagerank"].sum() if df["pagerank"].sum()>0 else 1.0)
        df["cheirank_norm"] = df["cheirank"] / (df["cheirank"].sum() if df["cheirank"].sum()>0 else 1.0)

        # Semantics: components + combined strings (for anchor hints)
        gsc_sem   = build_gsc_semantics(df_gsc, keep_query=keep_query)
        components= build_component_texts(df_pages, gsc_sem)
        combined_texts = {u: " ".join([components[u]["slug"], components[u]["title"], components[u]["h1"], components[u]["gsc"]]) for u in components.keys()}

        # Categories (weighted)
        if categories:
            cat_map, cat_scores = assign_categories_semantic_weighted(
                df_pages=df_pages,
                categories=categories,
                components=components,
                slug_w=slug_w, title_w=title_w, h1_w=h1_w, gsc_w=gsc_w,
                mode=sem_choice
            )
        else:
            cat_map, cat_scores = ({u:"" for u in df["url"]},{u:0.0 for u in df["url"]})
        df["category"] = df["url"].map(lambda u: cat_map.get(u,"")); df["category_score"] = df["url"].map(lambda u: float(cat_scores.get(u,0.0)))

        # Flags
        df = flag_pages(df, low_pr_q=low_pr_q, high_ch_q=high_ch_q)

        st.success("Analysis complete.")

        tabs = st.tabs([
            "Overview","Low PR candidates","Orphans","Backlinks but low PR",
            "Link hubs (high CheiRank)","All pages (200 only)","Excluded non-200 pages","Menu/Footer links (debug)","Suggestions"
        ])

        with tabs[0]:
            st.subheader("Top pages by PageRank")
            st.dataframe(df.sort_values("pagerank", ascending=False).head(25), use_container_width=True)
            st.subheader("Top pages by CheiRank")
            st.dataframe(df.sort_values("cheirank", ascending=False).head(25), use_container_width=True)

        with tabs[1]:
            th = df["pagerank"].quantile(low_pr_q)
            st.caption(f"Low PR threshold: pagerank ≤ {th:.4g} (quantile {low_pr_q})")
            st.dataframe(df[df["pagerank"]<=th].sort_values("pagerank"), use_container_width=True)

        with tabs[2]:
            st.caption("Pages with zero internal inlinks (200 only)")
            st.dataframe(df[df["inlinks"]==0].sort_values("pagerank", ascending=False), use_container_width=True)

        with tabs[3]:
            th = df["pagerank"].quantile(low_pr_q)
            sel = df[(df["backlinks_refcnt"]>0) & (df["pagerank"]<=th)]
            st.caption("Pages that have external backlinks but low internal PageRank — add internal links.")
            st.dataframe(sel.sort_values(["backlinks_refcnt","pagerank"], ascending=[False,True]), use_container_width=True)

        with tabs[4]:
            th = df["cheirank"].quantile(1 - high_ch_q)
            st.caption(f"High CheiRank threshold: cheirank ≥ {th:.4g} (top {int(high_ch_q*100)}%)")
            st.dataframe(df[df["cheirank"]>=th].sort_values("cheirank", ascending=False), use_container_width=True)

        with tabs[5]:
            st.dataframe(df.sort_values("pagerank", ascending=False), use_container_width=True)

        with tabs[6]:
            if df_pages_non200.shape[0] == 0:
                st.info("No non-200 pages detected in the Pages export.")
            else:
                show_cols = [c for c in ["url_norm","status_code","Title","H1-1","Address","URL"] if c in df_pages_non200.columns]
                st.caption("Excluded from analysis (Status Code != 200).")
                st.dataframe(df_pages_non200[show_cols].sort_values("status_code"), use_container_width=True)

        with tabs[7]:
            st.caption("Header/footer/nav rows (for transparency)")
            pos_col, path_col, elem_col, anch_col = detect_link_position_columns(df_in)
            if pos_col or path_col or elem_col or anch_col:
                classified = df_in.copy()
                classified["_cls"] = classified.apply(lambda r: classify_inlink_row(r, pos_col, path_col, elem_col, anch_col), axis=1)
                st.dataframe(classified[classified["_cls"]=="menu_footer"].head(300), use_container_width=True)
            else:
                st.info("No link position/path columns found in Inlinks to classify.")

        with tabs[8]:
            st.subheader("Internal link suggestions")
            sugg_df = suggest_internal_links(
                df_metrics=df,
                components=components,
                combined_texts=combined_texts,
                category_map=cat_map,
                prefer_same_category=prefer_same,
                same_category_only=same_only,
                homepage_url=homepage_input,
                exclude_homepage_source=exclude_home_src,
                low_pr_q=low_pr_q,
                top_k_sources=5,
                use_link_budget=use_cap,
                cap_weight=cap_w,
                semantic_mode=sem_choice,
                slug_w=slug_w, title_w=title_w, h1_w=h1_w, gsc_w=gsc_w
            )
            if sugg_df.empty:
                st.info("No targets met the criteria for suggestions.")
            else:
                st.markdown("**Compact view (one row per target)**")
                st.dataframe(sugg_df, use_container_width=True)
                st.download_button("Download compact CSV", sugg_df.to_csv(index=False).encode("utf-8"), "internal_link_suggestions_compact.csv", "text/csv")

                st.markdown("**Detailed view (one source per row)**")
                long_df = explode_suggestions(sugg_df)
                st.dataframe(long_df, use_container_width=True, height=min(900, 80 + 28*len(long_df)))
                st.download_button("Download detailed CSV", long_df.to_csv(index=False).encode("utf-8"), "internal_link_suggestions_detailed.csv", "text/csv")

        st.markdown("---")
        st.subheader("Download results")
        st.download_button("Download metrics CSV", df.to_csv(index=False).encode("utf-8"), "seo_rank_results.csv", "text/csv")

        st.markdown("---")
        st.subheader("Quick recommendations")
        orphan_cnt = int((df["inlinks"]==0).sum())
        low_th = df["pagerank"].quantile(low_pr_q); low_cnt = int((df["pagerank"]<=low_th).sum())
        bl_low_cnt = int(((df["backlinks_refcnt"]>0) & (df["pagerank"]<=low_th)).sum())
        hubs_cnt = int((df["cheirank"] >= df["cheirank"].quantile(1 - high_ch_q)).sum())
        rec = pd.DataFrame({
            "Item":["Orphans","Low internal PR","Backlinked but low PR","Link hubs"],
            "Count":[orphan_cnt, low_cnt, bl_low_cnt, hubs_cnt],
            "Action":[
                "Add contextual links from relevant categories/hubs",
                f"Promote with internal links from high-PR & semantically related pages (≤ {low_pr_q:.0%} quantile)",
                "Route equity from topically related hubs and evergreen guides",
                "Audit outlinks; reduce noise; keep contextual anchors"
            ]
        })
        st.dataframe(rec, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    st.info("Upload ALL FOUR files and enter your homepage URL.")
