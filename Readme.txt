# SEO PageRank & CheiRank Analyzer

Find low PageRank or orphan pages, then identify the best internal pages to link from using topical similarity, internal authority, hubness, and external signals.

---

## Contents
- [What this tool does](#what-this-tool-does)
- [How scoring works](#how-scoring-works)
  - [Suggestion score](#suggestion-score)
  - [PageRank personalization](#pagerank-personalization)
- [Inputs and column mapping](#inputs-and-column-mapping)
- [Options that change results](#options-that-change-results)
- [Outputs](#outputs)
- [Glossary](#glossary)
- [Quick start](#quick-start)
- [Tips and workflow](#tips-and-workflow)
- [Limits and caveats](#limits-and-caveats)

---

## What this tool does
- Ingests **4 exports**  
  1) Screaming Frog **Pages (HTML)**  
  2) Screaming Frog **All Inlinks**  
  3) **Ahrefs Backlinks**  
  4) **Google Search Console** (GSC) query export  
- Builds an **internal-only link graph**. Your **homepage URL** defines the domain scope; external links are removed.
- **Homepage as a source is configurable:** a checkbox (enabled by default) **excludes the homepage from link suggestions**. Untick it to allow the homepage to appear as a suggested source. *(This does not change the graph itself, only the suggestion filter.)*
- Computes **PageRank** (page importance) and **CheiRank** (hubness). CheiRank = PageRank on the reversed graph.
- Understands topics from **GSC queries + Title + Meta Description + H1 + H2**. Default similarity engine is **TF-IDF**; **Embeddings** are optional.
- Assigns **categories** to each URL by comparing its text (URL path + Title + Description) to your category list.
- Generates **internal linking suggestions** for weak pages: **low PR**, **orphans**, or **“has backlinks but low PR.”**

---

## How scoring works

### Suggestion score
For each weak **target** page, candidate **sources** are ranked by:

score = 0.35 * semantic_similarity(target, source)
0.35 * PR_norm(source)
0.15 * CH_norm(source)
0.15 * Ahrefs_norm(source)
[+ cap_weight * capacity(source) if Link Budget is enabled]

- **semantic_similarity** — TF-IDF by default; **Embeddings** if selected and available.  
- **PR_norm / CH_norm** — normalized PageRank and CheiRank (columns sum to 1).  
- **Ahrefs_norm** — external support for the **source** page from the Ahrefs export (grouped links or unique ref entities).  
- **capacity** — normalized `PR_norm / (outlinks + 1)`; rewards sources with authority **and** fewer outlinks.

**Extra rules**
- **Prefer same-category sources** — **×1.15** multiplier on the score (nudge, not a filter).
- **Restrict to same category only** — hard filter; only same-category sources are eligible.
- **Homepage source toggle** — when **Exclude homepage as a source** is checked (default), the homepage will **not** be suggested; untick to allow it.  
- Pages with **zero outlinks** get a small penalty.

### PageRank personalization
Ahrefs strength is used to **personalize PageRank**, so URLs with more external support receive slightly more probability mass during PR computation.

**Optional quality weighting**
- Toggle **“Weight Ahrefs by DR/UR (quality)”** to multiply each Ahrefs row by DR and/or UR of the referring domain/page.  
  We apply square-root scaling to avoid domination by outliers:  
  `weight_quality = sqrt(DR/100) * sqrt(UR/100)`  
  This affects both **PR personalization** and the **Ahrefs (15%)** component in suggestion scoring.

---

## Inputs and column mapping

The app auto-detects common column names. Typical fields:

- **Pages (HTML)**: `Address` or `URL`, plus `Title`, `Meta Description`, `H1`, optionally `H2-*`.
- **All Inlinks**: `From`/`Source`, `To`/`Destination`/`Target URL`.  
  If available, position fields like `Link Position`, `DOM Path`, `Element`, `Anchor Text` help distinguish **contextual** vs **menu/footer** links.
- **Ahrefs Backlinks**: `Target URL`, and either `Links in group` **or** unique `Referring domain`/`Referring page URL`.  
  Optional quality fields: `DR`, `UR`.
- **GSC**: `Page`, `Query`, `Impressions`.

**URL normalization:** lowercased, fragments removed; query parameters removed by default (toggle available).

---

## Options that change results

- **Semantics engine**
  - **TF-IDF (default)** — fast and reliable.
  - **Embeddings** — better matching if the model is available.
- **Prefer same-category sources** — +15% boost to same-category sources (nudge).
- **Restrict to same category only** — hard filter to same category.
- **Which links count for PR**
  - **contextual** (default) — PR built from in-content links.
  - **all** — include header/footer/navigation.
  - **menu_footer** — only non-content links (debug / sensitivity checks).
- **Exclude homepage as a source** — **on by default**; when on, the homepage is never suggested as a source. Turn **off** to allow homepage suggestions. *(Graph scope is still defined by the homepage URL either way.)*
- **Keep URL query parameters** — off by default. Turn on only if parameters represent indexable content.
- **Link Budget** — off by default. When on, adds `PR_norm / (outlinks + 1)` as a capacity term to the suggestion score (weight configurable).
- **Ahrefs DR/UR weighting** — off by default. When on, the Ahrefs signal is weighted by DR/UR and flows into PR personalization and the 15% Ahrefs term.

---

## Outputs

- **Overview**
  - Top pages by **PageRank**
  - Top pages by **CheiRank**
- **Problem sets**
  - **Low PR** pages (quantile threshold you choose)
  - **Orphans** (zero internal inlinks)
  - **Backlinks but low PR** (externally supported but weak internally)
  - **High CheiRank hubs**
- **Suggestions**
  - **Compact**: one row per target with a packed list of best sources  
  - **Detailed**: one row per source with **score, PR, CH, sim, capacity, category**, and **anchor hint**
- **Quick recommendations**
  - Counts and actions for **orphans**, **low PR**, **backlinks but low PR**, and **hubs**

---

## Glossary

- **inlinks** — internal links pointing **to** the page  
- **outlinks** — internal links going **out** from the page  
- **pagerank (PR)** — internal importance from the site’s link graph  
- **cheirank (CH)** — hub score reflecting useful outlinking  
- **pagerank_norm / cheirank_norm** — PR/CH normalized (columns sum to 1)  
- **Ahrefs strength** — external support aggregated from the Ahrefs export  
- **DR / UR** — Domain Rating / URL Rating from Ahrefs  
- **semantic similarity (sim)** — topical similarity between pages (0–1)  
- **capacity (cap)** — normalized `PR_norm / (outlinks + 1)` when Link Budget is on  
- **category / category_score** — assigned category and confidence  
- **orphans** — pages with zero internal inlinks
