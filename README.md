# News NER Explorer

A 100% open-source, no-API-key Named Entity Recognition pipeline for news
articles: RSS ingestion → full-text extraction → spaCy NER → Streamlit UI.

## Architecture

```
RSS / Google News search  →  feedparser        (headlines + links, no key)
        │
        ▼
Article link              →  newspaper4k       (full body text + metadata)
        │
        ▼
Article text               →  spaCy NER          (statistical model)
                              + EntityRuler       (your custom terms/labels)
        │
        ▼
Entities (+ sentences)      →  networkx + pyvis   (interactive knowledge graph)
        │
        ▼
Entities                   →  Streamlit UI       (highlighted text, graph, table,
                                                    CSV/JSON export)
```

| Layer | Library | Why |
|---|---|---|
| News ingestion | `feedparser` | Talks directly to any RSS feed or Google News RSS search URL — zero API key, no rate limit, no wrapper needed |
| Full-text extraction | `newspaper4k` + `googlenewsdecoder` | RSS entries are usually just a headline/snippet. `googlenewsdecoder` resolves Google News' signed redirect wrapper to the real publisher URL (Google no longer does a plain HTTP redirect there), then `newspaper4k` pulls the actual body text — NER runs on the full article, not the headline |
| NER | `spaCy` (`en_core_web_trf`, transformer-based) | RoBERTa-backed statistical NER — more accurate than the small CPU model, worth it for full-length article bodies. `en_core_web_sm` is available as a faster fallback in the sidebar |
| Custom entities | spaCy `EntityRuler` | Add domain terms (tickers, org aliases, watchlist names) the base model won't know, without retraining |
| Zero-shot custom labels (optional) | `GLiNER` | For labels you can't enumerate as a term list (e.g. "sanctioned entity") — heavier dependency, opt-in only |
| Knowledge graph | `networkx` + `pyvis` | Turns extracted entities into an interactive, human-readable graph — nodes are entities (merged across name variants), edges are same-sentence co-occurrence |
| UI | `Streamlit` | Fast interactive app: highlighted entities, knowledge graph, aggregate table, filters, export |

## Setup

```bash
pip install -r requirements.txt

# Transformer English model (default) isn't on PyPI's index — install separately:
python -m spacy download en_core_web_trf
# (or) pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl

# Needs a torch build matching your hardware, e.g. CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: smaller/faster fallback model, selectable in the app's sidebar
python -m spacy download en_core_web_sm
```

> `en_core_web_trf` is noticeably slower and heavier than `en_core_web_sm` (needs torch, more RAM,
> no GPU required but one helps). If you're prototyping on modest hardware, switch to the small
> model via the "NER model" dropdown in the sidebar — the rest of the pipeline is unaffected.

## A note on Google News links

Google News RSS entries don't link straight to the publisher — they link to a signed
`news.google.com/rss/articles/...` wrapper that Google's own JS resolves client-side. A plain
`requests.get(..., allow_redirects=True)` just downloads that wrapper shell (0 characters of
real article text), which is why full-text extraction can silently fail on Google News items
specifically. `resolve_real_url()` in `news_fetcher.py` uses the `googlenewsdecoder` package to
replicate Google's internal resolution step and get the actual publisher URL before handing it
to `newspaper4k`. Non-Google-News links (Reuters, BBC, custom RSS, etc.) are untouched by this
and go straight to `newspaper4k` as before.

If a specific publisher still returns a short/empty scrape after that (paywalls, JS-only pages),
the app falls back to the RSS summary and flags the article as **⚠️ headline/summary only** —
that's expected for sites that block scrapers outright, not a bug.

### Troubleshooting: everything falls back to "headline/summary only"

If *every* article shows the same generic short-scrape warning, the decode step itself is
probably failing rather than individual publishers blocking scrapers. Run:

```bash
python debug_decode.py
```

This calls `googlenewsdecoder` directly (bypassing the Streamlit UI) and prints the raw
error instead of the app's summarized fallback message. Common causes it'll help you spot:

- **Not installed**: `pip install googlenewsdecoder`
- **Corporate proxy/firewall** blocking POST requests to `news.google.com/_/DotsSplashUi/*`
  (common on locked-down office/campus networks)
- **Rate limiting (429)** from resolving many links back-to-back — this unofficial endpoint
  has no published quota, so pulling 20-30 articles at once can trip it
- **Google changed the internal endpoint** — these are unofficial, reverse-engineered
  decoders with no stability guarantee; try `pip install --upgrade googlenewsdecoder` first

Since the app also has a Reuters/BBC/NYT built-in feed option that doesn't go through Google
News at all, those are a good way to confirm the rest of the pipeline (scraping, NER,
transformer model) is working independently of the Google decode step.

## Run

```bash
streamlit run app.py
```

Then, in the sidebar:
1. Pick a source — a built-in feed, a **Google News search** (type any topic,
   no key required), or paste your own RSS URL.
2. Choose how many articles to pull.
3. Optionally add **custom entities**: a label (e.g. `TICKER`) and a list of
   terms (e.g. `NVDA`, `OpenAI`) — these get added to spaCy's pipeline as an
   `EntityRuler` so they're recognized even though the base model has never
   seen them.
4. Click **Fetch & Analyze**.

You'll get four tabs:
- **Articles** — each article with entities highlighted inline
- **Entity summary** — aggregated counts by type, bar chart
- **Knowledge graph** — an interactive graph of how the extracted entities relate
- **Export** — download everything as CSV or JSON

### The knowledge graph tab

Each entity becomes a **node**, merged across minor surface-form variants first —
"Biden", "President Biden", and "Biden's" all collapse into one node instead of
three disconnected ones. Two entities get an **edge** whenever they're mentioned
in the same *sentence* somewhere across the fetched articles; that's a much
cleaner relationship signal than "appeared in the same article", so it's the
only mode this app uses.

- Node size = how often that entity was mentioned; node color = entity type
  (same palette as the highlighted-text view).
- Edge thickness = how many times the pair was co-mentioned; hover an edge to
  see one example sentence.
- Controls above the graph let you raise the minimum co-mention count (to prune
  one-off pairings), cap the number of entities shown (to keep dense articles
  readable), and toggle the drag physics on/off.
- It's fully interactive in the browser — drag nodes apart, scroll to zoom, pan
  to explore — powered by `pyvis`/`vis.js`, no extra setup required.
- A "Strongest relationships" table below the graph lists the top co-mentioned
  pairs in plain text, for when a table is faster to scan than a graph.

## Extending it

- **Zero-shot custom entity types** (labels you can't list as fixed terms,
  e.g. "cyber threat actor"): use `get_gliner_entities()` in `ner_engine.py`.
  Requires `pip install gliner` — kept optional since it pulls in
  torch/transformers.
- **Bigger/more accurate model**: swap `en_core_web_sm` for
  `en_core_web_trf` (transformer-based, slower, more accurate) in
  `load_pipeline()`.
- **Geopolitical entity mapping**: filter to `GPE`/`LOC`/`NORP` in the
  sidebar type filter, then feed the exported CSV into a mapping tool.
- **Scheduled monitoring**: call `news_fetcher.fetch_articles()` on a cron
  job instead of through the UI, and persist results to a database.

## Notes on the "free tier" APIs

Commercial news APIs (NewsData.io, NewsAPI, etc.) work fine but come with
daily credit caps and license restrictions on redistributing article text.
This pipeline avoids that entirely: RSS is public and unlimited, and
`newspaper4k` scrapes the article page you already have permission to view
in a browser. The trade-off is that some publishers block scrapers or sit
behind paywalls — the code degrades gracefully to the RSS summary in that
case rather than failing.
