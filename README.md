# News NER Explorer

**Turn any RSS news feed into an interactive knowledge graph of people, organizations, and places — 100% open-source, no paid APIs, no keys.**

Point it at a topic or a feed and it pulls real articles (not just headlines), runs Named Entity Recognition over the full text, merges entity variants into single identities, and renders an interactive graph of who and what is connected — all in your browser, with CSV/JSON export.

---

## What makes this different from a typical "spaCy demo"

A lot of NER projects stop at "highlight entities in a text box." This one builds a **complete pipeline**, and each stage solves a real problem the naive version runs into:

**1. Ingestion → Scraping**
RSS feeds (including Google News search) only give you a headline and a one-line summary — not the article. Google News links are also redirect wrappers, not real URLs, and resolving them requires decoding a signed internal payload (not a normal HTTP redirect). This project:
- Resolves Google News redirect links via signature decoding, with a fallback to legacy base64 decoding for older link formats
- Scrapes the real article body with a realistic browser User-Agent to get past basic bot-blocking that a lot of publishers apply to unidentified scrapers
- Detects "successful but useless" scrapes (paywalls, cookie walls, JS-only shells) by a minimum character threshold, and falls back to the RSS summary automatically — the UI tells you exactly which one you got, and why, per article

**2. NER**
Runs spaCy's statistical NER pipeline over the *actual* article text, with a runtime-swappable model: a lightweight CNN model by default (fits comfortably on free hosting), or an optional transformer model for higher accuracy when you have the RAM. You can also layer in your own custom entity types (tickers, product names, watchlist terms) via a rule-based ruler that extends the model instead of overriding it.

**3. Co-occurrence graph**
Most "entity graph" demos connect anything that appeared in the same article — a weak, noisy signal. This one:
- Connects entities only when they're mentioned in the **same sentence**, a meaningfully stronger relationship signal
- **Normalizes entity identity**, not just entity text — "Biden," "President Biden," and "Biden's" all collapse into a single node instead of fragmenting the graph into duplicates
- Renders as a fully interactive graph (drag, zoom, hover for mention counts and context) with adjustable thresholds for minimum co-mentions and max nodes shown

**4. Export**
One-click CSV/JSON export of every extracted entity, filtered to whatever entity types you've selected — ready to drop into a spreadsheet or another pipeline.

## Features

- Fetch from curated feeds (Reuters, BBC, NYT) or any custom RSS URL, or search Google News by topic
- Full-article scraping with automatic paywall/bot-block detection and graceful fallback
- Runtime model switch: `en_core_web_sm` (fast, free-hosting friendly) or `en_core_web_trf` (transformer, more accurate)
- Custom entity types via a rule-based EntityRuler layer
- Optional zero-shot NER path (GLiNER) for arbitrary custom labels, kept out of the default install so it doesn't bloat lightweight deployments
- Per-article highlighted-entity view, aggregate entity-type breakdown with charts, and an interactive knowledge graph
- CSV/JSON export

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Ingestion | feedparser (RSS/Atom) |
| Scraping | newspaper4k + custom Google News redirect resolution |
| NER | spaCy (`en_core_web_sm` / `en_core_web_trf`), optional GLiNER for zero-shot |
| Graph | networkx + pyvis (vis.js) |
| Data handling | pandas |

## Project layout

```
news-ner-explorer/
├── app.py                Streamlit UI — entry point
├── news_fetcher.py         RSS parsing, Google News link decoding, full-article scraping
├── ner_engine.py             spaCy NER pipeline + custom EntityRuler + optional GLiNER
├── knowledge_graph.py          Entity normalization, co-occurrence graph, pyvis rendering
├── requirements.txt               Core deps (includes en_core_web_sm)
└── requirements-trf.txt             Optional: transformer model deps
```

## Getting started

```bash
git clone <your-repo-url>
cd news-ner-explorer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

Want the more accurate transformer model too?
```bash
pip install -r requirements-trf.txt
```
Then pick "Transformer (en_core_web_trf)" in the app's sidebar model dropdown.

## Model choice

- **`en_core_web_sm`** (default) — ~12MB, fast, no `torch` dependency, fits comfortably on free hosting (e.g. Streamlit Community Cloud's ~1GB tier).
- **`en_core_web_trf`** (optional) — ~440MB, RoBERTa-based, more accurate but slow and RAM-hungry on CPU. Use locally, on Colab, or on a paid tier with more resources.

## Limitations

This is a prototype built on scraping and open-source models, so it inherits their honest limits:
- Scraping is best-effort: heavily paywalled or JS-rendered sites will fall back to the RSS summary rather than the full article (the UI flags this per article).
- Google News link decoding depends on an unofficial internal Google endpoint that can change or rate-limit without notice; a legacy fallback covers older link formats only.
- `en_core_web_sm` trades accuracy for speed/size — expect more mislabeled or missed entities than the transformer model, especially on ambiguous names.

## License

MIT
