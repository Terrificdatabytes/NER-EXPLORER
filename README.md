# News NER Explorer

Free, open-source pipeline: RSS ingestion (feedparser) → full-article extraction (newspaper4k) → Named Entity Recognition (spaCy) → interactive knowledge graph (pyvis). No paid APIs, no API keys.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — entry point, run with `streamlit run app.py` |
| `news_fetcher.py` | RSS parsing + full-article scraping (Google News link decoding, bot-block workarounds) |
| `ner_engine.py` | spaCy NER pipeline + custom EntityRuler support |
| `knowledge_graph.py` | Builds the entity co-occurrence graph and renders it with pyvis |
| `requirements.txt` | Core dependencies, including `en_core_web_sm` — light enough for free hosting |
| `requirements-trf.txt` | **Optional** extra deps for the more accurate but heavier `en_core_web_trf` transformer model |

## Run locally

```bash
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

- **`en_core_web_sm`** (default) — ~12MB, fast, no `torch` dependency, fits comfortably in free hosting.
- **`en_core_web_trf`** (optional) — ~440MB, RoBERTa-based, more accurate but slow on CPU and RAM-hungry. Use locally or on hosting with more resources; select it via `requirements-trf.txt` + the sidebar dropdown.
