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

## Deploy for free — Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick this repo/branch, set the main file to `app.py`.
4. Deploy. Streamlit Cloud installs from `requirements.txt` automatically.

Notes:
- The free tier's memory (roughly ~1GB) is why `requirements.txt` defaults to `en_core_web_sm` rather than the transformer model — `en_core_web_trf` + torch generally won't fit or will run very slowly. Don't add `requirements-trf.txt` on this tier.
- Free apps sleep after inactivity and wake on the next visit (a few seconds' delay) — this is normal.
- No secrets/API keys are needed for this app, so there's nothing to configure in Streamlit Cloud's "Secrets" panel.

## Deploy for free — Hugging Face Spaces (alternative)

Hugging Face's free CPU tier policies for Streamlit-type Spaces have shifted over time — check your account's current options before relying on this route. If available: create a Space, choose the Streamlit SDK, push this repo's contents to it (HF Spaces work as their own git remote), and it builds from `requirements.txt` automatically the same way.

## Model choice

- **`en_core_web_sm`** (default) — ~12MB, fast, no `torch` dependency, fits comfortably in free hosting.
- **`en_core_web_trf`** (optional) — ~440MB, RoBERTa-based, more accurate but slow on CPU and RAM-hungry. Use locally or on hosting with more resources; select it via `requirements-trf.txt` + the sidebar dropdown.
