"""
app.py
-------
Streamlit UI for the open-source NER pipeline.

Run with:  streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import knowledge_graph as kg
from ner_engine import DEFAULT_MODEL, LABEL_DESCRIPTIONS, build_ruler_pattern, load_pipeline, render_doc_html, run_ner
from news_fetcher import NewsItem, fetch_articles, fetch_feed_entries, google_news_search_url, enrich_with_full_text

st.set_page_config(page_title="News NER Explorer", page_icon="", layout="wide")

DEFAULT_FEEDS = {
    "Google News: search a topic": None,  # built dynamically
    "Reuters — World": "http://feeds.reuters.com/Reuters/worldNews",
    "BBC — World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "NYT — Home Page": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Custom RSS URL": None,
}

MODEL_OPTIONS = {
    "Small (en_core_web_sm) — fast, low-RAM, works on free hosting": "en_core_web_sm",
    "Transformer (en_core_web_trf) — more accurate, slower, RAM-heavy": "en_core_web_trf",
}

# ------------------------------------------------------------------ #
# Sidebar: source selection
# ------------------------------------------------------------------ #
st.sidebar.title("🗞️ News NER Explorer")
st.sidebar.caption("100% open-source pipeline: feedparser + newspaper4k + spaCy")

source_choice = st.sidebar.selectbox("News source", list(DEFAULT_FEEDS.keys()))

if source_choice == "Google News: search a topic":
    query = st.sidebar.text_input("Search query", value="artificial intelligence regulation")
    feed_url = google_news_search_url(query) if query else None
elif source_choice == "Custom RSS URL":
    feed_url = st.sidebar.text_input("RSS feed URL", value="")
else:
    feed_url = DEFAULT_FEEDS[source_choice]

num_articles = st.sidebar.slider("Number of articles", 1, 30, 8)
fetch_full_text = st.sidebar.checkbox(
    "Fetch full article text (newspaper4k)", value=True,
    help="On = scrape and run NER over the full article body. Off = NER runs on the short "
         "RSS headline/summary only — leave this on unless you need speed over coverage.",
)

st.sidebar.divider()
model_label = st.sidebar.selectbox("NER model", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[model_label]

st.sidebar.divider()
st.sidebar.subheader("Custom entities (optional)")
st.sidebar.caption("Add domain-specific terms spaCy's base model might miss, e.g. tickers or product names.")
custom_label = st.sidebar.text_input("Label", value="", placeholder="e.g. TICKER")
custom_terms_raw = st.sidebar.text_area("Terms (one per line)", value="", placeholder="NVDA\nOpenAI\nxAI")

custom_patterns = []
if custom_label and custom_terms_raw.strip():
    terms = [t.strip() for t in custom_terms_raw.splitlines() if t.strip()]
    custom_patterns = build_ruler_pattern(custom_label.upper(), terms)

st.sidebar.divider()
all_labels = sorted(LABEL_DESCRIPTIONS.keys()) + ([custom_label.upper()] if custom_label else [])
label_filter = st.sidebar.multiselect(
    "Entity types to show",
    options=all_labels,
    default=["PERSON", "ORG", "GPE", "DATE", "MONEY", "EVENT"] + ([custom_label.upper()] if custom_label else []),
)

run = st.sidebar.button(" Fetch & Analyze", type="primary", width='stretch')

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
st.title("Named Entity Recognition for News Articles")
st.caption("Free RSS ingestion → full-text extraction → spaCy NER, no API keys required.")

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    if not feed_url:
        st.error("Please provide a feed URL or search query in the sidebar.")
    else:
        with st.spinner(f"Fetching up to {num_articles} articles and loading {model_name}..."):
            nlp = load_pipeline(model_name=model_name, custom_patterns=custom_patterns or None)
            items = fetch_articles(feed_url, limit=num_articles, full_text=fetch_full_text)

            all_rows = []
            article_results = []
            for item in items:
                # Always prefer the full scraped article body over the headline/summary.
                text_for_ner = item.text or item.summary or item.title
                ents, doc = run_ner(text_for_ner, nlp)
                article_results.append((item, ents, text_for_ner, doc))
                for e in ents:
                    all_rows.append(
                        {"article": item.title, "text": e.text, "label": e.label, "source": item.source}
                    )

            st.session_state.results = {
                "articles": article_results,
                "table": pd.DataFrame(all_rows),
                "nlp": nlp,
            }

results = st.session_state.results

if results is None:
    st.info("Set a source in the sidebar and click **Fetch & Analyze** to begin.")
else:
    articles = results["articles"]
    df = results["table"]

    n_fetched = len(articles)
    n_full_text = sum(1 for item, _, _, _ in articles if item.is_full_text)
    n_entities = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Articles fetched", n_fetched)
    c2.metric("Full article text extracted", f"{n_full_text}/{n_fetched}")
    c3.metric("Entities extracted", n_entities)

    tab1, tab2, tab3, tab4 = st.tabs([" Articles", " Entity summary", " Knowledge graph", "⬇ Export"])

    with tab1:
        for item, ents, text_for_ner, doc in articles:
            badge = " full article" if item.is_full_text else "⚠️ headline/summary only"
            with st.expander(f"**{item.title}**  —  {item.source or 'unknown source'}  ·  {badge}"):
                st.caption(item.published or "no date")
                if item.link:
                    st.markdown(f"[Open original article]({item.link})")
                if item.fetch_error:
                    st.warning(item.fetch_error[:200])

                html = render_doc_html(doc, labels_filter=set(label_filter) if label_filter else None)
                st.markdown(html, unsafe_allow_html=True)

    with tab2:
        if df.empty:
            st.write("No entities extracted.")
        else:
            filtered = df[df["label"].isin(label_filter)] if label_filter else df
            if filtered.empty:
                st.write("No entities match the selected type filter.")
            else:
                counts = (
                    filtered.groupby(["label", "text"])
                    .size()
                    .reset_index(name="count")
                    .sort_values(["label", "count"], ascending=[True, False])
                )
                st.dataframe(counts, width='stretch', hide_index=True)

                st.subheader("Entity type distribution")
                type_counts = filtered["label"].value_counts()
                st.bar_chart(type_counts)

    with tab3:
        st.caption(
            "Nodes are entities merged across name variants (e.g. \"Biden\" and \"President Biden\" "
            "collapse into one). An edge means two entities were mentioned in the same sentence "
            "somewhere in the fetched articles — that's a much stronger signal than just sharing an "
            "article. **Drag** nodes to rearrange, **scroll** to zoom, **hover** for details."
        )

        docs_for_graph = [(item.title, doc) for item, _, _, doc in articles if doc is not None]

        if not docs_for_graph:
            st.info("No article text was available to build a graph from.")
        else:
            gcol1, gcol2, gcol3 = st.columns(3)
            min_weight = gcol1.slider("Minimum co-mentions per edge", 1, 5, 1)
            max_nodes = gcol2.slider("Max entities shown", 10, 150, 60, step=10)
            physics_on = gcol3.checkbox("Enable drag physics", value=True)

            graph_label_filter = set(label_filter) if label_filter else None
            G = kg.build_graph(
                docs_for_graph,
                labels_filter=graph_label_filter,
                min_edge_weight=min_weight,
                max_nodes=max_nodes,
            )

            gm1, gm2 = st.columns(2)
            gm1.metric("Entities in graph", G.number_of_nodes())
            gm2.metric("Relationships", G.number_of_edges())

            if G.number_of_nodes() == 0:
                st.warning(
                    "No entities survive the current filters — widen the entity type filter in the "
                    "sidebar or lower the co-mention threshold above."
                )
            else:
                html = kg.render_pyvis_html(G, physics=physics_on)
                _iframe_supports_srcdoc = False
                if hasattr(st, "iframe"):
                    try:
                        import inspect
                        _iframe_supports_srcdoc = "srcdoc" in inspect.signature(st.iframe).parameters
                    except (TypeError, ValueError):
                        _iframe_supports_srcdoc = False

                if _iframe_supports_srcdoc:
                    # Newer Streamlit: st.components.v1.html is deprecated in favor of this.
                    st.iframe(srcdoc=html, height=680, scrolling=True)
                else:
                    components.html(html, height=680, scrolling=True)

                rel_rows = kg.top_relationships(G, n=15)
                if rel_rows:
                    st.subheader("Strongest relationships")
                    st.dataframe(pd.DataFrame(rel_rows), width='stretch', hide_index=True)

    with tab4:
        if df.empty:
            st.write("Nothing to export yet.")
        else:
            filtered = df[df["label"].isin(label_filter)] if label_filter else df
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(filtered.to_dict(orient="records"), indent=2).encode("utf-8")

            colA, colB = st.columns(2)
            colA.download_button("Download CSV", csv_bytes, file_name="entities.csv", mime="text/csv",
                                  width='stretch')
            colB.download_button("Download JSON", json_bytes, file_name="entities.json", mime="application/json",
                                  width='stretch')
