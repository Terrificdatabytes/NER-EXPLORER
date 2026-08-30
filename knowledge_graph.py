"""
knowledge_graph.py
--------------------
Build and render an interactive, human-readable knowledge graph from the
entities already extracted by ner_engine.py.

Nodes  = entities (people, orgs, places, ...), merged across minor
         surface-form variants — "Biden" / "President Biden" / "Joe
         Biden's" all collapse to one node — so the graph reads like a
         map of *things*, not a map of *strings*.
Edges  = sentence-level co-occurrence: two entities mentioned in the
         same sentence somewhere across the fetched articles. This is a
         much cleaner relationship signal than "appeared in the same
         article", so it's the default and only mode exposed in the UI.

Rendered with pyvis (a thin wrapper over vis.js) as a self-contained
HTML fragment that Streamlit embeds via st.components.v1.html — fully
interactive in the browser: drag nodes, scroll to zoom/pan, hover a
node or edge for a tooltip with mention counts and example context.
No extra JS needs to be written or shipped by this app.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

# Titles/honorifics stripped when normalizing an entity's identity, so
# "President Biden" and "Biden" merge into the same node instead of
# showing up as two disconnected people.
_TITLE_PREFIXES = (
    "president", "vice president", "vp", "senator", "sen.", "rep.",
    "representative", "gov.", "governor", "dr.", "dr", "mr.", "mr",
    "mrs.", "ms.", "prof.", "professor", "secretary", "sec.",
    "chancellor", "prime minister", "pm", "ceo", "cfo", "cto",
    "general", "gen.", "colonel", "col.", "judge", "justice",
)

_WS_RE = re.compile(r"\s+")

# Consistent color per spaCy label so the graph's palette lines up with
# the highlighted-text view in the Articles tab.
LABEL_COLORS = {
    "PERSON": "#f5a742", "ORG": "#4f9cf0", "GPE": "#5cbf7a", "LOC": "#3fb0a8",
    "DATE": "#b98cd8", "TIME": "#c9a0e8", "MONEY": "#e8c04a", "PERCENT": "#e8a94a",
    "NORP": "#e86f8c", "FAC": "#9aa5b1", "PRODUCT": "#4fd0e0", "EVENT": "#e85f5f",
    "WORK_OF_ART": "#d88fd8", "LAW": "#8f9fd8", "LANGUAGE": "#7ecbe0",
    "QUANTITY": "#c2c26b", "ORDINAL": "#a8a8a8", "CARDINAL": "#9a9a9a",
}
_DEFAULT_COLOR = "#cccccc"


def _normalize(text: str) -> str:
    """Collapse an entity mention down to a merge key (identity, not display)."""
    t = text.strip().strip("\"'")
    for apos in ("'s", "\u2019s"):
        if t.lower().endswith(apos):
            t = t[: -len(apos)]
    t_low = t.lower()
    for prefix in sorted(_TITLE_PREFIXES, key=len, reverse=True):
        if t_low.startswith(prefix + " "):
            t = t[len(prefix) + 1:]
            t_low = t.lower()
            break
    t = _WS_RE.sub(" ", t).strip()
    return t.lower()


@dataclass
class _NodeAccum:
    label: str
    surface_forms: Counter = field(default_factory=Counter)
    count: int = 0
    articles: set = field(default_factory=set)


def build_graph(
    docs: list[tuple[str, object]],
    labels_filter: set[str] | None = None,
    min_edge_weight: int = 1,
    max_nodes: int = 60,
) -> nx.Graph:
    """
    docs: list of (article_title, spaCy Doc) pairs. Each Doc must already
          have .ents populated (i.e. it came out of the NER pipeline) —
          this function does NOT run NER itself, so building the graph
          costs nothing extra on top of what the app already computed.
    labels_filter: only entities with these labels become nodes/edges;
          None means "all labels".
    min_edge_weight: drop co-occurrence edges seen fewer than this many
          times across all fetched articles' sentences.
    max_nodes: keep only the top-N entities by total mention count, so a
          batch of long articles doesn't turn into an unreadable hairball.
    """
    nodes: dict[str, _NodeAccum] = {}
    edge_weights: Counter = Counter()
    edge_examples: dict[tuple[str, str], str] = {}

    def touch(ent_text: str, ent_label: str, article: str) -> str | None:
        key = _normalize(ent_text)
        if not key:
            return None
        acc = nodes.setdefault(key, _NodeAccum(label=ent_label))
        acc.count += 1
        acc.surface_forms[ent_text] += 1
        acc.articles.add(article)
        return key

    for article_title, doc in docs:
        if doc is None:
            continue
        for sent in doc.sents:
            sent_ents = [e for e in sent.ents if not labels_filter or e.label_ in labels_filter]
            keys = []
            for e in sent_ents:
                k = touch(e.text, e.label_, article_title)
                if k:
                    keys.append(k)
            keys = list(dict.fromkeys(keys))  # de-dupe repeats within the sentence, keep order
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pair = tuple(sorted((keys[i], keys[j])))
                    edge_weights[pair] += 1
                    edge_examples.setdefault(pair, sent.text.strip()[:220])

    # Keep only the most-mentioned entities so the rendered graph stays legible.
    top_keys = {k for k, _ in sorted(nodes.items(), key=lambda kv: kv[1].count, reverse=True)[:max_nodes]}

    G = nx.Graph()
    for key in top_keys:
        acc = nodes[key]
        display = acc.surface_forms.most_common(1)[0][0]
        G.add_node(key, label=acc.label, display=display, count=acc.count, articles=sorted(acc.articles))

    for (a, b), weight in edge_weights.items():
        if a in top_keys and b in top_keys and weight >= min_edge_weight:
            G.add_edge(a, b, weight=weight, example=edge_examples.get((a, b), ""))

    # Drop nodes left with no surviving edges after filtering (unless that
    # would empty the whole graph, e.g. only one entity was ever mentioned).
    isolated = [n for n, d in G.degree() if d == 0]
    if isolated and G.number_of_nodes() > len(isolated):
        G.remove_nodes_from(isolated)

    return G


def render_pyvis_html(G: nx.Graph, height: str = "650px", physics: bool = True) -> str:
    """Render the graph to a self-contained interactive HTML string."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#111318", font_color="#eaeaea", notebook=False)

    if G.number_of_nodes() == 0:
        net.add_node("empty", label="No entities to show — widen your filters", color="#555555")
    else:
        max_count = max((d["count"] for _, d in G.nodes(data=True)), default=1)
        for node_id, data in G.nodes(data=True):
            size = 12 + 28 * (data["count"] / max_count) ** 0.5
            color = LABEL_COLORS.get(data["label"], _DEFAULT_COLOR)
            # Plain text, not HTML: this vis-network build renders tooltip
            # titles as literal text rather than parsing markup, so any
            # <b>/<br> tags here would show up as raw characters.
            title = (
                f"{data['display']}\n"
                f"Type: {data['label']}\n"
                f"Mentions: {data['count']}\n"
                f"Articles: {len(data['articles'])}"
            )
            net.add_node(node_id, label=data["display"], title=title, color=color, size=size)

        max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
        for a, b, data in G.edges(data=True):
            width = 1 + 6 * (data["weight"] / max_w)
            example = data["example"].replace('"', "'")
            title = f"Co-mentioned {data['weight']}x\ne.g. \u201c{example}\u201d"
            net.add_edge(a, b, value=width, title=title, color="#4a4f5a")

    physics_block = (
        """
        "physics": {
          "enabled": true,
          "solver": "barnesHut",
          "barnesHut": {
            "gravitationalConstant": -6000,
            "centralGravity": 0.3,
            "springLength": 140,
            "springConstant": 0.02,
            "damping": 0.5,
            "avoidOverlap": 0.2
          },
          "minVelocity": 0.75,
          "stabilization": {"enabled": true, "iterations": 300, "fit": true}
        }
        """
        if physics
        else '"physics": {"enabled": false}'
    )
    net.set_options(
        """
        {
          "interaction": {"hover": true, "tooltipDelay": 80},
          %s
        }
        """
        % physics_block
    )
    html = net.generate_html(notebook=False)

    # vis-network's default tooltip CSS collapses whitespace, so a plain-text
    # multi-line title (built with \n above) would otherwise render as one
    # long run-on line. pre-line preserves those line breaks without needing
    # any HTML markup in the title itself.
    tooltip_style = (
        "<style>.vis-tooltip{white-space:pre-line !important;"
        "max-width:320px;line-height:1.4;}</style>"
    )
    if "</head>" in html:
        html = html.replace("</head>", tooltip_style + "</head>", 1)
    else:
        html = tooltip_style + html

    return html


def top_relationships(G: nx.Graph, n: int = 15) -> list[dict]:
    """Top-N (entity_a, entity_b, co-mentions) rows for a plain summary table."""
    edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:n]
    return [
        {
            "entity_a": G.nodes[a]["display"],
            "entity_b": G.nodes[b]["display"],
            "co-mentions": data["weight"],
        }
        for a, b, data in edges
    ]
