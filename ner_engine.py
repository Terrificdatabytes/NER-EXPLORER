"""
ner_engine.py
--------------
Core NER engine: spaCy statistical model + a custom EntityRuler layer
for domain-specific entities the base model doesn't know (e.g. ticker
symbols, org aliases, custom watchlist terms).

Default model swapped to en_core_web_sm (CNN-based, ~12MB, no torch
dependency, fast + low-RAM) so the app fits inside free hosting tiers
like Streamlit Community Cloud (~1GB RAM). en_core_web_trf (RoBERTa
transformer, ~440MB, needs spacy-transformers + torch, meaningfully
more accurate but slow/RAM-hungry on CPU) is kept as an optional
selection in the app's model dropdown for local/Colab use where
resources aren't constrained.

A second, optional GLiNER path is included but not imported by default,
since it pulls in its own torch/transformers weights and is meaningfully
heavier still. Enable it only if zero-shot custom labels (e.g.
"sanctioned entity", "product name") are actually needed — see
get_gliner_entities() below.
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy

# Models available to switch between at runtime (e.g. via the app's
# sidebar dropdown). "sm" is the default: light enough for free hosting.
# "trf" is opt-in for environments (local machine, Colab) where accuracy
# matters more than footprint.
AVAILABLE_MODELS = {
    "en_core_web_sm": "Fast, lightweight (~12MB, CPU-friendly, no torch needed)",
    "en_core_web_trf": "Transformer-based, most accurate, slower & memory-heavy (~440MB, needs torch)",
}

DEFAULT_MODEL = "en_core_web_sm"

# Human-readable labels for spaCy's default entity types.
LABEL_DESCRIPTIONS = {
    "PERSON": "People, including fictional",
    "ORG": "Companies, agencies, institutions",
    "GPE": "Countries, cities, states",
    "LOC": "Non-GPE locations (mountains, water bodies)",
    "DATE": "Absolute or relative dates/periods",
    "TIME": "Times smaller than a day",
    "MONEY": "Monetary values",
    "PERCENT": "Percentages",
    "NORP": "Nationalities, religious/political groups",
    "FAC": "Buildings, airports, highways, bridges",
    "PRODUCT": "Objects, vehicles, products",
    "EVENT": "Named hurricanes, battles, wars, sports events",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "LAW": "Named documents made into laws",
    "LANGUAGE": "Any named language",
    "QUANTITY": "Measurements, weight, distance",
    "ORDINAL": "'first', 'second', etc.",
    "CARDINAL": "Numerals that don't fall under another type",
}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


_NLP_CACHE: dict[str, "spacy.language.Language"] = {}


def load_pipeline(model_name: str = DEFAULT_MODEL, custom_patterns: list[dict] | None = None):
    """
    Load (and cache) a spaCy pipeline, optionally adding an EntityRuler
    with custom patterns. The ruler runs BEFORE the statistical NER
    component and is set to not overwrite spans the model is confident
    about, so custom rules extend the model rather than fighting it.

    Works with either the lightweight model (en_core_web_sm, default —
    the one to use for free/hosted deployments) or the transformer
    model (en_core_web_trf — heavier, opt in via the app's dropdown for
    local/Colab use where RAM isn't constrained). Pass model_name
    explicitly to switch.

    Raises a clear error if a requested model isn't installed, rather
    than letting spaCy's OSError bubble up unexplained (useful if a
    hosting environment skipped installing trf to save space).
    """
    cache_key = f"{model_name}:{len(custom_patterns or [])}"
    if cache_key in _NLP_CACHE:
        return _NLP_CACHE[cache_key]

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Model '{model_name}' isn't installed in this environment. "
            f"Available models should be one of: {list(AVAILABLE_MODELS)}. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    # Full news articles can be long; make sure spaCy won't silently
    # truncate/reject them (default max_length is already ~1M chars,
    # but this makes the intent explicit and future-proofs longer text).
    nlp.max_length = max(nlp.max_length, 2_000_000)

    if custom_patterns:
        if "entity_ruler" in nlp.pipe_names:
            ruler = nlp.get_pipe("entity_ruler")
        else:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(custom_patterns)

    _NLP_CACHE[cache_key] = nlp
    return nlp


def run_ner(text: str, nlp) -> tuple[list[Entity], "spacy.tokens.Doc | None"]:
    """
    Run the pipeline once and return both the flat Entity list and the
    underlying spaCy Doc. Callers that need sentence-level structure
    (e.g. knowledge_graph.py's co-occurrence graph) should use this
    instead of extract_entities(), so the model only runs once per
    article rather than once per consumer.
    """
    if not text or not text.strip():
        return [], None
    doc = nlp(text)
    ents = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]
    return ents, doc


def extract_entities(text: str, nlp) -> list[Entity]:
    """Run the pipeline and return a flat list of Entity objects."""
    ents, _ = run_ner(text, nlp)
    return ents


def render_html(text: str, nlp, labels_filter: set[str] | None = None) -> str:
    """
    Return displaCy-rendered inline HTML with entities highlighted.
    If labels_filter is given, only those labels are shown/highlighted
    (others are stripped from the rendered doc, not from the source text).
    """
    from spacy import displacy

    if not text or not text.strip():
        return "<p><em>No text to display.</em></p>"

    doc = nlp(text)
    if labels_filter:
        doc.ents = [e for e in doc.ents if e.label_ in labels_filter]

    return displacy.render(doc, style="ent", jupyter=False)


def render_doc_html(doc, labels_filter: set[str] | None = None) -> str:
    """
    Like render_html, but takes a Doc already produced by run_ner()
    instead of re-running NER on the same text. Filtering copies the
    doc first so the original (which may still be in use for the
    knowledge graph) is left untouched.
    """
    from spacy import displacy

    if doc is None or not doc.text.strip():
        return "<p><em>No text to display.</em></p>"

    if labels_filter:
        doc = doc.copy()
        doc.ents = [e for e in doc.ents if e.label_ in labels_filter]

    return displacy.render(doc, style="ent", jupyter=False)


def build_ruler_pattern(label: str, terms: list[str]) -> list[dict]:
    """
    Helper to turn a flat list of terms into EntityRuler patterns for a
    given label, e.g. build_ruler_pattern("ORG", ["OpenAI", "xAI"]).
    Matches are case-sensitive token patterns; multi-word terms are
    split on whitespace automatically.
    """
    patterns = []
    for term in terms:
        tokens = term.split()
        patterns.append({"label": label, "pattern": [{"LOWER": t.lower()} for t in tokens]})
    return patterns


# ---------------------------------------------------------------------
# Optional zero-shot path (GLiNER). Not imported at module load time —
# call get_gliner_entities() explicitly, and pip install gliner first.
# ---------------------------------------------------------------------

_GLINER_MODEL = None


def get_gliner_entities(text: str, labels: list[str], threshold: float = 0.5) -> list[Entity]:
    """
    Zero-shot NER against arbitrary custom labels the base spaCy model
    doesn't support out of the box (e.g. "sanctioned entity", "cyber
    threat actor", "drug name"). Requires: pip install gliner
    """
    global _GLINER_MODEL
    if _GLINER_MODEL is None:
        from gliner import GLiNER  # local import: optional heavy dependency

        _GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

    raw = _GLINER_MODEL.predict_entities(text, labels, threshold=threshold)
    return [Entity(text=r["text"], label=r["label"], start=r["start"], end=r["end"]) for r in raw]
