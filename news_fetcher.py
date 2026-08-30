"""
news_fetcher.py
----------------
Free, open-source news ingestion layer.

- feedparser talks directly to any RSS feed, including Google News RSS
  search URLs (no API key, no wrapper library needed).
- newspaper4k pulls the full article body + metadata from each link,
  since RSS entries usually only carry a headline/summary.

No paid API, no rate-limited key required.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import quote_plus

import feedparser
from newspaper import Article, ArticleException, Config

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

_TAG_RE = re.compile(r"<[^>]+>")

# newspaper4k's default downloader sends no real User-Agent, which a lot of
# publishers (NDTV, Hindustan Times, etc.) reject outright with a 403 as
# basic bot protection. A realistic browser UA + a couple of standard
# headers gets past that in most cases without doing anything deceptive —
# it's the same UA string any Chrome browser sends.
_BROWSER_CONFIG = Config()
_BROWSER_CONFIG.headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
# Must be set AFTER .headers — newspaper4k's headers setter replaces the
# whole dict, so setting browser_user_agent first would get wiped out.
_BROWSER_CONFIG.browser_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_BROWSER_CONFIG.request_timeout = 12


def strip_html(raw: str) -> str:
    """
    RSS 'summary' fields (Google News especially) often contain literal
    HTML markup (<a>, <font>, &nbsp;, etc.) rather than plain text.
    Strip tags and unescape entities so downstream NER/display sees
    clean text instead of markup.
    """
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_via_googlenewsdecoder(link: str, timeout: int) -> tuple[str | None, str | None]:
    """Primary method: googlenewsdecoder's signature/batchexecute resolution."""
    try:
        from googlenewsdecoder import new_decoderv1
    except ImportError:
        return None, "googlenewsdecoder not installed (pip install googlenewsdecoder)"

    try:
        result = new_decoderv1(link, interval=2)
    except Exception as exc:
        return None, f"googlenewsdecoder raised: {exc}"

    if result.get("status") and result.get("decoded_url"):
        return result["decoded_url"], None
    return None, f"googlenewsdecoder returned no URL: {result.get('message', result)}"


def _decode_legacy_base64(link: str) -> tuple[str | None, str | None]:
    """
    Fallback for the OLDER Google News URL format, where the encoded
    article ID directly embeds the destination URL as base64 (no
    network call needed). Only works for links matching that older
    "CBMibG..." style; newer signed IDs will fail here and that's
    expected — the caller moves on to other strategies.
    """
    import base64

    try:
        path = link.split("/articles/")[1].split("?")[0]
        # Google's base64 variant needs standard padding + URL-safe swap.
        padded = path + "=" * (-len(path) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        match = re.search(rb"https?://[^\x00-\x1f\"']+", decoded)
        if match:
            return match.group(0).decode("utf-8", errors="ignore"), None
    except Exception as exc:
        return None, f"legacy base64 decode failed: {exc}"
    return None, "legacy base64 decode found no embedded URL (likely the newer signed format)"


def resolve_real_url(link: str, timeout: int = 10) -> tuple[str, str | None]:
    """
    Google News RSS links point at a news.google.com redirect wrapper —
    NOT a plain HTTP redirect in most cases. Google signs each link and
    resolves the real destination via an internal JS endpoint, so a
    normal requests.get(..., allow_redirects=True) just downloads the
    wrapper page itself (which is why newspaper4k can end up scraping 0
    characters if this step is skipped or fails silently).

    Tries, in order:
      1. googlenewsdecoder (signature + batchexecute resolution — works
         for the current signed URL format, but depends on an
         unofficial internal Google endpoint that can rate-limit or
         change without notice).
      2. Legacy base64 decode (works only for the older unsigned URL
         format some feeds still emit).

    Returns (url, diagnostic). `url` is the resolved link, or the
    original Google link if every strategy failed. `diagnostic` is None
    on success, or a human-readable reason otherwise — surfaced to the
    UI so failures are debuggable instead of a silent fallback.
    """
    if "news.google.com" not in link:
        return link, None

    url, err1 = _decode_via_googlenewsdecoder(link, timeout)
    if url:
        return url, None

    url, err2 = _decode_legacy_base64(link)
    if url:
        return url, None

    return link, f"Google News URL decode failed — {err1}"


MIN_ARTICLE_CHARS = 200  # below this, treat the scrape as "not really the article body"


@dataclass
class NewsItem:
    title: str
    link: str
    published: str = ""
    source: str = ""
    summary: str = ""
    text: str = ""
    authors: list = field(default_factory=list)
    fetch_error: str | None = None
    is_full_text: bool = False  # True only when `text` is the real scraped article body


def google_news_search_url(query: str) -> str:
    """Build a Google News RSS search URL for an arbitrary query — no API key."""
    return GOOGLE_NEWS_RSS.format(query=quote_plus(query))


def fetch_feed_entries(feed_url: str, limit: int = 15) -> list[NewsItem]:
    """Parse any RSS/Atom feed URL into lightweight NewsItem stubs (no full text yet)."""
    parsed = feedparser.parse(feed_url)
    items: list[NewsItem] = []
    for entry in parsed.entries[:limit]:
        items.append(
            NewsItem(
                title=getattr(entry, "title", "").strip(),
                link=getattr(entry, "link", ""),
                published=getattr(entry, "published", ""),
                source=getattr(getattr(entry, "source", None), "title", "") or parsed.feed.get("title", ""),
                summary=strip_html(getattr(entry, "summary", "")),
            )
        )
    return items


def _scrape_article(url: str) -> Article:
    """
    Build and download an Article, using a real browser User-Agent to get
    past basic bot-blocking (403s from sites like NDTV, Hindustan Times).
    Falls back to newspaper4k's default config if the browser-UA request
    itself errors out, in case a particular site reacts badly to specific
    headers rather than the lack of one.
    """
    try:
        article = Article(url, config=_BROWSER_CONFIG)
        article.download()
        if article.download_state == 2:  # 2 = success in newspaper's internal enum
            article.parse()
            return article
    except Exception:
        pass

    # Fallback: bare config, in case the browser headers themselves were the problem.
    article = Article(url)
    article.download()
    article.parse()
    return article


def enrich_with_full_text(item: NewsItem, timeout: int = 10, delay: float = 0.0) -> NewsItem:
    """
    Download + parse the full article body via newspaper4k so NER runs
    against the actual article, not just the RSS headline/summary.

    A scrape that "succeeds" but returns almost nothing (paywall stub,
    cookie-consent page, JS-only shell) is treated the same as a hard
    failure — anything under MIN_ARTICLE_CHARS falls back to the RSS
    summary, and item.is_full_text tells the caller which one it got.
    """
    try:
        real_url, decode_error = resolve_real_url(item.link, timeout=timeout)
        item.link = real_url  # so "Open original article" also points at the real page

        article = _scrape_article(real_url)
        scraped = article.text.strip()

        if len(scraped) >= MIN_ARTICLE_CHARS:
            item.text = scraped
            item.is_full_text = True
        else:
            item.text = item.summary
            item.is_full_text = False
            if decode_error:
                # We never even got a resolved publisher URL — say so plainly
                # instead of the generic "paywall or JS" guess.
                item.fetch_error = f"{decode_error}. Fell back to RSS summary."
            else:
                item.fetch_error = (
                    f"Resolved to {real_url}, but scraped body was too short "
                    f"({len(scraped)} chars) — likely a paywall or JS-rendered "
                    "page; using RSS summary instead."
                )

        item.authors = article.authors or []
        if not item.published and article.publish_date:
            item.published = str(article.publish_date)
    except ArticleException as exc:
        item.fetch_error = str(exc)
        item.text = item.summary
        item.is_full_text = False
    except Exception as exc:  # network hiccups, timeouts, malformed pages
        item.fetch_error = str(exc)
        item.text = item.summary
        item.is_full_text = False
    finally:
        if delay:
            time.sleep(delay)
    return item


def fetch_articles(feed_url: str, limit: int = 15, full_text: bool = True) -> list[NewsItem]:
    """End-to-end: RSS feed -> list of NewsItem, optionally with full article text."""
    items = fetch_feed_entries(feed_url, limit=limit)
    if full_text:
        items = [enrich_with_full_text(it) for it in items]
    return items
