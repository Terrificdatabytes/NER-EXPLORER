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

import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from html import unescape
from urllib.parse import quote_plus

import feedparser
from newspaper import Article, ArticleException, Config

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with newspaper4k anyway
    requests = None

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


def _decode_via_googlenewsdecoder(
    link: str, timeout: int, retries: int = 3
) -> tuple[str | None, str | None]:
    """
    Primary method: googlenewsdecoder's signature/batchexecute resolution.

    This hits an unofficial internal Google endpoint, which on shared
    hosting (Streamlit Community Cloud, etc.) gets rate-limited (HTTP 429)
    far more aggressively than it does from a residential IP — especially
    when several articles are decoded back-to-back. A few retries with
    exponential backoff + jitter absorb transient 429s/connection resets
    without needing any extra dependency.

    googlenewsdecoder >=0.2 renamed its entry point from `new_decoderv1`
    (returning {"status", "decoded_url"}) to `gnewsdecoder` (returning
    {"success", "decoded_url"}). Both are tried so this keeps working
    whichever major version ends up installed, instead of silently
    reporting "not installed" for what's actually an API mismatch.
    """
    try:
        from googlenewsdecoder import gnewsdecoder as _decode_fn

        def _call(u):
            r = _decode_fn(u, interval=1)
            return bool(r.get("success")), r.get("decoded_url"), r.get("message", r)
    except ImportError:
        try:
            from googlenewsdecoder import new_decoderv1 as _decode_fn

            def _call(u):
                r = _decode_fn(u, interval=1)
                return bool(r.get("status")), r.get("decoded_url"), r.get("message", r)
        except ImportError as exc:
            # Report the REAL import failure (e.g. an incompatible
            # dependency like selectolax>=1.0 removing the backend
            # googlenewsdecoder's parser still imports) instead of a
            # blanket "not installed" that hides what's actually wrong.
            return None, f"googlenewsdecoder import failed: {exc}"

    last_error = None
    for attempt in range(retries):
        try:
            ok, decoded_url, message = _call(link)
        except Exception as exc:
            last_error = f"googlenewsdecoder raised: {exc}"
        else:
            if ok and decoded_url:
                return decoded_url, None
            last_error = f"googlenewsdecoder returned no URL: {message}"
            # A clean "no URL" response (bad/expired link) won't improve on
            # retry; only back off+retry for exception-y / rate-limit cases.
            if "429" not in str(last_error) and "rate" not in str(last_error).lower():
                break

        if attempt < retries - 1:
            time.sleep((2**attempt) + random.uniform(0, 0.5))

    return None, last_error


def _decode_via_direct_redirect(link: str, timeout: int) -> tuple[str | None, str | None]:
    """
    Fallback: just request the Google News link like a browser and let
    normal HTTP redirects run. This doesn't work for the current signed
    `/rss/articles/...` format (Google resolves those client-side via JS,
    not a 30x), but it costs nothing to try, it works for some older/short
    link shapes, and — unlike googlenewsdecoder — it doesn't depend on an
    internal batchexecute endpoint that cloud-host IP ranges get throttled
    on, so it's a useful independent failure mode.
    """
    if requests is None:
        return None, "requests not available for direct-redirect fallback"

    try:
        resp = requests.get(
            link,
            headers={"User-Agent": _BROWSER_CONFIG.browser_user_agent},
            timeout=timeout,
            allow_redirects=True,
        )
        final_url = resp.url
        if final_url and "news.google.com" not in final_url:
            return final_url, None
        return None, "direct redirect stayed on news.google.com (signed link, needs JS decode)"
    except Exception as exc:
        return None, f"direct redirect failed: {exc}"


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
         change without notice). Retried a few times with backoff since
         a single 429 shouldn't sink the whole article.
      2. Legacy base64 decode (works only for the older unsigned URL
         format some feeds still emit, no network call needed).
      3. Plain HTTP redirect-follow, in case the link shape doesn't need
         JS-side resolution at all.

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

    url, err3 = _decode_via_direct_redirect(link, timeout)
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
    published_dt: datetime | None = None  # parsed publish time, when the feed gives one


def google_news_search_url(
    query: str, start_date: "date | None" = None, end_date: "date | None" = None
) -> str:
    """
    Build a Google News RSS search URL for an arbitrary query — no API key.

    Google News' search supports `after:YYYY-MM-DD` / `before:YYYY-MM-DD`
    operators directly in the query string, so date-wise fetching for this
    source is pushed server-side (Google only returns matching articles)
    rather than filtered after the fact.
    """
    full_query = query
    if start_date:
        full_query += f" after:{start_date.isoformat()}"
    if end_date:
        # before: is exclusive, so bump by one day to make the picked
        # end date inclusive, matching how a person reads a date range.
        from datetime import timedelta

        full_query += f" before:{(end_date + timedelta(days=1)).isoformat()}"
    return GOOGLE_NEWS_RSS.format(query=quote_plus(full_query))


def _parse_entry_datetime(entry) -> datetime | None:
    """Best-effort conversion of a feedparser entry's publish time to a naive datetime."""
    struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not struct:
        return None
    try:
        return datetime(*struct[:6])
    except (TypeError, ValueError):
        return None


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
                published_dt=_parse_entry_datetime(entry),
            )
        )
    return items


def filter_items_by_date(
    items: list[NewsItem], start_date: "date | None", end_date: "date | None"
) -> list[NewsItem]:
    """
    Keep only items published within [start_date, end_date] (inclusive on
    both ends). Used as a client-side filter for feeds that have no
    server-side date query (anything that isn't a Google News search), and
    as a safety net for Google News too, since `after:`/`before:` are a
    best-effort filter on Google's side rather than a guarantee.

    An item with no parseable publish date is kept rather than dropped —
    some publishers' RSS entries omit it, and silently discarding them
    would make "no articles found" indistinguishable from "filter too
    strict".
    """
    if not start_date and not end_date:
        return items

    kept = []
    for item in items:
        if item.published_dt is None:
            kept.append(item)
            continue
        d = item.published_dt.date()
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        kept.append(item)
    return kept


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


def fetch_articles(
    feed_url: str,
    limit: int = 15,
    full_text: bool = True,
    start_date: "date | None" = None,
    end_date: "date | None" = None,
    request_delay: float = 0.4,
) -> list[NewsItem]:
    """
    End-to-end: RSS feed -> list of NewsItem, optionally with full article
    text and filtered to a date range.

    When a date range is set, more raw entries are pulled than `limit`
    before filtering (a feed isn't necessarily date-sorted, and most
    non-Google feeds have no date query of their own) so the person still
    gets up to `limit` matching articles instead of an under-filled page.

    `request_delay` is a small pause between each article's full-text
    fetch. Hammering Google's decode endpoint and publisher sites back to
    back is what tends to trip rate limiting on shared cloud IPs (the
    actual cause of "0 full articles" runs); a few hundred ms between
    requests costs little and meaningfully improves the success rate.
    """
    pool_limit = limit if not (start_date or end_date) else max(limit * 4, 40)
    items = fetch_feed_entries(feed_url, limit=pool_limit)
    items = filter_items_by_date(items, start_date, end_date)
    items = items[:limit]

    if full_text:
        enriched = []
        for idx, it in enumerate(items):
            enriched.append(enrich_with_full_text(it, delay=request_delay if idx > 0 else 0.0))
        items = enriched
    return items
