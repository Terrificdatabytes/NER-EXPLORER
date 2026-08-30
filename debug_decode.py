"""
debug_decode.py
-----------------
Standalone diagnostic for Google News link decoding — run this directly
(not through Streamlit) to see the RAW error from googlenewsdecoder,
since the app normally shows a summarized version of this.

Usage:
    py debug_decode.py "https://news.google.com/rss/articles/CBMi...."

Or with no argument, it fetches one live link from a Google News search
and tries to decode it.
"""

import sys


def main():
    if len(sys.argv) > 1:
        link = sys.argv[1]
    else:
        import feedparser
        from news_fetcher import google_news_search_url

        print("No URL given — pulling one live link from Google News search...")
        feed = feedparser.parse(google_news_search_url("technology"))
        if not feed.entries:
            print("Could not fetch any entries. Check your internet connection.")
            return
        link = feed.entries[0].link
        print(f"Using: {feed.entries[0].title}\n{link}\n")

    print("=" * 70)
    print("1) Checking googlenewsdecoder package...")
    try:
        import googlenewsdecoder
        print(f"   Installed OK (version: {getattr(googlenewsdecoder, '__version__', 'unknown')})")
    except ImportError as exc:
        print(f"   NOT INSTALLED: {exc}")
        print("   Fix: pip install googlenewsdecoder")
        return

    print("=" * 70)
    print("2) Calling new_decoderv1() directly...")
    from googlenewsdecoder import new_decoderv1
    try:
        result = new_decoderv1(link, interval=2)
        print(f"   Raw result: {result}")
        if result.get("status"):
            print(f"\n   SUCCESS -> {result.get('decoded_url')}")
        else:
            print(f"\n   FAILED -> {result.get('message')}")
    except Exception as exc:
        print(f"   EXCEPTION: {type(exc).__name__}: {exc}")
        print("\n   Common causes:")
        print("   - Corporate proxy/firewall blocking POST to news.google.com/_/DotsSplashUi/*")
        print("   - Rate limiting (429) from fetching too many links too fast")
        print("   - Google changed the internal endpoint again (check for a package update:")
        print("       pip install --upgrade googlenewsdecoder)")

    print("=" * 70)
    print("3) Testing full resolve_real_url() (includes legacy-format fallback)...")
    from news_fetcher import resolve_real_url
    real_url, diagnostic = resolve_real_url(link)
    print(f"   Resolved URL: {real_url}")
    print(f"   Diagnostic:   {diagnostic}")


if __name__ == "__main__":
    main()
