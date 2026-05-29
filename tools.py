"""
tools.py
--------
Tool layer for the Newsletter Agent.

Three tools are exposed to the agent's reasoning loop:

  1. research_news(topic, max_results)  -> list[dict]
       Live web search via Tavily if TAVILY_API_KEY is set,
       otherwise an automatic fallback to free public RSS feeds.

  2. summarize_articles(...)            -> handled by the LLM in agent.py
       (kept as a thin helper here for clarity / reuse)

  3. build_newsletter_html(...)         -> str
       Deterministic HTML generator. Turns the LLM's structured
       newsletter content into a clean, self-contained HTML email.

Everything here is plain Python so it is easy to read and test in isolation.
"""

from __future__ import annotations

import os
import html
import datetime as _dt
from typing import List, Dict


# --------------------------------------------------------------------------- #
#  Tool 1:  Research                                                          #
# --------------------------------------------------------------------------- #

# A small set of high-signal, free AI / AI-agent RSS feeds used as a fallback
# when no Tavily key is available.
_RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://huggingface.co/blog/feed.xml",
    "https://www.marktechpost.com/feed/",
]


def _research_via_tavily(topic: str, max_results: int) -> List[Dict]:
    """Use the Tavily search API. Raises if the package/key is unavailable."""
    from tavily import TavilyClient  # imported lazily so RSS-only users don't need it

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(
        query=topic,
        topic="news",
        max_results=max_results,
        search_depth="advanced",
        days=10,  # keep it "this week"-ish
    )
    articles: List[Dict] = []
    for r in resp.get("results", []):
        articles.append(
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "source": (r.get("url", "").split("/")[2] if r.get("url") else ""),
                "content": r.get("content", ""),
                "published": r.get("published_date", ""),
            }
        )
    return articles


def _research_via_rss(topic: str, max_results: int) -> List[Dict]:
    """Free fallback: pull recent entries from curated AI RSS feeds."""
    import feedparser

    articles: List[Dict] = []
    for feed_url in _RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue
        source = feed_url.split("/")[2]
        for entry in parsed.entries[:5]:
            summary = entry.get("summary", "") or entry.get("description", "")
            # strip the worst of the HTML for a clean snippet
            summary = _strip_tags(summary)[:600]
            articles.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "source": source,
                    "content": summary,
                    "published": entry.get("published", ""),
                }
            )

    # Lightweight relevance ranking: prefer entries mentioning agent keywords.
    keywords = ["agent", "agentic", "autonomous", "llm", "ai", "model", "openai",
                "anthropic", "langchain", "tool", "reasoning"]

    def score(a: Dict) -> int:
        text = (a["title"] + " " + a["content"]).lower()
        return sum(text.count(k) for k in keywords)

    articles.sort(key=score, reverse=True)
    return articles[: max_results * 2]  # return a bit extra; LLM picks the best


def research_news(topic: str, max_results: int = 7) -> List[Dict]:
    """
    Return a list of recent article dicts: {title, url, source, content, published}.

    Tries Tavily first (if TAVILY_API_KEY present), otherwise RSS feeds.
    Always returns a list (possibly empty) and never raises for the caller.
    """
    use_tavily = bool(os.environ.get("TAVILY_API_KEY"))
    if use_tavily:
        try:
            results = _research_via_tavily(topic, max_results)
            if results:
                return results
        except Exception as e:  # pragma: no cover - network/dep issues
            print(f"[tools] Tavily failed ({e}); falling back to RSS.")
    return _research_via_rss(topic, max_results)


# --------------------------------------------------------------------------- #
#  Small HTML helpers                                                          #
# --------------------------------------------------------------------------- #

def _strip_tags(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text or "").strip()


# --------------------------------------------------------------------------- #
#  Tool 3:  HTML newsletter generator                                         #
# --------------------------------------------------------------------------- #

def build_newsletter_html(
    subject: str,
    intro: str,
    items: List[Dict],
    sign_off: str = "— The AI Agent Weekly Team",
) -> str:
    """
    Render a clean, self-contained HTML newsletter.

    `items` is a list of dicts: {title, summary, url, source}
    Returns a full HTML document string.
    """
    today = _dt.date.today().strftime("%B %d, %Y")

    cards = []
    for i, it in enumerate(items, 1):
        title = html.escape(it.get("title", "Untitled"))
        summary = html.escape(it.get("summary", ""))
        url = html.escape(it.get("url", "#"))
        source = html.escape(it.get("source", ""))
        cards.append(
            f"""
        <tr><td style="padding:18px 0;border-bottom:1px solid #eceff3;">
          <div style="font-size:13px;color:#7b8794;letter-spacing:.04em;text-transform:uppercase;">
            Story {i}{f' &middot; {source}' if source else ''}
          </div>
          <a href="{url}" style="text-decoration:none;">
            <h3 style="margin:6px 0 8px;font-size:18px;line-height:1.35;color:#111827;">{title}</h3>
          </a>
          <p style="margin:0 0 10px;font-size:15px;line-height:1.6;color:#374151;">{summary}</p>
          <a href="{url}" style="font-size:14px;color:#4f46e5;text-decoration:none;font-weight:600;">
            Read more &rarr;
          </a>
        </td></tr>"""
        )

    cards_html = "".join(cards)
    intro_html = html.escape(intro)
    subject_html = html.escape(subject)
    sign_off_html = html.escape(sign_off)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{subject_html}</title>
</head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7;padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="640" cellpadding="0" cellspacing="0"
             style="max-width:640px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;
                    box-shadow:0 1px 3px rgba(16,24,40,.08);">
        <!-- header -->
        <tr><td style="background:linear-gradient(135deg,#4f46e5,#7c3aed);padding:32px 32px 26px;">
          <div style="font-size:13px;color:#e0e7ff;letter-spacing:.08em;text-transform:uppercase;">
            AI Agent Weekly &middot; {today}
          </div>
          <h1 style="margin:8px 0 0;font-size:26px;line-height:1.25;color:#ffffff;">{subject_html}</h1>
        </td></tr>
        <!-- intro -->
        <tr><td style="padding:26px 32px 4px;">
          <p style="margin:0;font-size:16px;line-height:1.7;color:#374151;">{intro_html}</p>
        </td></tr>
        <!-- stories -->
        <tr><td style="padding:4px 32px 8px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
            {cards_html}
          </table>
        </td></tr>
        <!-- footer -->
        <tr><td style="padding:22px 32px 30px;">
          <p style="margin:0 0 14px;font-size:15px;color:#374151;">{sign_off_html}</p>
          <p style="margin:0;font-size:12px;color:#9aa5b1;line-height:1.6;">
            You are receiving this because you subscribed to AI Agent Weekly.
            <br>This newsletter was generated autonomously by an AI agent.
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
