"""Web search and content scraping for MLX LM."""

import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from ddgs import DDGS


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
SEARCH_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
MAX_SCRAPE_BYTES = 200_000
MAX_CLEAN_TEXT = 10_000


def search_web(query: str, num_results: int = 5) -> List[dict]:
    """Search DuckDuckGo and return list of {title, url, snippet}."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract clean text from the main content area of a page."""
    candidates = []

    # Try common main-content selectors
    for sel in ("article", 'main', '[role="main"]', ".mw-parser-output",
                ".post-content", ".entry-content", ".article-body",
                "#article", "#content", ".content", "#mw-content-text"):
        els = soup.select(sel)
        for el in els:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                candidates.append(text)

    if candidates:
        return max(candidates, key=len)

    # Fallback: use body
    body = soup.find("body") or soup
    return body.get_text(separator="\n", strip=True)


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip boilerplate lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip navigation/cookie boilerplate
        if re.match(
            r"^(skip to|jump to|navigation|main menu|search|"
            r"tools|what links here|related changes|special pages|"
            r"permanent link|page information|cite this page|"
            r"wikidata item|download as pdf|print/export|"
            r"cookie statement|privacy policy|about|"
            r"disclaimers|contact us|terms of use|"
            r"the free encyclopedia|this page was last edited)",
            stripped,
            re.I,
        ):
            continue
        # Skip short or empty lines
        if len(stripped) < 3:
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned)


def scrape_url(url: str) -> Optional[str]:
    """Fetch and extract readable text content from a URL."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type:
            return None

        soup = BeautifulSoup(resp.content, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "button", "iframe", "noscript",
                         "svg", "canvas", "audio", "video", "menu"]):
            tag.decompose()

        text = _extract_main_content(soup)
        text = _clean_text(text)

        if len(text.encode("utf-8")) > MAX_CLEAN_TEXT:
            text = text[:MAX_CLEAN_TEXT]

        return text.strip() or None

    except Exception:
        return None
