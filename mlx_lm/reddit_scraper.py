"""Reddit post and comment scraper via old.reddit.com HTML."""

import re
import requests
from bs4 import BeautifulSoup
from typing import Optional

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT = 15


def _soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def _extract_comments(soup: BeautifulSoup, depth: int = 0, max_depth: int = 3) -> list[dict]:
    """Recursively extract comments from a Reddit post page."""
    if depth > max_depth:
        return []
    comments = []
    for entry in soup.find_all("div", class_="entry"):
        # Skip non-comment entries
        if not entry.find("div", class_="usertext-body"):
            continue
        author_el = entry.find("a", class_="author")
        body_el = entry.find("div", class_="md")
        score_el = entry.find("span", class_="score")
        if not body_el:
            continue
        comments.append({
            "author": author_el.text.strip() if author_el else "[deleted]",
            "body": body_el.get_text("\n", strip=True),
            "score": score_el.text.strip() if score_el else "",
            "replies": [],
        })
        # Find child comments (nested <div class="child">)
        child = entry.find_next_sibling("div", class_="child")
        if child:
            comments[-1]["replies"] = _extract_comments(child, depth + 1, max_depth)
    return comments


def scrape_post(url: str, max_comments: int = 50) -> Optional[dict]:
    """Scrape a Reddit post (text + top comments) via old.reddit.com.
    
    Accepts any Reddit URL format and converts to old.reddit.com.
    """
    # Convert any Reddit URL to old.reddit.com format
    m = re.search(r"reddit\.com/r/([^/]+)/comments/([^/]+)", url)
    if not m:
        return None
    subreddit, post_id = m.group(1), m.group(2)
    old_url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}/"

    soup = _soup(old_url)
    if not soup:
        return None

    # Post title
    title_el = soup.find("a", class_="title")
    title = title_el.text.strip() if title_el else ""

    # Post body (selftext)
    selftext_el = soup.find("div", class_="expando")
    selftext = ""
    if selftext_el:
        md = selftext_el.find("div", class_="md")
        if md:
            selftext = md.get_text("\n", strip=True)

    # Post metadata
    score_el = soup.find("span", class_="score")
    author_el = soup.find("a", class_="author")
    domain_el = soup.find("span", class_="domain")

    # Comments
    comment_area = soup.find("div", class_="commentarea")
    comments = []
    if comment_area:
        # Find top-level comment entries (inside the comment listing)
        top_level = comment_area.find("div", class_="sitetable")
        if top_level:
            comments = _extract_comments(top_level, max_depth=3)

    # Flatten comments up to max_comments
    flat = []
    def _flatten(clist, d=0):
        for c in clist:
            flat.append({"author": c["author"], "body": c["body"], "score": c["score"], "depth": d})
            if len(flat) >= max_comments:
                return
            _flatten(c.get("replies", []), d + 1)
    _flatten(comments)

    return {
        "title": title,
        "subreddit": subreddit,
        "post_id": post_id,
        "url": old_url,
        "selftext": selftext,
        "score": score_el.text.strip() if score_el else "",
        "author": author_el.text.strip() if author_el else "",
        "domain": domain_el.text.strip().strip("()") if domain_el else "",
        "comments": flat,
        "comment_count": len(flat),
    }


def format_post_for_llm(post: dict, max_chars: int = 8000) -> str:
    """Format a Reddit post + comments as a readable text block for LLM consumption."""
    parts = []
    parts.append(f"# {post['title']}")
    parts.append(f"r/{post['subreddit']} | by u/{post['author']} | {post['score']}")
    if post["selftext"]:
        parts.append(f"\n{post['selftext']}")
    if post["comments"]:
        parts.append(f"\n--- Comments ({post['comment_count']}) ---")
        for c in post["comments"]:
            indent = "  " * c["depth"]
            parts.append(f"{indent}u/{c['author']} ({c['score']}): {c['body']}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    return text


def search_reddit(query: str, subreddit: str = "", max_results: int = 10) -> list[dict]:
    """Search Reddit via DDGS for posts matching query, optionally in a subreddit.
    
    Returns list of {title, url, snippet} for matching posts.
    """
    from .web_search import search_web
    site = f"site:reddit.com/r/{subreddit}" if subreddit else "site:reddit.com"
    results = search_web(f"{query} {site}", num_results=max_results)
    # Filter to actual Reddit post URLs
    posts = []
    for r in results:
        if re.search(r"reddit\.com/r/[^/]+/comments/", r["url"]):
            posts.append(r)
    return posts
