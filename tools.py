"""
tools.py — Free, keyless web-search tool for the GloryTecks AI Research Crew.

The Researcher agent calls this tool to "Act" on the world: it queries the live
web via DuckDuckGo (through the free `ddgs` package — no API key, no signup) and
returns a clean, readable block of results that the agents can reason over.

This maps to the "Perceive" step in the Anatomy-of-an-Agent loop: the agent
reaches out and senses the real, current world before it reasons.
"""

from __future__ import annotations

import time

# CrewAI's @tool decorator turns a plain Python function into something an agent
# can call on its own during a run.
from crewai.tools import tool

try:
    # `ddgs` is the CURRENT package name (it was formerly `duckduckgo-search`).
    # It needs no API key and no account — perfect for a 100%-free demo.
    from ddgs import DDGS
except Exception:  # pragma: no cover - only hit if the package isn't installed
    DDGS = None


def _run_search(query: str, max_results: int = 5) -> list[dict]:
    """Low-level call to DuckDuckGo. Returns a list of result dicts (may be []).

    Each result dict typically has the keys: 'title', 'href', 'body'.
    """
    if DDGS is None:
        # Turned into a friendly message by the tool wrapper below.
        raise RuntimeError("The 'ddgs' package isn't installed. Run: pip install ddgs")

    # `with DDGS() as ...` makes sure the underlying HTTP session is closed cleanly.
    with DDGS() as ddgs:
        # .text() does a normal web search. We ask for the top few hits only,
        # which keeps the demo fast and the prompt to the LLM small.
        return list(ddgs.text(query, max_results=max_results))


@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web for the given query and return the top results as text.

    Each result includes a title, a short snippet, and its source URL. Use this
    tool whenever you need current, real-world information about a topic.
    """
    last_error = None

    # Try once, then retry a single time. DuckDuckGo will occasionally rate-limit
    # ("throttle") rapid calls; a short pause + one retry fixes most of those.
    for attempt in range(2):
        try:
            results = _run_search(query, max_results=5)
            if results:
                lines = []
                for i, r in enumerate(results, start=1):
                    # Different ddgs versions use slightly different key names, so
                    # we read defensively and fall back gracefully.
                    title = r.get("title") or "Untitled"
                    url = r.get("href") or r.get("url") or ""
                    snippet = r.get("body") or r.get("snippet") or ""
                    lines.append(f"{i}. {title}\n   {snippet}\n   Source: {url}")
                return "\n\n".join(lines)
            # Empty list -> fall through to the retry / no-results message.
        except Exception as e:  # network hiccup, throttling, import error, etc.
            last_error = e

        # Brief back-off before the single retry so we don't hammer the endpoint.
        if attempt == 0:
            time.sleep(1.5)

    # If we get here, both attempts failed OR both returned nothing. We NEVER
    # raise — we return a clear, self-explanatory string so the agent (and the
    # UI) can handle it gracefully instead of crashing with a stack trace.
    if last_error is not None:
        return (
            f"SEARCH_ERROR: the web search could not be completed ({last_error}). "
            "Please try again in a moment."
        )
    return (
        "NO_RESULTS: the web search returned nothing for this query. "
        "Try rephrasing the topic."
    )
