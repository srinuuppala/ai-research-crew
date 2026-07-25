"""
agents.py — The three-agent CrewAI pipeline for GloryTecks.

This file is the heart of the demo. It maps directly onto the
"Anatomy of an Agent" loop from the slides:

    Perceive  ->  Researcher searches the live web (via the web_search tool)
    Reason    ->  Analyst extracts the key insights from the raw findings
    Act       ->  Writer produces the final, polished markdown report

The three agents run one after another (a sequential Crew), and each hands its
output to the next — exactly the hand-off the class watches on screen.
"""

from __future__ import annotations

import os


# ===========================================================================
# COMPATIBILITY PATCH — must run BEFORE CrewAI is imported.
# ---------------------------------------------------------------------------
# Known CrewAI bug (issue #5886): CrewAI adds a 'cache_breakpoint' marker to
# messages for prompt caching, but only strips it for Anthropic models — not
# for Groq. Groq then rejects the request with:
#     'cache_breakpoint' is unsupported ... invalid_request_error
# We strip that field ourselves at the litellm chokepoint (which every Groq
# call flows through), so the crew runs cleanly. This is a no-op once applied.
# ===========================================================================
def _apply_groq_cache_patch() -> None:
    try:
        import litellm
    except Exception:
        return  # litellm not installed yet; nothing to patch

    if getattr(litellm, "_glorytecks_patched", False):
        return

    def _strip(messages):
        """Remove the unsupported 'cache_breakpoint' key from each message."""
        if not messages:
            return messages
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and "cache_breakpoint" in m:
                m = {k: v for k, v in m.items() if k != "cache_breakpoint"}
            cleaned.append(m)
        return cleaned

    for name in ("completion", "acompletion"):
        orig = getattr(litellm, name, None)
        if orig is None:
            continue

        def make_wrapper(orig_fn):
            def wrapper(*args, **kwargs):
                # messages usually arrive as a keyword arg...
                if kwargs.get("messages"):
                    kwargs["messages"] = _strip(kwargs["messages"])
                # ...but handle the positional case too, just in case.
                elif len(args) >= 2 and isinstance(args[1], list):
                    args = list(args)
                    args[1] = _strip(args[1])
                return orig_fn(*args, **kwargs)
            return wrapper

        setattr(litellm, name, make_wrapper(orig))

    litellm._glorytecks_patched = True


_apply_groq_cache_patch()  # <-- runs now, before the crewai import below


# Core CrewAI building blocks (imported AFTER the patch is applied).
from crewai import Agent, Task, Crew, Process, LLM

# Our free, keyless search tool (defined in tools.py).
from tools import web_search


def build_llm(model_id: str, api_key: str) -> LLM:
    """Create a CrewAI LLM backed by Groq's free API.

    `model_id` is a litellm-style string such as:
        'groq/openai/gpt-oss-20b'        (fast)
        'groq/openai/gpt-oss-120b'       (higher quality, slower)
        'groq/llama-3.1-8b-instant'      (legacy fast)
        'groq/llama-3.3-70b-versatile'   (legacy quality)

    The 'groq/' prefix tells CrewAI (via litellm) to route the call to Groq.
    We both pass the key explicitly AND export it to the environment, because
    different layers of the stack look for it in different places.
    """
    _apply_groq_cache_patch()  # belt-and-suspenders: safe to call again
    os.environ["GROQ_API_KEY"] = api_key  # litellm/groq reads this env var
    return LLM(
        model=model_id,
        api_key=api_key,
        temperature=0.4,   # mostly grounded, a touch of flexibility
        max_tokens=1200,   # plenty for a short report; keeps the demo fast
    )


def build_agents(llm: LLM):
    """Define the three agents. Each has a role, a goal, and a backstory that
    shapes how it behaves. Only the Researcher gets the web_search tool."""

    researcher = Agent(
        role="Web Researcher",
        goal="Find current, credible, source-backed information about: {topic}",
        backstory=(
            "A relentless digital investigator who scours the live web and always "
            "keeps the source URL for every claim. Never makes facts up."
        ),
        tools=[web_search],       # <- the only agent that can touch the web
        llm=llm,
        verbose=True,             # prints its thinking to the terminal (great for teaching)
        allow_delegation=False,   # keep the pipeline simple and predictable
    )

    analyst = Agent(
        role="Insight Analyst",
        goal="Distil raw research into the most important, non-obvious insights",
        backstory=(
            "A sharp analyst who cuts through noise and surfaces the 4-6 insights "
            "that actually matter, ignoring filler and stating things plainly."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Report Writer",
        goal="Turn insights into a clean, well-structured markdown report",
        backstory=(
            "A crisp technical writer who produces skimmable reports with a "
            "headline, a short summary, key points, and a list of sources."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, analyst, writer


def build_crew(llm: LLM, task_callback=None) -> Crew:
    """Wire the agents and their tasks into a sequential Crew.

    `task_callback` (optional) is called once after EACH task finishes. The
    Streamlit app uses it to light up the Researcher -> Analyst -> Writer status
    cards live, so the audience sees the hand-off happen in real time.

    The literal `{topic}` in the task descriptions is filled in at run time by
    `crew.kickoff(inputs={"topic": ...})`.
    """
    researcher, analyst, writer = build_agents(llm)

    # 1) PERCEIVE — gather raw findings from the live web.
    research_task = Task(
        description=(
            "Research the topic: '{topic}'. Use the web_search tool to gather "
            "current information. Collect 5-8 concrete findings, each with its "
            "source URL. If the search returns an error or nothing, say so clearly "
            "instead of inventing facts."
        ),
        expected_output=(
            "A bulleted list of 5-8 findings about the topic, each 1-2 sentences, "
            "with the source URL in parentheses at the end of each bullet."
        ),
        agent=researcher,
    )

    # 2) REASON — turn raw findings into a few sharp insights.
    analysis_task = Task(
        description=(
            "Read the researcher's findings and extract the 4-6 MOST important, "
            "non-obvious insights about '{topic}'. Prefer insights that connect the "
            "dots or reveal a trend over simply restating a single fact."
        ),
        expected_output=(
            "A numbered list of 4-6 key insights, each 1-2 sentences, ordered by "
            "importance."
        ),
        agent=analyst,
        context=[research_task],  # <- receives the researcher's output
    )

    # 3) ACT — write the final, presentable report.
    writing_task = Task(
        description=(
            "Using the analyst's insights, write a polished markdown report on "
            "'{topic}'. Structure it as: a '# ' headline, a 2-3 sentence "
            "**Summary**, a '## Key Points' section (bulleted), and a "
            "'## Sources' section listing the URLs gathered during research. "
            "Keep it tight and skimmable."
        ),
        expected_output=(
            "A complete markdown report with a headline, summary, key points, and a "
            "sources list."
        ),
        agent=writer,
        context=[analysis_task, research_task],  # <- sees insights AND raw sources
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,   # run strictly in order: 1 -> 2 -> 3
        task_callback=task_callback,  # fires after each task (drives the UI cards)
        verbose=True,
    )