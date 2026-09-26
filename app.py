"""
app.py — Streamlit UI + orchestration for the GloryTecks AI Research Crew.

Run it with:   streamlit run app.py

What happens on screen:
  1. The user types a topic (or taps an example chip).
  2. Three agents run in sequence: Researcher -> Analyst -> Writer.
  3. Three status cards move  idle -> running -> done  so the class watches
     the hand-off happen live.
  4. The final markdown report is shown, with a Download button.

Everything runs on free tools: Groq's free LLM API + keyless DuckDuckGo search.
"""

from __future__ import annotations

import os
import queue
import threading

import streamlit as st

from agents import build_llm, build_crew

# Streamlit runs our crew in a background thread so the UI can update live.
# add_script_run_ctx lets that thread play nicely with Streamlit. If the import
# ever moves in a future Streamlit version, we fall back to a harmless no-op.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except Exception:  # pragma: no cover
    def add_script_run_ctx(thread):
        return thread


# ---------------------------------------------------------------------------
# Model choices. Groq's free-tier model list changes fairly often, so these are
# kept in one place — if a name stops working, edit it here (see GUIDE.md).
# As of build time, Groq recommends the GPT-OSS pair over the older Llama pair.
# ---------------------------------------------------------------------------
MODELS = {
    "Fast · GPT-OSS 20B (recommended)": "groq/openai/gpt-oss-20b",
    "Quality · GPT-OSS 120B (slower)": "groq/openai/gpt-oss-120b",
    "Legacy · Llama 3.1 8B Instant": "groq/llama-3.1-8b-instant",
    "Legacy · Llama 3.3 70B Versatile": "groq/llama-3.3-70b-versatile",
    "Optional . qwen3.8-27b": "groq/qwen/qwen3.8-27b",
    "option 2 . gpt-oss-safeguard-20b": "openai/gpt-oss-safeguard-20b",
    "option 3(rec) . llama-prompt-guard-2-86m": "meta-llama/llama-prompt-guard-2-86m",
}

EXAMPLE_TOPICS = [
    "Agentic AI trends 2026",
    "LangGraph vs CrewAI",
    "Groq LPU vs GPU inference",
]

# What each running card should say while it works.
RUNNING_MSG = [
    "Searching the live web…",
    "Reading findings and extracting insights…",
    "Writing the final report…",
]


# ===========================================================================
# Page config + GloryTecks theme (deep near-black green + neon-green accents)
# ===========================================================================
st.set_page_config(
    page_title="GloryTecks AI Research Crew",
    page_icon="🟢",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* ---- Base canvas: near-black deep green ---- */
      .stApp {
        background: radial-gradient(1200px 600px at 20% -10%, #0D1F16 0%, #0A1710 55%, #081109 100%);
        color: #D6EFE0;
      }
      /* Sidebar in a slightly different dark green */
      section[data-testid="stSidebar"] > div {
        background: #0B1B12;
        border-right: 1px solid rgba(34,229,138,0.18);
      }
      /* Headings in bright white/neon */
      h1, h2, h3, h4 { color: #EAFFF4 !important; letter-spacing: .3px; }

      /* ---- Neon-green primary buttons ---- */
      .stButton > button {
        background: linear-gradient(180deg, #22E58A 0%, #2ECC71 100%);
        color: #05130C;
        font-weight: 700;
        border: 0;
        border-radius: 12px;
        padding: .55rem 1.1rem;
        box-shadow: 0 6px 18px rgba(34,229,138,0.25);
        transition: transform .05s ease, box-shadow .2s ease;
      }
      .stButton > button:hover {
        box-shadow: 0 8px 26px rgba(34,229,138,0.45);
        transform: translateY(-1px);
        color: #05130C;
      }

      /* Text input styling */
      .stTextInput input, .stTextInput input:focus {
        background: #10241A;
        color: #EAFFF4;
        border: 1px solid rgba(34,229,138,0.35);
        border-radius: 10px;
      }

      /* ---- Agent status cards ---- */
      .agent-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 8px 0 4px; }
      @media (max-width: 900px) { .agent-grid { grid-template-columns: 1fr; } }

      .agent-card {
        background: #10241A;
        border: 1px solid rgba(34,229,138,0.22);
        border-radius: 16px;
        padding: 16px 16px 14px;
        min-height: 118px;
      }
      .agent-card.running {
        border-color: #22E58A;
        box-shadow: 0 0 0 1px #22E58A, 0 0 26px rgba(34,229,138,0.35);
        animation: glow 1.2s ease-in-out infinite alternate;
      }
      .agent-card.done { border-color: rgba(34,229,138,0.55); }
      @keyframes glow {
        from { box-shadow: 0 0 0 1px #22E58A, 0 0 14px rgba(34,229,138,0.25); }
        to   { box-shadow: 0 0 0 1px #22E58A, 0 0 30px rgba(34,229,138,0.55); }
      }

      .agent-head { display: flex; justify-content: space-between; align-items: center; }
      .agent-name { font-weight: 800; font-size: 1.02rem; color: #EAFFF4; }
      .agent-sub { color: #8FC7AC; font-size: .82rem; margin: 2px 0 8px; }
      .agent-summary { color: #C7E9D7; font-size: .9rem; line-height: 1.35; }

      .agent-badge {
        font-size: .68rem; font-weight: 800; letter-spacing: .5px;
        padding: 3px 9px; border-radius: 999px;
      }
      .agent-badge.idle    { background: #16311F; color: #6FAF8C; }
      .agent-badge.running { background: #22E58A; color: #05130C; }
      .agent-badge.done    { background: #0F3A24; color: #22E58A; border: 1px solid #22E58A; }

      /* ---- GloryTecks logo block ---- */
      .gt-logo { font-size: 1.6rem; font-weight: 900; color: #EAFFF4; line-height: 1; }
      .gt-logo .glow { color: #22E58A; }
      .gt-tag { color: #22E58A; font-weight: 800; letter-spacing: 3px; font-size: .72rem; margin-top: 4px; }

      /* Report container */
      .report-wrap {
        background: #0E211800; border: 1px solid rgba(34,229,138,0.18);
        border-radius: 16px; padding: 6px 20px 12px; margin-top: 8px;
      }

      /* Example chip caption */
      .chip-label { color: #8FC7AC; font-size: .8rem; margin: 2px 0 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Small helpers
# ===========================================================================
def _esc(text: str) -> str:
    """Escape the handful of characters that would break our injected HTML."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _one_line(text: str, limit: int = 150) -> str:
    """Collapse an agent's output into a short one-line summary for its card."""
    flat = " ".join((text or "").split())
    if len(flat) > limit:
        flat = flat[:limit].rstrip() + "…"
    return flat or "Done."


def render_cards(slot, states, summaries):
    """Draw the three agent cards into a placeholder (`slot`).

    states / summaries are 3-item lists. state is 'idle' | 'running' | 'done'.
    Calling this repeatedly on the same slot is how the live animation works.
    """
    names = ["🔎 Researcher", "🧠 Analyst", "✍️ Writer"]
    subs = [
        "Perceive · searches the live web",
        "Reason · extracts key insights",
        "Act · writes the final report",
    ]
    badge_text = {"idle": "IDLE", "running": "RUNNING", "done": "DONE"}

    html = ['<div class="agent-grid">']
    for name, state, summary, sub in zip(names, states, summaries, subs):
        html.append(
            f"""
            <div class="agent-card {state}">
              <div class="agent-head">
                <span class="agent-name">{name}</span>
                <span class="agent-badge {state}">{badge_text[state]}</span>
              </div>
              <div class="agent-sub">{sub}</div>
              <div class="agent-summary">{_esc(summary)}</div>
            </div>"""
        )
    html.append("</div>")
    slot.markdown("".join(html), unsafe_allow_html=True)


# ===========================================================================
# Sidebar — logo, API key, model choice
# ===========================================================================
with st.sidebar:
    st.markdown(
        '<div class="gt-logo">Glory<span class="glow">Tecks</span></div>'
        '<div class="gt-tag">LEARN · BUILD · GLOW</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### 🔑 Groq API key")
    api_key = st.text_input(
        "Paste your free Groq key",
        type="password",
        value=st.session_state.get("api_key", os.environ.get("GROQ_API_KEY", "")),
        help="Free, no credit card. Grab one below — it starts with 'gsk_'.",
        label_visibility="collapsed",
        placeholder="gsk_...",
    )
    st.session_state["api_key"] = api_key
    st.caption("Get a free key → [console.groq.com/keys](https://console.groq.com/keys)")

    st.markdown("#### ⚙️ Model")
    model_label = st.radio(
        "Model quality",
        list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    model_id = MODELS[model_label]
    st.caption(f"Using `{model_id}`")

    st.markdown("---")
    st.caption(
        "Perceive → **search** · Reason → **analyze** · Act → **write**. "
        "Three agents, one free stack."
    )


# ===========================================================================
# Main panel — header, topic input, example chips, Run button
# ===========================================================================
st.markdown(
    '<div class="gt-logo" style="font-size:2.1rem;">Glory<span class="glow">Tecks</span> '
    'AI Research Crew</div>'
    '<div class="agent-sub" style="font-size:.95rem;margin-top:6px;">'
    "A live multi-agent demo: three AI agents research any topic on the open web, "
    "analyze it, and write you a report — on a 100% free stack.</div>",
    unsafe_allow_html=True,
)
st.write("")

# Keep the topic in session state so example chips can fill it in.
st.session_state.setdefault("topic", "")

# --- Example chips (safe live-demo fallbacks). Handled BEFORE the text box so
#     a click can pre-fill it cleanly on the next rerun. ---
st.markdown('<div class="chip-label">Try an example:</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLE_TOPICS))
for col, example in zip(chip_cols, EXAMPLE_TOPICS):
    if col.button(example, key=f"chip_{example}", use_container_width=True):
        st.session_state["topic"] = example
        st.rerun()

topic = st.text_input(
    "Research topic",
    key="topic",
    placeholder="e.g. latest trends in agentic AI",
)

run_clicked = st.button("🚀 Run the Crew", type="primary")

# A single placeholder that the three status cards are drawn into.
cards_slot = st.empty()
report_slot = st.container()


def run_crew_live(topic: str, model_id: str, api_key: str):
    """Build the crew, run it in a background thread, and animate the cards.

    Returns the final markdown report (str) or raises nothing — errors are
    returned as a special ('__error__', message) tuple for the caller to show.
    """
    # A thread-safe queue the background thread pushes task completions onto.
    event_q: "queue.Queue[str]" = queue.Queue()
    holder: dict = {}

    # This callback runs INSIDE the worker thread, so it must NOT touch Streamlit.
    # It only drops the finished task's text onto the queue; the main thread reads
    # it and updates the UI. That separation is what keeps Streamlit happy.
    def on_task_done(task_output):
        try:
            text = getattr(task_output, "raw", None) or str(task_output)
        except Exception:
            text = ""
        event_q.put(text)

    llm = build_llm(model_id, api_key)
    crew = build_crew(llm, task_callback=on_task_done)

    def worker():
        try:
            result = crew.kickoff(inputs={"topic": topic})
            holder["result"] = getattr(result, "raw", None) or str(result)
        except Exception as e:  # captured and surfaced as a friendly banner
            holder["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    add_script_run_ctx(thread)  # let the thread cooperate with Streamlit
    thread.start()

    # Initial state: Researcher running, others waiting.
    states = ["running", "idle", "idle"]
    summaries = [RUNNING_MSG[0], "Waiting…", "Waiting…"]
    render_cards(cards_slot, states, summaries)

    # Live loop: each time a task finishes, mark it done and start the next one.
    done = 0
    while thread.is_alive() or not event_q.empty():
        try:
            text = event_q.get(timeout=0.25)
        except queue.Empty:
            continue
        summaries[done] = _one_line(text)
        states[done] = "done"
        done += 1
        if done < 3:
            states[done] = "running"
            summaries[done] = RUNNING_MSG[done]
        render_cards(cards_slot, states, summaries)

    thread.join()

    if "error" in holder:
        return ("__error__", str(holder["error"]))
    return holder.get("result", "")


# ===========================================================================
# Orchestration
# ===========================================================================
if run_clicked:
    # --- Friendly validation, never a raw crash ---
    if not api_key or not api_key.strip():
        st.error("🔑 Please paste your free Groq API key in the sidebar first. "
                 "Grab one at console.groq.com/keys — no credit card needed.")
    elif not topic or not topic.strip():
        st.warning("✏️ Type a topic (or tap an example) before running the crew.")
    else:
        try:
            with st.spinner("The crew is on it… (usually 30-45 seconds)"):
                report = run_crew_live(topic.strip(), model_id, api_key.strip())

            # Handle the error tuple returned by run_crew_live.
            if isinstance(report, tuple) and report and report[0] == "__error__":
                msg = report[1].lower()
                if "auth" in msg or "api key" in msg or "401" in msg or "invalid" in msg:
                    st.error("🔑 Groq rejected that API key. Double-check it "
                             "(it starts with 'gsk_') at console.groq.com/keys.")
                elif "rate" in msg or "429" in msg or "quota" in msg:
                    st.error("⏳ Hit Groq's free rate limit. Wait a minute, or "
                             "switch to the Fast model in the sidebar, then retry.")
                else:
                    st.error("Something went wrong during the run:\n\n"
                             f"`{report[1]}`\n\nTry again, or switch models in the sidebar.")
            elif not report or not str(report).strip():
                st.warning("The crew finished but produced an empty report. "
                           "Try a different topic or the Quality model.")
            else:
                # Success — stash it so the page can re-show it after reruns
                # (e.g. when the Download button is pressed).
                st.session_state["report"] = report
                st.session_state["report_topic"] = topic.strip()
        except Exception as e:  # last-resort catch-all — still no stack trace on screen
            st.error(f"Unexpected error: `{e}`. Please try again.")

    # Render whatever we produced this run.
    if st.session_state.get("report"):
        with report_slot:
            st.markdown("### 📄 Final report")
            st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
            st.markdown(st.session_state["report"])
            st.markdown("</div>", unsafe_allow_html=True)
            fname = (st.session_state.get("report_topic", "report")
                     .lower().replace(" ", "-")[:40] or "report")
            st.download_button(
                "⬇️ Download report (.md)",
                data=st.session_state["report"],
                file_name=f"glorytecks-{fname}.md",
                mime="text/markdown",
            )

# On a plain rerun (e.g. after a download click), keep showing the last result
# and leave the cards in their finished state so nothing flickers away.
elif st.session_state.get("report"):
    render_cards(
        cards_slot,
        ["done", "done", "done"],
        ["Gathered live findings.", "Extracted key insights.", "Report ready."],
    )
    with report_slot:
        st.markdown("### 📄 Final report")
        st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
        st.markdown(st.session_state["report"])
        st.markdown("</div>", unsafe_allow_html=True)
        fname = (st.session_state.get("report_topic", "report")
                 .lower().replace(" ", "-")[:40] or "report")
        st.download_button(
            "⬇️ Download report (.md)",
            data=st.session_state["report"],
            file_name=f"glorytecks-{fname}.md",
            mime="text/markdown",
        )
else:
    # First load: show the three idle cards so the layout is visible.
    render_cards(
        cards_slot,
        ["idle", "idle", "idle"],
        ["Waiting for a topic…", "Waiting…", "Waiting…"],
    )
