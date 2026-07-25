# GloryTecks AI Research Crew 🟢

A **live-demo-ready multi-agent AI app** for a classroom. A user types any topic;
three AI agents research it on the open web, analyze it, and write a polished
report — running on a **100% free stack** (Groq's free LLM API + keyless
DuckDuckGo search + Streamlit).

> **Anatomy of an Agent, on screen:**
> **Perceive → search** (Researcher) · **Reason → analyze** (Analyst) · **Act → write** (Writer)

---

## ⚡ 60-second setup

1. **Get a free Groq key** (no credit card): <https://console.groq.com/keys> → copy the `gsk_...` key.
2. **Install** (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run:**
   ```bash
   streamlit run app.py
   ```
4. In the browser tab that opens, paste your key in the sidebar, type a topic
   (or tap an example chip), and hit **🚀 Run the Crew**.

That's it — no other signup, no other key.

---

## 🎓 What the trainer says while it runs

> "Watch the three cards. The **Researcher** *perceives* — it searches the live
> web. It hands off to the **Analyst**, which *reasons* — pulling out the
> insights that matter. That goes to the **Writer**, which *acts* — producing the
> report you can download. That's the exact Perceive → Reason → Act loop from the
> slide, running for real."

---

## 📁 Files

| File | What it does |
|------|--------------|
| `app.py` | Streamlit UI, theme, and live orchestration |
| `agents.py` | The 3 CrewAI agents + tasks + sequential crew |
| `tools.py` | The free, keyless web-search tool (`ddgs`) |
| `requirements.txt` | Dependencies (all free) |
| `.env.example` | Optional: put your key in `.env` instead of the sidebar |
| `GUIDE.md` | **Full walkthrough** — setup, demo script, how the code works, troubleshooting |

**New to this? Read `GUIDE.md` — it explains everything from zero.**

---

## ⚠️ Note on models

Groq's free model list changes often. The app defaults to **GPT-OSS 20B / 120B**
(Groq's current recommended free models) and also offers the older Llama pair.
If a model name ever stops working, open `app.py`, find the `MODELS` dict near the
top, and swap in any current model from <https://console.groq.com/docs/models>.
