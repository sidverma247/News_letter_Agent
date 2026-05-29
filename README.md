# 🤖 Autonomous Newsletter Agent

A mini **autonomous AI agent** that turns a plain-English goal —
*"Create a weekly newsletter on latest AI agent news and send it to our subscribers"* —
into a finished newsletter, with **zero further input**.

It is built as an explicit **LangGraph** state machine driven by **Grok (xAI)**, with real
tool use, a self-reflection/critique loop, and a toggle between **Fully Autonomous** and
**Human-in-the-Loop** modes. A small **Flask** web UI lets you watch every step live.

---

## What it does (agentic pipeline)

```
   plan ─▶ research ─▶ write ─▶ self_reflect ─┬─▶ revise ─┐
                                              │           │
                                              └──▶ finalize◀┘
```

| Step | What happens | Tool / reasoning |
|------|--------------|------------------|
| **plan** | LLM converts the goal into a concrete plan (search queries, audience, tone, # stories) | Grok reasoning |
| **research** | Fetches the latest AI-agent news | **Tool:** `research_news` (Tavily API, with automatic RSS fallback) |
| **write** | LLM selects the top 5–7 stories and writes original summaries + subject + intro | Grok reasoning |
| **self_reflect** | The agent **critiques its own draft** (score /10, issues, suggestions) and decides whether to revise | Self-reflection |
| **revise** | LLM rewrites the draft using its own critique (and human feedback if any). Loops back to re-critique | Grok reasoning |
| **finalize** | Renders the final newsletter | **Tool:** `build_newsletter_html` → HTML + Markdown |

This satisfies the assignment's core requirements:

- **Multi-step reasoning** — planning → research → writing → review → output, as explicit graph nodes.
- **Tool use (3 tools)** — web search (`research_news`), the LLM summarizer, and the HTML generator (`build_newsletter_html`).
- **Self-reflection / critique** — the `self_reflect` node grades the draft and triggers up to `MAX_REVISIONS` automatic improvement passes.
- **Autonomous** — a single call, `run_newsletter_agent(goal)`, does the whole job.
- **Mode toggle** — `mode="autonomous"` vs `mode="human_in_loop"`.
- **"Sending"** — the final newsletter is saved as `output/newsletter.html` / `.md` and the email subject + body are printed.

---

## Project structure

```
newsletter_agent/
├── agent.py            # LangGraph agent + run_newsletter_agent() entrypoint
├── tools.py            # research (Tavily/RSS) + HTML newsletter generator
├── app.py              # Flask web frontend (live SSE log, mode toggle, HITL gates)
├── run_cli.py          # command-line runner
├── templates/
│   └── index.html      # single-file web UI
├── requirements.txt
├── .env.example
└── output/             # generated newsletter.html / .md / result.json (created on first run)
```

---

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure keys** — copy `.env.example` to `.env` and fill it in:

   ```bash
   cp .env.example .env
   ```

   ```ini
   XAI_API_KEY=your_grok_api_key        # REQUIRED — get one at https://console.x.ai
   GROK_MODEL=grok-3                    # e.g. grok-3, grok-3-mini, grok-4
   TAVILY_API_KEY=                      # OPTIONAL — live web search; falls back to RSS if blank
   FLASK_PORT=8080
   ```

   > **No Grok key?** The agent still runs in a clearly-labelled **offline mode** (extractive
   > summaries from the research feeds, no LLM reasoning) so you can see the full pipeline.
   > For real, high-quality summaries, set `XAI_API_KEY`.

   > **No Tavily key?** Research automatically falls back to curated free **RSS feeds**
   > (TechCrunch AI, The Verge, VentureBeat, Hugging Face, MarkTechPost). No action needed.

---

## Run it

### Live Hosted Link

```
# open https://news-letter-agent.onrender.com/
```

### Web UI (recommended)

```bash
python app.py
# open http://localhost:5000
```

You get a goal box, a **Fully Autonomous / Human-in-the-Loop** toggle, a live activity log
(plan → research → write → reflect → revise → output), and the finished newsletter rendered
inline with HTML / Markdown / "Plan & critique" tabs and download links.

In **Human-in-the-Loop** mode the page pauses at two gates — after the **plan** and after the
**draft + self-critique** — and lets you *Approve* or send *feedback* to trigger a revision.

### Command line

```bash
# Fully autonomous, default goal
python run_cli.py

# Custom goal
python run_cli.py --goal "Create a weekly newsletter on latest AI agent news"

# Human-in-the-loop (approve plan & draft in the terminal)
python run_cli.py --mode human_in_loop
```
### Output
<img width="1470" height="880" alt="Screenshot 2026-05-29 at 7 03 36 PM" src="https://github.com/user-attachments/assets/5fb64a1c-6280-420f-8651-bda608d19932" />

<img width="1470" height="880" alt="Screenshot 2026-05-29 at 7 03 47 PM" src="https://github.com/user-attachments/assets/fab5aeca-1ada-4344-862a-c98ca4b5e7b2" />





### As a library — the single autonomous call

```python
from agent import run_newsletter_agent

result = run_newsletter_agent(
    "Create a weekly newsletter on latest AI agent news and send it to our subscribers.",
    mode="autonomous",          # or "human_in_loop"
    save_dir="output",          # writes newsletter.html / .md / result.json
)

print(result["subject"])
print(result["markdown"])
# result also contains: html, newsletter (structured), plan, critique, events
```

For human-in-the-loop in code, pass a callback:

```python
def review(stage, data):
    if stage == "plan":
        return {}                                   # approve plan as-is
    return {"action": "revise", "feedback": "Make the subject punchier"}

run_newsletter_agent(goal, mode="human_in_loop", human_review=review)
```

---

## How the modes differ

| | Fully Autonomous | Human-in-the-Loop |
|---|---|---|
| Plan | accepted automatically | you can edit / approve it |
| Draft | improved only via the agent's **own** self-critique | you can approve or inject feedback that forces a revision |
| Human input | none | two checkpoints |
| Entry | `run_newsletter_agent(goal)` | `run_newsletter_agent(goal, mode="human_in_loop", human_review=cb)` |

---

## Tech stack

- **LangGraph** — explicit, inspectable agent state machine (nodes + conditional edges + revise loop).
- **Grok (xAI)** via `langchain-xai` (`ChatXAI`), with an OpenAI-compatible fallback to `https://api.x.ai/v1`.
- **Tavily** for live web search, **feedparser** for the free RSS fallback.
- **Flask** + a single HTML template (Server-Sent Events for the live log) for the frontend.

---

## Notes & design choices

- **Robustness:** every LLM call and network call is wrapped so a failure degrades gracefully
  (offline summaries / RSS fallback) instead of crashing the run.
- **Self-improvement is bounded** by `MAX_REVISIONS` (default 2) to keep runs fast and avoid loops.
- **"Sending" is simulated** by saving the rendered email (`output/newsletter.html`) and printing
  the subject + body — swapping in a real SMTP / provider send is a one-function change in `finalize`.
