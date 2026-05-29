"""
agent.py
--------
The autonomous **Newsletter Agent**, built as an explicit LangGraph state machine.

Pipeline (true multi-step reasoning):

    plan  ->  research  ->  write  ->  self_reflect  -+-> revise -+
                                                       |           |
                                                       +-> finalize <-

  * plan          The LLM turns the plain-English goal into a concrete plan
                  (topic queries, audience, tone, # of stories).
  * research      Tool use: research_news() (Tavily or RSS).
  * write         The LLM summarizes the top stories + writes intro/subject.
  * self_reflect  The agent CRITIQUES its own draft (self-reflection step) and
                  decides whether another revision pass is warranted.
  * revise        The LLM rewrites the draft using its own critique (and any
                  human feedback in Human-in-the-Loop mode).
  * finalize      Tool use: build_newsletter_html() -> final deliverable.

Modes
-----
  "autonomous"      -> runs end to end with no human input.
  "human_in_loop"   -> calls the supplied `human_review` callback at two gates
                       (after planning, and after the draft + self-critique).

Public entrypoint
-----------------
    run_newsletter_agent(goal, mode="autonomous",
                         human_review=None, on_event=None) -> dict
"""

from __future__ import annotations

import os
import json
import datetime as _dt
from typing import List, Dict, Optional, Callable, TypedDict

from langgraph.graph import StateGraph, START, END

from tools import research_news, build_newsletter_html

MAX_REVISIONS = 2  # how many self-improvement passes the agent may take


# --------------------------------------------------------------------------- #
#  LLM (Grok / xAI) initialisation                                            #
# --------------------------------------------------------------------------- #

def _get_llm():
    """
    Return a chat model bound to Grok (xAI).

    Prefers langchain_xai.ChatXAI; falls back to the OpenAI-compatible client
    pointed at https://api.x.ai/v1. Returns None if no key is configured so the
    app can run in a clearly-labelled degraded/offline mode.
    """
    api_key = os.environ.get("XAI_API_KEY")
    model = os.environ.get("GROK_MODEL", "grok-3")
    if not api_key:
        return None
    try:
        from langchain_xai import ChatXAI
        return ChatXAI(model=model, api_key=api_key, temperature=0.4)
    except Exception:
        # Fallback: Grok is OpenAI-API compatible.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            temperature=0.4,
        )


def _llm_json(llm, system: str, user: str) -> dict:
    """Call the LLM and parse a JSON object from its reply (robust to fences)."""
    from langchain_core.messages import SystemMessage, HumanMessage

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = resp.content if hasattr(resp, "content") else str(resp)
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        # pull the contents of the first fenced block
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


# --------------------------------------------------------------------------- #
#  Graph state                                                                #
# --------------------------------------------------------------------------- #

class AgentState(TypedDict, total=False):
    goal: str
    mode: str                       # "autonomous" | "human_in_loop"
    plan: dict                      # {queries, audience, tone, num_stories}
    raw_articles: List[Dict]
    newsletter: dict                # {subject, intro, items, sign_off}
    critique: dict                  # {score, issues, suggestions, needs_revision}
    revision_count: int
    human_feedback: str
    html: str
    markdown: str
    events: List[dict]


# --------------------------------------------------------------------------- #
#  Event + callback plumbing                                                  #
# --------------------------------------------------------------------------- #

# Module-level handles set per-run (kept simple; the graph is run synchronously).
_on_event: Optional[Callable[[dict], None]] = None
_human_review: Optional[Callable[[str, dict], dict]] = None
_llm = None


def _emit(stage: str, title: str, detail: str = "", data: Optional[dict] = None):
    evt = {
        "stage": stage,
        "title": title,
        "detail": detail,
        "data": data or {},
        "ts": _dt.datetime.now().strftime("%H:%M:%S"),
    }
    if _on_event:
        try:
            _on_event(evt)
        except Exception:
            pass
    return evt


# --------------------------------------------------------------------------- #
#  Nodes                                                                       #
# --------------------------------------------------------------------------- #

def node_plan(state: AgentState) -> AgentState:
    goal = state["goal"]
    _emit("plan", "Planning", f"Interpreting goal: “{goal}”")

    if _llm is None:
        plan = {
            "queries": ["latest AI agent news", "autonomous AI agents", "LLM agent frameworks"],
            "audience": "AI practitioners and enthusiasts",
            "tone": "informative, concise, upbeat",
            "num_stories": 6,
        }
    else:
        system = (
            "You are an autonomous newsletter agent. Convert the user's goal into a "
            "concrete research + writing plan. Respond ONLY with JSON."
        )
        user = (
            f"GOAL: {goal}\n\n"
            "Return JSON with keys: "
            '"queries" (list of 2-4 web search queries to find the freshest, most relevant news), '
            '"audience" (string), "tone" (string), "num_stories" (int between 5 and 7).'
        )
        try:
            plan = _llm_json(_llm, system, user)
        except Exception as e:
            _emit("plan", "Planning fallback", f"LLM planning failed ({e}); using defaults.")
            plan = {
                "queries": ["latest AI agent news this week"],
                "audience": "AI practitioners",
                "tone": "informative",
                "num_stories": 6,
            }

    plan.setdefault("num_stories", 6)
    plan["num_stories"] = max(5, min(7, int(plan["num_stories"])))
    _emit("plan", "Plan ready",
          f"{plan['num_stories']} stories • audience: {plan.get('audience','')} • "
          f"queries: {', '.join(plan.get('queries', []))[:160]}",
          data=plan)

    # ---- Human-in-the-loop gate #1 ----
    if state.get("mode") == "human_in_loop" and _human_review:
        decision = _human_review("plan", {"plan": plan, "goal": goal})
        if decision.get("plan"):
            plan = decision["plan"]
            _emit("plan", "Plan updated by human", "Using human-edited plan.", data=plan)

    return {"plan": plan, "revision_count": 0}


def node_research(state: AgentState) -> AgentState:
    plan = state["plan"]
    queries = plan.get("queries") or [state["goal"]]
    n = plan["num_stories"]
    _emit("research", "Researching", f"Searching the web / feeds for {len(queries)} queries…")

    collected: List[Dict] = []
    seen = set()
    per_query = max(4, n)
    for q in queries:
        for art in research_news(q, max_results=per_query):
            key = art.get("url") or art.get("title")
            if key and key not in seen:
                seen.add(key)
                collected.append(art)

    _emit("research", "Research complete",
          f"Found {len(collected)} candidate articles.",
          data={"count": len(collected),
                "titles": [a["title"] for a in collected[:12]]})
    return {"raw_articles": collected}


def node_write(state: AgentState) -> AgentState:
    plan = state["plan"]
    articles = state.get("raw_articles", [])
    n = plan["num_stories"]
    _emit("write", "Writing", f"Selecting & summarizing the top {n} stories…")

    if _llm is None or not articles:
        items = []
        for a in articles[:n]:
            snippet = (a.get("content") or "").strip().replace("\n", " ")
            items.append({
                "title": a.get("title", "Untitled"),
                "summary": (snippet[:240] + "…") if len(snippet) > 240 else snippet,
                "url": a.get("url", "#"),
                "source": a.get("source", ""),
            })
        newsletter = {
            "subject": "AI Agent Weekly — Your Roundup of the Latest in Autonomous AI",
            "intro": "Here are this week's most notable developments in AI agents.",
            "items": items,
            "sign_off": "— The AI Agent Weekly Team",
        }
    else:
        # Give the LLM a compact view of the candidates.
        compact = [
            {"title": a.get("title", ""), "url": a.get("url", ""),
             "source": a.get("source", ""),
             "content": (a.get("content", "") or "")[:700]}
            for a in articles[:20]
        ]
        system = (
            "You are an expert tech newsletter writer. From the candidate articles, "
            "select the most relevant and recent stories about AI agents / autonomous AI. "
            "Write tight, original 2-3 sentence summaries (no copying). Respond ONLY with JSON."
        )
        user = (
            f"AUDIENCE: {plan.get('audience')}\nTONE: {plan.get('tone')}\n"
            f"NUMBER OF STORIES: {n}\n\n"
            f"CANDIDATES (JSON):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
            "Return JSON: {\"subject\": str, \"intro\": str (2-3 sentences), "
            "\"items\": [{\"title\": str, \"summary\": str, \"url\": str, \"source\": str}], "
            "\"sign_off\": str }. Use exactly the requested number of items, best stories first."
        )
        try:
            newsletter = _llm_json(_llm, system, user)
        except Exception as e:
            _emit("write", "Writing fallback", f"LLM writing failed ({e}); using extractive summaries.")
            items = [{
                "title": a.get("title", "Untitled"),
                "summary": (a.get("content", "") or "")[:240],
                "url": a.get("url", "#"),
                "source": a.get("source", ""),
            } for a in articles[:n]]
            newsletter = {"subject": "AI Agent Weekly", "intro": "This week in AI agents.",
                          "items": items, "sign_off": "— The AI Agent Weekly Team"}

    newsletter.setdefault("sign_off", "— The AI Agent Weekly Team")
    _emit("write", "Draft ready",
          f"Drafted “{newsletter.get('subject','')}” with {len(newsletter.get('items', []))} stories.",
          data={"subject": newsletter.get("subject"),
                "items": [i.get("title") for i in newsletter.get("items", [])]})
    return {"newsletter": newsletter}


def node_self_reflect(state: AgentState) -> AgentState:
    """The agent critiques its own draft — the self-reflection step."""
    newsletter = state["newsletter"]
    _emit("reflect", "Self-reflection", "Critiquing the draft for quality & relevance…")

    if _llm is None:
        critique = {"score": 7, "issues": [], "suggestions": [],
                    "needs_revision": False}
    else:
        system = (
            "You are a ruthless newsletter editor. Critique the draft for relevance to "
            "'latest AI agent news', clarity, redundancy, subject-line strength, and summary "
            "quality. Be specific and constructive. Respond ONLY with JSON."
        )
        user = (
            f"DRAFT (JSON):\n{json.dumps(newsletter, ensure_ascii=False)}\n\n"
            "Return JSON: {\"score\": int 1-10, \"issues\": [str], "
            "\"suggestions\": [str], \"needs_revision\": bool}. "
            "Set needs_revision=true only if score < 8."
        )
        try:
            critique = _llm_json(_llm, system, user)
        except Exception as e:
            critique = {"score": 8, "issues": [f"critique failed: {e}"],
                        "suggestions": [], "needs_revision": False}

    _emit("reflect", f"Self-critique: {critique.get('score','?')}/10",
          "; ".join(critique.get("issues", [])) or "No major issues found.",
          data=critique)

    # ---- Human-in-the-loop gate #2 ----
    if state.get("mode") == "human_in_loop" and _human_review:
        decision = _human_review("draft", {"newsletter": newsletter, "critique": critique})
        fb = decision.get("feedback", "").strip()
        if decision.get("action") == "revise" or fb:
            critique = dict(critique)
            critique["needs_revision"] = True
            return {"critique": critique, "human_feedback": fb}
        # human approved as-is
        critique = dict(critique)
        critique["needs_revision"] = False

    return {"critique": critique}


def node_revise(state: AgentState) -> AgentState:
    newsletter = state["newsletter"]
    critique = state.get("critique", {})
    human_fb = state.get("human_feedback", "")
    count = state.get("revision_count", 0) + 1
    _emit("revise", f"Revising (pass {count})", "Applying critique to improve the draft…")

    if _llm is None:
        return {"revision_count": count, "human_feedback": ""}

    system = (
        "You are revising a newsletter using editorial critique and any human feedback. "
        "Improve clarity, relevance and the subject line. Keep the same JSON shape. "
        "Respond ONLY with JSON."
    )
    user = (
        f"CURRENT DRAFT:\n{json.dumps(newsletter, ensure_ascii=False)}\n\n"
        f"EDITOR CRITIQUE:\n{json.dumps(critique, ensure_ascii=False)}\n\n"
        f"HUMAN FEEDBACK: {human_fb or '(none)'}\n\n"
        "Return the improved newsletter as JSON with the same keys "
        "(subject, intro, items[{title,summary,url,source}], sign_off)."
    )
    try:
        revised = _llm_json(_llm, system, user)
        revised.setdefault("sign_off", newsletter.get("sign_off", "— The AI Agent Weekly Team"))
        _emit("revise", "Revision complete", "Draft improved.", data={"subject": revised.get("subject")})
        return {"newsletter": revised, "revision_count": count, "human_feedback": ""}
    except Exception as e:
        _emit("revise", "Revision skipped", f"Revision failed ({e}); keeping previous draft.")
        return {"revision_count": count, "human_feedback": ""}


def node_finalize(state: AgentState) -> AgentState:
    newsletter = state["newsletter"]
    _emit("output", "Generating newsletter", "Rendering HTML + Markdown deliverables…")

    html_doc = build_newsletter_html(
        subject=newsletter.get("subject", "AI Agent Weekly"),
        intro=newsletter.get("intro", ""),
        items=newsletter.get("items", []),
        sign_off=newsletter.get("sign_off", "— The AI Agent Weekly Team"),
    )
    md = _to_markdown(newsletter)
    _emit("output", "Newsletter ready", "Deliverables generated.", data={"subject": newsletter.get("subject")})
    return {"html": html_doc, "markdown": md}


def _to_markdown(n: dict) -> str:
    today = _dt.date.today().strftime("%B %d, %Y")
    lines = [f"# {n.get('subject','AI Agent Weekly')}",
             f"*AI Agent Weekly · {today}*", "",
             n.get("intro", ""), ""]
    for i, it in enumerate(n.get("items", []), 1):
        src = f" — *{it.get('source')}*" if it.get("source") else ""
        lines.append(f"## {i}. {it.get('title','Untitled')}{src}")
        lines.append(it.get("summary", ""))
        lines.append(f"[Read more]({it.get('url','#')})")
        lines.append("")
    lines.append(n.get("sign_off", "— The AI Agent Weekly Team"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Conditional routing after self-reflection                                  #
# --------------------------------------------------------------------------- #

def _route_after_reflect(state: AgentState) -> str:
    critique = state.get("critique", {})
    if critique.get("needs_revision") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"
    return "finalize"


# --------------------------------------------------------------------------- #
#  Graph construction                                                          #
# --------------------------------------------------------------------------- #

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("research", node_research)
    g.add_node("write", node_write)
    g.add_node("self_reflect", node_self_reflect)
    g.add_node("revise", node_revise)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "self_reflect")
    g.add_conditional_edges("self_reflect", _route_after_reflect,
                            {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "self_reflect")   # re-critique after revising
    g.add_edge("finalize", END)
    return g.compile()


# --------------------------------------------------------------------------- #
#  Public entrypoint                                                           #
# --------------------------------------------------------------------------- #

def run_newsletter_agent(
    goal: str,
    mode: str = "autonomous",
    human_review: Optional[Callable[[str, dict], dict]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run the full agent for a plain-English goal.

    Parameters
    ----------
    goal          : e.g. "Create a weekly newsletter on latest AI agent news…"
    mode          : "autonomous" or "human_in_loop"
    human_review  : callback(stage, data) -> dict, used only in human_in_loop mode.
                    stage == "plan"  -> may return {"plan": <edited plan>}
                    stage == "draft" -> may return {"action": "approve"|"revise",
                                                     "feedback": str}
    on_event      : callback(event_dict) for live progress (UI logging).
    save_dir      : if given, writes newsletter.html / newsletter.md / result.json there.

    Returns a dict: {subject, html, markdown, newsletter, plan, critique, events}
    """
    global _on_event, _human_review, _llm
    _on_event = on_event
    _human_review = human_review
    _llm = _get_llm()

    if _llm is None:
        _emit("plan", "No Grok key found",
              "XAI_API_KEY not set — running in degraded offline mode "
              "(extractive summaries, no LLM reasoning).")

    events: List[dict] = []
    # capture events into a list as well as forwarding them
    orig = _on_event

    def _capture(e):
        events.append(e)
        if orig:
            orig(e)

    _on_event = _capture

    app = build_graph()
    final_state = app.invoke(
        {"goal": goal, "mode": mode, "events": []},
        config={"recursion_limit": 50},
    )

    result = {
        "subject": final_state["newsletter"].get("subject", ""),
        "html": final_state.get("html", ""),
        "markdown": final_state.get("markdown", ""),
        "newsletter": final_state.get("newsletter", {}),
        "plan": final_state.get("plan", {}),
        "critique": final_state.get("critique", {}),
        "events": events,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "newsletter.html"), "w", encoding="utf-8") as f:
            f.write(result["html"])
        with open(os.path.join(save_dir, "newsletter.md"), "w", encoding="utf-8") as f:
            f.write(result["markdown"])
        with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "html"}, f,
                      ensure_ascii=False, indent=2)
        _emit("output", "Saved", f"Files written to {save_dir}")

    return result
