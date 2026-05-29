"""
app.py
------
Flask frontend for the Newsletter Agent.

Features
  * Plain-English goal box.
  * Toggle: Fully Autonomous  vs  Human-in-the-Loop.
  * Live step-by-step log streamed over Server-Sent Events (SSE) so you can
    watch the agent plan → research → write → reflect → revise → output.
  * In Human-in-the-Loop mode the page pauses at two gates (plan & draft) and
    lets you approve or send feedback.
  * Final newsletter rendered inline (HTML preview) with a Markdown view and
    a download link.

Run:  python app.py    then open http://localhost:5000
"""

from __future__ import annotations

import os
import json
import uuid
import queue
import threading

from flask import Flask, request, jsonify, Response, render_template, send_file
from dotenv import load_dotenv

load_dotenv()

from agent import run_newsletter_agent

app = Flask(__name__)

# job_id -> dict(events: Queue, result, status, reviews: dict, lock)
JOBS: dict = {}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {
        "events": queue.Queue(),
        "result": None,
        "status": "running",
        # pending review gates: stage -> {"data":..., "event":Event, "response":...}
        "reviews": {},
    }
    return jid


def _make_human_review(jid: str):
    """Return a human_review callback that blocks until the web client responds."""
    job = JOBS[jid]

    def human_review(stage: str, data: dict) -> dict:
        ev = threading.Event()
        job["reviews"][stage] = {"event": ev, "response": None}
        # tell the browser to show a review panel
        job["events"].put({
            "stage": "review_request",
            "title": f"Awaiting human review: {stage}",
            "detail": "Review the content and approve or send feedback.",
            "data": {"review_stage": stage, "payload": data},
        })
        ev.wait()  # block worker thread until /review is posted
        return job["reviews"][stage]["response"] or {}

    return human_review


def _worker(jid: str, goal: str, mode: str):
    job = JOBS[jid]

    def on_event(evt):
        job["events"].put(evt)

    human_review = _make_human_review(jid) if mode == "human_in_loop" else None
    try:
        result = run_newsletter_agent(
            goal, mode=mode,
            human_review=human_review,
            on_event=on_event,
            save_dir=OUTPUT_DIR,
        )
        job["result"] = result
        job["status"] = "done"
        job["events"].put({"stage": "done", "title": "Done", "detail": "",
                           "data": {"subject": result["subject"]}})
    except Exception as e:  # pragma: no cover
        job["status"] = "error"
        job["events"].put({"stage": "error", "title": "Agent error",
                           "detail": str(e), "data": {}})


@app.route("/")
def index():
    grok = bool(os.environ.get("XAI_API_KEY"))
    tavily = bool(os.environ.get("TAVILY_API_KEY"))
    return render_template("index.html", grok_ready=grok, tavily_ready=tavily)


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True)
    goal = (data.get("goal") or "").strip()
    mode = data.get("mode", "autonomous")
    if not goal:
        return jsonify({"error": "goal is required"}), 400
    jid = _new_job()
    threading.Thread(target=_worker, args=(jid, goal, mode), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/stream/<jid>")
def stream(jid: str):
    job = JOBS.get(jid)
    if not job:
        return "unknown job", 404

    def gen():
        while True:
            try:
                evt = job["events"].get(timeout=60)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("stage") in ("done", "error"):
                break
        # send final result payload
        if job.get("result"):
            payload = {"stage": "result", "title": "result",
                       "data": job["result"]}
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/review/<jid>", methods=["POST"])
def review(jid: str):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    data = request.get_json(force=True)
    stage = data.get("stage")
    gate = job["reviews"].get(stage)
    if not gate:
        return jsonify({"error": "no pending review for stage"}), 400
    gate["response"] = data.get("decision", {})
    gate["event"].set()
    return jsonify({"ok": True})


@app.route("/download/<kind>")
def download(kind: str):
    fname = {"html": "newsletter.html", "md": "newsletter.md"}.get(kind)
    if not fname:
        return "bad kind", 400
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "FLASK_PORT",
            8080
        )
    )

    print(
        f"\n  Newsletter Agent UI  ->  http://localhost:{port}\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )