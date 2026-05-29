"""
run_cli.py
----------
Command-line entrypoint for the Newsletter Agent.

Examples
--------
  # Fully autonomous (default goal):
  python run_cli.py

  # Custom goal:
  python run_cli.py --goal "Create a weekly newsletter on latest AI agent news"

  # Human-in-the-loop (you approve the plan and the draft in the terminal):
  python run_cli.py --mode human_in_loop
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from agent import run_newsletter_agent

DEFAULT_GOAL = ("Create a weekly newsletter on latest AI agent news and "
                "send it to our subscribers.")


def cli_event(evt):
    icon = {
        "plan": "🧭", "research": "🔎", "write": "✍️", "reflect": "🪞",
        "revise": "🔧", "output": "📤", "review_request": "🙋", "done": "✅",
        "error": "❌",
    }.get(evt.get("stage"), "•")
    print(f"  {icon} [{evt.get('ts','')}] {evt.get('title','')}"
          + (f" — {evt['detail']}" if evt.get("detail") else ""))


def cli_human_review(stage, data):
    print("\n" + "=" * 70)
    if stage == "plan":
        print("HUMAN REVIEW · PLAN")
        print(json.dumps(data["plan"], indent=2))
        ans = input("Approve plan? [Y/n] (n lets you edit num_stories): ").strip().lower()
        if ans == "n":
            try:
                n = int(input("How many stories (5-7)? ").strip())
                plan = dict(data["plan"]); plan["num_stories"] = max(5, min(7, n))
                return {"plan": plan}
            except ValueError:
                pass
        return {}
    else:  # draft
        print("HUMAN REVIEW · DRAFT")
        nl = data["newsletter"]
        print(f"Subject: {nl.get('subject')}\n")
        for i, it in enumerate(nl.get("items", []), 1):
            print(f"  {i}. {it.get('title')}")
        print(f"\nSelf-critique score: {data['critique'].get('score')}/10")
        fb = input("Press Enter to approve, or type feedback to revise: ").strip()
        if fb:
            return {"action": "revise", "feedback": fb}
        return {"action": "approve"}


def main():
    p = argparse.ArgumentParser(description="Autonomous Newsletter Agent")
    p.add_argument("--goal", default=DEFAULT_GOAL)
    p.add_argument("--mode", choices=["autonomous", "human_in_loop"],
                   default="autonomous")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    print(f"\n🤖 Newsletter Agent — mode: {args.mode}\n   Goal: {args.goal}\n")
    result = run_newsletter_agent(
        args.goal,
        mode=args.mode,
        human_review=cli_human_review if args.mode == "human_in_loop" else None,
        on_event=cli_event,
        save_dir=args.out,
    )

    print("\n" + "=" * 70)
    print("📧 NEWSLETTER READY")
    print("=" * 70)
    print(f"Subject: {result['subject']}")
    print(f"Self-critique: {result['critique'].get('score','?')}/10")
    print(f"\nSaved to ./{args.out}/  (newsletter.html, newsletter.md, result.json)")
    print("\n--- Markdown preview ---\n")
    print(result["markdown"])


if __name__ == "__main__":
    main()
