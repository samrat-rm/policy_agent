"""Evaluation harness.

    python -m eval.run_eval                      # run everything
    python -m eval.run_eval --tag baseline       # label this run
    python -m eval.run_eval --limit 5            # quick smoke test

Writes a JSON record and a markdown summary to eval/results/ so you can
compare runs after changing the prompt, the model, or the chunk size.

Why this exists: "it seemed to work" is not a claim you can defend. A score
you can reproduce, and watch move when you change something, is.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import LLM_MODEL
from app.graph import AGENT, SYSTEM_PROMPT

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Amounts are money, so allow a rupee of float slop but nothing meaningful.
AMOUNT_TOLERANCE = 1.0


def load_gold(limit: int | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def run_one(question: str) -> dict:
    """Invoke the graph directly so we can inspect the full state."""
    start = time.perf_counter()
    state = AGENT.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "answer": None,
            "tool_loops": 0,
        }
    )
    latency = time.perf_counter() - start

    messages = state["messages"]
    tool_calls = sum(
        len(getattr(m, "tool_calls", []) or []) for m in messages
    )
    # Rough proxy for tokens. Swap in real usage metadata if your provider
    # returns it — this is only here to make cost visible, not exact.
    chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)

    return {
        "answer_obj": state["answer"],
        "latency": latency,
        "tool_calls": tool_calls,
        "approx_tokens": chars // 4,
    }


def score(gold: dict, result: dict) -> dict:
    """Score one question. Returns the verdict plus why it failed."""
    ans = result["answer_obj"]
    failures: list[str] = []

    if ans is None:
        return {"passed": False, "failures": ["no answer returned"]}

    # 1. Did it get the number right?
    expected_amount = gold.get("expected_amount")
    if expected_amount is not None:
        if ans.amount is None:
            failures.append(f"expected amount {expected_amount}, got none")
        elif abs(ans.amount - expected_amount) > AMOUNT_TOLERANCE:
            failures.append(f"expected {expected_amount}, got {ans.amount}")

    # 2. Did the guardrail fire correctly?
    #    This is the one people forget. An always-false flag is as useless
    #    as an always-true one.
    expected_supported = gold.get("expected_supported")
    if expected_supported is not None:
        if ans.supported_by_policy != expected_supported:
            failures.append(
                f"supported_by_policy expected {expected_supported}, "
                f"got {ans.supported_by_policy}"
            )

    # 3. For non-numeric answers, check the key terms appear.
    lowered = ans.answer.lower()
    for kw in gold.get("expected_keywords", []):
        if kw.lower() not in lowered:
            failures.append(f"missing keyword '{kw}'")

    # 4. Did it cite anything at all when it claimed support?
    if ans.supported_by_policy and not ans.sources:
        failures.append("claimed policy support but cited no sources")

    return {"passed": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run", help="label for this run")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    gold = load_gold(args.limit)
    print(f"Running {len(gold)} questions against {LLM_MODEL}\n")

    records = []
    for i, row in enumerate(gold, 1):
        print(f"[{i}/{len(gold)}] {row['id']} ... ", end="", flush=True)
        try:
            result = run_one(row["question"])
            verdict = score(row, result)
        except Exception as exc:
            result = {
                "answer_obj": None,
                "latency": 0.0,
                "tool_calls": 0,
                "approx_tokens": 0,
            }
            verdict = {"passed": False, "failures": [f"exception: {exc}"]}

        ans = result["answer_obj"]
        records.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "expected_amount": row.get("expected_amount"),
                "got_amount": ans.amount if ans else None,
                "expected_supported": row.get("expected_supported"),
                "got_supported": ans.supported_by_policy if ans else None,
                "answer": ans.answer if ans else "",
                "sources": ans.sources if ans else [],
                "passed": verdict["passed"],
                "failures": verdict["failures"],
                "latency": round(result["latency"], 2),
                "tool_calls": result["tool_calls"],
                "approx_tokens": result["approx_tokens"],
            }
        )
        print("PASS" if verdict["passed"] else f"FAIL — {verdict['failures'][0]}")

    write_report(records, args.tag)


def write_report(records: list[dict], tag: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    passed = sum(r["passed"] for r in records)
    total = len(records)
    latencies = sorted(r["latency"] for r in records)

    by_category: dict[str, list[bool]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r["passed"])

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.0f}%" if d else "n/a"

    p50 = statistics.median(latencies) if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0

    summary = {
        "tag": tag,
        "timestamp": stamp,
        "model": LLM_MODEL,
        "score": f"{passed}/{total}",
        "accuracy_pct": round(100 * passed / total, 1) if total else 0,
        "latency_p50": round(p50, 2),
        "latency_p95": round(p95, 2),
        "latency_mean": round(statistics.mean(latencies), 2) if latencies else 0,
        "avg_tool_calls": round(
            statistics.mean([r["tool_calls"] for r in records]), 1
        )
        if records
        else 0,
        "by_category": {
            k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_category.items())
        },
    }

    json_path = RESULTS_DIR / f"{stamp}-{tag}.json"
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# Eval run: {tag}",
        "",
        f"- Model: `{LLM_MODEL}`",
        f"- Score: **{passed}/{total}** ({summary['accuracy_pct']}%)",
        f"- Latency p50 / p95: {summary['latency_p50']}s / {summary['latency_p95']}s",
        f"- Avg tool calls per question: {summary['avg_tool_calls']}",
        "",
        "## By category",
        "",
        "| Category | Score |",
        "|---|---|",
    ]
    for k, v in summary["by_category"].items():
        lines.append(f"| {k} | {v} |")

    failures = [r for r in records if not r["passed"]]
    if failures:
        lines += ["", "## Failures", "", "| ID | Expected | Got | Why |", "|---|---|---|---|"]
        for r in failures:
            why = r["failures"][0] if r["failures"] else ""
            lines.append(
                f"| {r['id']} | {r['expected_amount']} | {r['got_amount']} | {why} |"
            )

    md_path = RESULTS_DIR / f"{stamp}-{tag}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"SCORE: {passed}/{total}  ({summary['accuracy_pct']}%)")
    print(f"Latency p50 {summary['latency_p50']}s / p95 {summary['latency_p95']}s")
    for k, v in summary["by_category"].items():
        print(f"  {k}: {v}")
    print(f"\nWrote {md_path.name} and {json_path.name}")


if __name__ == "__main__":
    main()
