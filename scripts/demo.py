"""Interactive CLI.

    python -m scripts.demo                 # interactive
    python -m scripts.demo "your question" # one-shot
"""

import sys
import time

from app.graph import ask


def show(question: str) -> None:
    start = time.perf_counter()
    result = ask(question)
    elapsed = time.perf_counter() - start

    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    if result.amount is not None:
        print(f"Amount: {result.amount}")
    if result.reasoning:
        print(f"Why: {result.reasoning}")
    print(f"Sources: {', '.join(result.sources) or 'none'}")
    print(f"Supported by policy: {result.supported_by_policy}")
    print(f"Latency: {elapsed:.2f}s\n")


SAMPLES = [
    "I bought a monitor for 40000. How much do I get back?",
    "How many sick days do I get and do I need a doctor's note?",
    "I spent 12000 on an office chair and 30000 on a desk. Total reimbursement?",
    "Does the company pay for my dog's vet bills?",
]


def main() -> None:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return

    print("Policy agent. Type a question, 'samples' to run examples, or 'q' to quit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"q", "quit", "exit"}:
            break
        if q == "samples":
            for s in SAMPLES:
                show(s)
            continue
        if q:
            show(q)


if __name__ == "__main__":
    main()
