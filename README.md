# Policy Agent

A multi-step LLM agent that answers company policy questions requiring both
document retrieval and computation. Built with LangGraph, Chroma, and OpenAI.

Example: *"I bought a monitor for ₹40,000, how much do I get back?"* requires
retrieving the reimbursement rule (80%, capped at ₹25,000), computing
40000 × 0.8 = 32000, then applying the cap to arrive at ₹25,000. The number of
steps is decided by the model at runtime, not hardcoded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY

python -m app.ingest        # chunk, embed, persist to Chroma
python -m scripts.demo      # interactive CLI
```

Type `samples` in the CLI to run the example questions.

## Architecture

```
START ──> agent ──┬──> tools ──┐
                  │            │  (loop)
                  │  <─────────┘
                  └──> finalize ──> END
```

- **agent** — asks the model what to do next, with tools bound
- **tools** — executes `search_policies` or `calculate`, returns the result
- **conditional edge** — routes back to `agent` if a tool call is pending,
  otherwise forwards to `finalize`
- **finalize** — forces the answer into a Pydantic schema with a
  `supported_by_policy` flag, which is the hallucination guardrail

The loop is why this is a graph rather than a chain: a chain cannot route back
to a previous step.

## Design notes

**Header-aware chunking.** Policy rules live under headings. Splitting purely
on character count strands numbers away from the rule they belong to, which is
the most common cause of bad retrieval.

**No `eval()` in the calculator.** Model output is untrusted input. The
calculator parses to an AST and permits a fixed operator allowlist.

**Loop cap.** `MAX_TOOL_LOOPS` bounds the agent. Without it a confused model
can call tools indefinitely.

**Structured output as a guardrail.** Asking the model to declare whether the
retrieved policy actually supports its answer makes hallucination a field your
code can branch on, rather than something buried in confident prose.

## Still to build

- [ ] FastAPI wrapper (`/query`, `/health`) with request logging
- [ ] Eval set: 20 gold questions, accuracy scoring, latency and cost per query
- [ ] Prompt iteration with before/after eval numbers
