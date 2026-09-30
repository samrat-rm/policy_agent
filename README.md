# Policy Agent

A multi-step LLM agent that answers company policy questions requiring both
document retrieval and computation. Built with LangGraph, Chroma, and a
provider-agnostic model layer (OpenAI or local Ollama).

**Eval: 19/20 (95%), p50 latency 4.4s** on `gpt-4o-mini`.

Example: *"I bought a monitor for ₹40,000, how much do I get back?"* requires
retrieving the reimbursement rule (80%, capped at ₹25,000), computing
40000 × 0.8 = 32000, then applying the cap to arrive at ₹25,000. The number of
steps is decided by the model at runtime, not hardcoded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # set providers and OPENAI_API_KEY

python -m app.ingest        # chunk, embed, persist to Chroma
python -m scripts.demo      # interactive CLI — type `samples`
```

```bash
uvicorn app.api:app --reload      # REST API at /docs
python -m eval.run_eval           # 20-question eval, writes to eval/results/
python -m scripts.check_retrieval # retrieval only, no LLM
```

### Providers

Set in `.env`. The two are independent:

```
LLM_PROVIDER=openai        # or ollama
EMBEDDING_PROVIDER=ollama  # or openai
```

Running chat on OpenAI and embeddings on local Ollama keeps iteration fast
while spending nothing on the part that runs once. Changing
`EMBEDDING_PROVIDER` invalidates the vector store — delete `chroma_db/` and
re-ingest, since embedding dimensions differ between models.

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
  `supported_by_policy` flag, the hallucination guardrail

The loop is why this is a graph rather than a chain: a chain cannot route back
to a previous step.

## Evaluation

20 gold questions across four categories, scored on amount accuracy, guardrail
correctness, keyword presence, and whether sources were cited.

| Category | Score |
|---|---|
| compute_cap (percentage + cap) | 7/7 |
| compute_multi (several items) | 1/1 |
| lookup_amount | 7/7 |
| factual | 3/3 |
| unanswerable (guardrail) | 1/2 |

**Known failure — q19.** The agent correctly refuses in prose but sets
`supported_by_policy: true`. Sources are extracted deterministically from
retrieval output, and a cited source reads as grounding — but semantic search
always returns its nearest matches whether or not they are relevant.
Retrieval returning a document is not the same as that document answering the
question.

## Design notes

**Header-aware chunking.** Policy rules live under headings. Splitting purely
on character count strands numbers away from the rule they belong to, which is
the most common cause of bad retrieval.

**No `eval()` in the calculator.** Model output is untrusted input. The
calculator parses to an AST and permits a fixed operator allowlist.

**Loop cap.** `MAX_TOOL_LOOPS` bounds the agent. Without it a confused model
can call tools indefinitely.

**Deterministic extraction over prompting.** Source filenames and fallback
amounts are parsed from the transcript in code rather than requested from the
model. Anything the code can do reliably should not be delegated to the model.

**No `with_structured_output`.** That method relies on constrained decoding,
which some backends do not implement. Prompting for JSON and parsing it works
everywhere, with a fallback path for malformed output.

## Debugging notes

Three failures worth recording, since each taught something the code now
reflects.

**A small model ignoring its context.** `phi4-mini` answered 10 sick days when
the retrieved chunk said 12, replied in dollars when every document uses
rupees, and cited a filename that does not exist. Diagnosed by testing
retrieval in isolation (`scripts/check_retrieval.py`) — the correct chunk
returned at similarity 0.4998 with a clear margin, proving the pipeline was
sound and isolating the fault to the model.

**Optional schema fields silently dropped.** `amount` and `sources` returned
empty on every question. Fields with defaults are optional in the generated
JSON schema, and smaller models skip optional fields. Fixed by removing
defaults and adding a deterministic fallback.

**A fix that broke something else.** Extracting sources from retrieval output
solved empty citations but caused the guardrail to report false grounding,
since the model reads a cited source as evidence. This is the q19 failure
above and it is still open.

## Still to build

- [ ] Fix the q19 guardrail failure and re-run the eval
- [ ] Deploy (Railway or Render)
- [ ] Expand the gold set beyond 20 questions