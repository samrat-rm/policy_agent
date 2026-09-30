"""The agent graph.

Shape:

    START -> agent -> (tools -> agent)* -> finalize -> END

The loop is the point. `agent` asks the model what to do; if it wants a tool,
the conditional edge routes to `tools`, which executes and sends the result
back. That repeats until the model has enough to answer. The number of steps
is decided at runtime, not hardcoded — that is what makes this an agent rather
than a chain, and why it needs a graph, since a chain cannot loop back.

`finalize` then forces the answer into a validated schema. That is the
guardrail.

Provider note: this version does NOT use `with_structured_output`. That method
relies on constrained decoding, which some backends (HuggingFace serverless
among them) do not implement. Prompting for JSON and parsing it ourselves
works on every provider, at the cost of needing a fallback when the model
returns something malformed.
"""

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.llm import get_chat_model
from app.schemas import PolicyAnswer
from app.tools import TOOLS

MAX_TOOL_LOOPS = 4
AGENT_RETRIES = 2

llm = get_chat_model(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a company policy assistant.

Rules you must follow:
- Always call search_policies before answering any question about company \
policy. Never answer from memory, and never use rates or limits that did not \
come back from a search.
- Always call calculate for arithmetic. Do not compute in your head.

MATCHING THE RIGHT RULE
A retrieved section usually lists several items, each with its own rate and its
own cap. Find the line for the specific item in the question and use only that
line. Do not reuse a neighbouring item's rate just because it appeared first.
If the exact item is not listed, say so rather than substituting a similar one.

APPLYING A PERCENTAGE AND A CAP
Rules often have BOTH a percentage AND a cap. Apply them in this order:
  Step 1: compute the percentage of the PURCHASE PRICE.
  Step 2: if that result is greater than the cap, the answer is the cap.
Worked example: an item costs 40000 and its rule is "80% capped at 25000".
  Step 1: 40000 * 0.8 = 32000.
  Step 2: 32000 > 25000, so the answer is 25000.
Never compute a percentage of the cap. The cap is a ceiling, not a base.

MULTIPLE ITEMS
Handle each item on its own: find its rule, compute its percentage, apply its
cap. Only then add the results together. Give the final total as one number.

If the policy documents do not cover the question, say so plainly. Do not guess
or fill gaps with general knowledge.
"""

FINALIZE_PROMPT = """Turn the working below into a final answer.

Respond with ONLY a JSON object. No prose, no markdown fences, no explanation.

{{
  "answer": "the answer in two or three plain sentences",
  "amount": <the final number, digits only, no currency symbol or commas, or null>,
  "supported_by_policy": <true or false>,
  "reasoning": "one line on which rule was applied and how"
}}

"amount": set this whenever the question asks how much, how many, or what
amount. If several items were involved, give the combined TOTAL as a single
number, not the individual parts.

"supported_by_policy": set this to true ONLY if the retrieved text explicitly
addresses the subject of the question. A document being retrieved does not mean
it answers the question — semantic search always returns its closest matches,
relevant or not. If the question's subject is absent from the retrieved text,
set this to false even when a source was cited, and even when you are confident
the real-world answer is no.

Working:
{transcript}
"""


class AgentState(TypedDict):
    """State flows between nodes.

    `add_messages` is a reducer: messages returned by a node get appended
    rather than replacing the list. Without it each node would wipe the
    history and the tool loop would lose its memory.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    answer: PolicyAnswer | None
    tool_loops: int


# ---------------------------------------------------------------------------
# Helpers. Anything your code can do deterministically, it should — never
# delegate to the model what a regex can do reliably.
# ---------------------------------------------------------------------------

def _extract_sources(messages: list[AnyMessage]) -> list[str]:
    """Pull document filenames out of the retrieval tool's output."""
    seen: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            for name in re.findall(r"(\w+\.md)", str(m.content)):
                if name not in seen:
                    seen.append(name)
    return seen


def _extract_json(text: str) -> dict | None:
    """Parse JSON from a model response, tolerating fences and stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _extract_amount(text: str) -> float | None:
    """Last number in the text. Used when the model omits the amount field."""
    nums = re.findall(r"₹?\s*([\d,]+(?:\.\d+)?)", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """Ask the model what to do next.

    The retry exists because some backends occasionally return malformed
    tool_calls (args as a list rather than a dict), which fails validation
    inside LangChain before your code sees it. Retrying usually clears it.
    """
    last_error: Exception | None = None
    for _ in range(AGENT_RETRIES):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {
                "messages": [response],
                "tool_loops": state.get("tool_loops", 0) + 1,
            }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"agent_node failed after {AGENT_RETRIES} attempts: {last_error}"
    )


def should_continue(state: AgentState) -> str:
    """Conditional edge: pending tool call -> tools, otherwise -> finalize.

    The loop cap matters. Without it a confused model will call tools forever.
    """
    if state.get("tool_loops", 0) >= MAX_TOOL_LOOPS:
        return "finalize"

    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    """Force the answer into a validated schema.

    Asking the model to declare `supported_by_policy` makes hallucination a
    field your code can branch on, rather than something invisible inside a
    paragraph of confident prose.
    """
    messages = state["messages"]
    transcript = "\n".join(
        f"{m.type}: {m.content}" for m in messages if getattr(m, "content", None)
    )

    raw = llm.invoke(
        [HumanMessage(content=FINALIZE_PROMPT.format(transcript=transcript))]
    )
    parsed = _extract_json(str(raw.content))

    if parsed is None:
        # Unparseable output. Degrade gracefully rather than crashing the run —
        # a partial answer beats an exception in an eval sweep.
        return {
            "answer": PolicyAnswer(
                answer=str(raw.content)[:500],
                amount=_extract_amount(str(raw.content)),
                sources=_extract_sources(messages),
                supported_by_policy=False,
                reasoning="JSON parsing failed; fell back to raw text.",
            )
        }

    answer_text = str(parsed.get("answer", ""))
    amount = parsed.get("amount")
    if isinstance(amount, str):
        amount = _extract_amount(amount)
    if amount is None:
        amount = _extract_amount(answer_text)

    return {
        "answer": PolicyAnswer(
            answer=answer_text,
            amount=amount,
            sources=_extract_sources(messages),
            supported_by_policy=bool(parsed.get("supported_by_policy", False)),
            reasoning=str(parsed.get("reasoning", "")),
        )
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")  # the loop back
    graph.add_edge("finalize", END)

    return graph.compile()


AGENT = build_graph()


def ask(question: str) -> PolicyAnswer:
    """Single entry point, used by the CLI, the API and the eval harness."""
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
    return state["answer"]