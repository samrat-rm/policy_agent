"""The agent graph.

Shape:

    START -> agent -> (tools -> agent)* -> finalize -> END

The loop in the middle is the whole idea. The `agent` node asks the model what
to do. If the model wants to call a tool, the conditional edge routes to
`tools`, which executes it and sends the result back to `agent`. The model sees
the result and decides again. That repeats until it has enough to answer.

That is a "multi-step workflow": the number of steps is decided at runtime by
the model, not hardcoded by you. It is also why this needs a graph rather than
a linear chain — a chain cannot loop back on itself.

The `finalize` node then forces the answer into a validated schema, which is
where the guardrail lives.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import LLM_MODEL, OLLAMA_BASE_URL
from app.schemas import PolicyAnswer
from app.tools import TOOLS

MAX_TOOL_LOOPS = 6

SYSTEM_PROMPT = """You are a company policy assistant.

Rules you must follow:
- Always call search_policies before answering any question about company \
policy. Never answer from memory.
- Always call calculate for arithmetic. Do not compute in your head.
- Many questions need both: find the rule, then apply it to the numbers given.
- Watch for caps. A percentage reimbursement is often capped at a maximum \
amount, so compute the percentage and then apply the cap.
- If the policy documents do not cover the question, say so plainly. Do not \
guess or fill gaps with general knowledge.
"""


class AgentState(TypedDict):
    """State is the data that flows between nodes.

    `add_messages` is a reducer: when a node returns messages, they are
    appended to the list rather than replacing it. Without it, each node would
    wipe the conversation history and the tool loop would lose its memory.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    answer: PolicyAnswer | None
    tool_loops: int


llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Ask the model what to do next. It either calls a tool or replies."""
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "tool_loops": state.get("tool_loops", 0) + 1,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge: tool call pending -> tools, otherwise -> finalize.

    The loop guard matters. Without a cap, a confused model can call tools
    forever and burn your budget. Every agent you ship needs one.
    """
    if state.get("tool_loops", 0) >= MAX_TOOL_LOOPS:
        return "finalize"

    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    """Force the answer into a validated schema.

    This is the guardrail. Asking the model to declare `supported_by_policy`
    and to cite sources makes hallucination visible to your code instead of
    invisible inside a paragraph of confident prose. Your application can then
    refuse, log, or escalate.
    """
    structured = llm.with_structured_output(PolicyAnswer)

    transcript = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if m.content
    )
    result = structured.invoke(
        [
            SystemMessage(
                content=(
                    "Turn the working below into a final answer. Set "
                    "supported_by_policy to false if the retrieved policy text "
                    "does not actually answer the question. Cite the document "
                    "names you used."
                )
            ),
            HumanMessage(content=transcript),
        ]
    )
    return {"answer": result}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")  # the loop back
    graph.add_edge("finalize", END)

    return graph.compile()


AGENT = build_graph()


def ask(question: str) -> PolicyAnswer:
    """Single entry point. Used by the CLI, the API, and the eval harness."""
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