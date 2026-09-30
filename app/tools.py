"""Tools the agent can call.

A "tool" is just a Python function plus a description the model can read.
The @tool decorator turns the signature and docstring into a JSON schema that
gets sent to the model. The model never runs your code — it returns the name
of a function and the arguments, and your code decides whether to execute it.

Write the docstrings for the model, not for humans. They are the only thing
it sees when deciding what to call.
"""

import ast
import operator

from langchain_core.tools import tool

from app.config import TOP_K
from app.ingest import get_vectorstore

# Cached so we don't reopen Chroma on every call.
_store = None


def _get_store():
    global _store
    if _store is None:
        _store = get_vectorstore()
    return _store


@tool
def search_policies(query: str) -> str:
    """Search the company policy documents for relevant rules and numbers.

    Use this whenever a question refers to company policy: expenses,
    reimbursement, travel, leave, or allowances. Pass a focused search phrase
    rather than the user's full sentence.
    """
    results = _get_store().similarity_search(query, k=TOP_K)
    if not results:
        return "No relevant policy sections found."

    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        header = f"[{i}] {source}" + (f" — {section}" if section else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


# Only these operations are allowed through the calculator. Never use eval()
# on model-generated strings — the model is an untrusted input source.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unary operator not allowed")
        return op(_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Use this for any arithmetic — percentages, caps, totals. Do not do mental
    maths. Pass a plain expression such as "4500 * 0.8" or "min(25000, 40000)"
    expressed as arithmetic, for example "4500 * 0.8".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


TOOLS = [search_policies, calculate]
