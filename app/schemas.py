"""Output schemas.

Free-text model output is unusable in an application: you cannot branch on it,
test it, or show it in a UI reliably. Forcing the model into a Pydantic schema
turns "a paragraph that might be right" into typed fields your code can check.
"""

from pydantic import BaseModel, Field


class PolicyAnswer(BaseModel):
    """A validated answer to a policy question."""

    answer: str = Field(
        description="The answer in plain language, two or three sentences."
    )
    amount: float | None = Field(
        default=None,
        description="The final monetary amount, if the question asked for one.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Policy document names the answer relied on.",
    )
    supported_by_policy: bool = Field(
        description=(
            "False if the policy documents do not actually answer the "
            "question. Setting this honestly matters more than appearing "
            "helpful."
        )
    )
    reasoning: str = Field(
        default="",
        description="One line on which rule was applied and how.",
    )
