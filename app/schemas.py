"""Output schemas.

Free-text model output is unusable in an application: you cannot branch on it,
test it, or render it reliably. A Pydantic schema turns "a paragraph that might
be right" into typed fields your code can check.

Note there are no defaults on `amount` or `sources`. A field with a default is
optional in the generated JSON schema, and smaller models routinely skip
optional fields entirely. Making them required forces the key to be emitted.
"""

from pydantic import BaseModel, Field


class PolicyAnswer(BaseModel):
    """A validated answer to a policy question."""

    answer: str = Field(
        description="The answer in plain language, two or three sentences."
    )
    amount: float | None = Field(
        description=(
            "The final numeric answer. REQUIRED whenever the question asks "
            "how much, how many, or what amount. Null ONLY when the question "
            "has no numeric answer. Digits only, no currency symbol or commas."
        )
    )
    sources: list[str] = Field(
        description="Policy file names used, e.g. ['expense_policy.md']."
    )
    supported_by_policy: bool = Field(
        description=(
            "False if the policy documents do not actually answer the "
            "question. Setting this honestly matters more than appearing "
            "helpful."
        )
    )
    reasoning: str = Field(
        default="", description="One line on which rule was applied and how."
    )