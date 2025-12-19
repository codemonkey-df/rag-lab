"""
Pydantic models for Self-RAG structured outputs.

These models define the structure for LLM responses in the Self-RAG pipeline,
providing type safety and validation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    """Decision on whether retrieval is needed for a question."""

    needs_retrieval: bool = Field(
        description="Whether retrieval is needed to answer the question"
    )
    reasoning: str = Field(
        description="Brief explanation for the decision (in the same language as the question)"
    )


class RelevanceEvaluation(BaseModel):
    """Evaluation of document relevance to a question."""

    relevant: bool = Field(
        description="Whether the document is relevant to the question"
    )
    reasoning: str = Field(
        description="Brief explanation for the relevance assessment (in the same language as the question)"
    )


class SupportAssessment(BaseModel):
    """Assessment of how well a response is supported by context."""

    support_level: Literal["Fully supported", "Partially supported", "No support"] = (
        Field(description="Level of support the context provides for the response")
    )
    reasoning: str = Field(
        description="Brief explanation for the support assessment (in the same language as the question)"
    )


class UtilityEvaluation(BaseModel):
    """Evaluation of response utility for answering the question."""

    utility_score: int = Field(
        ge=1, le=5, description="Utility score from 1 (low) to 5 (high)"
    )
    reasoning: str = Field(
        description="Brief explanation for the utility score (in the same language as the question)"
    )
