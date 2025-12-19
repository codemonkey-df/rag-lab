"""
Pydantic models for CRAG structured outputs.

These models define the structure for LLM responses in the CRAG pipeline,
providing type safety and validation.
"""

from pydantic import BaseModel, Field


class RelevanceScoreInput(BaseModel):
    """Relevance score for a single document."""

    relevance_score: float = Field(
        ...,
        description="The relevance score of the document to the query, between 0 and 1",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation for the relevance score (in the same language as the question)",
    )


class QueryRewriterInput(BaseModel):
    """Rewritten query optimized for web search."""

    query: str = Field(
        ...,
        description="The rewritten query optimized for web search (in the same language as the original query)",
    )


class KnowledgeRefinementInput(BaseModel):
    """Key points extracted from a document."""

    key_points: str = Field(
        ...,
        description="Key points extracted from the document (in the same language as the document)",
    )
